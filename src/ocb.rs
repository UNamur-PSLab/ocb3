//! OCB3 encryption/decryption (x86-64 SIMD implementation).
//!
//! # Compile-time feature detection
//!
//! This module is only compiled on x86_64 targets built with the `avx` and
//! `aes` target features. The check happens at compile time in `lib.rs`
//! (`#[cfg(target_feature = ...)]`): on any other target the
//! crate fails to build with an explanatory message instead of emitting code
//! that would trap at run time.
use crate::DecryptIO;
use crate::EncryptIO;
use crate::OCB_TAG_LEN;
use crate::Placement;
use crate::PlainText;
use crate::Result;
use crate::aes::AesKey;
use crate::avx::*;
use core::arch::x86_64::*;
use core::sync::atomic::{AtomicU64, Ordering, compiler_fence};

const L_TABLE_SZ: usize = 16;
const BPI: usize = 8;
const PAGE_SIZE_4K: usize = 4096;
const PAGE_OFFSET_MASK: usize = PAGE_SIZE_4K - 1;

/// OCB3 context: the AES-128 key schedule (encrypt and decrypt round keys)
/// plus the key-derived `L*`, `L$` and `L` table used to build the offsets.
#[repr(C, align(16))]
pub struct AEctx {
    lstar: __m128i,
    ldollar: __m128i,
    l: [__m128i; L_TABLE_SZ],
    decrypt_key: AesKey,
    encrypt_key: AesKey,
    ktop: KtopCache,
}

struct KtopCache {
    seq: AtomicU64,
    cached_top: [AtomicU64; 2],
    ktop_str: [AtomicU64; 3],
}

impl Default for KtopCache {
    fn default() -> Self {
        Self {
            seq: AtomicU64::new(0),
            cached_top: [AtomicU64::new(0), AtomicU64::new(0)],
            ktop_str: [AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0)],
        }
    }
}

impl AEctx {
    /// Creates an OCB3 context from a 16-byte AES-128 key.
    #[inline]
    pub fn new(key: &[u8]) -> Self {
        unsafe { Self::new_inner(key) }
    }

    #[target_feature(enable = "avx,aes")]
    unsafe fn new_inner(key: &[u8]) -> Self {
        // Safety: needs to be 16-bytes aligned.
        let zero = zero_block();
        let mut ctx = AEctx {
            lstar: zero,
            ldollar: zero,
            l: [zero; L_TABLE_SZ],
            decrypt_key: AesKey::default(),
            encrypt_key: AesKey::default(),
            ktop: KtopCache::default(),
        };

        // Key dependent values
        ctx.encrypt_key.aes_128_key_expansion(key);
        ctx.decrypt_key.aes_set_key_fast(&ctx.encrypt_key);
        ctx.lstar = ctx.encrypt_key.aes_encrypt_block(zero);
        let mut tmp_blk = swap_if_le(ctx.lstar);
        tmp_blk = double(tmp_blk);
        ctx.ldollar = swap_if_le(tmp_blk);
        tmp_blk = double(tmp_blk);
        ctx.l[0] = swap_if_le(tmp_blk);

        for i in 1..L_TABLE_SZ {
            tmp_blk = double(tmp_blk);
            ctx.l[i] = swap_if_le(tmp_blk);
        }

        ctx
    }

    #[inline(always)]
    fn gen_offset_from_nonce(&self, nonce: &[u8; 12]) -> __m128i {
        let n_hi = unsafe { core::ptr::read_unaligned(nonce.as_ptr() as *const u32) };
        let n_lo = unsafe { core::ptr::read_unaligned(nonce.as_ptr().add(4) as *const u64) };
        let idx = (nonce[11] & 0x3f) as i32;
        let lo_qword = 0x0100_0000u64 | ((n_hi as u64) << 32);
        let hi_qword = n_lo & !(0x3fu64 << 56);

        let seq1 = self.ktop.seq.load(Ordering::Acquire);
        // 63 over 64 times, this should resolve to true:
        if (seq1 & 1) == 0 {
            let top0 = self.ktop.cached_top[0].load(Ordering::Relaxed);
            let top1 = self.ktop.cached_top[1].load(Ordering::Relaxed);
            if top0 == lo_qword && top1 == hi_qword {
                let k0 = self.ktop.ktop_str[0].load(Ordering::Relaxed);
                let k1 = self.ktop.ktop_str[1].load(Ordering::Relaxed);
                let k2 = self.ktop.ktop_str[2].load(Ordering::Relaxed);
                compiler_fence(Ordering::Acquire);
                if self.ktop.seq.load(Ordering::Relaxed) == seq1 {
                    return gen_offset(&[k0, k1, k2], idx);
                }
            }
        }

        // Slow path, compute ktop, update and gen offset
        let mut bl = unsafe { _mm_set_epi64x(hi_qword as i64, lo_qword as i64) };
        bl = self.encrypt_key.aes_encrypt_block(bl);
        let [k0, k1] = m128i_to_bswap_u64x2(bl);
        let k2 = k0 ^ (k0 << 8) ^ (k1 >> 56);
        let ktop_str = [k0, k1, k2];

        // Cache update is no writer active.
        if (seq1 & 1) == 0
            && self
                .ktop
                .seq
                .compare_exchange_weak(seq1, seq1 + 1, Ordering::Acquire, Ordering::Relaxed)
                .is_ok()
        {
            self.ktop.cached_top[0].store(lo_qword, Ordering::Relaxed);
            self.ktop.cached_top[1].store(hi_qword, Ordering::Relaxed);
            self.ktop.ktop_str[0].store(k0, Ordering::Relaxed);
            self.ktop.ktop_str[1].store(k1, Ordering::Relaxed);
            self.ktop.ktop_str[2].store(k2, Ordering::Relaxed);
            self.ktop.seq.store(seq1 + 2, Ordering::Release);
        }

        gen_offset(&ktop_str, idx)
    }

    #[inline(always)]
    fn load_partial_block(ptr: *const u8, len: usize) -> __m128i {
        // On x86-64, virtual memory pages are at least 4 KiB. If the 16-byte load stays
        // within the same page, overreading partial bytes cannot cross into an unmapped
        // page and will never cause a page fault. Otherwise we safely copy up to page
        // boundary.
        if ((ptr as usize) & PAGE_OFFSET_MASK) <= PAGE_SIZE_4K - 16 {
            loadu_block(ptr as *const __m128i)
        } else {
            load_padded_tail_cold(ptr, len)
        }
    }

    #[inline(always)]
    fn load_tail_padded(ptr: *const u8, len: usize, mask: __m128i) -> __m128i {
        // On x86-64, virtual memory pages are at least 4 KiB. If the 16-byte load stays
        // within the same page, overreading partial bytes cannot cross into an unmapped
        // page and will never cause a page fault. Otherwise we safely copy up to page
        // boundary.
        if ((ptr as usize) & PAGE_OFFSET_MASK) <= PAGE_SIZE_4K - 16 {
            let raw = loadu_block(ptr as *const __m128i);
            unsafe { _mm_or_si128(_mm_and_si128(raw, mask), TAIL_PAD_80.get(len)) }
        } else {
            load_padded_tail_cold(ptr, len)
        }
    }

    #[allow(clippy::pointers_in_nomem_asm_block)]
    #[inline(never)]
    fn process_ad(&self, ad: &[u8]) -> __m128i {
        let num_bpi = ad.len() / (BPI * 16);
        let mut ad_blocks_num: usize = 0;
        let mut ad_ptr = ad.as_ptr();
        let mut ad_offset = zero_block();
        let mut ad_checksum = zero_block();

        for _ in 0..num_bpi {
            ad_blocks_num += BPI;
            let tz = ad_blocks_num.trailing_zeros() as usize;
            let mut l_ptr = self.l.as_ptr();
            // Keep l_ptr in register.
            unsafe {
                core::arch::asm!(
                    "/* {0} */",
                    inout(reg) l_ptr,
                    options(pure, nomem, nostack, preserves_flags)
                );
            }
            let mut rk_ptr = self.encrypt_key.rd_key.as_ptr();
            let k0 = load_block(rk_ptr);
            let l0 = load_block(l_ptr);
            // SAFETY: self.l contains 16 elements; we are accessing the first 3.
            let (l1, l2) = unsafe { (load_block(l_ptr.add(1)), load_block(l_ptr.add(2))) };

            let offset_k = xor_block(ad_offset, k0);
            let oa0_k = xor_block(offset_k, l0);
            let oa1_k = xor_block(oa0_k, l1);
            let oa2_k = xor_block(offset_k, l1);
            let oa3_k = xor_block(oa2_k, l2);
            let oa4_k = xor_block(oa1_k, l2);
            let oa5_k = xor_block(oa0_k, l2);
            let oa6_k = xor_block(offset_k, l2);
            // SAFETY: `ad_blocks_num < 2^16` in OCB3, ensuring `tz < 16` and in-bounds for `self.l`.
            let l_tz = unsafe { *l_ptr.add(tz) };
            ad_offset = xor_block(xor_block(ad_offset, l2), l_tz);
            let oa7_k = xor_block(oa6_k, l_tz);

            let b0 = xor_block(oa0_k, loadu_block(ad_ptr as *const __m128i));
            let b1 = xor_block(oa1_k, loadu_block_with_offset(ad_ptr, 16));
            let b2 = xor_block(oa2_k, loadu_block_with_offset(ad_ptr, 32));
            let b3 = xor_block(oa3_k, loadu_block_with_offset(ad_ptr, 48));
            let b4 = xor_block(oa4_k, loadu_block_with_offset(ad_ptr, 64));
            let b5 = xor_block(oa5_k, loadu_block_with_offset(ad_ptr, 80));
            let b6 = xor_block(oa6_k, loadu_block_with_offset(ad_ptr, 96));
            let b7 = xor_block(oa7_k, loadu_block_with_offset(ad_ptr, 112));

            unsafe {
                core::arch::asm!(
                    "/* {0} {1} */",
                    inout(xmm_reg) ad_offset,
                    inout(reg) rk_ptr,
                        options(pure, nomem, nostack, preserves_flags)
                );
            }

            let ct =
                AesKey::aes_encrypt_8_blocks_prewhitened(rk_ptr, [b0, b1, b2, b3, b4, b5, b6, b7]);
            let sum0 = xor_block(xor_block(ct[0], ct[1]), xor_block(ct[2], ct[3]));
            let sum1 = xor_block(xor_block(ct[4], ct[5]), xor_block(ct[6], ct[7]));
            ad_checksum = xor_block(ad_checksum, xor_block(sum0, sum1));

            unsafe {
                ad_ptr = ad_ptr.add(128);
            }
        }

        let l0 = self.l[0];
        let l1 = self.l[1];
        let l2 = self.l[2];
        let mut remaining = ad.len() % (BPI * 16);
        if remaining >= 64 {
            let oa0 = xor_block(ad_offset, l0);
            let oa1 = xor_block(oa0, l1);
            let oa2 = xor_block(ad_offset, l1);
            let oa3 = xor_block(oa2, l2);
            ad_offset = oa3;

            let ad0 = loadu_block(ad_ptr as *const __m128i);
            let ad1 = loadu_block_with_offset(ad_ptr, 16);
            let ad2 = loadu_block_with_offset(ad_ptr, 32);
            let ad3 = loadu_block_with_offset(ad_ptr, 48);

            let ct = self.encrypt_key.aes_encrypt_4_blocks([
                xor_block(oa0, ad0),
                xor_block(oa1, ad1),
                xor_block(oa2, ad2),
                xor_block(oa3, ad3),
            ]);
            let sum = xor_block(xor_block(ct[0], ct[1]), xor_block(ct[2], ct[3]));
            ad_checksum = xor_block(ad_checksum, sum);
            remaining -= 64;
            // SAFETY: in range since remaining >= 64
            unsafe {
                ad_ptr = ad_ptr.add(64);
            }
        }
        if remaining >= 32 {
            // Block k+1 (odd: ntz is 0) and block k+2 (even: k is 0 or 4, so k+2 is 2 or 6, ntz is 1)
            let oa0 = xor_block(ad_offset, l0);
            let oa1 = xor_block(oa0, l1);
            ad_offset = oa1;

            let ad0 = loadu_block(ad_ptr as *const __m128i);
            let ad1 = loadu_block_with_offset(ad_ptr, 16);
            let ct = self
                .encrypt_key
                .aes_encrypt_2_blocks([xor_block(oa0, ad0), xor_block(oa1, ad1)]);
            ad_checksum = xor_block(ad_checksum, xor_block(ct[0], ct[1]));
            // SAFETY: in range since remaining >= 32
            unsafe {
                ad_ptr = ad_ptr.add(32);
            }
            remaining -= 32;
        }
        if remaining >= 16 {
            ad_offset = xor_block(ad_offset, l0);
            let ad0 = loadu_block(ad_ptr as *const __m128i);
            let ta0 = xor_block(ad_offset, ad0);
            let ct = self.encrypt_key.aes_encrypt_block(ta0);
            ad_checksum = xor_block(ad_checksum, ct);
            // SAFETY: in range since remaining >= 16
            unsafe {
                ad_ptr = ad_ptr.add(16);
            }
            remaining -= 16;
        }
        if remaining > 0 {
            let mask = TAIL_STORE_MASK.get(remaining);
            let ad0 = Self::load_tail_padded(ad_ptr, remaining, mask);
            ad_offset = xor_block(ad_offset, self.lstar);
            let ta = xor_block(ad_offset, ad0);
            let ct = self.encrypt_key.aes_encrypt_block(ta);
            ad_checksum = xor_block(ad_checksum, ct);
        }
        ad_checksum
    }

    #[inline(always)]
    fn enc_block(&self, oa: __m128i, pt_ptr: *const u8, ct_ptr: *mut u8, checksum: &mut __m128i) {
        let pt = loadu_block(pt_ptr as *const __m128i);
        let t = xor_block(oa, pt);
        let ct = self.encrypt_key.aes_encrypt_block(t);
        storeu_block(ct_ptr as *mut __m128i, xor_block(ct, oa));
        *checksum = xor_block(*checksum, pt);
    }

    #[inline(always)]
    fn enc_single_block_local(
        &self,
        pt_ptr: *const u8,
        ct_ptr: *mut u8,
        block_num: &mut u32,
        offset: __m128i,
        checksum: &mut __m128i,
    ) -> __m128i {
        *block_num += 1;
        let tz = block_num.trailing_zeros() as usize;
        // SAFETY: `block_num <= MAX_PT_SIZE / 16 < 2^16`, ensuring `ntz < 16` and in-bounds for `self.l`.
        let oa = xor_block(offset, table_get(&self.l, tz));
        self.enc_block(oa, pt_ptr, ct_ptr, checksum);
        oa
    }

    /// Encrypts `num_bpi` groups of 8 blocks (`128` bytes each) starting at an 8-block aligned
    /// `block_num`. Uses all 16 XMM registers with zero stack spills via prewhitened offsets
    /// and a telescoping offset-whitened checksum.
    #[allow(clippy::pointers_in_nomem_asm_block)]
    #[target_feature(enable = "avx,aes")]
    #[inline(never)]
    unsafe fn enc_bpi_loop(
        &self,
        mut pt_ptr: *const u8,
        mut ct_ptr: *mut u8,
        num_bpi: usize,
        mut block_num: u32,
        mut offset: __m128i,
        mut checksum: __m128i,
    ) -> (__m128i, __m128i) {
        debug_assert_eq!(block_num % 8, 0);
        let init_offset = offset;

        for _ in 0..num_bpi {
            block_num += BPI as u32;
            let tz = block_num.trailing_zeros() as usize;
            let mut l_ptr = self.l.as_ptr();
            unsafe {
                // l0, l1, l2 are loop invariant but must not be declared outside of the loop
                // or the compiler considers them living through the whole loop as it should be,
                // but that creates register pressure. Instead, we load them before we encrypt,
                // and reload them again when needed after the encryption. This way, we never
                // spill registers on the stack.
                //
                // The asm below is no-op, but prevents the compiler to put l0, l1, l2 outside of
                // the loop. This is however brittle as it depends for the *current* compiler
                // heuristics ...
                // TODO Emit the whole loop as asm?
                core::arch::asm!(
                    "/* {0} */",
                    inout(reg) l_ptr,
                    // This asm does nothing, so tell the compiler.
                    options(pure, nomem, nostack, preserves_flags)
                );
            }
            let mut rk_ptr = self.encrypt_key.rd_key.as_ptr();
            let k0 = load_block(rk_ptr);
            let l0 = load_block(l_ptr);
            let (l1, l2) = unsafe { (load_block(l_ptr.add(1)), load_block(l_ptr.add(2))) };

            let ok = xor_block(offset, k0);
            let oa0_k = xor_block(ok, l0);
            let oa1_k = xor_block(oa0_k, l1);
            let oa2_k = xor_block(ok, l1);
            let oa3_k = xor_block(oa2_k, l2);
            let oa4_k = xor_block(oa1_k, l2);
            let oa5_k = xor_block(oa0_k, l2);
            let oa6_k = xor_block(ok, l2);
            // SAFETY: `block_num <= MAX_PT_SIZE / 16 < 2^16`, ensuring `tz < 16` and in-bounds for `self.l`.
            let l_tz = unsafe { *l_ptr.add(tz) };
            let oa7_k = xor_block(oa6_k, l_tz);

            let b0 = xor_block(oa0_k, loadu_block(pt_ptr as *const __m128i));
            let b1 = xor_block(oa1_k, loadu_block_with_offset(pt_ptr, 16));
            let b2 = xor_block(oa2_k, loadu_block_with_offset(pt_ptr, 32));
            let b3 = xor_block(oa3_k, loadu_block_with_offset(pt_ptr, 48));
            let b4 = xor_block(oa4_k, loadu_block_with_offset(pt_ptr, 64));
            let b5 = xor_block(oa5_k, loadu_block_with_offset(pt_ptr, 80));
            let b6 = xor_block(oa6_k, loadu_block_with_offset(pt_ptr, 96));
            let b7 = xor_block(oa7_k, loadu_block_with_offset(pt_ptr, 112));

            // Telescoping checksum: b0 ^ .. ^ b7 == (pt0 ^ .. ^ pt7) ^ offset ^ oa7.
            let sum0 = xor_block(xor_block(b0, b1), xor_block(b2, b3));
            let sum1 = xor_block(xor_block(b4, b5), xor_block(b6, b7));
            checksum = xor_block(checksum, xor_block(sum0, sum1));
            unsafe {
                core::arch::asm!(
                    "/* {0} {1} */",
                    inout(xmm_reg) checksum,
                    inout(reg) rk_ptr,
                    // This asm does nothing, so tell the compiler.
                    options(pure, nomem, nostack, preserves_flags)
                );
            }

            let ct =
                AesKey::aes_encrypt_8_blocks_prewhitened(rk_ptr, [b0, b1, b2, b3, b4, b5, b6, b7]);

            // Reloading l0, l1, l2 again, reusing registers, and heuristically
            // preventing LLVM to move these variables outside of the loop.
            let mut l_ptr2 = self.l.as_ptr();
            unsafe {
                core::arch::asm!(
                    "/* {0} */",
                    inout(reg) l_ptr2,
                    options(pure, nomem, nostack, preserves_flags)
                );
            }
            let l0 = load_block(l_ptr2);
            let (l1, l2) = unsafe { (load_block(l_ptr2.add(1)), load_block(l_ptr2.add(2))) };

            let oa0 = xor_block(offset, l0);
            let oa1 = xor_block(oa0, l1);
            let oa2 = xor_block(offset, l1);
            let oa3 = xor_block(oa2, l2);
            let oa4 = xor_block(oa1, l2);
            let oa5 = xor_block(oa0, l2);
            let oa6 = xor_block(offset, l2);
            let oa7 = unsafe { xor_block(oa6, *l_ptr2.add(tz)) };

            storeu_block(ct_ptr as *mut __m128i, xor_block(ct[0], oa0));
            storeu_block_with_offset(ct_ptr, xor_block(ct[1], oa1), 16);
            storeu_block_with_offset(ct_ptr, xor_block(ct[2], oa2), 32);
            storeu_block_with_offset(ct_ptr, xor_block(ct[3], oa3), 48);
            storeu_block_with_offset(ct_ptr, xor_block(ct[4], oa4), 64);
            storeu_block_with_offset(ct_ptr, xor_block(ct[5], oa5), 80);
            storeu_block_with_offset(ct_ptr, xor_block(ct[6], oa6), 96);
            storeu_block_with_offset(ct_ptr, xor_block(ct[7], oa7), 112);

            offset = oa7;
            unsafe {
                pt_ptr = pt_ptr.add(BPI * 16);
                ct_ptr = ct_ptr.add(BPI * 16);
            }
        }

        checksum = xor_block(checksum, xor_block(init_offset, offset));
        (offset, checksum)
    }

    /// Decrypts `num_bpi` groups of 8 blocks (`128` bytes each) using all 16 XMM
    /// registers
    #[allow(clippy::pointers_in_nomem_asm_block)]
    #[inline(never)]
    fn dec_bpi_loop(
        &self,
        mut ct_ptr: *const u8,
        mut pt_ptr: *mut u8,
        num_bpi: usize,
        mut offset: __m128i,
    ) -> (__m128i, __m128i) {
        let mut block_num = 0u32;
        let init_offset = offset;
        let mut checksum = zero_block();

        for _ in 0..num_bpi {
            block_num += BPI as u32;
            let tz = block_num.trailing_zeros() as usize;
            let mut l_ptr = self.l.as_ptr();
            // Same as in enc_bpi_loop; we try to prevent LLVM
            // to move loop independent variables up the loop.
            unsafe {
                core::arch::asm!(
                    "/* {0} */",
                    inout(reg) l_ptr,
                    // This asm does nothing, so tell the compiler.
                    options(pure, nomem, nostack, preserves_flags)
                );
            }
            let mut rk_ptr = self.decrypt_key.rd_key.as_ptr();
            let k0 = load_block(rk_ptr);
            let l0 = load_block(l_ptr);
            let (l1, l2) = unsafe { (load_block(l_ptr.add(1)), load_block(l_ptr.add(2))) };

            let ok = xor_block(offset, k0);
            let oa0_k = xor_block(ok, l0);
            let oa1_k = xor_block(oa0_k, l1);
            let oa2_k = xor_block(ok, l1);
            let oa3_k = xor_block(oa2_k, l2);
            let oa4_k = xor_block(oa1_k, l2);
            let oa5_k = xor_block(oa0_k, l2);
            let oa6_k = xor_block(ok, l2);
            // SAFETY: `block_num <= MAX_PT_SIZE / 16 < 2^16`, ensuring `tz < 16`
            // and in-bounds for `self.l`.
            let l_tz = unsafe { *l_ptr.add(tz) };

            let oa7_k = xor_block(oa6_k, l_tz);

            let b0 = xor_block(oa0_k, loadu_block(ct_ptr as *const __m128i));
            let b1 = xor_block(oa1_k, loadu_block_with_offset(ct_ptr, 16));
            let b2 = xor_block(oa2_k, loadu_block_with_offset(ct_ptr, 32));
            let b3 = xor_block(oa3_k, loadu_block_with_offset(ct_ptr, 48));
            let b4 = xor_block(oa4_k, loadu_block_with_offset(ct_ptr, 64));
            let b5 = xor_block(oa5_k, loadu_block_with_offset(ct_ptr, 80));
            let b6 = xor_block(oa6_k, loadu_block_with_offset(ct_ptr, 96));
            let b7 = xor_block(oa7_k, loadu_block_with_offset(ct_ptr, 112));

            unsafe {
                core::arch::asm!(
                    "/* {0} {1} {2} */",
                    inout(xmm_reg) offset,
                    inout(xmm_reg) checksum,
                    inout(reg) rk_ptr,
                    options(pure, nomem, nostack, preserves_flags)
                );
            }

            let d =
                AesKey::aes_decrypt_8_blocks_prewhitened(rk_ptr, [b0, b1, b2, b3, b4, b5, b6, b7]);

            let sum0 = xor_block(xor_block(d[0], d[1]), xor_block(d[2], d[3]));
            let sum1 = xor_block(xor_block(d[4], d[5]), xor_block(d[6], d[7]));
            checksum = xor_block(checksum, xor_block(sum0, sum1));

            let mut l_ptr2 = self.l.as_ptr();
            unsafe {
                core::arch::asm!(
                    "/* {0} */",
                    inout(reg) l_ptr2,
                    options(pure, nomem, nostack, preserves_flags)
                );
            }
            let l0 = load_block(l_ptr2);
            let (l1, l2) = unsafe { (load_block(l_ptr2.add(1)), load_block(l_ptr2.add(2))) };

            let oa0 = xor_block(offset, l0);
            let oa1 = xor_block(oa0, l1);
            let oa2 = xor_block(offset, l1);
            let oa3 = xor_block(oa2, l2);
            let oa4 = xor_block(oa1, l2);
            let oa5 = xor_block(oa0, l2);
            let oa6 = xor_block(offset, l2);
            let oa7 = unsafe { xor_block(oa6, *l_ptr2.add(tz)) };

            storeu_block(pt_ptr as *mut __m128i, xor_block(d[0], oa0));
            storeu_block_with_offset(pt_ptr, xor_block(d[1], oa1), 16);
            storeu_block_with_offset(pt_ptr, xor_block(d[2], oa2), 32);
            storeu_block_with_offset(pt_ptr, xor_block(d[3], oa3), 48);
            storeu_block_with_offset(pt_ptr, xor_block(d[4], oa4), 64);
            storeu_block_with_offset(pt_ptr, xor_block(d[5], oa5), 80);
            storeu_block_with_offset(pt_ptr, xor_block(d[6], oa6), 96);
            storeu_block_with_offset(pt_ptr, xor_block(d[7], oa7), 112);

            offset = oa7;
            unsafe {
                ct_ptr = ct_ptr.add(128);
                pt_ptr = pt_ptr.add(128);
            }
        }

        checksum = xor_block(checksum, xor_block(init_offset, offset));
        (offset, checksum)
    }

    /// Encrypts `full_blocks` starting from an 8-block aligned `block_num`, using the 8-block
    /// loop followed by `4 -> 2 -> 1` parallel leftover blocks with constant offsets.
    #[inline(always)]
    fn enc_aligned_blocks(
        &self,
        mut pt_ptr: *const u8,
        mut ct_ptr: *mut u8,
        full_blocks: usize,
        block_num: &mut u32,
        mut offset: __m128i,
        checksum: &mut __m128i,
    ) -> (*const u8, *mut u8, __m128i) {
        let num_bpi = full_blocks / BPI;
        if num_bpi > 0 {
            unsafe {
                (offset, *checksum) =
                    self.enc_bpi_loop(pt_ptr, ct_ptr, num_bpi, *block_num, offset, *checksum);
                let bytes = num_bpi * (BPI * 16);
                pt_ptr = pt_ptr.add(bytes);
                ct_ptr = ct_ptr.add(bytes);
            }
            *block_num += (num_bpi * BPI) as u32;
        }
        let mut rem_blocks = full_blocks % BPI;
        *block_num += rem_blocks as u32;

        let l0 = self.l[0];
        let l1 = self.l[1];
        let l2 = self.l[2];

        if rem_blocks >= 4 {
            let oa0 = xor_block(offset, l0);
            let oa1 = xor_block(oa0, l1);
            let oa2 = xor_block(offset, l1);
            let oa3 = xor_block(oa2, l2);
            offset = oa3;

            let pt0 = loadu_block(pt_ptr as *const __m128i);
            let pt1 = loadu_block_with_offset(pt_ptr, 16);
            let pt2 = loadu_block_with_offset(pt_ptr, 32);
            let pt3 = loadu_block_with_offset(pt_ptr, 48);

            let pt_sum = xor_block(xor_block(pt0, pt1), xor_block(pt2, pt3));
            *checksum = xor_block(*checksum, pt_sum);

            let ct = self.encrypt_key.aes_encrypt_4_blocks([
                xor_block(oa0, pt0),
                xor_block(oa1, pt1),
                xor_block(oa2, pt2),
                xor_block(oa3, pt3),
            ]);

            storeu_block(ct_ptr as *mut __m128i, xor_block(ct[0], oa0));
            storeu_block_with_offset(ct_ptr, xor_block(ct[1], oa1), 16);
            storeu_block_with_offset(ct_ptr, xor_block(ct[2], oa2), 32);
            storeu_block_with_offset(ct_ptr, xor_block(ct[3], oa3), 48);

            unsafe {
                pt_ptr = pt_ptr.add(64);
                ct_ptr = ct_ptr.add(64);
            }
            rem_blocks -= 4;
        }
        if rem_blocks >= 2 {
            let oa0 = xor_block(offset, l0);
            let oa1 = xor_block(oa0, l1);
            offset = oa1;

            let pt0 = loadu_block(pt_ptr as *const __m128i);
            let pt1 = loadu_block_with_offset(pt_ptr, 16);
            *checksum = xor_block(*checksum, xor_block(pt0, pt1));

            let ct = self
                .encrypt_key
                .aes_encrypt_2_blocks([xor_block(oa0, pt0), xor_block(oa1, pt1)]);

            storeu_block(ct_ptr as *mut __m128i, xor_block(ct[0], oa0));
            storeu_block_with_offset(ct_ptr, xor_block(ct[1], oa1), 16);

            unsafe {
                pt_ptr = pt_ptr.add(32);
                ct_ptr = ct_ptr.add(32);
            }
            rem_blocks -= 2;
        }
        if rem_blocks >= 1 {
            offset = xor_block(offset, l0);
            self.enc_block(offset, pt_ptr, ct_ptr, checksum);
            unsafe {
                pt_ptr = pt_ptr.add(16);
                ct_ptr = ct_ptr.add(16);
            }
        }

        (pt_ptr, ct_ptr, offset)
    }

    #[inline(always)]
    fn enc_unaligned_contiguous(
        &self,
        mut pt_ptr: *const u8,
        mut ct_ptr: *mut u8,
        mut full_blocks: usize,
        block_num: &mut u32,
        mut offset: __m128i,
        checksum: &mut __m128i,
    ) -> (*const u8, *mut u8, __m128i) {
        while !(*block_num).is_multiple_of(BPI as u32) && full_blocks > 0 {
            offset = self.enc_single_block_local(pt_ptr, ct_ptr, block_num, offset, checksum);
            //SAFETY: full_blocks indicate how many 16-bytes blocks we may advance until
            //we end up to a multiple of BPI.
            unsafe {
                pt_ptr = pt_ptr.add(16);
                ct_ptr = ct_ptr.add(16);
            }
            full_blocks -= 1;
        }

        if full_blocks > 0 {
            (pt_ptr, ct_ptr, offset) =
                self.enc_aligned_blocks(pt_ptr, ct_ptr, full_blocks, block_num, offset, checksum);
        }

        (pt_ptr, ct_ptr, offset)
    }

    /// Encrypts the final partial block (`len < 16`) with `0x80` padding and L*.
    /// Returns the updated `(ct_ptr, offset, checksum)`.
    #[inline(always)]
    fn enc_tail(
        &self,
        data_ptr: *const u8,
        data_len: usize,
        ct_ptr: *mut u8,
        offset: __m128i,
        checksum: __m128i,
    ) -> (*mut u8, __m128i, __m128i) {
        let mask = TAIL_STORE_MASK.get(data_len);
        let pt_padded = Self::load_tail_padded(data_ptr, data_len, mask);
        let checksum = xor_block(checksum, pt_padded);
        let offset = xor_block(offset, self.lstar);
        let pad = self.encrypt_key.aes_encrypt_block(offset);
        let ct_block = xor_block(pt_padded, pad);

        //SAFETY: the ciphertext slice is required to be 16 bytes longer than the plaintext, so
        //this cannot fall outside of the allocation. In case of InPlace encryption, this design
        //encrypts up to the size of the input buffer - 16, so this is safe to do.
        let old = loadu_block(ct_ptr as *const __m128i);
        let blended = blendv_block(old, ct_block, mask);
        storeu_block(ct_ptr as *mut __m128i, blended);

        (unsafe { ct_ptr.add(data_len) }, offset, checksum)
    }

    #[allow(clippy::too_many_arguments)]
    #[inline(always)]
    fn enc_scatter(
        &self,
        mut pt1_ptr: *const u8,
        mut pt2_ptr: *const u8,
        pt1_len: usize,
        pt2_len: usize,
        mut ct_ptr: *mut u8,
        mut offset: __m128i,
        mut checksum: __m128i,
        mut block_num: u32,
        ad: &[u8],
    ) -> Result<()> {
        let full_blocks_pt1 = pt1_len / 16;
        let rem_pt1 = pt1_len & 15;

        (pt1_ptr, ct_ptr, offset) = self.enc_aligned_blocks(
            pt1_ptr,
            ct_ptr,
            full_blocks_pt1,
            &mut block_num,
            offset,
            &mut checksum,
        );

        if rem_pt1 > 0 {
            if pt2_len == 0 {
                (ct_ptr, offset, checksum) =
                    self.enc_tail(pt1_ptr, rem_pt1, ct_ptr, offset, checksum);
            } else if rem_pt1 + pt2_len < 16 {
                let pt2b = Self::load_tail_padded(pt2_ptr, pt2_len, TAIL_STORE_MASK.get(pt2_len));
                // Shift pt2b left by rem_pt1 bytes and blend with pt1 using the
                // same control mask
                let shuf = shift_left_mask(rem_pt1);
                let pt2b = shuffle_block(pt2b, shuf);
                let pt1 = Self::load_partial_block(pt1_ptr, rem_pt1);
                let blended = blendv_block(pt2b, pt1, shuf);
                checksum = xor_block(checksum, blended);
                offset = xor_block(offset, self.lstar);
                let pad = self.encrypt_key.aes_encrypt_block(offset);
                let ct_block = xor_block(blended, pad);
                storeu_block(ct_ptr as *mut __m128i, ct_block);
                unsafe {
                    ct_ptr = ct_ptr.add(rem_pt1 + pt2_len);
                }
            } else {
                // Reassemble one full block from pt1 tail + pt2 head
                let need = 16 - rem_pt1;
                let shuf = shift_left_mask(rem_pt1);
                let pt2b = shuffle_block(Self::load_partial_block(pt2_ptr, need), shuf);
                let pt1 = Self::load_partial_block(pt1_ptr, rem_pt1);
                let block = blendv_block(pt2b, pt1, shuf);

                block_num += 1;
                let tz = block_num.trailing_zeros() as usize;
                offset = xor_block(offset, table_get(&self.l, tz));
                let ct = self.encrypt_key.aes_encrypt_block(xor_block(offset, block));
                storeu_block(ct_ptr as *mut __m128i, xor_block(ct, offset));
                checksum = xor_block(checksum, block);
                unsafe {
                    ct_ptr = ct_ptr.add(16);
                    pt2_ptr = pt2_ptr.add(need);
                }

                let remaining_pt2 = pt2_len - need;
                let full_blocks_pt2 = remaining_pt2 / 16;
                let rem_pt2 = remaining_pt2 & 15;

                (pt2_ptr, ct_ptr, offset) = self.enc_unaligned_contiguous(
                    pt2_ptr,
                    ct_ptr,
                    full_blocks_pt2,
                    &mut block_num,
                    offset,
                    &mut checksum,
                );
                if rem_pt2 > 0 {
                    (ct_ptr, offset, checksum) =
                        self.enc_tail(pt2_ptr, rem_pt2, ct_ptr, offset, checksum);
                }
            }
        } else {
            // rem_pt1 == 0, pt2 continues directly
            let full_blocks_pt2 = pt2_len / 16;
            let rem_pt2 = pt2_len & 15;

            (pt2_ptr, ct_ptr, offset) = self.enc_unaligned_contiguous(
                pt2_ptr,
                ct_ptr,
                full_blocks_pt2,
                &mut block_num,
                offset,
                &mut checksum,
            );
            if rem_pt2 > 0 {
                (ct_ptr, offset, checksum) =
                    self.enc_tail(pt2_ptr, rem_pt2, ct_ptr, offset, checksum);
            }
        }
        self.compute_and_store_tag(ct_ptr, offset, checksum, ad);
        return Ok(());
    }

    /// Compute and store the final tag
    #[inline(always)]
    fn compute_and_store_tag(
        &self,
        ct_ptr: *mut u8,
        offset: __m128i,
        checksum: __m128i,
        ad: &[u8],
    ) {
        let t = xor_block(xor_block(offset, self.ldollar), checksum);
        let mut tag = self.encrypt_key.aes_encrypt_block(t);
        if !ad.is_empty() {
            tag = xor_block(tag, self.process_ad(ad));
        }
        storeu_block(ct_ptr as *mut __m128i, tag);
    }

    /// Decrypts and verifies `io` using `nonce` and associated data `ad`. On a
    /// tag mismatch everything that may have been written to the destination
    /// is wiped and `Err(CryptoError::TagInvalid)` is returned.
    pub fn ae_decrypt(&self, io: DecryptIO, nonce: &[u8; 12], ad: &[u8]) -> Result<()> {
        // SAFETY: this crate is only compiled for x86_64 targets with the
        // `avx` and `aes` target features enabled (see the
        // module-level note), so the feature requirement of `ae_decrypt_inner`
        // is met. The pointer invariants it relies on are established by
        // `io.validate()`, which is its first operation.
        unsafe { self.ae_decrypt_inner(io, nonce, ad) }
    }

    /// # Safety
    ///
    /// Requires the `avx` and `aes` target features to be
    /// enabled at compile time for the calling context; this crate guarantees that in `lib.rs`'s
    /// module declaration.
    #[target_feature(enable = "avx,aes")]
    unsafe fn ae_decrypt_inner(&self, io: DecryptIO, nonce: &[u8; 12], ad: &[u8]) -> Result<()> {
        io.validate()?;
        if ad.len() > crate::MAX_PT_SIZE {
            return Err(crate::CryptoError::PlaintextTooLong);
        }
        let mut offset = self.gen_offset_from_nonce(nonce);
        let mut checksum = _mm_setzero_si128();
        let io_len = io.len();

        let (mut ct_ptr, mut pt_ptr, fail_zero_len, is_out_of_place) = match io {
            DecryptIO::InPlace(buf) => {
                let pt_ptr = buf.as_mut_ptr();
                let ct_ptr = pt_ptr as *const u8;
                (ct_ptr, pt_ptr, buf.len(), false)
            },
            DecryptIO::OutOfPlace { in_ct, out_pt } => (
                in_ct.as_ptr(),
                out_pt.as_mut_ptr(),
                in_ct.len() - OCB_TAG_LEN as usize,
                true
            ),
        };
        let out_start = pt_ptr;

        let pt_len = io_len - OCB_TAG_LEN as usize;
        let num_bpi = pt_len / (BPI * 16);

        if num_bpi > 0 {
            (offset, checksum) = self.dec_bpi_loop(ct_ptr, pt_ptr, num_bpi, offset);
            let bytes = num_bpi * (BPI * 16);
            unsafe {
                ct_ptr = ct_ptr.add(bytes);
                pt_ptr = pt_ptr.add(bytes);
            }
        }

        // Leftover full blocks (0..7) after the 8-block groups.
        let l0 = self.l[0];
        let l1 = self.l[1];
        let l2 = self.l[2];
        let mut full_blocks = (pt_len % (BPI * 16)) / 16;

        if full_blocks >= 4 {
            let oa0 = xor_block(offset, l0);
            let oa1 = xor_block(oa0, l1);
            let oa2 = xor_block(offset, l1);
            let oa3 = xor_block(oa2, l2);
            offset = oa3;

            let ct0 = loadu_block(ct_ptr as *const __m128i);
            let ct1 = loadu_block_with_offset(ct_ptr, 16);
            let ct2 = loadu_block_with_offset(ct_ptr, 32);
            let ct3 = loadu_block_with_offset(ct_ptr, 48);

            let d = self.decrypt_key.aes_decrypt_4_blocks([
                xor_block(oa0, ct0),
                xor_block(oa1, ct1),
                xor_block(oa2, ct2),
                xor_block(oa3, ct3),
            ]);

            let p0 = xor_block(d[0], oa0);
            let p1 = xor_block(d[1], oa1);
            let p2 = xor_block(d[2], oa2);
            let p3 = xor_block(d[3], oa3);

            let ptsum = xor_block(xor_block(p0, p1), xor_block(p2, p3));
            checksum = xor_block(checksum, ptsum);

            storeu_block(pt_ptr as *mut __m128i, p0);
            storeu_block_with_offset(pt_ptr, p1, 16);
            storeu_block_with_offset(pt_ptr, p2, 32);
            storeu_block_with_offset(pt_ptr, p3, 48);

            unsafe {
                ct_ptr = ct_ptr.add(64);
                pt_ptr = pt_ptr.add(64);
            }
            full_blocks -= 4;
        }
        if full_blocks >= 2 {
            let oa0 = xor_block(offset, l0);
            let oa1 = xor_block(oa0, l1);
            offset = oa1;

            let ct0 = loadu_block(ct_ptr as *const __m128i);
            let ct1 = loadu_block_with_offset(ct_ptr, 16);

            let d = self
                .decrypt_key
                .aes_decrypt_2_blocks([xor_block(oa0, ct0), xor_block(oa1, ct1)]);

            let p0 = xor_block(d[0], oa0);
            let p1 = xor_block(d[1], oa1);
            checksum = xor_block(checksum, xor_block(p0, p1));

            storeu_block(pt_ptr as *mut __m128i, p0);
            storeu_block_with_offset(pt_ptr, p1, 16);

            unsafe {
                ct_ptr = ct_ptr.add(32);
                pt_ptr = pt_ptr.add(32);
            }
            full_blocks -= 2;
        }
        if full_blocks >= 1 {
            offset = xor_block(offset, l0);
            let ct = loadu_block(ct_ptr as *const __m128i);
            let d = self.decrypt_key.aes_decrypt_block(xor_block(offset, ct));
            let pt = xor_block(d, offset);
            storeu_block(pt_ptr as *mut __m128i, pt);
            checksum = xor_block(checksum, pt);
            unsafe {
                ct_ptr = ct_ptr.add(16);
                pt_ptr = pt_ptr.add(16);
            }
        }

        let remaining = pt_len & 15;
        if remaining > 0 {
            // SAFETY: `in_ct` always contains `remaining` ciphertext bytes followed by 16 tag
            // bytes (`remaining + 16 >= 17` readable bytes), so a 16-byte load at `ct_ptr` is
            // always in-bounds of the input slice.
            let raw = loadu_block(ct_ptr as *const __m128i);
            offset = xor_block(offset, self.lstar);
            let pad = self.encrypt_key.aes_encrypt_block(offset);
            let mask = TAIL_STORE_MASK.get(remaining);
            let pt_zero_padded = _mm_and_si128(xor_block(raw, pad), mask);
            let pt_padded = _mm_or_si128(pt_zero_padded, TAIL_PAD_80.get(remaining));
            checksum = xor_block(checksum, pt_padded);
            let old = loadu_block(pt_ptr as *const __m128i);
            // SAFETY: blends the plaintext with the tag in the case of in-place decryption. In
            // the out-of-place case, the destination buffer is at least pt_len.
            let blended = blendv_block(old, pt_zero_padded, mask);
            if !is_out_of_place {
                storeu_block(pt_ptr as *mut __m128i, blended);
            } else {
                let blended = unsafe {
                    std::slice::from_raw_parts((&raw const blended).cast::<u8>(), 16)
                };
                (0..remaining).for_each(|i| {
                    unsafe {
                        pt_ptr.add(i).write(blended[i]);
                    }
                });
            }
            unsafe {
                ct_ptr = ct_ptr.add(remaining);
            }
        }

        // Tag verification: `T = E_K((offset ^ L$) ^ checksum) ^ ad_checksum`.
        let tag = loadu_block(ct_ptr as *const __m128i);
        let mut t = xor_block(xor_block(offset, self.ldollar), checksum);
        t = self.encrypt_key.aes_encrypt_block(t);
        if !ad.is_empty() {
            t = xor_block(t, self.process_ad(ad));
        }
        if equal_blocks(t, tag) == 0 {
            // Wipe everything we may have written to the destination.
            // SAFETY: out_start is the start of the destination buffer and fail_zero_len
            // is the number of bytes this call may have written there.
            unsafe {
                out_start.write_bytes(0, fail_zero_len);
            }
            return Err(crate::CryptoError::TagInvalid);
        }

        Ok(())
    }

    /// Encrypt `io` using `nonce` and associated data `ad`. This function is atomic, and depending
    /// on `io`, it does in place encryption and then appends a tag, or out of place encryption and
    /// then appends a tag.
    ///
    /// Note that `io` may contain a scattered plaintext of two &[u8] for the out of place
    /// encryption. This is designed for encrypted transports encrypting protocol-level control
    /// information and application-level data.
    pub fn ae_encrypt(&self, io: EncryptIO, nonce: &[u8; 12], ad: &[u8]) -> Result<()> {
        // SAFETY: this crate is only compiled for x86_64 targets with the
        // `avx` and `aes` target features enabled (see the
        // module-level note), so the feature requirement of `ae_encrypt_inner`
        // is met. The pointer invariants it relies on are established by
        // `io.validate()`, which is its first operation.
        unsafe { self.ae_encrypt_inner(io, nonce, ad) }
    }

    /// # Safety
    ///
    /// Requires the `avx` and `aes` target features to be
    /// enabled at compile time for the calling context; this crate guarrantees that in `lib.rs`'s
    /// module declaration.
    #[target_feature(enable = "avx,aes")]
    unsafe fn ae_encrypt_inner(&self, io: EncryptIO, nonce: &[u8; 12], ad: &[u8]) -> Result<()> {
        io.validate()?;
        if ad.len() > crate::MAX_PT_SIZE {
            return Err(crate::CryptoError::PlaintextTooLong);
        }
        let mut offset = self.gen_offset_from_nonce(nonce);
        let mut checksum = _mm_setzero_si128();

        let mut block_num = 0u32;

        let (mut pt_ptr, mut ct_ptr, pt_len) = match io {
            EncryptIO::InPlace(buf) => {
                // The buffer holds the plaintext followed by a 16-byte tag slot.
                let ct_ptr = buf.as_mut_ptr();
                let pt_ptr = ct_ptr as *const u8;
                (
                    pt_ptr,
                    ct_ptr,
                    buf.len() - OCB_TAG_LEN as usize,
                )
            }
            EncryptIO::OutOfPlace {
                in_pt: PlainText::Contiguous(pt),
                out,
            } => (pt.as_ptr(), out.as_mut_ptr(), pt.len()),
            EncryptIO::OutOfPlace {
                in_pt: PlainText::Scattered(pt1, pt2),
                out,
            } => {
                let pt1_ptr = pt1.as_ptr();
                let pt2_ptr = pt2.as_ptr();
                let pt1_len = pt1.len();
                let pt2_len = pt2.len();
                let ct_ptr = out.as_mut_ptr();

                return self.enc_scatter(
                    pt1_ptr, pt2_ptr, pt1_len, pt2_len, ct_ptr, offset, checksum, block_num, ad,
                );
            }

            EncryptIO::MixedPlace { out, part, at } => {
                let ct_ptr = out.as_mut_ptr();
                let payload = out.len() - OCB_TAG_LEN as usize;
                let part_len = part.len();
                //SAFETY: EncryptIO::validate() guarantees part_len + OCB_TAG_LEN <= out.len(),
                //so `payload - part_len` cannot underflow and `ct_ptr.add(part_len)` stays
                //inside `out`.
                let (head_ptr, head_len, tail_ptr, tail_len) = match at {
                    Placement::Head => (
                        part.as_ptr(),
                        part_len,
                        unsafe { ct_ptr.add(part_len) } as *const u8,
                        payload - part_len,
                    ),
                    Placement::Tail => (
                        ct_ptr as *const u8,
                        payload - part_len,
                        part.as_ptr(),
                        part_len,
                    ),
                };

                return self.enc_scatter(
                    head_ptr, tail_ptr, head_len, tail_len, ct_ptr, offset, checksum, block_num, ad,
                );
            }
        };

        let full_blocks = pt_len / 16;
        let rem = pt_len & 15;

        (pt_ptr, ct_ptr, offset) = self.enc_aligned_blocks(
            pt_ptr,
            ct_ptr,
            full_blocks,
            &mut block_num,
            offset,
            &mut checksum,
        );
        if rem > 0 {
            (ct_ptr, offset, checksum) = self.enc_tail(pt_ptr, rem, ct_ptr, offset, checksum);
        }
        self.compute_and_store_tag(ct_ptr, offset, checksum, ad);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{EncryptIO, Placement, PlainText};
    // Note:
    // These tests were generated using Qwen-3.8-27B on local hardware, feeding the LLM
    // with the output test vectors of the original C implementation, and asking it to exercice
    // our InPlace/OutOfPLace and Scattered API on the test vectors.

    include!("test_vectors/spec_vectors.rs");
    /// Spec test data: pt[i] = ad[i] = i.
    fn spec_data(len: usize) -> Vec<u8> {
        (0..len).map(|i| i as u8).collect()
    }

    /// Encrypt out-of-place with contiguous plaintext. Returns (ct, tag).
    fn enc_contiguous(ctx: &AEctx, nonce: &[u8; 12], pt: &[u8], ad: &[u8]) -> (Vec<u8>, [u8; 16]) {
        let mut out = vec![0u8; pt.len() + 16];
        ctx.ae_encrypt(
            EncryptIO::OutOfPlace {
                in_pt: PlainText::Contiguous(pt),
                out: &mut out,
            },
            nonce,
            ad,
        )
        .expect("encrypt failed");
        let (ct, tag) = out.split_at(pt.len());
        (ct.to_vec(), tag.try_into().expect("tag is 16 bytes"))
    }

    /// Encrypt in place. Returns (ct, tag).
    fn enc_in_place(ctx: &AEctx, nonce: &[u8; 12], pt: &[u8], ad: &[u8]) -> (Vec<u8>, [u8; 16]) {
        let mut buf = pt.to_vec();
        buf.extend([0u8; 16]);
        ctx.ae_encrypt(EncryptIO::InPlace(&mut buf), nonce, ad)
            .expect("encrypt failed");
        let (ct, tag) = buf.split_at(pt.len());
        (ct.to_vec(), tag.try_into().expect("tag is 16 bytes"))
    }

    /// Encrypt out-of-place with scattered plaintext split at `split`. Returns (ct, tag).
    fn enc_scattered(
        ctx: &AEctx,
        nonce: &[u8; 12],
        pt: &[u8],
        split: usize,
        ad: &[u8],
    ) -> (Vec<u8>, [u8; 16]) {
        let mut out = vec![0u8; pt.len() + 16];
        ctx.ae_encrypt(
            EncryptIO::OutOfPlace {
                in_pt: PlainText::Scattered(&pt[..split], &pt[split..]),
                out: &mut out,
            },
            nonce,
            ad,
        )
        .expect("encrypt failed");
        let (ct, tag) = out.split_at(pt.len());
        (ct.to_vec(), tag.try_into().expect("tag is 16 bytes"))
    }

    /// Encrypt mixed-place with the externally supplied half of length `split` at
    /// `at`; the other half already sits in `out`. Returns (ct, tag).
    fn enc_mixed(
        ctx: &AEctx,
        nonce: &[u8; 12],
        pt: &[u8],
        split: usize,
        at: Placement,
        ad: &[u8],
    ) -> (Vec<u8>, [u8; 16]) {
        // 0xAA, not 0x00: if the implementation reads the half that must come from
        // `part` out of `out` instead, it encrypts canary bytes and every vector fails.
        let mut out = vec![0xAAu8; pt.len() + 16];
        let part = match at {
            Placement::Head => {
                out[split..pt.len()].copy_from_slice(&pt[split..]);
                &pt[..split]
            }
            Placement::Tail => {
                out[..split].copy_from_slice(&pt[..split]);
                &pt[split..]
            }
        };
        ctx.ae_encrypt(
            EncryptIO::MixedPlace {
                out: &mut out,
                part,
                at,
            },
            nonce,
            ad,
        )
        .expect("encrypt failed");
        let (ct, tag) = out.split_at(pt.len());
        (ct.to_vec(), tag.try_into().expect("tag is 16 bytes"))
    }

    /// Check one vector in every API mode: in place, out-of-place contiguous,
    /// out-of-place scattered at every split point, and mixed place at every split
    /// point with the external half on either end.
    fn check_vector(ptl: usize, adl: usize, ct_exp: &[u8], tag_exp: &[u8]) {
        let (pt, ad) = (spec_data(ptl), spec_data(adl));

        let ctx = AEctx::new(&SPEC_KEY);
        let (ct, tag) = enc_in_place(&ctx, &SPEC_NONCE, &pt, &ad);
        assert_eq!(
            ct.as_slice(),
            ct_exp,
            "in-place ct mismatch for ptl={} adl={}",
            ptl,
            adl
        );
        assert_eq!(
            &tag[..],
            tag_exp,
            "in-place tag mismatch for ptl={} adl={}",
            ptl,
            adl
        );

        let ctx = AEctx::new(&SPEC_KEY);
        let (ct, tag) = enc_contiguous(&ctx, &SPEC_NONCE, &pt, &ad);
        assert_eq!(
            ct.as_slice(),
            ct_exp,
            "out-of-place ct mismatch for ptl={} adl={}",
            ptl,
            adl
        );
        assert_eq!(
            &tag[..],
            tag_exp,
            "out-of-place tag mismatch for ptl={} adl={}",
            ptl,
            adl
        );

        let ctx = AEctx::new(&SPEC_KEY);
        for split in 0..=ptl {
            let (ct, tag) = enc_scattered(&ctx, &SPEC_NONCE, &pt, split, &ad);
            assert_eq!(
                ct.as_slice(),
                ct_exp,
                "scattered split={} ct mismatch for ptl={} adl={}",
                split,
                ptl,
                adl
            );
            assert_eq!(
                &tag[..],
                tag_exp,
                "scattered split={} tag mismatch for ptl={} adl={}",
                split,
                ptl,
                adl
            );
        }

        let ctx = AEctx::new(&SPEC_KEY);
        for split in 0..=ptl {
            for at in [Placement::Head, Placement::Tail] {
                let which = match at {
                    Placement::Head => "head",
                    Placement::Tail => "tail",
                };
                let (ct, tag) = enc_mixed(&ctx, &SPEC_NONCE, &pt, split, at, &ad);
                assert_eq!(
                    ct.as_slice(),
                    ct_exp,
                    "mixed {} split={} ct mismatch for ptl={} adl={}",
                    which,
                    split,
                    ptl,
                    adl
                );
                assert_eq!(
                    &tag[..],
                    tag_exp,
                    "mixed {} split={} tag mismatch for ptl={} adl={}",
                    which,
                    split,
                    ptl,
                    adl
                );
            }
        }
    }

    #[test]
    fn spec_vectors() {
        for &(ptl, adl, ct_exp, tag_exp) in &SPEC_VECTORS {
            check_vector(ptl, adl, ct_exp, tag_exp);
        }
    }

    /// Differential coverage for the scattered branch where the two tails
    /// form a single partial block (`rem_pt1 + pt2.len() < 16`), which the
    /// spec vectors (pt_len multiples of 8) do not exercise for
    /// `rem_pt1 >= 8`. Scattered encryption must match contiguous
    /// encryption byte for byte, and the result must decrypt back.
    #[test]
    fn scattered_two_tail_differential() {
        for ptl in 1..=64usize {
            let (pt, ad) = (spec_data(ptl), spec_data(ptl / 2));

            let ctx = AEctx::new(&SPEC_KEY);
            let (ct_cont, tag_cont) = enc_contiguous(&ctx, &SPEC_NONCE, &pt, &ad);

            let ctx = AEctx::new(&SPEC_KEY);
            for split in 0..=ptl {
                let (ct, tag) = enc_scattered(&ctx, &SPEC_NONCE, &pt, split, &ad);
                assert_eq!(
                    ct, ct_cont,
                    "scattered split={} ct mismatch for ptl={}",
                    split, ptl
                );
                assert_eq!(
                    tag, tag_cont,
                    "scattered split={} tag mismatch for ptl={}",
                    split, ptl
                );
            }

            // The (identical) ciphertext must decrypt back to the plaintext.
            let mut buf = ct_cont;
            buf.extend_from_slice(&tag_cont);
            let ctx = AEctx::new(&SPEC_KEY);
            ctx.ae_decrypt(DecryptIO::InPlace(&mut buf), &SPEC_NONCE, &ad)
                .expect("decrypt failed");
            assert_eq!(
                &buf[..ptl],
                pt.as_slice(),
                "decrypt roundtrip mismatch for ptl={}",
                ptl
            );
        }
    }

    /// The spec vectors use plaintext lengths that are multiples of 8, so their
    /// split points only ever give `rem_pt1` of 0 or 8. This covers the remaining
    /// blended-tail branches (`1 <= rem_pt1 + pt2_len < 16` and the full-block
    /// reassembly for every `rem_pt1`), the same way
    /// `scattered_two_tail_differential` does for scattered plaintext. Mixed
    /// encryption must match contiguous encryption byte for byte, and the result
    /// must decrypt back.
    #[test]
    fn mixed_two_tail_differential() {
        for ptl in 1..=64usize {
            let (pt, ad) = (spec_data(ptl), spec_data(ptl / 2));

            let ctx = AEctx::new(&SPEC_KEY);
            let (ct_cont, tag_cont) = enc_contiguous(&ctx, &SPEC_NONCE, &pt, &ad);

            let ctx = AEctx::new(&SPEC_KEY);
            for split in 0..=ptl {
                for at in [Placement::Head, Placement::Tail] {
                    let which = match at {
                        Placement::Head => "head",
                        Placement::Tail => "tail",
                    };
                    let (ct, tag) = enc_mixed(&ctx, &SPEC_NONCE, &pt, split, at, &ad);
                    assert_eq!(
                        ct, ct_cont,
                        "mixed {} split={} ct mismatch for ptl={}",
                        which, split, ptl
                    );
                    assert_eq!(
                        tag, tag_cont,
                        "mixed {} split={} tag mismatch for ptl={}",
                        which, split, ptl
                    );
                }
            }

            // The (identical) ciphertext must decrypt back to the plaintext.
            let mut buf = ct_cont;
            buf.extend_from_slice(&tag_cont);
            let ctx = AEctx::new(&SPEC_KEY);
            ctx.ae_decrypt(DecryptIO::InPlace(&mut buf), &SPEC_NONCE, &ad)
                .expect("decrypt failed");
            assert_eq!(
                &buf[..ptl],
                pt.as_slice(),
                "decrypt roundtrip mismatch for ptl={}",
                ptl
            );
        }
    }

    /// A single context is reused for every vector, as the C reference harness
    /// does; this exercises the per-call state resets.
    #[test]
    fn spec_vectors_reused_context() {
        let ctx = AEctx::new(&SPEC_KEY);
        for &(ptl, adl, ct_exp, tag_exp) in &SPEC_VECTORS {
            let (pt, ad) = (spec_data(ptl), spec_data(adl));
            let (ct, tag) = enc_contiguous(&ctx, &SPEC_NONCE, &pt, &ad);
            assert_eq!(
                ct.as_slice(),
                ct_exp,
                "reused context ct mismatch for ptl={} adl={}",
                ptl,
                adl
            );
            assert_eq!(
                &tag[..],
                tag_exp,
                "reused context tag mismatch for ptl={} adl={}",
                ptl,
                adl
            );
        }
    }

    include!("test_vectors/dec_vectors.rs");

    /// C-reference decryption vectors: pt[i] = i, ad[i] = (3i + 1) mod 256.
    fn dec_gen_ad(len: usize) -> Vec<u8> {
        (0..len).map(|i| ((i * 3 + 1) & 0xff) as u8).collect()
    }

    /// Decrypt C-generated vectors out of place and in place, and check that
    /// corrupted tag / ciphertext / AD are all rejected.
    #[test]
    fn dec_vectors_kat() {
        for &(ptl, adl, ct_tag) in DEC_VECTORS {
            let (pt_exp, ad) = (spec_data(ptl), dec_gen_ad(adl));

            // Out of place.
            let ctx = AEctx::new(&SPEC_KEY);
            let mut out = vec![0xffu8; ptl];
            ctx.ae_decrypt(
                DecryptIO::OutOfPlace {
                    in_ct: ct_tag,
                    out_pt: &mut out,
                },
                &SPEC_NONCE,
                &ad,
            )
            .unwrap_or_else(|e| panic!("out-of-place decrypt failed ptl={} adl={}: {e}", ptl, adl));
            assert_eq!(
                &out[..ptl],
                &pt_exp,
                "out-of-place pt mismatch for ptl={} adl={}",
                ptl,
                adl
            );

            // In place: the plaintext must come back and the tag must survive.
            let ctx = AEctx::new(&SPEC_KEY);
            let mut buf = ct_tag.to_vec();
            ctx.ae_decrypt(DecryptIO::InPlace(&mut buf), &SPEC_NONCE, &ad)
                .unwrap_or_else(|e| panic!("in-place decrypt failed ptl={} adl={}: {e}", ptl, adl));
            assert_eq!(
                &buf[..ptl],
                &pt_exp,
                "in-place pt mismatch for ptl={} adl={}",
                ptl,
                adl
            );
            assert_eq!(
                &buf[ptl..],
                &ct_tag[ptl..],
                "in-place tag clobbered for ptl={} adl={}",
                ptl,
                adl
            );

            // Corrupted tag must be rejected.
            let ctx = AEctx::new(&SPEC_KEY);
            let mut bad = ct_tag.to_vec();
            bad[ptl] ^= 0x01;
            assert!(
                matches!(
                    ctx.ae_decrypt(DecryptIO::InPlace(&mut bad), &SPEC_NONCE, &ad),
                    Err(crate::CryptoError::TagInvalid)
                ),
                "corrupted tag accepted for ptl={} adl={}",
                ptl,
                adl
            );

            // Corrupted ciphertext must be rejected.
            if ptl > 0 {
                let ctx = AEctx::new(&SPEC_KEY);
                let mut bad = ct_tag.to_vec();
                bad[ptl / 2] ^= 0x01;
                assert!(
                    matches!(
                        ctx.ae_decrypt(DecryptIO::InPlace(&mut bad), &SPEC_NONCE, &ad),
                        Err(crate::CryptoError::TagInvalid)
                    ),
                    "corrupted ct accepted for ptl={} adl={}",
                    ptl,
                    adl
                );
            }

            // Corrupted AD must be rejected.
            if adl > 0 {
                let ctx = AEctx::new(&SPEC_KEY);
                let mut bad_ad = ad.clone();
                bad_ad[adl / 2] ^= 0x01;
                assert!(
                    matches!(
                        ctx.ae_decrypt(
                            DecryptIO::InPlace(&mut ct_tag.to_vec()),
                            &SPEC_NONCE,
                            &bad_ad,
                        ),
                        Err(crate::CryptoError::TagInvalid)
                    ),
                    "corrupted ad accepted for ptl={} adl={}",
                    ptl,
                    adl
                );
            }
        }
    }

    /// Round-trip (encrypt then decrypt with this implementation) at every
    /// length 0..=1024 plus a few larger ones, in both modes. The KAT above
    /// pins absolute values against the C reference; this catches enc/dec
    /// asymmetries at lengths the KAT matrix does not cover.
    #[test]
    fn dec_roundtrip() {
        let ad_lens = [0usize, 1, 15, 16, 17, 100, 127, 128, 129];
        for ptl in 0..=1024 {
            for &adl in &ad_lens {
                let (pt, ad) = (spec_data(ptl), spec_data(adl));

                let ctx = AEctx::new(&SPEC_KEY);
                let (ct, tag) = enc_in_place(&ctx, &SPEC_NONCE, &pt, &ad);
                let mut blob = ct;
                blob.extend_from_slice(&tag);

                // In place.
                let mut buf = blob.clone();
                ctx.ae_decrypt(DecryptIO::InPlace(&mut buf), &SPEC_NONCE, &ad)
                    .unwrap_or_else(|e| {
                        panic!("in-place roundtrip failed ptl={} adl={}: {e}", ptl, adl)
                    });
                assert_eq!(
                    &buf[..ptl],
                    &pt,
                    "in-place roundtrip ptl={} adl={}",
                    ptl,
                    adl
                );
                assert_eq!(
                    &buf[ptl..],
                    &tag[..],
                    "in-place roundtrip tag ptl={} adl={}",
                    ptl,
                    adl
                );

                // Out of place.
                let mut out = vec![0u8; ptl + 16];
                ctx.ae_decrypt(
                    DecryptIO::OutOfPlace {
                        in_ct: &blob,
                        out_pt: &mut out,
                    },
                    &SPEC_NONCE,
                    &ad,
                )
                .unwrap_or_else(|e| panic!("o-o-p roundtrip failed ptl={} adl={}: {e}", ptl, adl));
                assert_eq!(&out[..ptl], &pt, "o-o-p roundtrip ptl={} adl={}", ptl, adl);
            }
        }
        for ptl in [2048usize, 4096, 8192] {
            let (pt, ad) = (spec_data(ptl), spec_data(129));
            let ctx = AEctx::new(&SPEC_KEY);
            let (ct, tag) = enc_in_place(&ctx, &SPEC_NONCE, &pt, &ad);
            let mut blob = ct;
            blob.extend_from_slice(&tag);
            let mut buf = blob.clone();
            ctx.ae_decrypt(DecryptIO::InPlace(&mut buf), &SPEC_NONCE, &ad)
                .unwrap_or_else(|e| panic!("in-place roundtrip failed ptl={}: {e}", ptl));
            assert_eq!(&buf[..ptl], &pt, "in-place roundtrip ptl={}", ptl);
        }
    }

    fn check_sync(_: &impl Sync) {}

    #[test]
    fn ctx_must_be_sync() {
        let ctx = AEctx::new(&SPEC_KEY);
        check_sync(&ctx);
    }

    #[test]
    fn ktop_cache_updated_once_per_64_sequential_nonces() {
        let ctx = AEctx::new(&SPEC_KEY);
        let mut nonce = [0u8; 12];
        let mut buf = [0u8; 16 + OCB_TAG_LEN as usize];

        assert_eq!(ctx.ktop.seq.load(Ordering::Relaxed), 0);

        nonce[11] = 0;
        ctx.ae_encrypt(EncryptIO::InPlace(&mut buf), &nonce, &[])
            .expect("encrypt failed");
        let mut cache = [
            ctx.ktop.cached_top[0].load(Ordering::Relaxed),
            ctx.ktop.cached_top[1].load(Ordering::Relaxed),
        ];
        for i in 1..256usize {
            nonce[11] = i as u8;
            ctx.ae_encrypt(EncryptIO::InPlace(&mut buf), &nonce, &[])
                .expect("encrypt failed");

            if i % 64 == 0 {
                assert!(
                    (ctx.ktop.cached_top[0].load(Ordering::Relaxed) != cache[0])
                        || (ctx.ktop.cached_top[1].load(Ordering::Relaxed) != cache[1])
                );
                cache = [
                    ctx.ktop.cached_top[0].load(Ordering::Relaxed),
                    ctx.ktop.cached_top[1].load(Ordering::Relaxed),
                ];
            } else {
                assert_eq!(ctx.ktop.cached_top[0].load(Ordering::Relaxed), cache[0]);
                assert_eq!(ctx.ktop.cached_top[1].load(Ordering::Relaxed), cache[1]);
            }

            // Each cache update increments `seq` by 2.
            // Update must occur only when `i % 64 == 0` (1 time out of 64).
            let expected_updates = (i / 64) + 1;
            let actual_updates = (ctx.ktop.seq.load(Ordering::Relaxed) / 2) as usize;
            assert_eq!(
                actual_updates, expected_updates,
                "unexpected cache update count at nonce[11]={i}"
            );
        }
    }
}
