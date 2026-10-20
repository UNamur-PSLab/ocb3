use crate::DecryptIO;
use crate::EncryptIO;
use crate::OCB_TAG_LEN;
use crate::PlainText;
use crate::Result;
use crate::aes::AesKey;
use crate::sse::*;
use core::arch::x86_64::*;

const L_TABLE_SZ: usize = 16;
const BPI: usize = 8;
const PAGE_SIZE_4K: usize = 4096;
const PAGE_OFFSET_MASK: usize = PAGE_SIZE_4K - 1;

#[repr(C, align(16))]
pub struct AEctx {
    lstar: __m128i,
    ldollar: __m128i,
    L: [__m128i; L_TABLE_SZ],
    cached_top: __m128i,
    decrypt_key: AesKey,
    encrypt_key: AesKey,
    ktop_str: [u64; 3],
}

impl AEctx {
    #[inline]
    pub fn new(key: &[u8]) -> Self {
        unsafe { Self::new_inner(key) }
    }

    #[target_feature(enable = "avx,aes,ssse3,sse4.1")]
    unsafe fn new_inner(key: &[u8]) -> Self {
        // Safety: needs to be 16-bytes aligned.
        let zero = zero_block();
        let mut ctx = AEctx {
            lstar: zero,
            ldollar: zero,
            L: [zero; L_TABLE_SZ],
            cached_top: zero,
            decrypt_key: AesKey::default(),
            encrypt_key: AesKey::default(),
            ktop_str: [0u64; 3],
        };

        // Key dependent values
        ctx.encrypt_key.AES_128_key_expansion(key);
        ctx.decrypt_key.AES_set_key_fast(&ctx.encrypt_key);
        ctx.lstar = ctx.encrypt_key.AES_encrypt_block(ctx.cached_top);
        let mut tmp_blk = swap_if_le(ctx.lstar);
        tmp_blk = double(tmp_blk);
        ctx.ldollar = swap_if_le(tmp_blk);
        tmp_blk = double(tmp_blk);
        ctx.L[0] = swap_if_le(tmp_blk);

        for i in 1..L_TABLE_SZ {
            tmp_blk = double(tmp_blk);
            ctx.L[i] = swap_if_le(tmp_blk);
        }

        ctx
    }

    #[inline(always)]
    fn gen_offset_from_nonce(&mut self, nonce: &[u8; 12]) -> __m128i {
        let n_hi = unsafe { core::ptr::read_unaligned(nonce.as_ptr() as *const u32) };
        let n_lo = unsafe { core::ptr::read_unaligned(nonce.as_ptr().add(4) as *const u64) };
        let idx = (nonce[11] & 0x3f) as i32;
        let lo_qword = 0x0100_0000u64 | ((n_hi as u64) << 32);
        let hi_qword = n_lo & !(0x3fu64 << 56);
        let mut bl = unsafe { _mm_set_epi64x(hi_qword as i64, lo_qword as i64) };
        if equal_blocks(bl, self.cached_top) == 0 {
            self.cached_top = bl;
            bl = self.encrypt_key.AES_encrypt_block(bl);
            [self.ktop_str[0], self.ktop_str[1]] = m128i_to_bswap_u64x2(bl);
            self.ktop_str[2] =
                self.ktop_str[0] ^ (self.ktop_str[0] << 8) ^ (self.ktop_str[1] >> 56);
        }

        gen_offset(&self.ktop_str, idx)
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
    #[target_feature(enable = "avx,aes,ssse3,sse4.1")]
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
            let mut l_ptr = self.L.as_ptr();
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
            let (l1, l2) = unsafe {
                (load_block(l_ptr.add(1)), load_block(l_ptr.add(2)))
            };

            let offset_k = xor_block(ad_offset, k0);
            let oa0_k = xor_block(offset_k, l0);
            let oa1_k = xor_block(oa0_k, l1);
            let oa2_k = xor_block(offset_k, l1);
            let oa3_k = xor_block(oa2_k, l2);
            let oa4_k = xor_block(oa1_k, l2);
            let oa5_k = xor_block(oa0_k, l2);
            let oa6_k = xor_block(offset_k, l2);
            // SAFETY: `ad_blocks_num < 2^16` in OCB3, ensuring `tz < 16` and in-bounds for `self.L`.
            let l_tz = unsafe {
                *l_ptr.add(tz)
            };
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

            let ct = AesKey::AES_encrypt_8_blocks_prewhitened(
                rk_ptr,
                [b0, b1, b2, b3, b4, b5, b6, b7],
            );
            let sum0 = xor_block(xor_block(ct[0], ct[1]), xor_block(ct[2], ct[3]));
            let sum1 = xor_block(xor_block(ct[4], ct[5]), xor_block(ct[6], ct[7]));
            ad_checksum = xor_block(ad_checksum, xor_block(sum0, sum1));

            unsafe {
                ad_ptr = ad_ptr.add(128);
            }
        }

        let l0 = self.L[0];
        let l1 = self.L[1];
        let l2 = self.L[2];
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

            let ct = self.encrypt_key.AES_encrypt_4_blocks([
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
                .AES_encrypt_2_blocks([xor_block(oa0, ad0), xor_block(oa1, ad1)]);
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
            let ct = self.encrypt_key.AES_encrypt_block(ta0);
            ad_checksum = xor_block(ad_checksum, ct);
            // SAFETY: in range since remaining >= 16
            unsafe {
                ad_ptr = ad_ptr.add(16);
            }
            remaining -= 16;
        }
        if remaining > 0 {
            let mask = TAIL_STORE_MASK.get(remaining);
            let ad0 = unsafe { Self::load_tail_padded(ad_ptr, remaining, mask) };
            ad_offset = xor_block(ad_offset, self.lstar);
            let ta = xor_block(ad_offset, ad0);
            let ct = self.encrypt_key.AES_encrypt_block(ta);
            ad_checksum = xor_block(ad_checksum, ct);
        }
        ad_checksum
    }

    #[inline(always)]
    fn enc_block(&self, oa: __m128i, pt_ptr: *const u8, ct_ptr: *mut u8, checksum: &mut __m128i) {
        let pt = loadu_block(pt_ptr as *const __m128i);
        let t = xor_block(oa, pt);
        let ct = self.encrypt_key.AES_encrypt_block(t);
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
        // SAFETY: `block_num <= MAX_PT_SIZE / 16 < 2^16`, ensuring `ntz < 16` and in-bounds for `self.L`.
        let oa = xor_block(offset, table_get(&self.L, tz));
        self.enc_block(oa, pt_ptr, ct_ptr, checksum);
        oa
    }

    /// Encrypts `num_bpi` groups of 8 blocks (`128` bytes each) starting at an 8-block aligned
    /// `block_num`. Uses all 16 XMM registers with zero stack spills via prewhitened offsets
    /// and a telescoping offset-whitened checksum.
    #[allow(clippy::pointers_in_nomem_asm_block)]
    #[target_feature(enable = "avx,aes,ssse3,sse4.1")]
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
            let mut l_ptr = self.L.as_ptr();
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
            let (l1, l2) = unsafe {
                (load_block(l_ptr.add(1)), load_block(l_ptr.add(2)))
            };

            let ok = xor_block(offset, k0);
            let oa0_k = xor_block(ok, l0);
            let oa1_k = xor_block(oa0_k, l1);
            let oa2_k = xor_block(ok, l1);
            let oa3_k = xor_block(oa2_k, l2);
            let oa4_k = xor_block(oa1_k, l2);
            let oa5_k = xor_block(oa0_k, l2);
            let oa6_k = xor_block(ok, l2);
            // SAFETY: `block_num <= MAX_PT_SIZE / 16 < 2^16`, ensuring `tz < 16` and in-bounds for `self.L`.
            let l_tz = unsafe {
                *l_ptr.add(tz)
            };
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

            let ct = AesKey::AES_encrypt_8_blocks_prewhitened(
                rk_ptr,
                [b0, b1, b2, b3, b4, b5, b6, b7],
            );

            // Reloading l0, l1, l2 again, reusing registers, and heuristically
            // preventing LLVM to move these variables outside of the loop.
            let mut l_ptr2 = self.L.as_ptr();
            unsafe {
                core::arch::asm!(
                    "/* {0} */",
                    inout(reg) l_ptr2,
                    options(pure, nomem, nostack, preserves_flags)
                );
            }
            let l0 = load_block(l_ptr2);
            let (l1, l2) = unsafe {
                (load_block(l_ptr2.add(1)), load_block(l_ptr2.add(2)))
            };

            let oa0 = xor_block(offset, l0);
            let oa1 = xor_block(oa0, l1);
            let oa2 = xor_block(offset, l1);
            let oa3 = xor_block(oa2, l2);
            let oa4 = xor_block(oa1, l2);
            let oa5 = xor_block(oa0, l2);
            let oa6 = xor_block(offset, l2);
            let oa7 = unsafe {
                xor_block(oa6, *l_ptr2.add(tz))
            };

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

    /// Decrypts `num_bpi` groups of 8 blocks (`128` bytes each) in lockstep using all 16 XMM
    /// registers and both AES pipelines.
    #[allow(clippy::pointers_in_nomem_asm_block)]
    #[target_feature(enable = "avx,aes,ssse3,sse4.1")]
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
            let mut l_ptr = self.L.as_ptr();
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
            let (l1, l2) = unsafe {
                (load_block(l_ptr.add(1)), load_block(l_ptr.add(2)))
            };

            let ok = xor_block(offset, k0);
            let oa0_k = xor_block(ok, l0);
            let oa1_k = xor_block(oa0_k, l1);
            let oa2_k = xor_block(ok, l1);
            let oa3_k = xor_block(oa2_k, l2);
            let oa4_k = xor_block(oa1_k, l2);
            let oa5_k = xor_block(oa0_k, l2);
            let oa6_k = xor_block(ok, l2);
            // SAFETY: `block_num <= MAX_PT_SIZE / 16 < 2^16`, ensuring `tz < 16`
            // and in-bounds for `self.L`.
            let l_tz = unsafe {
                *l_ptr.add(tz)
            };

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

            let d = AesKey::AES_decrypt_8_blocks_prewhitened(
                rk_ptr,
                [b0, b1, b2, b3, b4, b5, b6, b7],
            );

            let sum0 = xor_block(xor_block(d[0], d[1]), xor_block(d[2], d[3]));
            let sum1 = xor_block(xor_block(d[4], d[5]), xor_block(d[6], d[7]));
            checksum = xor_block(checksum, xor_block(sum0, sum1));

            let mut l_ptr2 = self.L.as_ptr();
            unsafe {
                core::arch::asm!(
                    "/* {0} */",
                    inout(reg) l_ptr2,
                    options(pure, nomem, nostack, preserves_flags)
                );
            }
            let l0 = load_block(l_ptr2);
            let (l1, l2) = unsafe {
                (load_block(l_ptr2.add(1)), load_block(l_ptr2.add(2)))
            };

            let oa0 = xor_block(offset, l0);
            let oa1 = xor_block(oa0, l1);
            let oa2 = xor_block(offset, l1);
            let oa3 = xor_block(oa2, l2);
            let oa4 = xor_block(oa1, l2);
            let oa5 = xor_block(oa0, l2);
            let oa6 = xor_block(offset, l2);
            let oa7 = unsafe {
                xor_block(oa6, *l_ptr2.add(tz))
            };

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
    /// lockstep loop followed by `4 -> 2 -> 1` parallel leftover blocks with constant offsets.
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

        let l0 = self.L[0];
        let l1 = self.L[1];
        let l2 = self.L[2];

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

            let ct = self.encrypt_key.AES_encrypt_4_blocks([
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
                .AES_encrypt_2_blocks([xor_block(oa0, pt0), xor_block(oa1, pt1)]);

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
        let pad = self.encrypt_key.AES_encrypt_block(offset);
        let ct_block = xor_block(pt_padded, pad);

        //SAFETY: the ciphertext slice is required to be 16 bytes longer than the plaintext, so
        //this cannot fall outside of the allocation. In case of InPlace encryption, this design
        //encrypts up to the size of the input buffer - 16, so this is safe to do.
        let old = loadu_block(ct_ptr as *const __m128i);
        let blended = blendv_block(old, ct_block, mask);
        storeu_block(ct_ptr as *mut __m128i, blended);

        (unsafe { ct_ptr.add(data_len) }, offset, checksum)
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
        let mut tag = self.encrypt_key.AES_encrypt_block(t);
        if !ad.is_empty() {
            unsafe {
                tag = xor_block(tag, self.process_ad(ad));
            }
        }
        storeu_block(ct_ptr as *mut __m128i, tag);
    }

    #[target_feature(enable = "avx,aes,ssse3,sse4.1")]
    pub fn ae_decrypt(&mut self, io: DecryptIO, nonce: &[u8; 12], ad: &[u8]) -> Result<()> {
        io.validate()?;
        if ad.len() > crate::MAX_PT_SIZE {
            return Err(crate::CryptoError::PlaintextTooLong);
        }
        let mut offset = self.gen_offset_from_nonce(nonce);
        let mut checksum = _mm_setzero_si128();
        let io_len = io.len();

        let (mut ct_ptr, mut pt_ptr, fail_zero_len) = match io {
            DecryptIO::InPlace(buf) => (buf.as_ptr(), buf.as_mut_ptr(), buf.len()),
            DecryptIO::OutOfPlace { in_ct, out_pt } => (
                in_ct.as_ptr(),
                out_pt.as_mut_ptr(),
                in_ct.len() - OCB_TAG_LEN as usize,
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
        let l0 = self.L[0];
        let l1 = self.L[1];
        let l2 = self.L[2];
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

            let d = self.decrypt_key.AES_decrypt_4_blocks([
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
                .AES_decrypt_2_blocks([xor_block(oa0, ct0), xor_block(oa1, ct1)]);

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
            let d = self.decrypt_key.AES_decrypt_block(xor_block(offset, ct));
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
            let pad = self.encrypt_key.AES_encrypt_block(offset);
            let mask = TAIL_STORE_MASK.get(remaining);
            let pt_zero_padded = _mm_and_si128(xor_block(raw, pad), mask);
            let pt_padded = _mm_or_si128(pt_zero_padded, TAIL_PAD_80.get(remaining));
            checksum = xor_block(checksum, pt_padded);
            let old = loadu_block(pt_ptr as *const __m128i);
            // SAFETY: blends the plaintext with the tag in the case of in-place decryption. In
            // the out-of-place case, the destination buffer is at least pt_len + OCB_TAG_LEN
            // bytes (see DecryptIO::validate).
            let blended = blendv_block(old, pt_zero_padded, mask);
            storeu_block(pt_ptr as *mut __m128i, blended);
            unsafe {
                ct_ptr = ct_ptr.add(remaining);
            }
        }

        // Tag verification: `T = E_K((offset ^ L$) ^ checksum) ^ ad_checksum`.
        let tag = loadu_block(ct_ptr as *const __m128i);
        let mut t = xor_block(xor_block(offset, self.ldollar), checksum);
        t = self.encrypt_key.AES_encrypt_block(t);
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

    #[target_feature(enable = "avx,aes,ssse3,sse4.1")]
    pub fn ae_encrypt(&mut self, io: EncryptIO, nonce: &[u8; 12], ad: &[u8]) -> Result<()> {
        io.validate()?;
        if ad.len() > crate::MAX_PT_SIZE {
            return Err(crate::CryptoError::PlaintextTooLong);
        }
        let mut offset = self.gen_offset_from_nonce(nonce);
        let mut checksum = _mm_setzero_si128();

        unsafe {
            let mut block_num = 0u32;

            let (mut pt_ptr, mut ct_ptr, pt_len) = match io {
                EncryptIO::InPlace(buf) => {
                    // The buffer holds the plaintext followed by a 16-byte tag slot.
                    (
                        buf.as_ptr(),
                        buf.as_mut_ptr(),
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
                    let mut pt1_ptr = pt1.as_ptr();
                    let mut pt2_ptr = pt2.as_ptr();
                    let mut ct_ptr = out.as_mut_ptr();

                    let full_blocks_pt1 = pt1.len() / 16;
                    let rem_pt1 = pt1.len() & 15;

                    (pt1_ptr, ct_ptr, offset) = self.enc_aligned_blocks(
                        pt1_ptr,
                        ct_ptr,
                        full_blocks_pt1,
                        &mut block_num,
                        offset,
                        &mut checksum,
                    );

                    if rem_pt1 > 0 {
                        if pt2.is_empty() {
                            (ct_ptr, offset, checksum) =
                                self.enc_tail(pt1_ptr, rem_pt1, ct_ptr, offset, checksum);
                        } else if rem_pt1 + pt2.len() < 16 {
                            let total_tail = rem_pt1 + pt2.len();
                            let mut tmp = [0u8; 16];
                            core::ptr::copy_nonoverlapping(pt1_ptr, tmp.as_mut_ptr(), rem_pt1);
                            core::ptr::copy_nonoverlapping(
                                pt2_ptr,
                                tmp.as_mut_ptr().add(rem_pt1),
                                pt2.len(),
                            );
                            (ct_ptr, offset, checksum) =
                                self.enc_tail(tmp.as_ptr(), total_tail, ct_ptr, offset, checksum);
                        } else {
                            // Reassemble one full block from pt1 tail + pt2 head
                            let need = 16 - rem_pt1;
                            let mut tmp = [0u8; 16];
                            core::ptr::copy_nonoverlapping(pt1_ptr, tmp.as_mut_ptr(), rem_pt1);
                            core::ptr::copy_nonoverlapping(
                                pt2_ptr,
                                tmp.as_mut_ptr().add(rem_pt1),
                                need,
                            );

                            offset = self.enc_single_block_local(
                                tmp.as_ptr(),
                                ct_ptr,
                                &mut block_num,
                                offset,
                                &mut checksum,
                            );
                            ct_ptr = ct_ptr.add(16);
                            pt2_ptr = pt2_ptr.add(need);

                            let remaining_pt2 = pt2.len() - need;
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
                        let full_blocks_pt2 = pt2.len() / 16;
                        let rem_pt2 = pt2.len() & 15;

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
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{EncryptIO, PlainText};
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
    fn enc_contiguous(
        ctx: &mut AEctx,
        nonce: &[u8; 12],
        pt: &[u8],
        ad: &[u8],
    ) -> (Vec<u8>, [u8; 16]) {
        let mut out = vec![0u8; pt.len() + 16];
        unsafe {
            ctx.ae_encrypt(
                EncryptIO::OutOfPlace {
                    in_pt: PlainText::Contiguous(pt),
                    out: &mut out,
                },
                nonce,
                ad,
            )
            .expect("encrypt failed");
        }
        let (ct, tag) = out.split_at(pt.len());
        (ct.to_vec(), tag.try_into().expect("tag is 16 bytes"))
    }

    /// Encrypt in place. Returns (ct, tag).
    fn enc_in_place(
        ctx: &mut AEctx,
        nonce: &[u8; 12],
        pt: &[u8],
        ad: &[u8],
    ) -> (Vec<u8>, [u8; 16]) {
        let mut buf = pt.to_vec();
        buf.extend([0u8; 16]);
        unsafe {
            ctx.ae_encrypt(EncryptIO::InPlace(&mut buf), nonce, ad)
                .expect("encrypt failed");
        }
        let (ct, tag) = buf.split_at(pt.len());
        (ct.to_vec(), tag.try_into().expect("tag is 16 bytes"))
    }

    /// Encrypt out-of-place with scattered plaintext split at `split`. Returns (ct, tag).
    fn enc_scattered(
        ctx: &mut AEctx,
        nonce: &[u8; 12],
        pt: &[u8],
        split: usize,
        ad: &[u8],
    ) -> (Vec<u8>, [u8; 16]) {
        let mut out = vec![0u8; pt.len() + 16];
        unsafe {
            ctx.ae_encrypt(
                EncryptIO::OutOfPlace {
                    in_pt: PlainText::Scattered(&pt[..split], &pt[split..]),
                    out: &mut out,
                },
                nonce,
                ad,
            )
            .expect("encrypt failed");
        }
        let (ct, tag) = out.split_at(pt.len());
        (ct.to_vec(), tag.try_into().expect("tag is 16 bytes"))
    }

    /// Check one vector in every API mode: in place, out-of-place contiguous,
    /// and out-of-place scattered at every split point.
    fn check_vector(ptl: usize, adl: usize, ct_exp: &[u8], tag_exp: &[u8]) {
        let (pt, ad) = (spec_data(ptl), spec_data(adl));

        let mut ctx = AEctx::new(&SPEC_KEY);
        let (ct, tag) = enc_in_place(&mut ctx, &SPEC_NONCE, &pt, &ad);
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

        let mut ctx = AEctx::new(&SPEC_KEY);
        let (ct, tag) = enc_contiguous(&mut ctx, &SPEC_NONCE, &pt, &ad);
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

        let mut ctx = AEctx::new(&SPEC_KEY);
        for split in 0..=ptl {
            let (ct, tag) = enc_scattered(&mut ctx, &SPEC_NONCE, &pt, split, &ad);
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
    }

    #[test]
    fn spec_vectors() {
        for &(ptl, adl, ct_exp, tag_exp) in &SPEC_VECTORS {
            check_vector(ptl, adl, ct_exp, tag_exp);
        }
    }

    /// A single context is reused for every vector, as the C reference harness
    /// does; this exercises the per-call state resets.
    #[test]
    fn spec_vectors_reused_context() {
        let mut ctx = AEctx::new(&SPEC_KEY);
        for &(ptl, adl, ct_exp, tag_exp) in &SPEC_VECTORS {
            let (pt, ad) = (spec_data(ptl), spec_data(adl));
            let (ct, tag) = enc_contiguous(&mut ctx, &SPEC_NONCE, &pt, &ad);
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
            let mut ctx = AEctx::new(&SPEC_KEY);
            let mut out = vec![0xffu8; ptl + 16];
            unsafe {
                ctx.ae_decrypt(
                    DecryptIO::OutOfPlace {
                        in_ct: ct_tag,
                        out_pt: &mut out,
                    },
                    &SPEC_NONCE,
                    &ad,
                )
            }
            .unwrap_or_else(|e| panic!("out-of-place decrypt failed ptl={} adl={}: {e}", ptl, adl));
            assert_eq!(
                &out[..ptl],
                &pt_exp,
                "out-of-place pt mismatch for ptl={} adl={}",
                ptl,
                adl
            );

            // In place: the plaintext must come back and the tag must survive.
            let mut ctx = AEctx::new(&SPEC_KEY);
            let mut buf = ct_tag.to_vec();
            unsafe { ctx.ae_decrypt(DecryptIO::InPlace(&mut buf), &SPEC_NONCE, &ad) }
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
            let mut ctx = AEctx::new(&SPEC_KEY);
            let mut bad = ct_tag.to_vec();
            bad[ptl] ^= 0x01;
            assert!(
                matches!(
                    unsafe { ctx.ae_decrypt(DecryptIO::InPlace(&mut bad), &SPEC_NONCE, &ad) },
                    Err(crate::CryptoError::TagInvalid)
                ),
                "corrupted tag accepted for ptl={} adl={}",
                ptl,
                adl
            );

            // Corrupted ciphertext must be rejected.
            if ptl > 0 {
                let mut ctx = AEctx::new(&SPEC_KEY);
                let mut bad = ct_tag.to_vec();
                bad[ptl / 2] ^= 0x01;
                assert!(
                    matches!(
                        unsafe { ctx.ae_decrypt(DecryptIO::InPlace(&mut bad), &SPEC_NONCE, &ad) },
                        Err(crate::CryptoError::TagInvalid)
                    ),
                    "corrupted ct accepted for ptl={} adl={}",
                    ptl,
                    adl
                );
            }

            // Corrupted AD must be rejected.
            if adl > 0 {
                let mut ctx = AEctx::new(&SPEC_KEY);
                let mut bad_ad = ad.clone();
                bad_ad[adl / 2] ^= 0x01;
                assert!(
                    matches!(
                        unsafe {
                            ctx.ae_decrypt(
                                DecryptIO::InPlace(&mut ct_tag.to_vec()),
                                &SPEC_NONCE,
                                &bad_ad,
                            )
                        },
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

                let mut ctx = AEctx::new(&SPEC_KEY);
                let (ct, tag) = enc_in_place(&mut ctx, &SPEC_NONCE, &pt, &ad);
                let mut blob = ct;
                blob.extend_from_slice(&tag);

                // In place.
                let mut buf = blob.clone();
                unsafe { ctx.ae_decrypt(DecryptIO::InPlace(&mut buf), &SPEC_NONCE, &ad) }
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
                unsafe {
                    ctx.ae_decrypt(
                        DecryptIO::OutOfPlace {
                            in_ct: &blob,
                            out_pt: &mut out,
                        },
                        &SPEC_NONCE,
                        &ad,
                    )
                }
                .unwrap_or_else(|e| panic!("o-o-p roundtrip failed ptl={} adl={}: {e}", ptl, adl));
                assert_eq!(&out[..ptl], &pt, "o-o-p roundtrip ptl={} adl={}", ptl, adl);
            }
        }
        for ptl in [2048usize, 4096, 8192] {
            let (pt, ad) = (spec_data(ptl), spec_data(129));
            let mut ctx = AEctx::new(&SPEC_KEY);
            let (ct, tag) = enc_in_place(&mut ctx, &SPEC_NONCE, &pt, &ad);
            let mut blob = ct;
            blob.extend_from_slice(&tag);
            let mut buf = blob.clone();
            unsafe { ctx.ae_decrypt(DecryptIO::InPlace(&mut buf), &SPEC_NONCE, &ad) }
                .unwrap_or_else(|e| panic!("in-place roundtrip failed ptl={}: {e}", ptl));
            assert_eq!(&buf[..ptl], &pt, "in-place roundtrip ptl={}", ptl);
        }
    }
}
