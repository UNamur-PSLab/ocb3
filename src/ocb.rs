#![allow(unsafe_op_in_unsafe_fn)]
use crate::EncryptIO;
use crate::PlainText;
use crate::Result;
use crate::aes::AesKey;
use crate::sse::*;
use core::arch::x86_64::*;
use std::env::consts;
use std::num;

// Note: this code cannot process ciphertext or plaintext larger than 2^20
const L_TABLE_SZ: usize = 16;
const BPI: usize = 8;

#[repr(C, align(16))]
struct AEctx {
    offset: __m128i,
    checksum: __m128i,
    Lstar: __m128i,
    Ldollar: __m128i,
    L: [__m128i; L_TABLE_SZ],
    ad_checksum: __m128i,
    ad_offset: __m128i,
    cached_Top: __m128i,
    decrypt_key: AesKey,
    encrypt_key: AesKey,
    KtopStr: [u64; 3],
}

impl AEctx {
    pub fn new(key: &[u8]) -> Self {
        // Safety: needs to be 16-bytes aligned.
        let mut ctx = unsafe {
            AEctx {
                offset: _mm_setzero_si128(),
                checksum: _mm_setzero_si128(),
                Lstar: _mm_setzero_si128(),
                Ldollar: _mm_setzero_si128(),
                L: [_mm_setzero_si128(); L_TABLE_SZ],
                ad_checksum: _mm_setzero_si128(),
                ad_offset: _mm_setzero_si128(),
                cached_Top: _mm_setzero_si128(),
                decrypt_key: AesKey::default(),
                encrypt_key: AesKey::default(),
                KtopStr: [0u64; 3],
            }
        };

        // Key dependent values
        ctx.encrypt_key.AES_128_key_expansion(key);
        ctx.decrypt_key.AES_set_key_fast(&ctx.encrypt_key);
        ctx.Lstar = ctx.encrypt_key.AES_encrypt_block(ctx.cached_Top);
        let mut tmp_blk = swap_if_le(ctx.Lstar);
        tmp_blk = double(tmp_blk);
        ctx.Ldollar = swap_if_le(tmp_blk);
        tmp_blk = double(tmp_blk);
        ctx.L[0] = swap_if_le(tmp_blk);

        for i in 0..L_TABLE_SZ {
            tmp_blk = double(tmp_blk);
            ctx.L[i] = swap_if_le(tmp_blk);
        }

        ctx
    }

    #[inline(always)]
    fn gen_offset_from_nonce(&mut self, nonce: &[u8; 12]) -> __m128i {
        let mut tmp = [0u8; 16];
        tmp[3] = 0x01;
        tmp[4..16].copy_from_slice(nonce);
        let idx: i32 = tmp[15] as i32 & 0x3f;
        tmp[15] &= 0xc0;
        unsafe {
            let mut bl = _mm_loadu_si128(tmp.as_ptr() as *const __m128i);
            if unequal_blocks(bl, self.cached_Top) {
                self.cached_Top = bl;
                bl = self.encrypt_key.AES_encrypt_block(bl);
                [self.KtopStr[0], self.KtopStr[1]] = m128i_to_bswap_u64x2(bl);
                self.KtopStr[2] =
                    self.KtopStr[0] ^ (self.KtopStr[0] << 8) ^ (self.KtopStr[1] >> 56);
            }

            gen_offset(&self.KtopStr, idx)
        }
    }

    #[inline(always)]
    fn process_ad(&mut self, ad: &[u8]) {
        let num_bpi = ad.len() / (BPI * 16);
        let mut ad_blocks_num: usize = 0;
        let l0 = self.L[0];
        let l1 = self.L[1];
        let l2 = self.L[2];
        let mut ad_ptr = ad.as_ptr();
        unsafe {
            for _ in 0..num_bpi {
                ad_blocks_num += BPI;
                let tz = ad_blocks_num.trailing_zeros() as usize;
                let oa = xor_block(self.ad_offset, l0);
                let ad0 = _mm_loadu_si128(ad_ptr as *const __m128i);
                let ta = xor_block(oa, ad0);
                let ct = self.encrypt_key.AES_encrypt_block(ta);
                self.ad_checksum = xor_block(self.ad_checksum, ct);
                let oa1 = xor_block(oa, l1);
                let ad1 = _mm_loadu_si128(ad_ptr.add(16) as *const __m128i);
                let ta1 = xor_block(oa1, ad1);
                let ct = self.encrypt_key.AES_encrypt_block(ta1);
                self.ad_checksum = xor_block(self.ad_checksum, ct);
                let oa2 = xor_block(self.ad_offset, l1);
                let ad2 = _mm_loadu_si128(ad_ptr.add(32) as *const __m128i);
                let ta2 = xor_block(oa2, ad2);
                let ct = self.encrypt_key.AES_encrypt_block(ta2);
                self.ad_checksum = xor_block(self.ad_checksum, ct);
                let oa3 = xor_block(oa2, l2);
                let ad3 = _mm_loadu_si128(ad_ptr.add(48) as *const __m128i);
                let ta3 = xor_block(oa3, ad3);
                let ct = self.encrypt_key.AES_encrypt_block(ta3);
                self.ad_checksum = xor_block(self.ad_checksum, ct);
                let oa4 = xor_block(oa1, l2);
                let ad4 = _mm_loadu_si128(ad_ptr.add(64) as *const __m128i);
                let ta4 = xor_block(oa4, ad4);
                let ct = self.encrypt_key.AES_encrypt_block(ta4);
                self.ad_checksum = xor_block(self.ad_checksum, ct);
                let oa5 = xor_block(oa, l2);
                let ad5 = _mm_loadu_si128(ad_ptr.add(80) as *const __m128i);
                let ta5 = xor_block(oa5, ad5);
                let ct = self.encrypt_key.AES_encrypt_block(ta5);
                self.ad_checksum = xor_block(self.ad_checksum, ct);
                let oa6 = xor_block(self.ad_offset, l2);
                let ad6 = _mm_loadu_si128(ad_ptr.add(96) as *const __m128i);
                let ta6 = xor_block(oa6, ad6);
                let ct = self.encrypt_key.AES_encrypt_block(ta6);
                self.ad_checksum = xor_block(self.ad_checksum, ct);
                self.ad_offset = xor_block(oa6, self.L[tz]);
                let ad7 = _mm_loadu_si128(ad_ptr.add(112) as *const __m128i);
                let ta7 = xor_block(self.ad_offset, ad7);
                let ct = self.encrypt_key.AES_encrypt_block(ta7);
                self.ad_checksum = xor_block(self.ad_checksum, ct);
                ad_ptr = ad_ptr.add(128);
            }

            let mut remaining = ad.len() % (BPI * 16);
            let mut k = 0;
            if remaining >= 64 {
                let bl = xor_block(self.ad_offset, l0);
                let ad0 = _mm_loadu_si128(ad_ptr as *const __m128i);
                let ta0 = xor_block(bl, ad0);
                let ct = self.encrypt_key.AES_encrypt_block(ta0);
                self.ad_checksum = xor_block(self.ad_checksum, ct);
                let bl = xor_block(bl, l1);
                let ad1 = _mm_loadu_si128(ad_ptr.add(16) as *const __m128i);
                let ta1 = xor_block(bl, ad1);
                let ct = self.encrypt_key.AES_encrypt_block(ta1);
                self.ad_checksum = xor_block(self.ad_checksum, ct);
                let ad2 = _mm_loadu_si128(ad_ptr.add(32) as *const __m128i);
                self.ad_offset = xor_block(self.ad_offset, l1);
                let ta2 = xor_block(self.ad_offset, ad2);
                let ct = self.encrypt_key.AES_encrypt_block(ta2);
                self.ad_checksum = xor_block(self.ad_checksum, ct);
                self.ad_offset = xor_block(self.ad_offset, l2);
                let ad3 = _mm_loadu_si128(ad_ptr.add(48) as *const __m128i);
                let ta3 = xor_block(self.ad_offset, ad3);
                let ct = self.encrypt_key.AES_encrypt_block(ta3);
                self.ad_checksum = xor_block(self.ad_checksum, ct);
                remaining -= 64;
                k += 4;
                ad_ptr = ad_ptr.add(64);
            }
            if remaining >= 32 {
                self.ad_offset = xor_block(self.ad_offset, l0);
                let ad0 = _mm_loadu_si128(ad_ptr as *const __m128i);
                let ta0 = xor_block(self.ad_offset, ad0);
                let ct = self.encrypt_key.AES_encrypt_block(ta0);
                self.ad_checksum = xor_block(self.ad_checksum, ct);
                let tz = (k + 2u8).trailing_zeros() as usize;
                self.ad_offset = xor_block(self.ad_offset, self.L[tz]);
                let ad1 = _mm_loadu_si128(ad_ptr.add(16) as *const __m128i);
                let ta1 = xor_block(self.ad_offset, ad1);
                let ct = self.encrypt_key.AES_encrypt_block(ta1);
                self.ad_checksum = xor_block(self.ad_checksum, ct);
                ad_ptr = ad_ptr.add(32);
                remaining -= 32;
            }
            if remaining >= 16 {
                self.ad_offset = xor_block(self.ad_offset, l0);
                let ad0 = _mm_loadu_si128(ad_ptr as *const __m128i);
                let ta0 = xor_block(self.ad_offset, ad0);
                let ct = self.encrypt_key.AES_encrypt_block(ta0);
                self.ad_checksum = xor_block(self.ad_checksum, ct);
                ad_ptr = ad_ptr.add(16);
                remaining -= 16;
            }
            if remaining > 0 {
                let mut tmp = [0u8; 16];
                core::ptr::copy(ad_ptr, tmp.as_mut_ptr(), remaining);
                tmp[remaining] = 0x80;
                self.ad_offset = xor_block(self.ad_offset, self.Lstar);
                let ad0 = _mm_loadu_si128(tmp.as_ptr() as *const __m128i);
                let ta = xor_block(self.ad_offset, ad0);
                let ct = self.encrypt_key.AES_encrypt_block(ta);
                self.ad_checksum = xor_block(self.ad_checksum, ct);
            }
        }
    }

    /// Encrypts one 16-byte block
    /// Returns oa (the block offset) and updates self.offset and self.checksum.
    #[inline(always)]
    unsafe fn enc_block(&mut self, l: __m128i, pt_ptr: *const u8, ct_ptr: *mut u8) -> __m128i {
        let oa = xor_block(self.offset, l);
        let pt = _mm_loadu_si128(pt_ptr as *const __m128i);
        let t = xor_block(oa, pt);
        let ct = self.encrypt_key.AES_encrypt_block(t);
        _mm_storeu_si128(ct_ptr as *mut __m128i, xor_block(ct, oa));
        self.checksum = xor_block(self.checksum, pt);
        self.offset = oa;
        oa
    }

    /// Encrypts a single block at `pt_ptr`, incrementing `block_num` and using `L[ntz(block_num)]`.
    #[inline(always)]
    unsafe fn enc_single_block(&mut self, pt_ptr: *const u8, ct_ptr: *mut u8, block_num: &mut u32) {
        *block_num += 1;
        let ntz = block_num.trailing_zeros() as usize;
        self.enc_block(self.L[ntz], pt_ptr, ct_ptr);
    }

    /// Encrypts a full BPI (8-block) group using the algebraic offset shortcut.
    /// `block_num` is the block count **before** this group and MUST be a multiple of 8.
    /// Returns the updated `block_num`.
    #[inline(always)]
    unsafe fn enc_bpi(&mut self, pt_ptr: *const u8, ct_ptr: *mut u8, block_num: u32) -> u32 {
        debug_assert_eq!(block_num % 8, 0);
        let block_num = block_num + BPI as u32;
        let l0 = self.L[0];
        let l1 = self.L[1];
        let l2 = self.L[2];
        let offset = self.offset;

        let oa0 = self.enc_block(l0, pt_ptr, ct_ptr);
        let oa1 = self.enc_block(l1, pt_ptr.add(16), ct_ptr.add(16));
        let _ = self.enc_block(l0, pt_ptr.add(32), ct_ptr.add(32));
        let _ = self.enc_block(l2, pt_ptr.add(48), ct_ptr.add(48));
        // algebraic shortcuts — valid only when starting at a multiple-of-8 boundary
        self.offset = oa1;
        let _ = self.enc_block(l2, pt_ptr.add(64), ct_ptr.add(64));
        self.offset = oa0;
        let _ = self.enc_block(l2, pt_ptr.add(80), ct_ptr.add(80));
        self.offset = offset;
        let oa6 = self.enc_block(l2, pt_ptr.add(96), ct_ptr.add(96));
        let ntz = block_num.trailing_zeros() as usize;
        self.offset = oa6;
        let _ = self.enc_block(self.L[ntz], pt_ptr.add(112), ct_ptr.add(112));

        block_num
    }

    /// Encrypts contiguous full blocks from `pt_ptr` into `ct_ptr`.
    /// First aligns to an 8-block boundary, then processes full BPI groups,
    /// then finishes any remaining full blocks.
    #[inline(always)]
    unsafe fn enc_contiguous(
        &mut self,
        mut pt_ptr: *const u8,
        mut ct_ptr: *mut u8,
        mut full_blocks: usize,
        block_num: &mut u32,
    ) -> (*const u8, *mut u8) {
        // Align to an 8-block boundary so enc_bpi can be safely used
        while !(*block_num).is_multiple_of(8) && full_blocks > 0 {
            self.enc_single_block(pt_ptr, ct_ptr, block_num);
            pt_ptr = pt_ptr.add(16);
            ct_ptr = ct_ptr.add(16);
            full_blocks -= 1;
        }

        // Process aligned full BPI groups using the original implementation algebraic shortcut
        let num_bpi = full_blocks / BPI;
        for _ in 0..num_bpi {
            *block_num = self.enc_bpi(pt_ptr, ct_ptr, *block_num);
            pt_ptr = pt_ptr.add(BPI * 16);
            ct_ptr = ct_ptr.add(BPI * 16);
        }
        full_blocks %= BPI;

        // Process remaining full blocks (< 8)
        while full_blocks > 0 {
            self.enc_single_block(pt_ptr, ct_ptr, block_num);
            pt_ptr = pt_ptr.add(16);
            ct_ptr = ct_ptr.add(16);
            full_blocks -= 1;
        }

        (pt_ptr, ct_ptr)
    }

    /// Encrypts the final partial block (`len < 16`) with `0x80` padding and L*.
    #[inline(always)]
    unsafe fn enc_tail(&mut self, data_ptr: *const u8, data_len: usize, ct_ptr: *mut u8) {
        let mut tmp = [0u8; 16];
        core::ptr::copy_nonoverlapping(data_ptr, tmp.as_mut_ptr(), data_len);
        tmp[data_len] = 0x80;
        let pt_padded = _mm_loadu_si128(tmp.as_ptr() as *const __m128i);
        self.checksum = xor_block(self.checksum, pt_padded);
        let t = xor_block(self.offset, self.Lstar);
        let pad = self.encrypt_key.AES_encrypt_block(t);
        let mut pad_bytes = [0u8; 16];
        _mm_storeu_si128(pad_bytes.as_mut_ptr() as *mut __m128i, pad);
        for i in 0..data_len {
            *ct_ptr.add(i) = tmp[i] ^ pad_bytes[i];
        }
        self.offset = t;
    }

    /// Compute and store the final tag
    #[inline(always)]
    unsafe fn compute_and_store_tag(&mut self, ct_ptr: *mut u8) {
        self.offset = xor_block(self.offset, self.Ldollar);
        let ta = xor_block(self.offset, self.checksum);
        let ct = self.encrypt_key.AES_encrypt_block(ta);
        self.offset = xor_block(ct, self.ad_checksum);
        _mm_storeu_si128(ct_ptr as *mut __m128i, self.offset);
    }

    /// Todo
    pub fn ae_encrypt(&mut self, io: EncryptIO, nonce: &[u8; 12], ad: &[u8]) -> Result<()> {
        io.validate()?;
        self.offset = self.gen_offset_from_nonce(nonce);
        unsafe {
            self.ad_offset = _mm_setzero_si128();
            self.checksum = _mm_setzero_si128();
        }
        if !ad.is_empty() {
            self.process_ad(ad);
        }

        unsafe {
            let mut block_num = 0u32;

            match io {
                EncryptIO::InPlace(buf) => {
                    let mut pt_ptr = buf.as_ptr();
                    let mut ct_ptr = buf.as_mut_ptr();
                    let full_blocks = buf.len() / 16;
                    let rem = buf.len() % 16;

                    (pt_ptr, ct_ptr) =
                        self.enc_contiguous(pt_ptr, ct_ptr, full_blocks, &mut block_num);
                    if rem > 0 {
                        self.enc_tail(pt_ptr, rem, ct_ptr);
                    }
                    self.compute_and_store_tag(ct_ptr);
                }
                EncryptIO::OutOfPlace { in_pt, out } => match in_pt {
                    PlainText::Contiguous(pt) => {
                        let mut pt_ptr = pt.as_ptr();
                        let mut ct_ptr = out.as_mut_ptr();
                        let full_blocks = pt.len() / 16;
                        let rem = pt.len() % 16;

                        (pt_ptr, ct_ptr) =
                            self.enc_contiguous(pt_ptr, ct_ptr, full_blocks, &mut block_num);
                        if rem > 0 {
                            self.enc_tail(pt_ptr, rem, ct_ptr);
                        }
                        self.compute_and_store_tag(ct_ptr);
                    }
                    PlainText::Scattered(pt1, pt2) => {
                        let mut pt1_ptr = pt1.as_ptr();
                        let mut pt2_ptr = pt2.as_ptr();
                        let mut ct_ptr = out.as_mut_ptr();

                        let full_blocks_pt1 = pt1.len() / 16;
                        let rem_pt1 = pt1.len() % 16;

                        (pt1_ptr, ct_ptr) =
                            self.enc_contiguous(pt1_ptr, ct_ptr, full_blocks_pt1, &mut block_num);

                        if rem_pt1 > 0 {
                            if pt2.is_empty() {
                                self.enc_tail(pt1_ptr, rem_pt1, ct_ptr);
                            } else if rem_pt1 + pt2.len() < 16 {
                                let total_tail = rem_pt1 + pt2.len();
                                let mut tmp = [0u8; 16];
                                core::ptr::copy_nonoverlapping(pt1_ptr, tmp.as_mut_ptr(), rem_pt1);
                                core::ptr::copy_nonoverlapping(
                                    pt2_ptr,
                                    tmp.as_mut_ptr().add(rem_pt1),
                                    pt2.len(),
                                );
                                self.enc_tail(tmp.as_ptr(), total_tail, ct_ptr);
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

                                self.enc_single_block(tmp.as_ptr(), ct_ptr, &mut block_num);
                                ct_ptr = ct_ptr.add(16);
                                pt2_ptr = pt2_ptr.add(need);

                                let remaining_pt2 = pt2.len() - need;
                                let full_blocks_pt2 = remaining_pt2 / 16;
                                let rem_pt2 = remaining_pt2 % 16;

                                (pt2_ptr, ct_ptr) = self.enc_contiguous(
                                    pt2_ptr,
                                    ct_ptr,
                                    full_blocks_pt2,
                                    &mut block_num,
                                );
                                if rem_pt2 > 0 {
                                    self.enc_tail(pt2_ptr, rem_pt2, ct_ptr);
                                }
                            }
                        } else {
                            // rem_pt1 == 0, pt2 continues directly
                            let full_blocks_pt2 = pt2.len() / 16;
                            let rem_pt2 = pt2.len() % 16;

                            (pt2_ptr, ct_ptr) = self.enc_contiguous(
                                pt2_ptr,
                                ct_ptr,
                                full_blocks_pt2,
                                &mut block_num,
                            );
                            if rem_pt2 > 0 {
                                self.enc_tail(pt2_ptr, rem_pt2, ct_ptr);
                            }
                        }
                        self.compute_and_store_tag(ct_ptr);
                    }
                },
            }
        }
        Ok(())
    }
}
