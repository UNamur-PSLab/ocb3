#![allow(unsafe_op_in_unsafe_fn)]
use crate::EncryptIO;
use crate::OCB_TAG_LEN;
use crate::PlainText;
use crate::Result;
use crate::aes::AesKey;
use crate::sse::*;
use core::arch::x86_64::*;

const L_TABLE_SZ: usize = 16;
const BPI: usize = 8;

#[repr(C, align(16))]
pub struct AEctx {
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

        for i in 1..L_TABLE_SZ {
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
            let mut ad_offset = self.ad_offset;
            let mut ad_checksum = self.ad_checksum;
            for _ in 0..num_bpi {
                ad_blocks_num += BPI;
                let tz = ad_blocks_num.trailing_zeros() as usize;
                let oa0 = xor_block(ad_offset, l0);
                let oa1 = xor_block(oa0, l1);
                let oa2 = xor_block(ad_offset, l1);
                let oa3 = xor_block(oa2, l2);

                let ad0 = _mm_loadu_si128(ad_ptr as *const __m128i);
                let ad1 = _mm_loadu_si128(ad_ptr.add(16) as *const __m128i);
                let ad2 = _mm_loadu_si128(ad_ptr.add(32) as *const __m128i);
                let ad3 = _mm_loadu_si128(ad_ptr.add(48) as *const __m128i);

                let ct_lo = self.encrypt_key.AES_encrypt_4_blocks([
                    xor_block(oa0, ad0),
                    xor_block(oa1, ad1),
                    xor_block(oa2, ad2),
                    xor_block(oa3, ad3),
                ]);

                let sum0 = xor_block(xor_block(ct_lo[0], ct_lo[1]), xor_block(ct_lo[2], ct_lo[3]));
                ad_checksum = xor_block(ad_checksum, sum0);

                let oa4 = xor_block(oa1, l2);
                let oa5 = xor_block(oa0, l2);
                let oa6 = xor_block(ad_offset, l2);
                ad_offset = xor_block(oa6, self.L[tz]);

                let ad4 = _mm_loadu_si128(ad_ptr.add(64) as *const __m128i);
                let ad5 = _mm_loadu_si128(ad_ptr.add(80) as *const __m128i);
                let ad6 = _mm_loadu_si128(ad_ptr.add(96) as *const __m128i);
                let ad7 = _mm_loadu_si128(ad_ptr.add(112) as *const __m128i);

                let ct_hi = self.encrypt_key.AES_encrypt_4_blocks([
                    xor_block(oa4, ad4),
                    xor_block(oa5, ad5),
                    xor_block(oa6, ad6),
                    xor_block(ad_offset, ad7),
                ]);

                let sum1 = xor_block(xor_block(ct_hi[0], ct_hi[1]), xor_block(ct_hi[2], ct_hi[3]));
                ad_checksum = xor_block(ad_checksum, sum1);

                ad_ptr = ad_ptr.add(128);
            }

            let mut remaining = ad.len() % (BPI * 16);
            let mut k = 0;
            if remaining >= 64 {
                let bl = xor_block(ad_offset, l0);
                let ad0 = _mm_loadu_si128(ad_ptr as *const __m128i);
                let ta0 = xor_block(bl, ad0);
                let ct = self.encrypt_key.AES_encrypt_block(ta0);
                ad_checksum = xor_block(ad_checksum, ct);
                let bl = xor_block(bl, l1);
                let ad1 = _mm_loadu_si128(ad_ptr.add(16) as *const __m128i);
                let ta1 = xor_block(bl, ad1);
                let ct = self.encrypt_key.AES_encrypt_block(ta1);
                ad_checksum = xor_block(ad_checksum, ct);
                let ad2 = _mm_loadu_si128(ad_ptr.add(32) as *const __m128i);
                ad_offset = xor_block(ad_offset, l1);
                let ta2 = xor_block(ad_offset, ad2);
                let ct = self.encrypt_key.AES_encrypt_block(ta2);
                ad_checksum = xor_block(ad_checksum, ct);
                ad_offset = xor_block(ad_offset, l2);
                let ad3 = _mm_loadu_si128(ad_ptr.add(48) as *const __m128i);
                let ta3 = xor_block(ad_offset, ad3);
                let ct = self.encrypt_key.AES_encrypt_block(ta3);
                ad_checksum = xor_block(ad_checksum, ct);
                remaining -= 64;
                k += 4;
                ad_ptr = ad_ptr.add(64);
            }
            if remaining >= 32 {
                ad_offset = xor_block(ad_offset, l0);
                let ad0 = _mm_loadu_si128(ad_ptr as *const __m128i);
                let ta0 = xor_block(ad_offset, ad0);
                let ct = self.encrypt_key.AES_encrypt_block(ta0);
                ad_checksum = xor_block(ad_checksum, ct);
                let tz = (k + 2u8).trailing_zeros() as usize;
                ad_offset = xor_block(ad_offset, self.L[tz]);
                let ad1 = _mm_loadu_si128(ad_ptr.add(16) as *const __m128i);
                let ta1 = xor_block(ad_offset, ad1);
                let ct = self.encrypt_key.AES_encrypt_block(ta1);
                ad_checksum = xor_block(ad_checksum, ct);
                ad_ptr = ad_ptr.add(32);
                remaining -= 32;
            }
            if remaining >= 16 {
                ad_offset = xor_block(ad_offset, l0);
                let ad0 = _mm_loadu_si128(ad_ptr as *const __m128i);
                let ta0 = xor_block(ad_offset, ad0);
                let ct = self.encrypt_key.AES_encrypt_block(ta0);
                ad_checksum = xor_block(ad_checksum, ct);
                ad_ptr = ad_ptr.add(16);
                remaining -= 16;
            }
            if remaining > 0 {
                let mut tmp = [0u8; 16];
                core::ptr::copy(ad_ptr, tmp.as_mut_ptr(), remaining);
                tmp[remaining] = 0x80;
                ad_offset = xor_block(ad_offset, self.Lstar);
                let ad0 = _mm_loadu_si128(tmp.as_ptr() as *const __m128i);
                let ta = xor_block(ad_offset, ad0);
                let ct = self.encrypt_key.AES_encrypt_block(ta);
                ad_checksum = xor_block(ad_checksum, ct);
            }
            self.ad_offset = ad_offset;
            self.ad_checksum = ad_checksum;
        }
    }

    #[inline(always)]
    unsafe fn enc_block(
        &mut self,
        oa: __m128i,
        pt_ptr: *const u8,
        ct_ptr: *mut u8,
        checksum: &mut __m128i,
    ) {
        let pt = _mm_loadu_si128(pt_ptr as *const __m128i);
        let t = xor_block(oa, pt);
        let ct = self.encrypt_key.AES_encrypt_block(t);
        _mm_storeu_si128(ct_ptr as *mut __m128i, xor_block(ct, oa));
        *checksum = xor_block(*checksum, pt);
    }

    #[inline(always)]
    unsafe fn enc_single_block(&mut self, pt_ptr: *const u8, ct_ptr: *mut u8, block_num: &mut u32) {
        *block_num += 1;
        let ntz = block_num.trailing_zeros() as usize;
        let oa = xor_block(self.offset, self.L[ntz]);
        let mut checksum = self.checksum;
        self.enc_block(oa, pt_ptr, ct_ptr, &mut checksum);
        self.checksum = checksum;
        self.offset = oa;
    }

    #[inline(always)]
    unsafe fn enc_single_block_local(
        &mut self,
        pt_ptr: *const u8,
        ct_ptr: *mut u8,
        block_num: &mut u32,
        offset: __m128i,
        checksum: &mut __m128i,
    ) -> __m128i {
        *block_num += 1;
        let ntz = block_num.trailing_zeros() as usize;
        let oa = xor_block(offset, self.L[ntz]);
        self.enc_block(oa, pt_ptr, ct_ptr, checksum);
        oa
    }

    #[inline(always)]
    unsafe fn enc_bpi(
        &mut self,
        pt_ptr: *const u8,
        ct_ptr: *mut u8,
        block_num: u32,
        offset: __m128i,
        checksum: &mut __m128i,
    ) -> (__m128i, u32) {
        debug_assert_eq!(block_num % 8, 0);
        let block_num = block_num + BPI as u32;
        let l0 = self.L[0];
        let l1 = self.L[1];
        let l2 = self.L[2];

        let oa0 = xor_block(offset, l0);
        let oa1 = xor_block(oa0, l1);
        let oa2 = xor_block(oa1, l0);
        let oa3 = xor_block(oa2, l2);

        let pt0 = _mm_loadu_si128(pt_ptr as *const __m128i);
        let pt1 = _mm_loadu_si128(pt_ptr.add(16) as *const __m128i);
        let pt2 = _mm_loadu_si128(pt_ptr.add(32) as *const __m128i);
        let pt3 = _mm_loadu_si128(pt_ptr.add(48) as *const __m128i);

        let pt_sum0 = xor_block(xor_block(pt0, pt1), xor_block(pt2, pt3));
        *checksum = xor_block(*checksum, pt_sum0);

        // Encrypting 4 blocks at once seems to avoid to spill a xmm register on the stack.
        // For 8 blocks, we needed 18 registers (16 for oa[0..7] and ta[0..7] (now directly xored)
        // and 2 for the offset and checksum, but we only had 16.
        //
        // Encrypting 4 blocks at once seems to still allow to fully occupy the processor's
        // pipelines.
        let ct_lo = self.encrypt_key.AES_encrypt_4_blocks([
            xor_block(oa0, pt0),
            xor_block(oa1, pt1),
            xor_block(oa2, pt2),
            xor_block(oa3, pt3),
        ]);

        _mm_storeu_si128(ct_ptr as *mut __m128i, xor_block(ct_lo[0], oa0));
        _mm_storeu_si128(ct_ptr.add(16) as *mut __m128i, xor_block(ct_lo[1], oa1));
        _mm_storeu_si128(ct_ptr.add(32) as *mut __m128i, xor_block(ct_lo[2], oa2));
        _mm_storeu_si128(ct_ptr.add(48) as *mut __m128i, xor_block(ct_lo[3], oa3));

        let oa4 = xor_block(oa1, l2);
        let oa5 = xor_block(oa0, l2);
        let oa6 = xor_block(offset, l2);
        let ntz = block_num.trailing_zeros() as usize;
        let oa7 = xor_block(oa6, self.L[ntz]);

        let pt4 = _mm_loadu_si128(pt_ptr.add(64) as *const __m128i);
        let pt5 = _mm_loadu_si128(pt_ptr.add(80) as *const __m128i);
        let pt6 = _mm_loadu_si128(pt_ptr.add(96) as *const __m128i);
        let pt7 = _mm_loadu_si128(pt_ptr.add(112) as *const __m128i);

        let pt_sum1 = xor_block(xor_block(pt4, pt5), xor_block(pt6, pt7));
        *checksum = xor_block(*checksum, pt_sum1);

        let ct_hi = self.encrypt_key.AES_encrypt_4_blocks([
            xor_block(oa4, pt4),
            xor_block(oa5, pt5),
            xor_block(oa6, pt6),
            xor_block(oa7, pt7),
        ]);

        _mm_storeu_si128(ct_ptr.add(64) as *mut __m128i, xor_block(ct_hi[0], oa4));
        _mm_storeu_si128(ct_ptr.add(80) as *mut __m128i, xor_block(ct_hi[1], oa5));
        _mm_storeu_si128(ct_ptr.add(96) as *mut __m128i, xor_block(ct_hi[2], oa6));
        _mm_storeu_si128(ct_ptr.add(112) as *mut __m128i, xor_block(ct_hi[3], oa7));

        (oa7, block_num)
    }

    #[inline(always)]
    unsafe fn enc_contiguous(
        &mut self,
        mut pt_ptr: *const u8,
        mut ct_ptr: *mut u8,
        mut full_blocks: usize,
        block_num: &mut u32,
    ) -> (*const u8, *mut u8) {
        let mut offset = self.offset;
        let mut checksum = self.checksum;

        while !(*block_num).is_multiple_of(8) && full_blocks > 0 {
            offset = self.enc_single_block_local(pt_ptr, ct_ptr, block_num, offset, &mut checksum);
            pt_ptr = pt_ptr.add(16);
            ct_ptr = ct_ptr.add(16);
            full_blocks -= 1;
        }

        let num_bpi = full_blocks / BPI;
        for _ in 0..num_bpi {
            (offset, *block_num) = self.enc_bpi(pt_ptr, ct_ptr, *block_num, offset, &mut checksum);
            pt_ptr = pt_ptr.add(BPI * 16);
            ct_ptr = ct_ptr.add(BPI * 16);
        }
        full_blocks %= BPI;

        while full_blocks > 0 {
            offset = self.enc_single_block_local(pt_ptr, ct_ptr, block_num, offset, &mut checksum);
            pt_ptr = pt_ptr.add(16);
            ct_ptr = ct_ptr.add(16);
            full_blocks -= 1;
        }

        self.offset = offset;
        self.checksum = checksum;

        (pt_ptr, ct_ptr)
    }

    /// Encrypts the final partial block (`len < 16`) with `0x80` padding and L*.
    /// Returns the `ct_ptr` advanced past the written ciphertext.
    #[inline(always)]
    unsafe fn enc_tail(
        &mut self,
        data_ptr: *const u8,
        data_len: usize,
        ct_ptr: *mut u8,
    ) -> *mut u8 {
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
        ct_ptr.add(data_len)
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

    #[target_feature(enable = "aes,ssse3,sse4.1")]
    pub fn ae_encrypt(&mut self, io: EncryptIO, nonce: &[u8; 12], ad: &[u8]) -> Result<()> {
        io.validate()?;
        self.offset = self.gen_offset_from_nonce(nonce);
        self.ad_offset = _mm_setzero_si128();
        self.ad_checksum = _mm_setzero_si128();
        self.checksum = _mm_setzero_si128();
        if !ad.is_empty() {
            self.process_ad(ad);
        }

        unsafe {
            let mut block_num = 0u32;

            match io {
                EncryptIO::InPlace(buf) => {
                    // The buffer holds the plaintext followed by a 16-byte tag slot.
                    let pt_len = buf.len() - OCB_TAG_LEN as usize;
                    let mut pt_ptr = buf.as_ptr();
                    let mut ct_ptr = buf.as_mut_ptr();
                    let full_blocks = pt_len / 16;
                    let rem = pt_len % 16;

                    (pt_ptr, ct_ptr) =
                        self.enc_contiguous(pt_ptr, ct_ptr, full_blocks, &mut block_num);
                    if rem > 0 {
                        ct_ptr = self.enc_tail(pt_ptr, rem, ct_ptr);
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
                            ct_ptr = self.enc_tail(pt_ptr, rem, ct_ptr);
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
                                ct_ptr = self.enc_tail(pt1_ptr, rem_pt1, ct_ptr);
                            } else if rem_pt1 + pt2.len() < 16 {
                                let total_tail = rem_pt1 + pt2.len();
                                let mut tmp = [0u8; 16];
                                core::ptr::copy_nonoverlapping(pt1_ptr, tmp.as_mut_ptr(), rem_pt1);
                                core::ptr::copy_nonoverlapping(
                                    pt2_ptr,
                                    tmp.as_mut_ptr().add(rem_pt1),
                                    pt2.len(),
                                );
                                ct_ptr = self.enc_tail(tmp.as_ptr(), total_tail, ct_ptr);
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
                                    ct_ptr = self.enc_tail(pt2_ptr, rem_pt2, ct_ptr);
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
                                ct_ptr = self.enc_tail(pt2_ptr, rem_pt2, ct_ptr);
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{EncryptIO, PlainText};
    // Note:
    // These tests were generated using Qwen-3.8-27B on local hardware, feeding the LLM
    // with the output test vectors of the original C implementation, and asking it to exercice
    // our InPlace/OutOfPLace and Scattered API on the test vectors.

    /// OCB3 spec test key and nonce (AES-128, 128-bit tag).
    const SPEC_KEY: [u8; 16] = [
        0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e,
        0x0f,
    ];
    const SPEC_NONCE: [u8; 12] = [
        0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b,
    ];

    /// Official OCB3 test vectors: (pt_len, ad_len, expected ct, expected tag).
    /// pt[i] = ad[i] = i for i in 0..len.
    /// Generated with the original C reference implementation from Ted Krovetz:
    /// for each length L it emits (L,L), then (0,L), then (L,0).
    static SPEC_VECTORS: [(usize, usize, &[u8], &[u8]); 16] = [
        (
            0,
            0,
            &[],
            &[
                0x19, 0x7B, 0x9C, 0x3C, 0x44, 0x1D, 0x3C, 0x83, 0xEA, 0xFB, 0x2B, 0xEF, 0x63, 0x3B,
                0x91, 0x82,
            ],
        ),
        (
            8,
            8,
            &[0x92, 0xB6, 0x57, 0x13, 0x0A, 0x74, 0xB8, 0x5A],
            &[
                0x16, 0xDC, 0x76, 0xA4, 0x6D, 0x47, 0xE1, 0xEA, 0xD5, 0x37, 0x20, 0x9E, 0x8A, 0x96,
                0xD1, 0x4E,
            ],
        ),
        (
            0,
            8,
            &[],
            &[
                0x98, 0xB9, 0x15, 0x52, 0xC8, 0xC0, 0x09, 0x18, 0x50, 0x44, 0xE3, 0x0A, 0x6E, 0xB2,
                0xFE, 0x21,
            ],
        ),
        (
            8,
            0,
            &[0x92, 0xB6, 0x57, 0x13, 0x0A, 0x74, 0xB8, 0x5A],
            &[
                0x97, 0x1E, 0xFF, 0xCA, 0xE1, 0x9A, 0xD4, 0x71, 0x6F, 0x88, 0xE8, 0x7B, 0x87, 0x1F,
                0xBE, 0xED,
            ],
        ),
        (
            16,
            16,
            &[
                0xBE, 0xA5, 0xE8, 0x79, 0x8D, 0xBE, 0x71, 0x10, 0x03, 0x1C, 0x14, 0x4D, 0xA0, 0xB2,
                0x61, 0x22,
            ],
            &[
                0x77, 0x6C, 0x99, 0x24, 0xD6, 0x72, 0x3A, 0x1F, 0xC4, 0x52, 0x45, 0x32, 0xAC, 0x3E,
                0x5B, 0xEB,
            ],
        ),
        (
            0,
            16,
            &[],
            &[
                0x7D, 0xDB, 0x8E, 0x6C, 0xEA, 0x68, 0x14, 0x86, 0x62, 0x12, 0x50, 0x96, 0x19, 0xB1,
                0x9C, 0xC6,
            ],
        ),
        (
            16,
            0,
            &[
                0xBE, 0xA5, 0xE8, 0x79, 0x8D, 0xBE, 0x71, 0x10, 0x03, 0x1C, 0x14, 0x4D, 0xA0, 0xB2,
                0x61, 0x22,
            ],
            &[
                0x13, 0xCC, 0x8B, 0x74, 0x78, 0x07, 0x12, 0x1A, 0x4C, 0xBB, 0x3E, 0x4B, 0xD6, 0xB4,
                0x56, 0xAF,
            ],
        ),
        (
            24,
            24,
            &[
                0xBE, 0xA5, 0xE8, 0x79, 0x8D, 0xBE, 0x71, 0x10, 0x03, 0x1C, 0x14, 0x4D, 0xA0, 0xB2,
                0x61, 0x22, 0xFC, 0xFC, 0xEE, 0x7A, 0x2A, 0x8D, 0x4D, 0x48,
            ],
            &[
                0x5F, 0xA9, 0x4F, 0xC3, 0xF3, 0x88, 0x20, 0xF1, 0xDC, 0x3F, 0x3D, 0x1F, 0xD4, 0xE5,
                0x5E, 0x1C,
            ],
        ),
        (
            0,
            24,
            &[],
            &[
                0x28, 0x20, 0x26, 0xDA, 0x30, 0x68, 0xBC, 0x9F, 0xA1, 0x18, 0x68, 0x1D, 0x55, 0x9F,
                0x10, 0xF6,
            ],
        ),
        (
            24,
            0,
            &[
                0xBE, 0xA5, 0xE8, 0x79, 0x8D, 0xBE, 0x71, 0x10, 0x03, 0x1C, 0x14, 0x4D, 0xA0, 0xB2,
                0x61, 0x22, 0xFC, 0xFC, 0xEE, 0x7A, 0x2A, 0x8D, 0x4D, 0x48,
            ],
            &[
                0x6E, 0xF2, 0xF5, 0x25, 0x87, 0xFD, 0xA0, 0xED, 0x97, 0xDC, 0x7E, 0xED, 0xE2, 0x41,
                0xDF, 0x68,
            ],
        ),
        (
            32,
            32,
            &[
                0xBE, 0xA5, 0xE8, 0x79, 0x8D, 0xBE, 0x71, 0x10, 0x03, 0x1C, 0x14, 0x4D, 0xA0, 0xB2,
                0x61, 0x22, 0xCE, 0xAA, 0xB9, 0xB0, 0x5D, 0xF7, 0x71, 0xA6, 0x57, 0x14, 0x9D, 0x53,
                0x77, 0x34, 0x63, 0xCB,
            ],
            &[
                0xB2, 0xA0, 0x40, 0xDD, 0x3B, 0xD5, 0x16, 0x43, 0x72, 0xD7, 0x6D, 0x7B, 0xB6, 0x82,
                0x42, 0x40,
            ],
        ),
        (
            0,
            32,
            &[],
            &[
                0xE1, 0xE0, 0x72, 0x63, 0x3B, 0xAD, 0xE5, 0x1A, 0x60, 0xE8, 0x59, 0x51, 0xD9, 0xC4,
                0x2A, 0x1B,
            ],
        ),
        (
            32,
            0,
            &[
                0xBE, 0xA5, 0xE8, 0x79, 0x8D, 0xBE, 0x71, 0x10, 0x03, 0x1C, 0x14, 0x4D, 0xA0, 0xB2,
                0x61, 0x22, 0xCE, 0xAA, 0xB9, 0xB0, 0x5D, 0xF7, 0x71, 0xA6, 0x57, 0x14, 0x9D, 0x53,
                0x77, 0x34, 0x63, 0xCB,
            ],
            &[
                0x4A, 0x3B, 0xAE, 0x82, 0x44, 0x65, 0xCF, 0xDA, 0xF8, 0xC4, 0x1F, 0xC5, 0x0C, 0x7D,
                0xF9, 0xD9,
            ],
        ),
        (
            40,
            40,
            &[
                0xBE, 0xA5, 0xE8, 0x79, 0x8D, 0xBE, 0x71, 0x10, 0x03, 0x1C, 0x14, 0x4D, 0xA0, 0xB2,
                0x61, 0x22, 0xCE, 0xAA, 0xB9, 0xB0, 0x5D, 0xF7, 0x71, 0xA6, 0x57, 0x14, 0x9D, 0x53,
                0x77, 0x34, 0x63, 0xCB, 0x68, 0xC6, 0x57, 0x78, 0xB0, 0x58, 0xA6, 0x35,
            ],
            &[
                0x65, 0x9C, 0x62, 0x32, 0x11, 0xDE, 0xEA, 0x0D, 0xE3, 0x0D, 0x2C, 0x38, 0x18, 0x79,
                0xF4, 0xC8,
            ],
        ),
        (
            0,
            40,
            &[],
            &[
                0x7A, 0xEB, 0x7A, 0x69, 0xA1, 0x68, 0x7D, 0xD0, 0x82, 0xCA, 0x27, 0xB0, 0xD9, 0xA3,
                0x70, 0x96,
            ],
        ),
        (
            40,
            0,
            &[
                0xBE, 0xA5, 0xE8, 0x79, 0x8D, 0xBE, 0x71, 0x10, 0x03, 0x1C, 0x14, 0x4D, 0xA0, 0xB2,
                0x61, 0x22, 0xCE, 0xAA, 0xB9, 0xB0, 0x5D, 0xF7, 0x71, 0xA6, 0x57, 0x14, 0x9D, 0x53,
                0x77, 0x34, 0x63, 0xCB, 0x68, 0xC6, 0x57, 0x78, 0xB0, 0x58, 0xA6, 0x35,
            ],
            &[
                0x06, 0x0C, 0x84, 0x67, 0xF4, 0xAB, 0xAB, 0x5E, 0x8B, 0x3C, 0x20, 0x67, 0xA2, 0xE1,
                0x15, 0xDC,
            ],
        ),
    ];

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

        for split in 0..=ptl {
            let mut ctx = AEctx::new(&SPEC_KEY);
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
}
