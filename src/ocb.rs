use crate::EncryptIO;
use crate::PlainText;
use crate::Result;
use crate::aes::AesKey;
use crate::sse::*;
use core::arch::x86_64::*;

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
    ad_blocks_processed: u32,
    blocks_processed: u32,
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
                ad_blocks_processed: 0,
                blocks_processed: 0,
            }
        };

        // Key dependent values
        ctx.encrypt_key.AES_128_key_expansion(key);
        ctx.decrypt_key.AES_set_key_fast(&ctx.encrypt_key);
        ctx.Lstar = ctx.encrypt_key.AES_encrypt_block(ctx.cached_Top);
        let mut tmp_blk = swap_if_le(ctx.Lstar);
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
        // 0x01000000 + 1 << 1
        tmp[0] = 0x02;
        tmp[1] = 0;
        tmp[2] = 0;
        tmp[4] = 0x01;
        tmp[4..16].copy_from_slice(nonce);
        let idx: i32 = tmp[15] as i32 & 0xc0 as i32; // The low 6 bits of the nonce
        unsafe {
            let mut bl = _mm_loadu_si128(tmp.as_ptr() as *const __m128i);
            if unequal_blocks(bl, self.cached_Top) {
                self.cached_Top = bl;
                bl = self.encrypt_key.AES_decrypt_block(bl);
                [self.KtopStr[0], self.KtopStr[1]] = m128i_to_bswap_u64x2(bl);
                self.KtopStr[2] =
                    self.KtopStr[0] ^ (self.KtopStr[0] << 8) ^ (self.KtopStr[1] >> 56);
            }

            gen_offset(&self.KtopStr, idx)
        }
    }

    #[inline(always)]
    fn process_ad(&mut self, ad: &[u8]) {}

    pub fn ae_encrypt(&mut self, io: EncryptIO, nonce: &[u8; 12], ad: &[u8]) -> Result<()> {
        // Return early if the io is inappropriate.
        io.validate()?;
        self.offset = self.gen_offset_from_nonce(nonce);
        // SAFETY: memory must be 16-bytes aligned
        unsafe {
            self.ad_offset = _mm_setzero_si128();
            self.checksum = _mm_setzero_si128();
        }

        if !ad.is_empty() {
            self.process_ad(ad);
        }

        let mut offset = self.offset;
        let mut checksum = self.checksum;

        let (
            mut pt1_ptr,
            mut pt2_ptr,
            mut ct_ptr,
            num_bpi_pt1,
            mut remaining_pt1,
            mut num_bpi_pt2,
            mut remaining_pt2,
        ) = match io {
            EncryptIO::InPlace(buf) => (
                buf.as_ptr(),
                std::ptr::null(),
                buf.as_mut_ptr(),
                buf.len() / BPI * 16,
                buf.len() % (BPI * 16),
                0,
                0,
            ),
            EncryptIO::OutOfPlace { in_pt, out } => match in_pt {
                PlainText::Contiguous(pt) => (
                    pt.as_ptr(),
                    std::ptr::null(),
                    out.as_mut_ptr(),
                    pt.len() / BPI * 16,
                    pt.len() % (BPI * 16),
                    0,
                    0,
                ),
                PlainText::Scattered(pt1, pt2) => (
                    pt1.as_ptr(),
                    pt2.as_ptr(),
                    out.as_mut_ptr(),
                    pt1.len() / BPI * 16,
                    pt1.len() % (BPI * 16),
                    pt2.len() / BPI * 16,
                    pt2.len() % (BPI * 16),
                ),
            },
        };

        unsafe {
            if num_bpi_pt1 > 0 {
                let mut oa = [_mm_setzero_si128(); BPI];
                let mut block_num = self.blocks_processed;
                oa[BPI - 1] = offset;
                for _ in 0..num_bpi_pt1 {
                    let mut ta = [_mm_setzero_si128(); BPI];
                    block_num += BPI as u32;
                    if BPI == 8 {
                        oa[0] = xor_block(oa[BPI - 1], self.L[0]);
                        let pt0 = _mm_loadu_si128(pt1_ptr as *const __m128i);
                        // Preparing inputs to the PRP
                        ta[0] = xor_block(oa[0], pt0);
                        let ct0 = self.encrypt_key.AES_encrypt_block(ta[0]);
                        let ct0 = xor_block(ct0, oa[0]);
                        _mm_storeu_si128(ct_ptr as *mut __m128i, ct0);
                        checksum = xor_block(checksum, pt0);
                        oa[1] = xor_block(oa[0], self.L[1]);
                        let pt1 = _mm_loadu_si128(pt1_ptr.add(16) as *const __m128i);
                        ta[1] = xor_block(oa[1], pt1);
                        let ct1 = self.encrypt_key.AES_encrypt_block(ta[1]);
                        let ct1 = xor_block(ct1, oa[1]);
                        _mm_storeu_si128(ct_ptr.add(16) as *mut __m128i, ct1);
                        checksum = xor_block(checksum, pt1);
                        oa[2] = xor_block(oa[1], self.L[0]);
                        let pt2 = _mm_loadu_si128(pt1_ptr.add(32) as *const __m128i);
                        ta[2] = xor_block(oa[2], pt2);
                        let ct2 = self.encrypt_key.AES_encrypt_block(ta[2]);
                        let ct2 = xor_block(ct2, oa[2]);
                        _mm_storeu_si128(ct_ptr.add(32) as *mut __m128i, ct2);
                        checksum = xor_block(checksum, pt2);
                        oa[3] = xor_block(oa[2], self.L[2]);
                        let pt3 = _mm_loadu_si128(pt1_ptr.add(48) as *const __m128i);
                        ta[3] = xor_block(oa[3], pt3);
                        let ct3 = self.encrypt_key.AES_encrypt_block(ta[3]);
                        let ct3 = xor_block(ct3, oa[3]);
                        _mm_storeu_si128(ct_ptr.add(48) as *mut __m128i, ct3);
                        checksum = xor_block(checksum, pt3);
                        // this allows the cpu to parallelize more efficiently, instead of using
                        // oa[3] and L[0]; this is equivalent tho.
                        oa[4] = xor_block(oa[1], self.L[2]);
                        let pt4 = _mm_loadu_si128(pt1_ptr.add(64) as *const __m128i);
                        ta[4] = xor_block(oa[4], pt4);
                        let ct4 = self.encrypt_key.AES_encrypt_block(ta[4]);
                        let ct4 = xor_block(ct4, oa[4]);
                        _mm_storeu_si128(ct_ptr.add(64) as *mut __m128i, ct4);
                        checksum = xor_block(checksum, pt4);
                        oa[5] = xor_block(oa[0], self.L[2]);
                        let pt5 = _mm_loadu_si128(pt1_ptr.add(80) as *const __m128i);
                        ta[5] = xor_block(oa[5], pt5);
                        let ct5 = self.encrypt_key.AES_encrypt_block(ta[5]);
                        let ct5 = xor_block(ct5, oa[5]);
                        _mm_storeu_si128(ct_ptr.add(80) as *mut __m128i, ct5);
                        checksum = xor_block(checksum, pt5);
                        oa[6] = xor_block(oa[7], self.L[2]);
                        let pt6 = _mm_loadu_si128(pt1_ptr.add(96) as *const __m128i);
                        ta[6] = xor_block(oa[6], pt6);
                        let ct6 = self.encrypt_key.AES_encrypt_block(ta[6]);
                        let ct6 = xor_block(ct6, oa[6]);
                        _mm_storeu_si128(ct_ptr.add(96) as *mut __m128i, ct6);
                        checksum = xor_block(checksum, pt6);
                        let ntz = block_num.trailing_zeros() as usize;
                        oa[7] = xor_block(oa[6], self.L[ntz]);
                        let pt7 = _mm_loadu_si128(pt1_ptr.add(112) as *const __m128i);
                        ta[7] = xor_block(oa[7], pt7);
                        let ct7 = self.encrypt_key.AES_encrypt_block(ta[7]);
                        let ct7 = xor_block(ct7, oa[7]);
                        _mm_storeu_si128(ct_ptr.add(112) as *mut __m128i, ct7);
                        checksum = xor_block(checksum, pt7);
                    }

                    pt1_ptr = pt1_ptr.add(128);
                    ct_ptr = ct_ptr.add(128);
                }
                self.offset = oa[BPI - 1];
                offset = self.offset;
                self.blocks_processed = block_num;
                self.checksum = checksum;
                
                let mut ta = [_mm_setzero_si128(); BPI + 1];
                let mut tmp = [0u8; 16];
                let mut k = 0;
                // We have to handle several cases: 
                //  - remaining_pt1 is > 0 && num_bpi_pt2 > 0 && remaining_pt2 > 0
                //  - remaining_pt1 is > 0 && num_bpi_pt2 > 0 && remaining_pt2 = 0
                //  - remaining_pt1 is > 0 && num_bpi_pt2 = 0 && remaining_pt2 > 0
                //  - remaining_pt1 is > 0 && num_npi_pt2 = 0 && remaining_pt2 = 0
                if remaining_pt1 >= 64 {
                    oa[0] = xor_block(offset, self.L[0]);
                    let pt0 = _mm_loadu_si128(pt1_ptr as *const __m128i);
                    ta[0] = xor_block(oa[0], pt0);
                    let ct0 = self.encrypt_key.AES_encrypt_block(ta[0]);
                    let ct0 = xor_block(ct0, oa[0]);
                    _mm_storeu_si128(ct_ptr as *mut __m128i, ct0);
                    checksum = xor_block(checksum, pt0);
                    oa[1] = xor_block(oa[0], self.L[1]);
                    let pt1 = _mm_loadu_si128(pt1_ptr.add(16) as *const __m128i);
                    ta[1] = xor_block(oa[1], pt1);
                    let ct1 = self.encrypt_key.AES_encrypt_block(ta[1]);
                    let ct1 = xor_block(ct1, oa[1]);
                    _mm_storeu_si128(ct_ptr.add(16) as *mut __m128i, ct1);
                    checksum = xor_block(checksum, pt1);
                    oa[2] = xor_block(oa[1], self.L[0]);
                    let pt2 = _mm_loadu_si128(pt1_ptr.add(32) as *const __m128i);
                    ta[2] = xor_block(oa[2], pt2);
                    let ct2 = self.encrypt_key.AES_encrypt_block(ta[2]);
                    let ct2 = xor_block(ct2, oa[2]);
                    _mm_storeu_si128(ct_ptr.add(32) as *mut __m128i, ct2);
                    checksum = xor_block(checksum, pt2);
                    oa[3] = xor_block(oa[2], self.L[2]);
                    offset = oa[3];
                    let pt3 = _mm_loadu_si128(pt1_ptr.add(48) as *const __m128i);
                    ta[3] = xor_block(offset, pt3);
                    let ct3 = self.encrypt_key.AES_encrypt_block(ta[3]);
                    let ct3 = xor_block(ct3, oa[3]);
                    _mm_storeu_si128(ct_ptr.add(48) as *mut __m128i, ct3);
                    checksum = xor_block(checksum, pt3);
                    remaining_pt1 -= 64;
                    k = 4;
                    pt1_ptr = pt1_ptr.add(64);
                    ct_ptr = ct_ptr.add(64);
                }
                if remaining_pt1 >= 32 {
                    oa[k] = xor_block(offset, self.L[0]);
                    let ptk0 = _mm_loadu_si128(pt1_ptr as *const __m128i);
                    ta[k] = xor_block(oa[k], ptk0);
                    let ctk0 = self.encrypt_key.AES_encrypt_block(ta[k]);
                    let ctk0 = xor_block(ctk0, oa[k]);
                    _mm_storeu_si128(ct_ptr as *mut __m128i, ctk0);
                    checksum = xor_block(checksum, ptk0);
                    oa[k + 1] = xor_block(oa[k], self.L[1]);
                    offset = oa[k + 1];
                    let ptk1 = _mm_loadu_si128(pt1_ptr.add(16) as *const __m128i);
                    ta[k + 1] = xor_block(oa[k + 1], ptk1);
                    let ctk1 = self.encrypt_key.AES_encrypt_block(ta[k + 1]);
                    let ctk1 = xor_block(ctk1, oa[k + 1]);
                    _mm_storeu_si128(ct_ptr.add(16) as *mut __m128i, ctk1);
                    checksum = xor_block(checksum, ptk1);
                    remaining_pt1 -= 32;
                    k += 2;
                    pt1_ptr = pt1_ptr.add(32);
                    ct_ptr = ct_ptr.add(32);
                }
                if remaining_pt1 >= 16 {
                    oa[k] = xor_block(offset, self.L[0]);
                    offset = oa[k];
                    let ptk0 = _mm_loadu_si128(pt1_ptr as *const __m128i);
                    ta[k] = xor_block(offset, ptk0);
                    let ctk0 = self.encrypt_key.AES_encrypt_block(ta[k]);
                    let ctk0 = xor_block(ctk0, oa[k]);
                    _mm_storeu_si128(ct_ptr as *mut __m128i, ctk0);
                    checksum = xor_block(checksum, ptk0);
                    remaining_pt1 -= 16;
                    k += 1;
                    pt1_ptr = pt1_ptr.add(16);
                    ct_ptr = ct_ptr.add(16);
                }
                if remaining_pt1 > 0  && num_bpi_pt2 > 0 {
                    core::ptr::copy_nonoverlapping(pt1_ptr, tmp.as_mut_ptr(), remaining_pt1);
                    core::ptr::copy_nonoverlapping(pt2_ptr, tmp.as_mut_ptr().add(remaining_pt1), 16 - remaining_pt1);
                    if remaining_pt2 < 16 - remaining_pt1 {
                        num_bpi_pt2 -= 1;
                        remaining_pt2 = 16 * BPI - (remaining_pt2 - (16 - remaining_pt1));
                    } else {
                        remaining_pt2 -= 16 - remaining_pt1;
                    }
                    let pt0 = _mm_loadu_si128(tmp.as_ptr() as *const __m128i);
                    oa[k] = xor_block(offset, self.L[0]);
                    offset = oa[k];
                    ta[k] = xor_block(offset, pt0);
                    let ctk0 = self.encrypt_key.AES_encrypt_block(ta[k]);
                    let ctk0 = xor_block(ctk0, oa[k]);
                    _mm_storeu_si128(ct_ptr as *mut __m128i, ctk0);
                    checksum = xor_block(checksum, pt0);
                    pt2_ptr = pt2_ptr.add(16 - remaining_pt1);
                    remaining_pt1 = 0;
                    ct_ptr = ct_ptr.add(16);
                    k+=1;
                }

                if remaining_pt1 > 0 && num_bpi_pt2 == 0 && remaining_pt2 > 0 {
                    // remainin_pt1 must be < 16 bytes; remaining_pt2 is 0 <= remaining_pt2 <
                    // 16*BPI
                    if remaining_pt1 + remaining_pt2 == 16 {
                        // we can reassemble then encrypt on a full block length (it should be a
                        // rare case.
                        core::ptr::copy_nonoverlapping(pt1_ptr, tmp.as_mut_ptr(), remaining_pt1);
                        core::ptr::copy_nonoverlapping(pt2_ptr, tmp.as_mut_ptr().add(remaining_pt1), remaining_pt2);
                        remaining_pt1 = 0;
                        remaining_pt2 = 0;
                        let pt0 = _mm_loadu_si128(tmp.as_ptr() as *const __m128i);
                        oa[k] = xor_block(offset, self.L[0]);
                        offset = oa[k];
                        ta[k] = xor_block(offset, pt0);
                        let ctk0 = self.encrypt_key.AES_encrypt_block(ta[k]);
                        let ctk0 = xor_block(ctk0, oa[k]);
                        _mm_storeu_si128(ct_ptr as *mut __m128i, ctk0);
                        checksum = xor_block(checksum, pt0);
                        ct_ptr = ct_ptr.add(16);
                    } else if remaining_pt1 + remaining_pt2 < 16 {
                        let mut enc_pad = [0u8; 16];
                        core::ptr::copy_nonoverlapping(pt1_ptr, tmp.as_mut_ptr(), remaining_pt1);
                        core::ptr::copy_nonoverlapping(pt2_ptr, tmp.as_mut_ptr().add(remaining_pt1), remaining_pt2);
                        tmp[remaining_pt1 + remaining_pt2] = 0x80;
                        let pt_padded = _mm_loadu_si128(tmp.as_ptr() as *const __m128i);
                        checksum = xor_block(checksum, pt_padded);
                        ta[k] = xor_block(offset, self.Lstar);
                        offset = ta[k];
                        let pad = self.encrypt_key.AES_encrypt_block(ta[k]);
                        _mm_storeu_si128(enc_pad.as_mut_ptr() as *mut __m128i, pad);
                        for i in 0..remaining_pt1+remaining_pt2 {
                            *ct_ptr.add(i) = tmp[i] ^ enc_pad[i];
                        }
                        k += 1;
                        remaining_pt1 = 0;
                        remaining_pt2 = 0;


                    } else { // > 16

                    }

                }

                if num_bpi_pt2 > 0 {

                }
                if remaining_pt2 > 0 {
                }

                self.offset = oa[BPI - 1];
                self.blocks_processed = block_num;
                self.checksum = checksum;
            }
        }
        Ok(())
    }
}
