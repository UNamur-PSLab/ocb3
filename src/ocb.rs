#![allow(unsafe_op_in_unsafe_fn)]
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
    fn process_ad(&mut self, ad: &[u8]) {}

    /// Encrypts one 16-byte block
    /// Returns (oa, new_checksum).
    #[inline(always)]
    unsafe fn enc_block(
        &mut self,
        offset: __m128i,
        l: __m128i,
        pt_ptr: *const u8,
        ct_ptr: *mut u8,
        checksum: __m128i,
    ) -> (__m128i, __m128i) {
        let oa = xor_block(offset, l);
        let pt = _mm_loadu_si128(pt_ptr as *const __m128i);
        let t = xor_block(oa, pt);
        let ct = self.encrypt_key.AES_encrypt_block(t);
        _mm_storeu_si128(ct_ptr as *mut __m128i, xor_block(ct, oa));
        (oa, xor_block(checksum, pt))
    }

    /// Encrypts a full BPI (8-block) group using the algebraic offset shortcut.
    /// `block_num` is the block count **before** this group.
    /// Returns `(new_offset, new_checksum, new_block_num)`.
    #[inline(always)]
    unsafe fn enc_bpi(
        &mut self,
        offset: __m128i,
        pt_ptr: *const u8,
        ct_ptr: *mut u8,
        block_num: u32,
        checksum: __m128i,
    ) -> (__m128i, __m128i, u32) {
        let block_num = block_num + BPI as u32;
        let l0 = self.L[0];
        let l1 = self.L[1];
        let l2 = self.L[2];
        let cs = checksum;

        let (oa0, cs) = self.enc_block(offset, l0, pt_ptr, ct_ptr, cs);
        let (oa1, cs) = self.enc_block(oa0, l1, pt_ptr.add(16), ct_ptr.add(16), cs);
        let (oa2, cs) = self.enc_block(oa1, l0, pt_ptr.add(32), ct_ptr.add(32), cs);
        let (_oa3, cs) = self.enc_block(oa2, l2, pt_ptr.add(48), ct_ptr.add(48), cs);
        // algebraic shortcuts — shorter dependency chains for OoO execution
        let (_oa4, cs) = self.enc_block(oa1, l2, pt_ptr.add(64), ct_ptr.add(64), cs);
        let (_oa5, cs) = self.enc_block(oa0, l2, pt_ptr.add(80), ct_ptr.add(80), cs);
        let (oa6, cs) = self.enc_block(offset, l2, pt_ptr.add(96), ct_ptr.add(96), cs);
        let ntz = block_num.trailing_zeros() as usize;
        let (oa7, cs) = self.enc_block(oa6, self.L[ntz], pt_ptr.add(112), ct_ptr.add(112), cs);

        (oa7, cs, block_num)
    }

    /// Encrypts full 16-byte blocks from the front of `remaining` bytes
    /// Returns `(new_offset, new_checksum, new_pt_ptr, new_ct_ptr, new_remaining)`.
    #[inline(always)]
    unsafe fn enc_full_blocks(
        &mut self,
        mut offset: __m128i,
        mut pt_ptr: *const u8,
        mut ct_ptr: *mut u8,
        mut remaining: usize,
        checksum: __m128i,
    ) -> (__m128i, __m128i, *const u8, *mut u8, usize) {
        let l0 = self.L[0];
        let l1 = self.L[1];
        let l2 = self.L[2];
        let cs = checksum;

        if remaining >= 64 {
            let (oa0, cs) = self.enc_block(offset, l0, pt_ptr, ct_ptr, cs);
            let (oa1, cs) = self.enc_block(oa0, l1, pt_ptr.add(16), ct_ptr.add(16), cs);
            let (oa2, cs) = self.enc_block(oa1, l0, pt_ptr.add(32), ct_ptr.add(32), cs);
            let (oa3, cs) = self.enc_block(oa2, l2, pt_ptr.add(48), ct_ptr.add(48), cs);
            offset = oa3;
            remaining -= 64;
            pt_ptr = pt_ptr.add(64);
            ct_ptr = ct_ptr.add(64);
        }
        if remaining >= 32 {
            let (oa0, cs) = self.enc_block(offset, l0, pt_ptr, ct_ptr, cs);
            let (oa1, cs) = self.enc_block(oa0, l1, pt_ptr.add(16), ct_ptr.add(16), cs);
            offset = oa1;
            remaining -= 32;
            pt_ptr = pt_ptr.add(32);
            ct_ptr = ct_ptr.add(32);
        }
        if remaining >= 16 {
            let (oa0, cs) = self.enc_block(offset, l0, pt_ptr, ct_ptr, cs);
            offset = oa0;
            remaining -= 16;
            pt_ptr = pt_ptr.add(16);
            ct_ptr = ct_ptr.add(16);
        }
        (offset, cs, pt_ptr, ct_ptr, remaining)
    }

    /// Encrypts the final partial block (`len < 16`) with `0x80` padding and L*.
    /// Returns `(new_offset, new_checksum)`.
    #[inline(always)]
    unsafe fn enc_tail(
        &mut self,
        data_ptr: *const u8,
        data_len: usize,
        offset: __m128i,
        ct_ptr: *mut u8,
        checksum: __m128i,
    ) -> (__m128i, __m128i) {
        let mut tmp = [0u8; 16];
        core::ptr::copy_nonoverlapping(data_ptr, tmp.as_mut_ptr(), data_len);
        tmp[data_len] = 0x80;
        let pt_padded = _mm_loadu_si128(tmp.as_ptr() as *const __m128i);
        let cs = xor_block(checksum, pt_padded);
        let t = xor_block(offset, self.Lstar);
        let pad = self.encrypt_key.AES_encrypt_block(t);
        let mut pad_bytes = [0u8; 16];
        _mm_storeu_si128(pad_bytes.as_mut_ptr() as *mut __m128i, pad);
        for i in 0..data_len {
            *ct_ptr.add(i) = tmp[i] ^ pad_bytes[i];
        }
        (t, cs)
    }

    // ---------------------------------------------------------------------
    //  Main encryption
    // ---------------------------------------------------------------------

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

        let (
            mut pt1_ptr,
            mut pt2_ptr,
            mut ct_ptr,
            num_bpi_pt1,
            mut rem_pt1,
            mut num_bpi_pt2,
            mut rem_pt2,
        ) = match io {
            EncryptIO::InPlace(buf) => (
                buf.as_ptr(),
                std::ptr::null(),
                buf.as_mut_ptr(),
                buf.len() / (BPI * 16),
                buf.len() % (BPI * 16),
                0,
                0,
            ),
            EncryptIO::OutOfPlace { in_pt, out } => match in_pt {
                PlainText::Contiguous(pt) => (
                    pt.as_ptr(),
                    std::ptr::null(),
                    out.as_mut_ptr(),
                    pt.len() / (BPI * 16),
                    pt.len() % (BPI * 16),
                    0,
                    0,
                ),
                PlainText::Scattered(pt1, pt2) => (
                    pt1.as_ptr(),
                    pt2.as_ptr(),
                    out.as_mut_ptr(),
                    pt1.len() / (BPI * 16),
                    pt1.len() % (BPI * 16),
                    pt2.len() / (BPI * 16),
                    pt2.len() % (BPI * 16),
                ),
            },
        };

        unsafe {
            let mut offset = self.offset;
            let mut checksum = self.checksum;
            let mut block_num = self.blocks_processed;

            // full BPI groups from pt1
            for _ in 0..num_bpi_pt1 {
                (offset, checksum, block_num) =
                    self.enc_bpi(offset, pt1_ptr, ct_ptr, block_num, checksum);
                pt1_ptr = pt1_ptr.add(BPI * 16);
                ct_ptr = ct_ptr.add(BPI * 16);
            }

            // full blocks from pt1 remainder
            if rem_pt1 >= 16 {
                (offset, checksum, pt1_ptr, ct_ptr, rem_pt1) =
                    self.enc_full_blocks(offset, pt1_ptr, ct_ptr, rem_pt1, checksum);
            }

            // handling pt1 to pt2 if any pt2 (rem_pt1 < 16 bytes left in pt1)
            if rem_pt1 > 0 {
                if num_bpi_pt2 == 0 && rem_pt2 == 0 {
                    // entire message tail is in pt1
                    (offset, checksum) = self.enc_tail(pt1_ptr, rem_pt1, offset, ct_ptr, checksum);
                    ct_ptr = ct_ptr.add(rem_pt1);
                } else if num_bpi_pt2 == 0 && rem_pt1 + rem_pt2 < 16 {
                    // combined tail spans both buffers, still < 16 bytes
                    let mut tmp = [0u8; 16];
                    core::ptr::copy_nonoverlapping(pt1_ptr, tmp.as_mut_ptr(), rem_pt1);
                    core::ptr::copy_nonoverlapping(pt2_ptr, tmp.as_mut_ptr().add(rem_pt1), rem_pt2);
                    (offset, checksum) =
                        self.enc_tail(tmp.as_ptr(), rem_pt1 + rem_pt2, offset, ct_ptr, checksum);
                    ct_ptr = ct_ptr.add(rem_pt1 + rem_pt2);
                    rem_pt2 = 0;
                } else {
                    // reassemble one full block from pt1 tail + pt2 head
                    let need = 16 - rem_pt1;
                    let mut tmp = [0u8; 16];
                    core::ptr::copy_nonoverlapping(pt1_ptr, tmp.as_mut_ptr(), rem_pt1);
                    core::ptr::copy_nonoverlapping(pt2_ptr, tmp.as_mut_ptr().add(rem_pt1), need);
                    if rem_pt2 < need {
                        num_bpi_pt2 -= 1;
                        rem_pt2 = BPI * 16 - need + rem_pt2;
                    } else {
                        rem_pt2 -= need;
                    }
                    pt2_ptr = pt2_ptr.add(need);

                    let l0 = self.L[0];
                    (offset, checksum) = self.enc_block(offset, l0, tmp.as_ptr(), ct_ptr, checksum);
                    ct_ptr = ct_ptr.add(16);
                }
                rem_pt1 = 0;
            }

            // full BPI groups from pt2
            for _ in 0..num_bpi_pt2 {
                (offset, checksum, block_num) =
                    self.enc_bpi(offset, pt2_ptr, ct_ptr, block_num, checksum);
                pt2_ptr = pt2_ptr.add(BPI * 16);
                ct_ptr = ct_ptr.add(BPI * 16);
            }

            // full blocks from pt2 remainder
            if rem_pt2 >= 16 {
                (offset, checksum, pt2_ptr, ct_ptr, rem_pt2) =
                    self.enc_full_blocks(offset, pt2_ptr, ct_ptr, rem_pt2, checksum);
            }

            // pt2 tail
            if rem_pt2 > 0 {
                (offset, checksum) = self.enc_tail(pt2_ptr, rem_pt2, offset, ct_ptr, checksum);
                ct_ptr = ct_ptr.add(rem_pt2);
            }

            self.offset = offset;
            self.checksum = checksum;
            self.blocks_processed = block_num;
        }
        Ok(())
    }
}
