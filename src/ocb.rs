use crate::aes::AesKey;
use core::arch::x86_64::*;
use crate::sse::*;
use crate::EncryptIO;
use crate::Result;

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
                self.KtopStr[2] = self.KtopStr[0] ^ (self.KtopStr[0] << 8) ^ (self.KtopStr[1] >> 56);
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
        // SAFETY: memory is 16-bytes aligned
        unsafe {
            self.ad_offset = _mm_setzero_si128();
            self.checksum = _mm_setzero_si128();
        }

        if !ad.is_empty() {
            self.process_ad(ad);
        }
        
        let mut offset = self.offset;
        let mut checksum  = self.checksum;

        let i = io.len() / (BPI * 16);
        
        Ok(())
    }
}
