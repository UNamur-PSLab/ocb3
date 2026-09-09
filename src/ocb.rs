use crate::aes::AesKey;
use core::arch::x86_64::*;
use crate::sse::*;

const L_TABLE_SZ: usize = 16;

#[repr(C, align(16))]
struct AEctx {
    offset: __m128i,
    checksum: __m128i,
    Lstar: __m128i,
    Ldollar: __m128i,
    L: [__m128i; L_TABLE_SZ],
    ad_checksum: __m128i,
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
}
