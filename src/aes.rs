use core::arch::x86_64::*;

// TODO extends this to AVX512
#[derive(Copy, Clone)]
#[repr(C, align(16))]
pub(crate) struct AesKey {
    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    pub(crate) rd_key: [__m128i; 11],
}

impl Default for AesKey {
    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    #[inline(always)]
    fn default() -> Self {
        // Safety: safe if Self is aligned on 16 bytes.
        unsafe {
            Self {
                rd_key: [_mm_setzero_si128(); 11],
            }
        }
    }
}

#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[inline(always)]
unsafe fn expand_step<const RCON: i32>(mut t1: __m128i) -> __m128i {
    unsafe {
        let mut t2 = _mm_aeskeygenassist_si128::<RCON>(t1);
        t2 = _mm_shuffle_epi32::<0xff>(t2);
        t1 = _mm_xor_si128(t1, _mm_slli_si128::<4>(t1));
        t1 = _mm_xor_si128(t1, _mm_slli_si128::<8>(t1));
        _mm_xor_si128(t1, t2)
    }
}

impl AesKey {
    #[inline(always)]
    pub(crate) fn aes_128_key_expansion(&mut self, userkey: &[u8]) {
        unsafe {
            let mut x0 = _mm_loadu_si128(userkey.as_ptr() as *const __m128i);
            self.rd_key[0] = x0;
            x0 = expand_step::<1>(x0);
            self.rd_key[1] = x0;
            x0 = expand_step::<2>(x0);
            self.rd_key[2] = x0;
            x0 = expand_step::<4>(x0);
            self.rd_key[3] = x0;
            x0 = expand_step::<8>(x0);
            self.rd_key[4] = x0;
            x0 = expand_step::<16>(x0);
            self.rd_key[5] = x0;
            x0 = expand_step::<32>(x0);
            self.rd_key[6] = x0;
            x0 = expand_step::<64>(x0);
            self.rd_key[7] = x0;
            x0 = expand_step::<128>(x0);
            self.rd_key[8] = x0;
            x0 = expand_step::<27>(x0);
            self.rd_key[9] = x0;
            x0 = expand_step::<54>(x0);
            self.rd_key[10] = x0;
        }
    }

    #[inline(always)]
    pub(crate) fn aes_set_key_fast(&mut self, other: &AesKey) {
        unsafe {
            self.rd_key[10] = other.rd_key[0];
            self.rd_key[9] = _mm_aesimc_si128(other.rd_key[1]);
            self.rd_key[8] = _mm_aesimc_si128(other.rd_key[2]);
            self.rd_key[7] = _mm_aesimc_si128(other.rd_key[3]);
            self.rd_key[6] = _mm_aesimc_si128(other.rd_key[4]);
            self.rd_key[5] = _mm_aesimc_si128(other.rd_key[5]);
            self.rd_key[4] = _mm_aesimc_si128(other.rd_key[6]);
            self.rd_key[3] = _mm_aesimc_si128(other.rd_key[7]);
            self.rd_key[2] = _mm_aesimc_si128(other.rd_key[8]);
            self.rd_key[1] = _mm_aesimc_si128(other.rd_key[9]);
            self.rd_key[0] = other.rd_key[10];
        }
    }

    /// AES encrypt on unaligned in/out memory. On modern processors, unaligned
    #[inline(always)]
    pub(crate) fn aes_encrypt_block(&self, mut in_pt: __m128i) -> __m128i {
        unsafe {
            let rnds = 10;
            in_pt = _mm_xor_si128(in_pt, self.rd_key[0]);
            for j in 1..rnds {
                in_pt = _mm_aesenc_si128(in_pt, self.rd_key[j]);
            }
            _mm_aesenclast_si128(in_pt, self.rd_key[rnds])
        }
    }

    #[inline(always)]
    pub(crate) fn aes_encrypt_2_blocks(&self, mut b: [__m128i; 2]) -> [__m128i; 2] {
        unsafe {
            let k0 = self.rd_key[0];
            b[0] = _mm_xor_si128(b[0], k0);
            b[1] = _mm_xor_si128(b[1], k0);

            for j in 1..10 {
                let kj = self.rd_key[j];
                b[0] = _mm_aesenc_si128(b[0], kj);
                b[1] = _mm_aesenc_si128(b[1], kj);
            }

            let k10 = self.rd_key[10];
            b[0] = _mm_aesenclast_si128(b[0], k10);
            b[1] = _mm_aesenclast_si128(b[1], k10);

            b
        }
    }

    #[inline(always)]
    pub(crate) fn aes_encrypt_4_blocks(&self, mut b: [__m128i; 4]) -> [__m128i; 4] {
        unsafe {
            let k0 = self.rd_key[0];
            b[0] = _mm_xor_si128(b[0], k0);
            b[1] = _mm_xor_si128(b[1], k0);
            b[2] = _mm_xor_si128(b[2], k0);
            b[3] = _mm_xor_si128(b[3], k0);

            for j in 1..10 {
                let kj = self.rd_key[j];
                b[0] = _mm_aesenc_si128(b[0], kj);
                b[1] = _mm_aesenc_si128(b[1], kj);
                b[2] = _mm_aesenc_si128(b[2], kj);
                b[3] = _mm_aesenc_si128(b[3], kj);
            }

            let k10 = self.rd_key[10];
            b[0] = _mm_aesenclast_si128(b[0], k10);
            b[1] = _mm_aesenclast_si128(b[1], k10);
            b[2] = _mm_aesenclast_si128(b[2], k10);
            b[3] = _mm_aesenclast_si128(b[3], k10);

            b
        }
    }

    /// Encrypts 8 blocks where `b[0..8]` have already been XORed with `rd_key[0]`.
    /// Passing `rk` as a pointer prevents LLVM's LICM from hoisting all 10 round keys into
    /// virtual registers across loop iterations, keeping total live XMM registers <= 16.
    #[inline(always)]
    pub(crate) fn aes_encrypt_8_blocks_prewhitened(
        rk: *const __m128i,
        mut b: [__m128i; 8],
    ) -> [__m128i; 8] {
        unsafe {
            for j in 1..10 {
                let kj = _mm_load_si128(rk.add(j));
                b[0] = _mm_aesenc_si128(b[0], kj);
                b[1] = _mm_aesenc_si128(b[1], kj);
                b[2] = _mm_aesenc_si128(b[2], kj);
                b[3] = _mm_aesenc_si128(b[3], kj);
                b[4] = _mm_aesenc_si128(b[4], kj);
                b[5] = _mm_aesenc_si128(b[5], kj);
                b[6] = _mm_aesenc_si128(b[6], kj);
                b[7] = _mm_aesenc_si128(b[7], kj);
            }

            let k10 = _mm_load_si128(rk.add(10));
            b[0] = _mm_aesenclast_si128(b[0], k10);
            b[1] = _mm_aesenclast_si128(b[1], k10);
            b[2] = _mm_aesenclast_si128(b[2], k10);
            b[3] = _mm_aesenclast_si128(b[3], k10);
            b[4] = _mm_aesenclast_si128(b[4], k10);
            b[5] = _mm_aesenclast_si128(b[5], k10);
            b[6] = _mm_aesenclast_si128(b[6], k10);
            b[7] = _mm_aesenclast_si128(b[7], k10);

            b
        }
    }

    #[inline(always)]
    pub(crate) fn aes_decrypt_2_blocks(&self, mut b: [__m128i; 2]) -> [__m128i; 2] {
        unsafe {
            let k0 = self.rd_key[0];
            b[0] = _mm_xor_si128(b[0], k0);
            b[1] = _mm_xor_si128(b[1], k0);

            for j in 1..10 {
                let kj = self.rd_key[j];
                b[0] = _mm_aesdec_si128(b[0], kj);
                b[1] = _mm_aesdec_si128(b[1], kj);
            }

            let k10 = self.rd_key[10];
            b[0] = _mm_aesdeclast_si128(b[0], k10);
            b[1] = _mm_aesdeclast_si128(b[1], k10);

            b
        }
    }

    #[inline(always)]
    pub(crate) fn aes_decrypt_4_blocks(&self, mut b: [__m128i; 4]) -> [__m128i; 4] {
        unsafe {
            let k0 = self.rd_key[0];
            b[0] = _mm_xor_si128(b[0], k0);
            b[1] = _mm_xor_si128(b[1], k0);
            b[2] = _mm_xor_si128(b[2], k0);
            b[3] = _mm_xor_si128(b[3], k0);

            for j in 1..10 {
                let kj = self.rd_key[j];
                b[0] = _mm_aesdec_si128(b[0], kj);
                b[1] = _mm_aesdec_si128(b[1], kj);
                b[2] = _mm_aesdec_si128(b[2], kj);
                b[3] = _mm_aesdec_si128(b[3], kj);
            }

            let k10 = self.rd_key[10];
            b[0] = _mm_aesdeclast_si128(b[0], k10);
            b[1] = _mm_aesdeclast_si128(b[1], k10);
            b[2] = _mm_aesdeclast_si128(b[2], k10);
            b[3] = _mm_aesdeclast_si128(b[3], k10);

            b
        }
    }

    /// Decrypts 8 blocks in lockstep where `b[0..8]` have already been XORed with `rd_key[0]`.
    #[inline(always)]
    pub(crate) fn aes_decrypt_8_blocks_prewhitened(
        rk: *const __m128i,
        mut b: [__m128i; 8],
    ) -> [__m128i; 8] {
        unsafe {
            for j in 1..10 {
                let kj = _mm_load_si128(rk.add(j));
                b[0] = _mm_aesdec_si128(b[0], kj);
                b[1] = _mm_aesdec_si128(b[1], kj);
                b[2] = _mm_aesdec_si128(b[2], kj);
                b[3] = _mm_aesdec_si128(b[3], kj);
                b[4] = _mm_aesdec_si128(b[4], kj);
                b[5] = _mm_aesdec_si128(b[5], kj);
                b[6] = _mm_aesdec_si128(b[6], kj);
                b[7] = _mm_aesdec_si128(b[7], kj);
            }

            let k10 = _mm_load_si128(rk.add(10));
            b[0] = _mm_aesdeclast_si128(b[0], k10);
            b[1] = _mm_aesdeclast_si128(b[1], k10);
            b[2] = _mm_aesdeclast_si128(b[2], k10);
            b[3] = _mm_aesdeclast_si128(b[3], k10);
            b[4] = _mm_aesdeclast_si128(b[4], k10);
            b[5] = _mm_aesdeclast_si128(b[5], k10);
            b[6] = _mm_aesdeclast_si128(b[6], k10);
            b[7] = _mm_aesdeclast_si128(b[7], k10);

            b
        }
    }

    #[inline(always)]
    pub(crate) fn aes_decrypt_block(&self, mut in_ct: __m128i) -> __m128i {
        unsafe {
            let rnds = 10;
            in_ct = _mm_xor_si128(in_ct, self.rd_key[0]);
            for j in 1..rnds {
                in_ct = _mm_aesdec_si128(in_ct, self.rd_key[j]);
            }
            _mm_aesdeclast_si128(in_ct, self.rd_key[10])
        }
    }
}
