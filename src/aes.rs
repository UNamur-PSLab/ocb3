use core::arch::x86_64::*;

// TODO extends this to AVX512
#[derive(Copy, Clone)]
#[repr(C, align(16))]
pub(crate) struct AesKey {
    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    rd_key: [__m128i; 11],
}

impl Default for AesKey {
    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
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
macro_rules! expand_assist {
    ($v1:expr, $v2:expr, $v3:expr, $v4:expr, $shuff:expr, $rcon:expr) => {{
        $v2 = _mm_aeskeygenassist_si128::<$rcon>($v4);
        $v3 = _mm_castps_si128(_mm_shuffle_ps::<16>(
            _mm_castsi128_ps($v3),
            _mm_castsi128_ps($v1),
        ));
        $v1 = _mm_xor_si128($v1, $v3);
        $v3 = _mm_castps_si128(_mm_shuffle_ps::<140>(
            _mm_castsi128_ps($v3),
            _mm_castsi128_ps($v1),
        ));
        $v1 = _mm_xor_si128($v1, $v3);
        $v2 = _mm_shuffle_epi32::<$shuff>($v2);
        $v1 = _mm_xor_si128($v1, $v2);
    }};
}

impl AesKey {
    pub(crate) fn AES_128_key_expansion(&mut self, userkey: &[u8]) {
        unsafe {
            let mut x0 = _mm_loadu_si128(userkey.as_ptr() as *const __m128i);

            self.rd_key[0] = x0;
            let mut x1 = _mm_setzero_si128();
            let mut x2 = x1;
            expand_assist!(x0, x1, x2, x0, 255, 1);
            self.rd_key[1] = x0;
            expand_assist!(x0, x1, x2, x0, 255, 2);
            self.rd_key[2] = x0;
            expand_assist!(x0, x1, x2, x0, 255, 4);
            self.rd_key[3] = x0;
            expand_assist!(x0, x1, x2, x0, 255, 8);
            self.rd_key[4] = x0;
            expand_assist!(x0, x1, x2, x0, 255, 16);
            self.rd_key[5] = x0;
            expand_assist!(x0, x1, x2, x0, 255, 32);
            self.rd_key[6] = x0;
            expand_assist!(x0, x1, x2, x0, 255, 64);
            self.rd_key[7] = x0;
            expand_assist!(x0, x1, x2, x0, 255, 128);
            self.rd_key[8] = x0;
            expand_assist!(x0, x1, x2, x0, 255, 27);
            self.rd_key[9] = x0;
            expand_assist!(x0, x1, x2, x0, 255, 54);
            self.rd_key[10] = x0;
        };
    }

    pub(crate) fn AES_set_key_fast(&mut self, other: &AesKey) {
        let mut round = 10;
        let mut i = 0;
        self.rd_key[round] = other.rd_key[i];
        round -= 1;
        i += 1;
        while round > 0 {
            unsafe {
                self.rd_key[round] = _mm_aesimc_si128(other.rd_key[i]);
            }
            round -= 1;
            i += 1;
        }
        self.rd_key[round] = other.rd_key[i];
    }

    /// AES encrypt on unaligned in/out memory. On modern processors, unaligned
    /// store and load perform equally to aligned ones on aligned memory.
    pub(crate) fn AES_encrypt_block(&mut self, mut in_pt: __m128i) -> __m128i {
        unsafe {
            let rnds = 10;
            in_pt = _mm_xor_si128(in_pt, self.rd_key[0]);
            for j in 1..rnds {
                in_pt = _mm_aesenc_si128(in_pt, self.rd_key[j]);
            }
            _mm_aesenclast_si128(in_pt, self.rd_key[rnds])
        }
    }

    pub(crate) fn AES_decrypt_block(&mut self, mut in_ct: __m128i) -> __m128i {
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
