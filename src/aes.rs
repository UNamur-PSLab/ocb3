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
    unsafe fn AES_128_key_expansion(userkey: &[u8], rd_key: &mut [__m128i; 11]) {
        unsafe {
            let mut x0 = _mm_loadu_si128(userkey.as_ptr() as *const __m128i);

            rd_key[0] = x0;
            let mut x1 = _mm_setzero_si128();
            let mut x2 = x1;
            expand_assist!(x0, x1, x2, x0, 255, 1);
            rd_key[1] = x0;
            expand_assist!(x0, x1, x2, x0, 255, 2);
            rd_key[2] = x0;
            expand_assist!(x0, x1, x2, x0, 255, 4);
            rd_key[3] = x0;
            expand_assist!(x0, x1, x2, x0, 255, 8);
            rd_key[4] = x0;
            expand_assist!(x0, x1, x2, x0, 255, 16);
            rd_key[5] = x0;
            expand_assist!(x0, x1, x2, x0, 255, 32);
            rd_key[6] = x0;
            expand_assist!(x0, x1, x2, x0, 255, 64);
            rd_key[7] = x0;
            expand_assist!(x0, x1, x2, x0, 255, 128);
            rd_key[8] = x0;
            expand_assist!(x0, x1, x2, x0, 255, 27);
            rd_key[9] = x0;
            expand_assist!(x0, x1, x2, x0, 255, 54);
            rd_key[10] = x0;
        }
    }
}
