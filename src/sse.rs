/// Requires SSE3
use core::arch::x86_64::*;

// from https://github.com/rust-lang-nursery/stdsimd/pull/479 not yet merged
// due to wrongtype bug in stdlib https://github.com/rust-lang/stdarch/issues/522
macro_rules! _MM_SHUFFLE_FIXME {
    ($z:expr, $y:expr, $x:expr, $w:expr) => {
        ($z << 6) | ($y << 4) | ($x << 2) | $w
    };
}

#[inline(always)]
pub(crate) unsafe fn zero_block() -> __m128i {
    unsafe { _mm_setzero_si128() }
}

#[inline(always)]
pub(crate) unsafe fn xor_block(a: __m128i, b: __m128i) -> __m128i {
    unsafe { _mm_xor_si128(a, b) }
}

#[inline(always)]
pub(crate) unsafe fn unequal_blocks(a: __m128i, b: __m128i) -> bool {
    unsafe { _mm_movemask_epi8(_mm_cmpeq_epi8(a, b)) != 0xffff }
}

/// Reverse the 16 bytes
#[inline(always)]
pub(crate) unsafe fn swap_if_le(block: __m128i) {
    unsafe {
        let mask = _mm_set_epi8(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        _mm_shuffle_epi8(block, mask);
    }
}

#[inline(always)]
pub(crate) unsafe fn gen_offset(KtopStr: &[u64; 3], bot: i32) -> __m128i {
    unsafe {
        let ptr_KtopStr = KtopStr.as_ptr();
        let hi = _mm_load_si128(ptr_KtopStr as *const __m128i);
        let mut lo = _mm_loadu_si128(ptr_KtopStr.add(1) as *const __m128i);
        let lshift = _mm_cvtsi32_si128(bot);
        let rshift = _mm_cvtsi32_si128(64i32 - bot);
        lo = _mm_xor_si128(_mm_sll_epi64(hi, lshift), _mm_srl_epi64(lo, rshift));
        _mm_shuffle_epi8(
            lo,
            _mm_set_epi8(8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7),
        )
    }
}

/// Constant time doubling in GF(2^128), requires block to represent
/// a polynomial where the first 32-bits is the most significant part
/// of the polynomial.
#[inline(always)]
pub(crate) unsafe fn double(mut block: __m128i) -> __m128i {
    unsafe {
        let mask = _mm_set_epi32(0x87, 1, 1, 1);
        // Get the sign for each dword
        let mut tmp = _mm_srai_epi32::<31>(block);
        tmp = _mm_and_si128(tmp, mask); // bitwise AND
        tmp = _mm_shuffle_epi32(tmp, _MM_SHUFFLE_FIXME!(2, 1, 0, 3)); // shift
        block = _mm_slli_epi32(block, 1);
        _mm_xor_si128(block, tmp) // applies carry
    }
}
