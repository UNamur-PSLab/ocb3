use thiserror::Error;

// Note: this code cannot process ciphertext or plaintext larger than 2^20 - 128
pub const MAX_PT_SIZE: usize = 1048448;
pub const OCB_KEY_LEN: u8 = 16;
pub const OCB_TAG_LEN: u8 = 16;
pub type Result<T> = std::result::Result<T, CryptoError>;

/// Represents a plaintext for encryption. It can either be contiguous, or be a composite
/// of two different memory buffers.
pub enum PlainText<'a> {
    Contiguous(&'a [u8]),
    Scattered(&'a [u8], &'a [u8]),
}

/// An encryption can either be in place or out of place. In the case of out of place encryption
/// we may as well have scattered plaintext.
pub enum EncryptIO<'a> {
    /// Encrypts up to .len() - OCB_TAG_LEN
    InPlace(&'a mut [u8]),
    OutOfPlace {
        in_pt: PlainText<'a>,
        out: &'a mut [u8],
    },
}

impl<'a> PlainText<'a> {
    #[inline]
    pub fn len(&self) -> usize {
        match self {
            Self::Contiguous(pt) => pt.len(),
            Self::Scattered(pt1, pt2) => pt1.len() + pt2.len(),
        }
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        match self {
            Self::Contiguous(buf) => buf.is_empty(),
            Self::Scattered(pt1, pt2) => pt1.is_empty() && pt2.is_empty(),
        }
    }
}

impl<'a> EncryptIO<'a> {
    /// The destination must be larger or equal size than the plaintext buffer.
    #[inline]
    pub fn validate(&self) -> Result<()> {
        match self {
            Self::InPlace(in_pt) => {
                if in_pt.len() < OCB_TAG_LEN as usize {
                    return Err(CryptoError::DestinationSizeUnfit);
                }
                if in_pt.len() - OCB_TAG_LEN as usize > MAX_PT_SIZE {
                    return Err(CryptoError::PlaintextTooLong);
                }
                Ok(())
            }
            Self::OutOfPlace { in_pt, out } => {
                let pt_len = in_pt.len();
                if pt_len > MAX_PT_SIZE {
                    return Err(CryptoError::PlaintextTooLong);
                }
                if pt_len + OCB_TAG_LEN as usize > out.len() {
                    return Err(CryptoError::DestinationSizeUnfit);
                }
                Ok(())
            }
        }
    }

    #[inline]
    pub fn len(&self) -> usize {
        match self {
            Self::InPlace(buf) => buf.len().saturating_sub(OCB_TAG_LEN as usize),
            Self::OutOfPlace { in_pt, .. } => in_pt.len(),
        }
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        match self {
            Self::InPlace(buf) => buf.is_empty(),
            Self::OutOfPlace { in_pt, .. } => in_pt.is_empty(),
        }
    }
}

pub enum DecryptIO<'a> {
    InPlace(&'a mut [u8]),
    OutOfPlace {
        in_ct: &'a [u8],
        out_pt: &'a mut [u8],
    },
}

impl<'a> DecryptIO<'a> {
    #[inline]
    fn validate(&self) -> Result<()> {
        match self {
            Self::OutOfPlace { in_ct, out_pt } => {
                if in_ct.len() < OCB_TAG_LEN as usize {
                    return Err(CryptoError::CiphertextTooSmall);
                }
                if in_ct.len() - OCB_TAG_LEN as usize > MAX_PT_SIZE {
                    return Err(CryptoError::PlaintextTooLong);
                }

                if (out_pt.len() + OCB_TAG_LEN as usize) < in_ct.len() {
                    return Err(CryptoError::DestinationSizeUnfit);
                }
            }
            Self::InPlace(in_ct) => {
                if in_ct.len() < OCB_TAG_LEN as usize {
                    return Err(CryptoError::CiphertextTooSmall);
                }
                if in_ct.len() - OCB_TAG_LEN as usize > MAX_PT_SIZE {
                    return Err(CryptoError::PlaintextTooLong);
                }
            }
        }
        Ok(())
    }

    /// Returns the length of the ciphertext.
    #[inline]
    pub fn len(&self) -> usize {
        match self {
            Self::InPlace(buf) => buf.len(),
            Self::OutOfPlace { in_ct, .. } => in_ct.len(),
        }
    }
}

#[derive(Error, Debug)]
pub enum CryptoError {
    #[error("Plaintext cannot be longer than {MAX_PT_SIZE} bytes")]
    PlaintextTooLong,
    #[error("The ciphertext cannot be smaller than the tag len")]
    CiphertextTooSmall,
    #[error(
        "The size of the destination buffer does fit. It needs to be sufficient to hold the size of the plaintext plus the size of tag"
    )]
    DestinationSizeUnfit,
    #[error("Bad tag observed during decryption")]
    TagInvalid,
    #[error("A provided option is unsupported")]
    UnSupported,
}

// The SIMD implementation is only compiled for x86_64 targets that have the `avx` and `aes` target
// features enabled at compile time. This is detected at compile time: on any other target the crate
// fails to build with an explanatory message instead of emitting code that would trap at run time.
#[cfg(all(target_arch = "x86_64", target_feature = "avx", target_feature = "aes",))]
mod aes;

#[cfg(all(target_arch = "x86_64", target_feature = "avx", target_feature = "aes",))]
pub mod ocb;

#[cfg(all(target_arch = "x86_64", target_feature = "avx", target_feature = "aes",))]
pub use ocb::AEctx;

#[cfg(all(target_arch = "x86_64", target_feature = "avx", target_feature = "aes",))]
mod avx;

// `not(doctest)`: rustdoc re-compiles this crate for the doctests without the
// `build.rustflags` from .cargo/config.toml, so the gate is not satisfied
// there; the doctest compilation must not fail on it.
#[cfg(not(all(target_arch = "x86_64", target_feature = "avx", target_feature = "aes",)))]
#[cfg(not(doctest))]
compile_error!(
    "ocb3 requires an x86_64 target with the `avx` and `aes` \
     target features enabled at compile time. Build with \
     RUSTFLAGS=\"-C target-feature=+avx,+aes"
);
