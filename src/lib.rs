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
}

impl<'a> EncryptIO<'a> {
    /// The destination must be larger or equal size than the plaintext buffer.
    #[inline]
    pub fn validate(&self) -> Result<()> {
        if self.len() > MAX_PT_SIZE {
            return Err(CryptoError::PlaintextTooLong);
        }
        match self {
            Self::InPlace(in_pt) => {
                if in_pt.len() < OCB_TAG_LEN as usize {
                    return Err(CryptoError::DestinationUnfit);
                }
                Ok(())
            }
            Self::OutOfPlace { in_pt, out } => {
                if in_pt.len() + OCB_TAG_LEN as usize > out.len() {
                    return Err(CryptoError::DestinationUnfit);
                }
                Ok(())
            }
        }
    }

    #[inline]
    pub fn len(&self) -> usize {
        match self {
            Self::InPlace(buf) => buf.len(),
            Self::OutOfPlace { in_pt, .. } => in_pt.len(),
        }
    }
}

#[derive(Error, Debug)]
pub enum CryptoError {
    #[error("Plaintext cannot be longer than {MAX_PT_SIZE} bytes")]
    PlaintextTooLong,
    #[error(
        "The size of the destination buffer does fit. It needs to be sufficient to hold the size of the plaintext plus the size of tag"
    )]
    DestinationUnfit,
    #[error("Bad tag observed during decryption")]
    TagInvalid,
    #[error("A provided option is unsupported")]
    UnSupported,
}

mod aes;
pub mod ocb;
#[cfg(target_arch = "x86_64")]
mod sse;

#[cfg(test)]
mod tests {
    use super::*;
}
