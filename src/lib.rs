use thiserror::Error;

pub const OCB_KEY_LEN: u8 = 16;
pub const OCB_TAG_LEN: u8 = 16;
pub type Result<T> = std::result::Result<T, CryptoError>;

/// Represents a plaintext for encryption. It can either be contiguous, or be a composite
/// of two different memory buffers.
pub enum PlainText<'a> {
    Contiguous(&'a [u8]),
    Scattered(&'a [u8], &'a [u8])
}

/// An encryption can either be in place or out of place. In the case of out of place encryption
/// we may as well have scattered plaintext.
pub enum EncryptIO<'a> {
    InPlace(&'a mut [u8]),
    OutOfPlace {
        in_pt: PlainText<'a>,
        out: &'a mut [u8],
    }
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
    
    #[inline]
    pub fn validate(&self) -> Result<()> {
        match self {
            Self::InPlace(_) => Ok(()),
            Self::OutOfPlace { in_pt, out } => {
                if in_pt.len() > out.len() {
                    return Err(CryptoError::DestinationUnfit);
                }
                Ok(())
            }
        }
    }
}


#[derive(Error, Debug)]
pub enum CryptoError {
    #[error("")]
    DestinationUnfit,
    #[error("Bad tag observed during decryption")]
    TagInvalid,
    #[error("A provided option is unsupported")]
    UnSupported,
}

mod aes;
mod ocb;
#[cfg(target_arch = "x86_64")]
mod sse;

#[cfg(test)]
mod tests {
    use super::*;
}
