use thiserror::Error;

pub const OCB_KEY_LEN: u8 = 16;
pub const OCB_TAG_LEN: u8 = 16;

#[derive(Error, Debug)]
pub enum CryptoError {
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
