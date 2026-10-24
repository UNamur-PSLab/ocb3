This library is a partial rewrite of Ted Krovetz's optimized [C
implementation](https://www.cs.ucdavis.edu/~rogaway/ocb/news/) of OCB3
using AVX intrinsics.

Beyond the Rust rewrite, the libraries exposes an API design tailored
for encrypted transport protocols such as QUIC, Tor or any protocol
mixing application data and protocol level controls during encryption.

## Encryption

The library has the current following constraints:

- plaintext and associated data are at most `MAX_PT_SIZE` bytes (2^20 - 128);
- the calling CPU must support AVX, AES-NI, SSSE3 and SSE4.1. There is
  no automatic detection currently.

We can encrypt using the API either **in place**, or **out of place**
from a contiguous buffer, or **out of place** from a scattered buffer.

**In place.** The buffer holds the plaintext followed by a 16-byte tag slot;
on return it holds the ciphertext followed by the tag:

```rust
use ocb3::{AEctx, EncryptIO};

let key: [u8; 16] = [0u8; 16];
let nonce: [u8; 12] = [0u8; 12];
let ad: &[u8] = b"protocol header"; // authenticated, not encrypted

let pt: &[u8] = b"some application data";
let mut buf = vec![0u8; pt.len() + ocb3::OCB_TAG_LEN as usize];
buf[..pt.len()].copy_from_slice(pt);

let mut ctx = AEctx::new(&key);
unsafe {
    ctx.ae_encrypt(EncryptIO::InPlace(&mut buf), &nonce, ad).unwrap();
}
// buf[..pt.len()] is now the ciphertext, buf[pt.len()..] the tag
```

**Out of place.** The destination must hold the plaintext plus the tag:

```rust
use ocb3::{AEctx, EncryptIO, PlainText};

let key: [u8; 16] = [0u8; 16];
let nonce: [u8; 12] = [0u8; 12];
let ad: &[u8] = b"protocol header";

let pt: &[u8] = b"some application data";
let mut ct = vec![0u8; pt.len() + ocb3::OCB_TAG_LEN as usize];

let mut ctx = AEctx::new(&key);
unsafe {
    ctx.ae_encrypt(
        EncryptIO::OutOfPlace { in_pt: PlainText::Contiguous(pt), out: &mut ct },
        &nonce,
        ad,
    )
    .unwrap();
}
```

**Scattered plaintext.** When the protocol-level control bytes and the
application data live in separate buffers, `PlainText::Scattered` encrypts
both as a single contiguous plaintext; there is no need to reassemble
before encryption:

```rust
use ocb3::{AEctx, EncryptIO, PlainText};

let key: [u8; 16] = [0u8; 16];
let nonce: [u8; 12] = [0u8; 12];
let ad: &[u8] = b"protocol header";

let control: &[u8] = b"frame header";
let payload: &[u8] = b"application data";
let mut ct = vec![0u8; control.len() + payload.len() + ocb3::OCB_TAG_LEN as usize];

let mut ctx = AEctx::new(&key);
unsafe {
    ctx.ae_encrypt(
        EncryptIO::OutOfPlace {
            in_pt: PlainText::Scattered(control, payload),
            out: &mut ct,
        },
        &nonce,
        ad,
    )
    .unwrap();
}
```

A size violation returns `Err(CryptoError::DestinationSizeUnfit)` or
`Err(CryptoError::PlaintextTooLong)`.

## Decryption

The input is `ciphertext || tag`, encrypted with the same key, nonce and
associated data. `ae_decrypt` verifies the tag with a constant-time
comparison: on a mismatch it returns `Err(CryptoError::TagInvalid)` and wipes
everything it may have written to the destination, so no partial plaintext
survives a failed call.

**In place.** The buffer `ciphertext || tag` becomes `plaintext || tag`; the
tag is left in place:

```rust
use ocb3::{AEctx, DecryptIO, EncryptIO};

let key: [u8; 16] = [0u8; 16];
let nonce: [u8; 12] = [0u8; 12];
let ad: &[u8] = b"protocol header";

let pt: &[u8] = b"some application data";
let mut buf = vec![0u8; pt.len() + ocb3::OCB_TAG_LEN as usize];
buf[..pt.len()].copy_from_slice(pt);

let mut ctx = AEctx::new(&key);
unsafe {
    ctx.ae_encrypt(EncryptIO::InPlace(&mut buf), &nonce, ad).unwrap();
    ctx.ae_decrypt(DecryptIO::InPlace(&mut buf), &nonce, ad).unwrap();
}
assert_eq!(&buf[..pt.len()], pt); // plaintext recovered, tag preserved
```

**Out of place.** The destination must hold the plaintext
(`in_ct.len() - OCB_TAG_LEN` bytes); the example allocates the input's full
size:

```rust
use ocb3::{AEctx, DecryptIO, EncryptIO, PlainText};

let key: [u8; 16] = [0u8; 16];
let nonce: [u8; 12] = [0u8; 12];
let ad: &[u8] = b"protocol header";

let pt: &[u8] = b"some application data";
let mut ct = vec![0u8; pt.len() + ocb3::OCB_TAG_LEN as usize];
let mut ctx = AEctx::new(&key);
unsafe {
    ctx.ae_encrypt(
        EncryptIO::OutOfPlace { in_pt: PlainText::Contiguous(pt), out: &mut ct },
        &nonce,
        ad,
    )
    .unwrap();
}

let mut out = vec![0u8; ct.len()];
unsafe {
    ctx.ae_decrypt(
        DecryptIO::OutOfPlace { in_ct: &ct, out_pt: &mut out },
        &nonce,
        ad,
    )
    .unwrap();
}
```

## Benchmarks

## AI Disclosure

This code was preliminary written by a human, yet the author used a
local AI model to improve the robustness of the implementation, finding
bugs, writing the tests, discussing optimizations such as LLVM code
hoisting in loops, and how to prevent it. While primarily written by a
human, it remains unclear to what extend this library would have been
different if AI was not used. We thanks all Jon Does from which the model
learnt and thanks to whom it turned to be helpful.
