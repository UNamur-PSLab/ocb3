This library is a partial rewrite of Ted Krovetz's optimized [C
implementation](https://www.cs.ucdavis.edu/~rogaway/ocb/news/) of OCB3
using AVX intrinsics.

Beyond the Rust rewrite, the libraries exposes an API design tailored
for encrypted transport protocols such as QUIC, Tor or any protocol
mixing application data and protocol level controls during encryption.

## Encryption

## Decryption

## Benchmarks

## AI Disclosure

This code was preliminary written by a human, yet the author used a
local AI model to improve the robustness of the implementation, finding
bugs, writing the tests, discussing optimizations such as LLVM code
hoisting in loops, and how to prevent it. While primarily written by a
human, it remains unclear to what extend this library would have been
different if AI was not used. We thanks all Jon Does from which the model
learnt and thanks to whom it turned to be helpful.
