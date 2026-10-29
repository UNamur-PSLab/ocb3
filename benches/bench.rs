use criterion::{BatchSize, BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use ocb3::{AEctx, DecryptIO, EncryptIO, PlainText};
use std::hint::black_box;

const SPEC_KEY: [u8; 16] = [
    0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f,
];

fn encrypt_in_place(in_pt: &mut [u8], nonce: &[u8; 12], aad: &[u8], ctx: &mut AEctx) {
    ctx.ae_encrypt(
        EncryptIO::InPlace(black_box(in_pt)),
        black_box(nonce),
        black_box(aad),
    )
    .expect("encrypt failed");
    black_box(in_pt);
}

fn encrypt_out_of_place(
    in_pt: &mut [u8],
    nonce: &[u8; 12],
    aad: &[u8],
    out: &mut [u8],
    ctx: &mut AEctx,
) {
    ctx.ae_encrypt(
        EncryptIO::OutOfPlace {
            in_pt: PlainText::Contiguous(black_box(in_pt)),
            out: black_box(out),
        },
        black_box(nonce),
        black_box(aad),
    )
    .expect("encrypt failed");
    black_box(out);
}

fn encrypt_out_of_place_scattered(
    in_pt1: &mut [u8],
    in_pt2: &mut [u8],
    nonce: &[u8; 12],
    aad: &[u8],
    out: &mut [u8],
    ctx: &mut AEctx,
) {
    ctx.ae_encrypt(
        EncryptIO::OutOfPlace {
            in_pt: PlainText::Scattered(black_box(in_pt1), black_box(in_pt2)),
            out: black_box(out),
        },
        black_box(nonce),
        black_box(aad),
    )
    .expect("encrypt failed");
    black_box(out);
}

fn decrypt_in_place(in_ct: &mut [u8], nonce: &[u8; 12], aad: &[u8], ctx: &mut AEctx) {
    ctx.ae_decrypt(
        DecryptIO::InPlace(black_box(in_ct)),
        black_box(nonce),
        black_box(aad),
    )
    .expect("decrypt failed");
    black_box(in_ct);
}

fn criterion_benchmark(c: &mut Criterion<CPUTime>) {
    let mut group = c.benchmark_group("encrypt/decrypt ");

    // Matching default OpenSSL's speed test sizes
    for size in [16, 64, 256, 1024, 8192, 16384].iter() {
        group.throughput(Throughput::Bytes(*size));
        group.bench_with_input(BenchmarkId::new("Encrypt in place", size), size, |b, s| {
            let mut ctx = AEctx::new(black_box(&SPEC_KEY));
            let mut nonce = [0u8; 12];
            b.iter_batched_ref(
                || {
                    nonce[11] = nonce[11].wrapping_add(1);
                    let in_pt = vec![0u8; *s as usize + 16];
                    (in_pt, nonce)
                },
                |(in_pt, iter_nonce)| encrypt_in_place(in_pt, iter_nonce, &[], &mut ctx),
                BatchSize::SmallInput,
            )
        });
        group.bench_with_input(
            BenchmarkId::new("Encrypt out of place contiguous", size),
            size,
            |b, s| {
                let mut ctx = AEctx::new(black_box(&SPEC_KEY));
                let mut nonce = [0u8; 12];
                b.iter_batched_ref(
                    || {
                        nonce[11] = nonce[11].wrapping_add(1);
                        let in_pt = vec![0u8; *s as usize];
                        let out = vec![0u8; *s as usize + 16];
                        (in_pt, out, nonce)
                    },
                    |(in_pt, out, iter_nonce)| {
                        encrypt_out_of_place(in_pt, iter_nonce, &[], out, &mut ctx)
                    },
                    BatchSize::SmallInput,
                )
            },
        );
        group.bench_with_input(
            BenchmarkId::new("Encrypt out of place scattered", size),
            size,
            |b, s| {
                let mut ctx = AEctx::new(black_box(&SPEC_KEY));
                let mut nonce = [0u8; 12];
                b.iter_batched_ref(
                    || {
                        nonce[11] = nonce[11].wrapping_add(1);
                        let in_pt1 = vec![0u8; (*s / 2) as usize];
                        let pt2 = in_pt1.clone();
                        let out = vec![0u8; *s as usize + 16];
                        (in_pt1, pt2, out, nonce)
                    },
                    |(in_pt1, pt2, out, iter_nonce)| {
                        encrypt_out_of_place_scattered(in_pt1, pt2, iter_nonce, &[], out, &mut ctx)
                    },
                    BatchSize::SmallInput,
                )
            },
        );

        group.bench_with_input(BenchmarkId::new("Decrypt in place", size), size, |b, s| {
            let mut setup_ctx = AEctx::new(black_box(&SPEC_KEY));
            let mut bench_ctx = AEctx::new(black_box(&SPEC_KEY));
            let mut nonce = [0u8; 12];
            b.iter_batched_ref(
                || {
                    nonce[11] = nonce[11].wrapping_add(1);
                    let mut in_ct = vec![0u8; *s as usize + 16];
                    setup_ctx
                        .ae_encrypt(EncryptIO::InPlace(&mut in_ct), &nonce, &[])
                        .expect("encrypt failed");
                    (in_ct, nonce)
                },
                |(in_ct, iter_nonce)| decrypt_in_place(in_ct, iter_nonce, &[], &mut bench_ctx),
                BatchSize::SmallInput,
            )
        });
    }

    for size in [40, 1200, 1424].iter() {
        let hdr_size = 25;
        group.throughput(Throughput::Bytes(*size));
        group.bench_with_input(
            BenchmarkId::new("QUIC-like Encrypt in place with 25 bytes AAD", size),
            size,
            |b, s| {
                let mut ctx = AEctx::new(black_box(&SPEC_KEY));
                let mut nonce = [0u8; 12];
                b.iter_batched_ref(
                    || {
                        nonce[11] = nonce[11].wrapping_add(1);
                        let in_pt = vec![0u8; *s as usize + 16];
                        let aad = vec![0u8; hdr_size];
                        (in_pt, aad, nonce)
                    },
                    |(in_pt, aad, iter_nonce)| encrypt_in_place(in_pt, iter_nonce, aad, &mut ctx),
                    BatchSize::SmallInput,
                )
            },
        );
        group.bench_with_input(
            BenchmarkId::new(
                "QUIC-like Encrypt out of place contiguous with 25 bytes AAD",
                size,
            ),
            size,
            |b, s| {
                let mut ctx = AEctx::new(black_box(&SPEC_KEY));
                let mut nonce = [0u8; 12];
                b.iter_batched_ref(
                    || {
                        nonce[11] = nonce[11].wrapping_add(1);
                        let in_pt = vec![0u8; *s as usize];
                        let out = vec![0u8; *s as usize + 16];
                        let aad = vec![0u8; hdr_size];
                        (in_pt, aad, out, nonce)
                    },
                    |(in_pt, aad, out, iter_nonce)| {
                        encrypt_out_of_place(in_pt, iter_nonce, aad, out, &mut ctx)
                    },
                    BatchSize::SmallInput,
                )
            },
        );
        // This should be a bit slower. pt1 and pt2 are not a multiple of the block size
        group.bench_with_input(
            BenchmarkId::new("QUIC-like Encrypt out of place scattered", size),
            size,
            |b, s| {
                let mut ctx = AEctx::new(black_box(&SPEC_KEY));
                let mut nonce = [0u8; 12];
                b.iter_batched_ref(
                    || {
                        nonce[11] = nonce[11].wrapping_add(1);
                        let pt1_size = if *s == 40 {
                            20
                        } else if *s == 1200 {
                            1130
                        } else {
                            1350
                        };

                        let in_pt1 = vec![0u8; pt1_size];
                        let pt2 = vec![0u8; *size as usize - pt1_size];
                        let out = vec![0u8; *s as usize + 16];
                        let aad = vec![0u8; hdr_size];
                        (in_pt1, pt2, aad, out, nonce)
                    },
                    |(in_pt1, pt2, aad, out, iter_nonce)| {
                        encrypt_out_of_place_scattered(in_pt1, pt2, iter_nonce, aad, out, &mut ctx)
                    },
                    BatchSize::SmallInput,
                )
            },
        );
        // Both plaintexts are multiple of the blocksize
        group.bench_with_input(
            BenchmarkId::new(
                "QUIC-like Encrypt out of place scattered, multiple of blocksize",
                size,
            ),
            size,
            |b, s| {
                let mut ctx = AEctx::new(black_box(&SPEC_KEY));
                let mut nonce = [0u8; 12];
                b.iter_batched_ref(
                    || {
                        nonce[11] = nonce[11].wrapping_add(1);
                        let pt1_size = if *s == 40 {
                            32
                        } else if *s == 1200 {
                            1136
                        } else {
                            1360
                        };

                        let in_pt1 = vec![0u8; pt1_size];
                        let pt2 = vec![0u8; *size as usize - pt1_size];
                        let out = vec![0u8; *s as usize + 16];
                        let aad = vec![0u8; hdr_size];
                        (in_pt1, pt2, aad, out, nonce)
                    },
                    |(in_pt1, pt2, aad, out, iter_nonce)| {
                        encrypt_out_of_place_scattered(in_pt1, pt2, iter_nonce, aad, out, &mut ctx)
                    },
                    BatchSize::SmallInput,
                )
            },
        );
    }
}

mod bench_util;
use bench_util::*;

criterion_group! {
    name = benches;
    config = Criterion::default()
        .with_measurement(CPUTime)
        .sample_size(500);
    targets = criterion_benchmark
}

criterion_main!(benches);
