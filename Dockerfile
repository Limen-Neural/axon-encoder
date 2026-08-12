# Verification image for contributors without a local Rust toolchain.
# Matches MSRV / rust-toolchain.toml (1.97.1). Library crate — not a runtime product.
#
#   docker build -t axon-encoder:dev .
#   docker run --rm axon-encoder:dev
#
FROM rust:1.97.1-slim-bookworm

WORKDIR /app

RUN rustup component add rustfmt clippy

# Copy sources and lockfile so `cargo test --locked` matches CI.
COPY . .

# One-command verification path (also used as container default).
CMD ["cargo", "test", "--all-features", "--locked"]
