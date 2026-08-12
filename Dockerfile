# Verification image for contributors without a local Rust toolchain.
# Keep `rust:1.97.1` in sync with Cargo.toml rust-version / rust-toolchain.toml
# / CI toolchain pin (see REVIEW.md MSRV pin rule).
#
# Cargo work runs in a `RUN` layer so Docker/GHA layer cache can reuse deps
# across commits that only touch docs or workflows.
#
#   docker build -t axon-encoder:dev .
#   docker run --rm axon-encoder:dev
#
FROM rust:1.97.1-slim-bookworm

WORKDIR /app

RUN rustup component add rustfmt clippy

# Dependency graph first (invalidates less often than full tree copies).
COPY Cargo.toml Cargo.lock rust-toolchain.toml ./
COPY src ./src
COPY tests ./tests
COPY benches ./benches
COPY examples ./examples

# Warm registry + compile + test in a cacheable layer.
RUN cargo test --all-features --locked

# Default re-check (fast when image layers are warm).
CMD ["cargo", "test", "--all-features", "--locked"]
