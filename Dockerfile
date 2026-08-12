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

RUN rustup component add rustfmt clippy \
    && useradd --system --create-home --uid 10001 --shell /usr/sbin/nologin encoder \
    && chown -R encoder:encoder /app /usr/local/cargo /usr/local/rustup

USER encoder

# Dependency graph first (invalidates less often than full tree copies).
COPY --chown=encoder:encoder Cargo.toml Cargo.lock rust-toolchain.toml ./
COPY --chown=encoder:encoder src ./src
COPY --chown=encoder:encoder tests ./tests
COPY --chown=encoder:encoder benches ./benches
COPY --chown=encoder:encoder examples ./examples

# Warm registry + compile + test in a cacheable layer (non-root).
RUN cargo test --all-features --locked

# Default re-check (fast when image layers are warm).
CMD ["cargo", "test", "--all-features", "--locked"]
