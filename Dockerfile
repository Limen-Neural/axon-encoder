# Multi-stage Docker for axon-encoder (examples runtime + CI builder).
# Keep `rust:1.97.1` in sync with Cargo.toml rust-version / rust-toolchain.toml
# / CI toolchain pin (see REVIEW.md "MSRV pin rule").
#
# Runtime (default): example binaries under /usr/local/bin
#   docker build -t axon-encoder:dev .
#   docker run --rm axon-encoder:dev
#
# Builder (tests / full toolchain):
#   docker build --target builder -t axon-encoder:builder .
#   docker run --rm axon-encoder:builder   # re-runs cargo test (CMD)

FROM rust:1.97.1-slim-bookworm AS builder

WORKDIR /app

# System toolchain stays root-owned under /usr/local/{cargo,rustup}.
# Writable Cargo registry/Git cache live under CARGO_HOME; build artifacts
# default to /app/target (WORKDIR), not under CARGO_HOME.
USER root
RUN rustup component add rustfmt clippy \
    && useradd --system --create-home --uid 10001 --shell /usr/sbin/nologin encoder \
    && mkdir -p /home/encoder/.cargo \
    && chown -R encoder:encoder /app /home/encoder

USER encoder
ENV CARGO_HOME=/home/encoder/.cargo
ENV PATH=/usr/local/cargo/bin:${PATH}

# Targeted copies so README/workflow/docs edits do not bust cargo layers.
COPY --chown=encoder:encoder Cargo.toml Cargo.lock rust-toolchain.toml ./
COPY --chown=encoder:encoder src ./src
COPY --chown=encoder:encoder tests ./tests
COPY --chown=encoder:encoder benches ./benches
COPY --chown=encoder:encoder examples ./examples

# Tests in a cacheable layer (parity with native CI / local builder rechecks).
RUN cargo test --all-features --locked

# Release examples (ndarray_encoding needs `ndarray`; use all-features).
# Copy only stable example names (skip cargo hash-suffixed intermediate bins).
RUN cargo build --release --examples --all-features --locked \
    && mkdir -p /app/out \
    && for ex in examples/*.rs; do \
         name="$(basename "${ex}" .rs)"; \
         bin="target/release/examples/${name}"; \
         test -x "${bin}" || { echo "missing example binary: ${name}" >&2; exit 1; }; \
         cp "${bin}" /app/out/; \
       done

# Default re-check when running the builder stage without args.
CMD ["cargo", "test", "--all-features", "--locked"]

# Runtime — minimal image for reproducible example runs / demos.
# Library consumers should depend on the crates.io package, not this image.
FROM debian:bookworm-slim

RUN useradd --system --create-home --uid 10001 --shell /usr/sbin/nologin encoder

COPY --from=builder /app/out/ /usr/local/bin/

USER encoder
WORKDIR /home/encoder

CMD ["ls", "/usr/local/bin"]
