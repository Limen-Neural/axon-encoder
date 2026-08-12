# Keep `rust:1.97.1` in sync with Cargo.toml rust-version / rust-toolchain.toml
# / CI toolchain pin (see REVIEW.md MSRV pin rule).
FROM rust:1.97.1-slim-bookworm

USER root
RUN apt-get update \
    && apt-get install -y --no-install-recommends git ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && rustup component add rustfmt clippy

# Non-root user for VS Code / Codespaces; own cargo/rustup caches.
RUN useradd -m -s /bin/bash -u 1000 vscode \
    && chown -R vscode:vscode /usr/local/cargo /usr/local/rustup

USER vscode
ENV CARGO_HOME=/usr/local/cargo
ENV RUSTUP_HOME=/usr/local/rustup
WORKDIR /workspace
