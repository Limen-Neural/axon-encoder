# Keep `rust:1.97.1` in sync with Cargo.toml rust-version / rust-toolchain.toml
# / CI toolchain pin (see REVIEW.md MSRV pin rule).
FROM rust:1.97.1-slim-bookworm

USER root
RUN apt-get update \
    && apt-get install -y --no-install-recommends git ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && rustup component add rustfmt clippy

# Non-root user for VS Code / Codespaces.
# Cargo home lives under /home/vscode so Dev Containers UID remapping keeps
# registry/cache writable when host UID != 1000.
RUN useradd -m -s /bin/bash -u 1000 vscode \
    && mkdir -p /home/vscode/.cargo /home/vscode/.rustup \
    && chown -R vscode:vscode /home/vscode

USER vscode
ENV CARGO_HOME=/home/vscode/.cargo
ENV RUSTUP_HOME=/home/vscode/.rustup
# Still use the image toolchain binaries from /usr/local.
ENV PATH=/usr/local/cargo/bin:/usr/local/rustup/toolchains/1.97.1-x86_64-unknown-linux-gnu/bin:${PATH}
WORKDIR /workspace
