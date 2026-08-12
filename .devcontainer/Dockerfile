# Keep `rust:1.97.1` in sync with Cargo.toml rust-version / rust-toolchain.toml
# / CI toolchain pin (see REVIEW.md MSRV pin rule).
FROM rust:1.97.1-slim-bookworm

USER root
# Do not pin exact apt revisions against floating bookworm/security indexes.
# hadolint ignore=DL3008
RUN apt-get update \
    && apt-get install -y --no-install-recommends git ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && rustup component add rustfmt clippy

# Non-root user for VS Code / Codespaces.
# Copy the image rustup/cargo trees into /home/vscode so Dev Containers UID
# remapping keeps both the toolchain and cargo caches writable (host UID != 1000).
RUN useradd -m -s /bin/bash -u 1000 vscode \
    && mkdir -p /home/vscode \
    && cp -a /usr/local/rustup /home/vscode/.rustup \
    && cp -a /usr/local/cargo /home/vscode/.cargo \
    && chown -R vscode:vscode /home/vscode

USER vscode
ENV RUSTUP_HOME=/home/vscode/.rustup
ENV CARGO_HOME=/home/vscode/.cargo
ENV PATH=${CARGO_HOME}/bin:${PATH}
WORKDIR /workspace
