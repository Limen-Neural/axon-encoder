# Keep `rust:1.97.1` in sync with Cargo.toml rust-version / rust-toolchain.toml
# / CI toolchain pin (see REVIEW.md MSRV pin rule).
FROM rust:1.97.1-slim-bookworm

USER root
# Do not pin exact apt revisions against floating bookworm/security indexes —
# superseded security builds disappear and break fresh Codespaces builds.
# hadolint ignore=DL3008
RUN apt-get update \
    && apt-get install -y --no-install-recommends git ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && rustup component add rustfmt clippy

# Non-root user for VS Code / Codespaces.
# CARGO_HOME lives under /home/vscode so Dev Containers UID remapping keeps
# registry/git caches writable when host UID != 1000.
# RUSTUP_HOME stays at the image default (/usr/local/rustup) so the preinstalled
# 1.97.1 toolchain and components remain visible to the rustup proxy.
RUN useradd -m -s /bin/bash -u 1000 vscode \
    && mkdir -p /home/vscode/.cargo/bin \
    && chown -R vscode:vscode /home/vscode \
    && chown -R vscode:vscode /usr/local/cargo /usr/local/rustup

USER vscode
ENV CARGO_HOME=/home/vscode/.cargo
# Do not override RUSTUP_HOME — empty home rustup would force a re-download.
# Prepend CARGO_HOME/bin so `cargo install` tools are on PATH.
ENV PATH=/home/vscode/.cargo/bin:/usr/local/cargo/bin:${PATH}
WORKDIR /workspace
