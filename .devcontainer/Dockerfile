FROM rust:1.97.1-slim-bookworm

# Components used by CI and REVIEW.md local quality gates.
RUN rustup component add rustfmt clippy

# Non-root user for VS Code / Codespaces.
RUN useradd -m -s /bin/bash -u 1000 vscode \
    && chown -R vscode:vscode /usr/local/cargo /usr/local/rustup

USER vscode
ENV CARGO_HOME=/usr/local/cargo
ENV RUSTUP_HOME=/usr/local/rustup
WORKDIR /workspace
