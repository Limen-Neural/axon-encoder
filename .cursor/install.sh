#!/usr/bin/env bash
# Cursor Cloud Agent install script for axon-encoder (`install` in .cursor/environment.json).
#
# Cursor runs this from the repository root during every Build, on its default
# Ubuntu base image (CPU only: cloud agents have no GPU), then snapshots the disk.
# It must be idempotent. Shell exports don't survive into agent runs, so the tools
# it installs are exposed through /etc/profile.d and /usr/local/bin.
# See https://cursor.com/docs/cloud-agent/setup
#
# Installs the Rust toolchain pinned by rust-toolchain.toml (plus the clippy and
# rustfmt components and the wasm32-unknown-unknown target used by CI) and warms
# the Cargo caches so fresh Builds can build, test, and lint without a cold start.
#
# Idempotent: safe to rerun on a stale snapshot; each step converges.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

STEP_START=$SECONDS

step() {
  printf '\n==> %s\n' "$1"
  STEP_START=$SECONDS
}

step_done() {
  printf '    done in %ds\n' "$((SECONDS - STEP_START))"
}

# --- System prerequisites ---------------------------------------------------
# A C linker and libc headers (gcc / libc6-dev via build-essential) plus curl
# and CA certificates for rustup. All are normally preinstalled; only
# install what is missing so warm runs do not touch apt.
required_packages=(build-essential curl ca-certificates)
missing_packages=()
for pkg in "${required_packages[@]}"; do
  if ! dpkg-query -W -f='${Status}' "$pkg" 2>/dev/null | grep -q 'install ok installed'; then
    missing_packages+=("$pkg")
  fi
done
if ((${#missing_packages[@]} > 0)); then
  step "Installing system packages: ${missing_packages[*]}"
  SUDO=()
  if [ "$(id -u)" -ne 0 ]; then
    SUDO=(sudo)
  fi
  "${SUDO[@]}" apt-get update -qq
  "${SUDO[@]}" apt-get install -y --no-install-recommends "${missing_packages[@]}"
  step_done
else
  step "System packages already present"
  step_done
fi

# --- Rust toolchain ---------------------------------------------------------
# The channel lives in rust-toolchain.toml and must match Cargo.toml
# rust-version and the CI pin (REVIEW.md "MSRV pin rule").
RUST_TOOLCHAIN="$(sed -n 's/^channel *= *"\(.*\)"/\1/p' rust-toolchain.toml)"
if [ -z "$RUST_TOOLCHAIN" ]; then
  echo "ERROR: could not read toolchain channel from rust-toolchain.toml" >&2
  exit 1
fi

# rustup-init cannot update this running script's environment, so put the
# cargo bin directory on PATH for the remaining steps.
export PATH="$HOME/.cargo/bin:$PATH"

if ! command -v rustup >/dev/null 2>&1; then
  step "Installing rustup"
  rustup_init="$(mktemp)"
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs -o "$rustup_init"
  sh "$rustup_init" -y --profile minimal --default-toolchain none
  rm -f "$rustup_init"
  step_done
else
  step "rustup already installed"
  step_done
fi

step "Installing Rust $RUST_TOOLCHAIN (clippy, rustfmt, wasm32-unknown-unknown)"
rustup toolchain install "$RUST_TOOLCHAIN" --profile minimal \
  --component clippy --component rustfmt \
  --target wasm32-unknown-unknown
# Make the pinned toolchain the default as well; inside the repository
# rust-toolchain.toml selects it regardless.
rustup default "$RUST_TOOLCHAIN" >/dev/null
step_done

# --- Cargo caches -----------------------------------------------------------
# Warm every build the repository's quality gate (REVIEW.md / CI) touches, so
# later runs only rebuild what changed. Cargo skips each step almost entirely
# on a warm snapshot.
step "Fetching locked dependencies"
cargo fetch --locked
step_done

step "Building library, tests, and examples (--all-features)"
cargo test --no-run --locked --all-features
step_done

# Plain `cargo test --locked` (default features) is the most common command.
step "Building library and tests (default features)"
cargo test --no-run --locked
step_done

# Warm the clippy cache for `cargo clippy --all-targets --all-features`.
step "Checking all targets with clippy"
cargo clippy --locked --all-targets --all-features
step_done

# Benchmarks build in the release profile; warming them keeps the documented
# regression workflow (`cargo bench`) from paying a full cold build.
step "Compiling benchmarks"
cargo bench --no-run --locked
step_done

# The CI wasm job's check: browser target with serde + wasm-js.
step "Checking the wasm32-unknown-unknown target"
cargo check --locked --target wasm32-unknown-unknown \
  --no-default-features --features serde,wasm-js
step_done

step "Building documentation (all features)"
RUSTDOCFLAGS='-D warnings' cargo doc --locked --no-deps --all-features
step_done

AS_ROOT=""
if [ "$(id -u)" -ne 0 ]; then
  AS_ROOT="sudo"
fi

# --- Expose the tools to later shells ---
# The PATH exports above last only for this script; Cursor starts the agent's shells
# separately. Login shells get these directories from /etc/profile.d, and every other
# shell finds the entry points through symlinks in /usr/local/bin (on the default PATH).
tool_dirs=("$HOME/.cargo/bin")
# shellcheck disable=SC2016 # $PATH must expand when the profile is sourced, not now.
printf 'export PATH=%q:$PATH\n' "$(IFS=:; echo "${tool_dirs[*]}")" |
  $AS_ROOT tee /etc/profile.d/cursor-env-axon-encoder.sh >/dev/null
for dir in "${tool_dirs[@]}"; do
  [ -d "$dir" ] || continue
  for tool in "$dir"/*; do
    name="${tool##*/}"
    dest="/usr/local/bin/$name"
    # Don't shadow a base-image command with the same name.
    if [ -f "$tool" ] && [ -x "$tool" ] && [ ! -e "$dest" ] && [ ! -L "$dest" ]; then
      $AS_ROOT ln -s "$tool" "$dest"
    fi
  done
done

printf '\n==> Setup complete in %ds\n' "$SECONDS"
