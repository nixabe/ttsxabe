# syntax=docker/dockerfile:1.7
#
# Two stages, and the split is the one `docs/DOCKER.md` argued for before any
# of this existed. There is no `build.rs`, no `cc` crate and no nvcc anywhere
# in the workspace: CUDA is reached at run time through `cudarc`'s dynamic
# loading, and every kernel is compiled from a string by NVRTC. So the builder
# needs a Rust toolchain and nothing else, and the runtime needs NVRTC and the
# driver API - which is the `runtime` image, not `devel`.
#
# What is deliberately *not* here is any model weight. `facebook/mms-tts-nan`
# is CC-BY-NC 4.0 and the translator CC-BY-NC-SA 4.0; this repository
# redistributes neither, `models/` is gitignored for that reason, and the tree
# is mounted at run time instead. `.dockerignore` enforces it from the other
# side - the 43 GB never even reaches the daemon.

# ------------------------------------------------------------------ builder
#
# `ubuntu:22.04` rather than the `rust:1-slim` the design sketch named, and the
# reason is glibc rather than taste. A dynamically linked binary requires the
# glibc it was built against *or newer*: `rust:1-slim` is Debian bookworm at
# 2.36 and the CUDA runtime image below is Ubuntu 22.04 at 2.35, so the obvious
# pairing compiles cleanly and then dies at exec with a missing `GLIBC_2.36`.
# Building on the runtime's own distribution removes the question rather than
# answering it. Nothing CUDA is installed in this stage.
FROM ubuntu:22.04 AS build

ENV DEBIAN_FRONTEND=noninteractive \
    CARGO_HOME=/usr/local/cargo \
    RUSTUP_HOME=/usr/local/rustup \
    PATH=/usr/local/cargo/bin:$PATH

# `gcc` and `libc6-dev` are for the linker, not for compiling any dependency:
# `reqwest` is taken with default features off precisely so that no TLS stack
# is built, and nothing else in the tree has a C half.
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        gcc libc6-dev curl ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# `--default-toolchain none`, because `rust-toolchain.toml` is the one source
# of truth for the channel and its components. Installing a toolchain here and
# then having the manifest override it would mean downloading two.
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
      | sh -s -- -y --no-modify-path --default-toolchain none

WORKDIR /build

# The manifest alone, first: it is the input to toolchain selection and it
# changes far less often than the sources, so the toolchain download is a
# cached layer rather than part of every build.
COPY rust-toolchain.toml ./
RUN rustup show

COPY Cargo.toml Cargo.lock ./
COPY crates ./crates

# Cache mounts rather than the usual dummy-`main.rs` trick, which with eighteen
# workspace members would be eighteen stanzas to keep in step with
# `Cargo.toml`. The binary has to be taken out of the target directory inside
# this same `RUN`: a cache mount is not part of the resulting layer, so
# anything left behind in it is not there to `COPY --from`.
#
# `--locked` because `Cargo.lock` is committed, and a build that quietly
# resolves something else is not the build that was tested.
#
# Only `xabe-engine` is built. The four benchmark binaries in the same crate
# need a card *and* a checkpoint to say anything, so they belong on the host
# that has both rather than in a deployment image.
RUN --mount=type=cache,target=/usr/local/cargo/registry,sharing=locked \
    --mount=type=cache,target=/build/target,sharing=locked \
    cargo build --release --locked -p xabe-engine --bin xabe-engine \
 && install -Dm755 target/release/xabe-engine /out/xabe-engine

# ------------------------------------------------------------------ runtime
#
# `runtime`, not `devel`: NVRTC compiles the kernels and `cudarc` resolves the
# driver API by name at run time, so headers and nvcc would be weight with no
# user. `libcuda.so.1` is not in this image and should not be - it comes from
# the host driver through the NVIDIA container runtime, which is why the image
# does not pin a driver version and the compose file asks for a GPU instead.
#
# 12.4.1 matches the `cuda-12040` feature `cudarc` is taken with.
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

# curl is here for the healthcheck and for nothing else. The alternative is a
# container with no way to answer whether it is up, since the engine has no
# self-check subcommand and `/health` is HTTP.
RUN apt-get update \
 && apt-get install -y --no-install-recommends curl \
 && rm -rf /var/lib/apt/lists/*

# Unprivileged, and uid 1000 specifically. The models are a read-only bind of a
# host directory, so the container's uid has to be able to read it as it stands
# - 1000 is the usual first human user and needs no `chown` of a 43 GB tree.
# Where that is wrong, compose's `user:` overrides it without a rebuild.
RUN useradd --uid 1000 --create-home --shell /usr/sbin/nologin xabe

COPY --from=build /out/xabe-engine /usr/local/bin/xabe-engine

# Where the model tree is expected. Created so that a run with no mount fails
# by naming a missing checkpoint rather than a missing directory.
RUN install -d -o xabe -g xabe /models

USER xabe

# The engine's own default in every example; the compose file publishes it.
EXPOSE 8000

# The binary itself, so `docker run ... --help` prints the real flag surface
# rather than a shell's, and `command:` in compose is an argument list. Every
# flag has an env twin (see docs/CLI.md), so the usual case sets no arguments
# at all and a stage is turned off by leaving its variables unset.
ENTRYPOINT ["/usr/local/bin/xabe-engine"]
