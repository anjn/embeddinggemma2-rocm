# syntax=docker/dockerfile:1
ARG ROCM_BASE=rocm/dev-ubuntu-24.04:7.2
FROM rust:1.90-bookworm AS tokenizer
WORKDIR /src/tokenizer
COPY tokenizer/ ./
RUN cargo build --locked --release
RUN mkdir -p /licenses && find /usr/local/cargo/registry/src -type f \
    \( -iname "license*" -o -iname "notice*" -o -iname "copying*" \) \
    -exec cp --parents {} /licenses/ \;

FROM ${ROCM_BASE} AS runtime
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-venv python3-pip cmake ninja-build g++ git ca-certificates curl migraphx \
    && rm -rf /var/lib/apt/lists/*
RUN python3 -m venv /opt/eg2-venv
ENV PATH=/opt/eg2-venv/bin:$PATH
RUN pip install --no-cache-dir 'torch==2.14.1+rocm7.2' 'torchvision==0.29.1+rocm7.2' \
    --index-url https://download.pytorch.org/whl/rocm7.2 --extra-index-url https://pypi.org/simple
WORKDIR /opt/eg2-src
COPY CMakeLists.txt LICENSE THIRD_PARTY_NOTICES.md THIRD_PARTY_LICENSES.txt ./
COPY src/ src/
COPY include/ include/
COPY kernels/ kernels/
COPY examples/ examples/
COPY --from=tokenizer /src/tokenizer/target/release/libeg2_tokenizer.a /opt/eg2-tokenizer.a
COPY --from=tokenizer /licenses/ /opt/eg2/share/licenses/tokenizer/
ARG GPU_ARCH=gfx1150
RUN cmake -S . -B /tmp/eg2-build -G Ninja \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/opt/eg2 \
    -DTORCH_ROOT=/opt/eg2-venv/lib/python3.12/site-packages/torch \
    -DEG2_GPU_ARCH=${GPU_ARCH} -DEG2_TOKENIZER_LIBRARY=/opt/eg2-tokenizer.a \
    && cmake --build /tmp/eg2-build -j 2 && cmake --install /tmp/eg2-build
COPY pyproject.toml ./
COPY python/ python/
RUN pip install --no-cache-dir '.[prepare,server,bench,test]'
ENV EG2_LIBRARY_PATH=/opt/eg2/lib/libeg2.so \
    PYTHONPATH=/opt/rocm/lib:/opt/rocm/lib/python3.12/site-packages \
    MIGRAPHX_DISABLE_MLIR=0 MIGRAPHX_MLIR_USE_SPECIFIC_OPS=attention \
    TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
COPY README.md README.ja.md ./
COPY docs/ docs/
COPY tools/ tools/
COPY tests/ tests/
COPY benchmarks/ benchmarks/
COPY tokenizer/ tokenizer/
WORKDIR /data
ENTRYPOINT ["eg2"]
CMD ["--help"]
