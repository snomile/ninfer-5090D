# syntax=docker/dockerfile:1

FROM nvidia/cuda:13.4.2-devel-ubuntu24.04 AS build

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
        cmake \
        libavcodec-dev \
        libavformat-dev \
        libavutil-dev \
        libcurl4-openssl-dev \
        libhwloc-dev \
        libjpeg-dev \
        libswscale-dev \
        liburing-dev \
        ninja-build \
        pkg-config \
        python3 \
        python3-jsonschema \
    && rm -rf /var/lib/apt/lists/*

# Link against the toolkit stub when the build stage has no host GPU driver.
RUN ln -s /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/libcuda.so.1

ARG NINFER_BUILD_RESERVE_MIB=6144
ARG NINFER_BUILD_TESTS=OFF
WORKDIR /src
COPY . .

RUN cmake -S . -B /build -G Ninja \
        -DCMAKE_BUILD_TYPE=Release -DNINFER_TARGET=rtx5090 \
        -DNINFER_BUILD_APPS=ON \
        -DBUILD_TESTING="${NINFER_BUILD_TESTS}" \
        -DNINFER_BUILD_BENCHMARKS=OFF

# Bound compilation by both physical and cgroup available memory.
RUN set -eu; \
    PHYSICAL_MIB="$(awk '/MemAvailable/{printf "%d", $2/1024}' /proc/meminfo)"; \
    AVAIL_MIB="$PHYSICAL_MIB"; \
    LIMIT="$(cat /sys/fs/cgroup/memory.max 2>/dev/null || true)"; \
    USED="$(cat /sys/fs/cgroup/memory.current 2>/dev/null || true)"; \
    if [ -z "$LIMIT" ]; then \
        LIMIT="$(cat /sys/fs/cgroup/memory/memory.limit_in_bytes 2>/dev/null || true)"; \
        USED="$(cat /sys/fs/cgroup/memory/memory.usage_in_bytes 2>/dev/null || true)"; \
    fi; \
    if [ -n "$LIMIT" ] && [ "$LIMIT" != max ] && \
       [ "$LIMIT" -lt 1000000000000 ] 2>/dev/null; then \
        CGROUP_MIB="$(( (LIMIT - ${USED:-0}) / 1048576 ))"; \
        [ "$CGROUP_MIB" -ge "$AVAIL_MIB" ] || AVAIL_MIB="$CGROUP_MIB"; \
    fi; \
    JOBS="$(( (AVAIL_MIB - NINFER_BUILD_RESERVE_MIB) / 1200 ))"; \
    if [ "$JOBS" -lt 1 ]; then \
        echo "Insufficient build memory: available=${AVAIL_MIB}MiB, reserve=${NINFER_BUILD_RESERVE_MIB}MiB" >&2; \
        exit 1; \
    fi; \
    NPROC="$(nproc)"; \
    [ "$JOBS" -le "$NPROC" ] || JOBS="$NPROC"; \
    echo "NInfer build: available=${AVAIL_MIB}MiB, reserve=${NINFER_BUILD_RESERVE_MIB}MiB, parallel=${JOBS}"; \
    cmake --build /build --parallel "$JOBS" --target ninfer ninfer-serve; \
    if [ "$NINFER_BUILD_TESTS" = ON ]; then \
        cmake --build /build --parallel "$JOBS" --target \
            ninfer_device_test ninfer_cpu_system_test ninfer_context_pool_test ninfer_kv_cache_test \
            ninfer_artifact_materialization_test ninfer_qwen3_5_vision_workspace_test \
            ninfer_qwen3_5_native_transactions_test ninfer_qwen3_5_state_image_test \
            ninfer_qwen_frontend_test ninfer_qwen_tool_constraints_test \
            ninfer_tool_call_parser_test ninfer_grammar_test ninfer_json_schema_test \
            ninfer_scheduler_test ninfer_context_cost_test ninfer_placement_policy_test \
            ninfer_resource_manager_test ninfer_context_capacity_test \
            ninfer_serve_options_test ninfer_cli_options_test ninfer_openai_schema_test \
            ninfer_openai_responses_test ninfer_anthropic_schema_test \
            ninfer_metrics_test ninfer_request_log_test ninfer_engine_failure_monitor_test; \
    fi; \
    rm /usr/local/cuda/lib64/libcuda.so.1

FROM nvidia/cuda:13.4.2-runtime-ubuntu24.04

ARG DEBIAN_FRONTEND=noninteractive
ARG NINFER_REVISION
LABEL org.opencontainers.image.source="https://github.com/snomile/ninfer-5090D" \
      org.opencontainers.image.revision="${NINFER_REVISION}"
RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
        ca-certificates \
        libavcodec60 \
        libavformat60 \
        libavutil58 \
        libcurl4t64 \
        libhwloc15 \
        libjpeg-turbo8 \
        libswscale7 \
        liburing2 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=build /build/apps/ninfer /usr/local/bin/ninfer
COPY --from=build /build/apps/ninfer-serve /usr/local/bin/ninfer-serve
COPY --from=build /build/lib/ /usr/local/lib/
RUN ldconfig

WORKDIR /workspace
EXPOSE 8080
STOPSIGNAL SIGTERM

CMD ["ninfer-serve", "--help"]
