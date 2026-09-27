# syntax=docker/dockerfile:1
#
# DuckDB v1.5.4 shell with the ball_tree extension (plus json, icu, parquet, autocomplete) as one static binary in
# an otherwise empty image.
#
#   docker build -t duckdb-ball-tree .
#   docker run --rm -v "$PWD":/data duckdb-ball-tree /data/my.duckdb -c "SELECT ..."
#
# Being static, the binary cannot dlopen, so extensions that are not built in (httpfs, ...) cannot be loaded.

FROM debian:bookworm-slim AS build
RUN apt-get update && apt-get install -y --no-install-recommends \
        make g++ cmake ninja-build python3 ca-certificates \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /src
COPY . .

# DuckDB takes its version string from git, and the build context has no .git, so state it.
# -static only for executables: the shared libraries and loadable extensions must stay position independent.
RUN cmake -G Ninja -S duckdb -B build \
        -DCMAKE_BUILD_TYPE=Release \
        -DEXTENSION_STATIC_BUILD=1 \
        -DDUCKDB_EXTENSION_CONFIGS="/src/extension_config.cmake;/src/extension_config_local.cmake" \
        -DOVERRIDE_GIT_DESCRIBE=v1.5.4-0-g08e34c447 \
        -DBUILD_UNITTESTS=OFF \
        -DCMAKE_EXE_LINKER_FLAGS=-static \
    && cmake --build build --target shell \
    && strip build/duckdb \
    && mkdir -p /rootfs/tmp /rootfs/data \
    && chmod 1777 /rootfs/tmp \
    && cp build/duckdb /rootfs/duckdb

# scratch has no /tmp, which DuckDB uses when a query spills to disk, and no $HOME for the shell's history file
FROM scratch
COPY --from=build /rootfs/ /
ENV HOME=/tmp
WORKDIR /data
ENTRYPOINT ["/duckdb"]
