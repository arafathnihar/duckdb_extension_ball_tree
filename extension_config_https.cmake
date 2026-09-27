# Extensions for the Docker image that can read https:// files: everything in extension_config_local.cmake, plus
# httpfs, which lives outside the DuckDB source tree. The git tag is the one DuckDB v1.5.4 itself pins
# (duckdb/.github/config/extensions/httpfs.cmake). Not used by CI.
duckdb_extension_load(autocomplete)
duckdb_extension_load(icu)
duckdb_extension_load(json)
duckdb_extension_load(httpfs
    GIT_URL https://github.com/duckdb/duckdb-httpfs
    GIT_TAG c3f215ab360f04dc3d3d5305fa81849c0121f111
)
