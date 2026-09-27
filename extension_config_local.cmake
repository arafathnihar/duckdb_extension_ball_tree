# Extra extensions for a local build of the DuckDB shell, so it can replace the stock CLI: the release CLI bundles these.
# Not used by CI. Build with:
#   GEN=ninja make release EXTRA_EXTENSION_CONFIGS=$PWD/extension_config_local.cmake
duckdb_extension_load(autocomplete)
duckdb_extension_load(icu)
duckdb_extension_load(json)
