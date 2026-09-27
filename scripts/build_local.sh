#!/bin/sh
# Build the DuckDB shell for everyday local use, as a replacement for the stock CLI:
#  - adds the extensions the stock CLI bundles (autocomplete, icu, json), see extension_config_local.cmake
#  - turns on extension autoloading/autoinstall, which the extension template's Makefile switches off
# Not used by CI. Run from the repository root; the result is build/release/duckdb.
set -e
cd "$(dirname "$0")/.."
ENABLE_EXTENSION_AUTOLOADING=1 ENABLE_EXTENSION_AUTOINSTALL=1 GEN=ninja \
    make release EXTRA_EXTENSION_CONFIGS="$PWD/extension_config_local.cmake"
