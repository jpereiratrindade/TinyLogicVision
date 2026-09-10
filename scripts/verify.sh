#!/usr/bin/env bash

set -euo pipefail

verify_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
verify_build="${verify_root}/build"
verify_tmp="$(mktemp -d "${TMPDIR:-/tmp}/tinyvision-verify.XXXXXX")"
trap 'rm -rf -- "${verify_tmp}"' EXIT

pass() {
    printf '%-20s %s\n' "$1" "$2"
}

fail() {
    local label="$1"
    local log="$2"
    printf '%-20s FAIL\n' "${label}" >&2
    if [[ -s "${log}" ]]; then
        printf '\n' >&2
        cat "${log}" >&2
    fi
    exit 1
}

run_step() {
    local label="$1"
    shift
    local log="${verify_tmp}/${label// /_}.log"
    if "$@" >"${log}" 2>&1; then
        pass "${label}" PASS
    else
        fail "${label}" "${log}"
    fi
}

run_test() {
    local label="$1"
    local test_name="$2"
    local log="${verify_tmp}/${test_name}.log"
    if ctest --test-dir "${verify_build}" -R "^${test_name}$" --output-on-failure \
        >"${log}" 2>&1 &&
        grep -Fq "100% tests passed, 0 tests failed out of 1" "${log}"; then
        pass "${label}" PASS
    else
        fail "${label}" "${log}"
    fi
}

run_witness() {
    local label="$1"
    local executable="$2"
    shift 2
    local log="${verify_tmp}/${label// /_}.log"
    if ! "${executable}" >"${log}" 2>&1; then
        fail "${label}" "${log}"
    fi
    local expected
    for expected in "$@"; do
        if ! grep -Fq "${expected}" "${log}"; then
            fail "${label}" "${log}"
        fi
    done
    pass "${label}" PASS
}

printf 'TinyLogicVision project verification\n\n'

run_step configure cmake -S "${verify_root}" -B "${verify_build}" \
    -G Ninja -DCMAKE_BUILD_TYPE=Release
run_step build cmake --build "${verify_build}" -j 4

run_test gradient_test gradient_test
run_test core_perf_test core_perf_test
run_test schema_model_test schema_model_test
run_test multichannel_test multichannel_test
run_test geo_test geo_test
run_test gdal_source_test gdal_source_test
run_test geo_dense_test geo_dense_test
run_test sentinel_test sentinel_test
run_test provenance_test provenance_test
run_test h3_test h3_test
run_test ablation_protocol_test ablation_protocol_test
run_test training_test training_test
run_test "TV-01A test" generalization_observatory_test
run_test "TV-01B test" spatial_coverage_experiment_test
run_test "TV-APP-00 test" application_pipeline_test
run_test "Dense test" dense_test
run_test "Dense streaming test" dense_streaming_test
run_test "CLI test" cli_test
run_test "Web GUI test" web_test
run_test "Workspace test" workspace_test
if ctest --test-dir "${verify_build}" -N | grep -Fq "gui_startup_test"; then
    run_test "GUI startup test" gui_startup_test
    run_test "GUI workspace test" gui_workspace_test
fi

run_witness "TV-00 witness" "${verify_build}/tinyvision" \
    "epoch=180 loss=0.000400 accuracy=100.000000%" \
    "TinyLogicVision TV-00 complete"
run_witness "TV-01A witness" "${verify_build}/tinyvision_generalization" \
    "final_validation_accuracy=46.88%" \
    "sealed_test_status=NOT_EVALUATED"

tv01b_log="${verify_tmp}/TV-01B_witness.log"
if ! "${verify_build}/tinyvision_spatial_coverage" >"${tv01b_log}" 2>&1; then
    fail "TV-01B witness" "${tv01b_log}"
fi
for tv01b_expected in \
    "Control A baseline_gate=PASS" \
    "Control A repetition=EXACT" \
    "Intervention B repetition=EXACT" \
    "preregistered_classification=STRONG SUPPORT" \
    "sealed_test_status=NOT_EVALUATED"; do
    if ! grep -Fq "${tv01b_expected}" "${tv01b_log}"; then
        fail "TV-01B witness" "${tv01b_log}"
    fi
done
pass "TV-01B witness" "STRONG SUPPORT"
run_witness "TV-APP-00 pipeline" "${verify_build}/application_pipeline_test" \
    "TV-APP-00 engineering pipeline PASS" \
    "classes=3 parameters=4707" \
    "repetition=EXACT" \
    "synthetic_test_status=NOT_EVALUATED"

printf '\n'
pass "SYNTHETIC TEST" "SEALED / NOT_EVALUATED"
pass "NATURAL RGB CLAIM" "NOT_EVALUATED"
pass "SENTINEL MULTIBAND CLAIM" "NOT_EVALUATED"
pass "SPECTRAL ABLATION" "NOT_EVALUATED"
