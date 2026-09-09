#!/usr/bin/env bash

# TinyLogicVision CLI test suite

test_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cli_bin="${test_root}/bin/tinyvision"
tmp_dir="$(mktemp -d "${TMPDIR:-/tmp}/tinyvision-cli-test.XXXXXX")"
cleanup() {
    rm -rf -- "${tmp_dir}"
}
trap cleanup EXIT

fail() {
    printf 'FAIL: %s\n' "$1" >&2
    exit 1
}

# 1. Test help outputs
"${cli_bin}" --help > "${tmp_dir}/help.log" 2>&1 || fail "tinyvision --help failed"
grep -Fq "TinyLogicVision CLI v0.1" "${tmp_dir}/help.log" || fail "--help missing header"
grep -Fq "Usage:" "${tmp_dir}/help.log" || fail "--help missing usage"

"${cli_bin}" -h > "${tmp_dir}/h.log" 2>&1 || fail "tinyvision -h failed"
grep -Fq "TinyLogicVision CLI v0.1" "${tmp_dir}/h.log" || fail "-h missing header"

# 2. Test no args and invalid commands
if "${cli_bin}" > "${tmp_dir}/no_args.log" 2>&1; then
    fail "tinyvision with no args should exit with error"
fi

if "${cli_bin}" unknown_command > "${tmp_dir}/unknown.log" 2>&1; then
    fail "tinyvision with unknown command should exit with error"
fi
grep -Fq "unknown command: unknown_command" "${tmp_dir}/unknown.log" || fail "missing unknown command error message"

# 3. Test wrong argument counts
if "${cli_bin}" train only_one_arg > "${tmp_dir}/train_wrong_args.log" 2>&1; then
    fail "tinyvision train with 1 arg should exit with error"
fi

if "${cli_bin}" classify only_one_arg > "${tmp_dir}/classify_wrong_args.log" 2>&1; then
    fail "tinyvision classify with 1 arg should exit with error"
fi

if "${cli_bin}" evaluate only_one_arg > "${tmp_dir}/evaluate_wrong_args.log" 2>&1; then
    fail "tinyvision evaluate with 1 arg should exit with error"
fi

# 4. Generate controlled fixture for end-to-end CLI workflow
fixture_dir="${tmp_dir}/fixture_dataset"
mkdir -p "${fixture_dir}"

python3 - <<'PYEOF' "${fixture_dir}"
import os, struct, sys, zlib

root = sys.argv[1]

def make_png(path, r, g, b, variation=0):
    width, height = 16, 16
    rows = []
    for y in range(height):
        row = bytearray([0])
        for x in range(width):
            delta = variation if (x + y) % 2 == 0 else -variation
            row.append(max(0, min(255, r + delta)))
            row.append(max(0, min(255, g + delta)))
            row.append(max(0, min(255, b + delta)))
        rows.append(bytes(row))
    raw = b''.join(rows)

    def chunk(tag, data):
        crc = zlib.crc32(tag + data) & 0xffffffff
        return struct.pack('>I', len(data)) + tag + data + struct.pack('>I', crc)

    ihdr = struct.pack('>IIBBBBB', width, height, 8, 2, 0, 0, 0)
    png = b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', ihdr) + chunk(b'IDAT', zlib.compress(raw)) + chunk(b'IEND', b'')

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        f.write(png)

classes = {
    'soil': (170, 100, 45),
    'vegetation': (30, 180, 45),
    'water': (25, 90, 200),
}

for split in ('train', 'dev', 'probe'):
    for name, (r, g, b) in classes.items():
        make_png(os.path.join(root, split, name, '01.png'), r, g, b, 4)
        make_png(os.path.join(root, split, name, '02.png'), r, g, b, 8)
PYEOF

[ -f "${fixture_dir}/train/vegetation/01.png" ] || fail "failed to generate PNG fixture"

# 5. Test path independence: invoke CLI from a different working directory
model_file="${tmp_dir}/cli_model.tlv"

(
    cd "${tmp_dir}" || exit 1
    # Train
    "${cli_bin}" train "${fixture_dir}" "${model_file}" > "${tmp_dir}/cli_train.log" 2>&1 \
        || fail "CLI train execution failed"
    grep -Fq "TinyLogicVision TV-APP-00 training" "${tmp_dir}/cli_train.log" || fail "missing train header"
    grep -Fq "model_saved=" "${tmp_dir}/cli_train.log" || fail "missing model_saved output"

    # Classify
    "${cli_bin}" classify "${model_file}" "${fixture_dir}/probe/vegetation/01.png" > "${tmp_dir}/cli_classify.log" 2>&1 \
        || fail "CLI classify execution failed"
    grep -Fq "predicted=" "${tmp_dir}/cli_classify.log" || fail "missing predicted class"
    grep -Fq "score=" "${tmp_dir}/cli_classify.log" || fail "missing prediction score"

    # Evaluate
    "${cli_bin}" evaluate "${model_file}" "${fixture_dir}/probe" > "${tmp_dir}/cli_evaluate.log" 2>&1 \
        || fail "CLI evaluate execution failed"
    grep -Fq "accuracy=" "${tmp_dir}/cli_evaluate.log" || fail "missing evaluate accuracy"
    grep -Fq "confusion rows=actual cols=predicted" "${tmp_dir}/cli_evaluate.log" || fail "missing confusion matrix"
) || fail "Subshell execution failed"

printf 'TinyLogicVision CLI tests PASS\n'
exit 0
