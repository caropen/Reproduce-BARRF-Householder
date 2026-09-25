#!/usr/bin/env bash

set -uo pipefail
export LC_ALL=C

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly BINARY_DIR="${SCRIPT_DIR}/external/householder-block-adaptive-range-finder"
readonly OUTPUT_DIR="${SCRIPT_DIR}/benchmark_results/adaptive_range_finder"
readonly LR_TOL="0.000000000e+00"
readonly ITERATIONS=6
readonly -a MATRIX_SIZES=(1024 2048 4096 16384)
readonly -a BLOCK_SIZES=(64 128 256)
readonly -a IMPLEMENTATIONS=(
    'sgeqrr3_cpu|testing_sgeqrr_cpu'
    'sgeqrr3_gpu_1_queue|testing_sgeqrr_gpu_1_queue'
    'sgeqrr3_gpu|testing_sgeqrr_gpu'
)

CURRENT_OUTPUT=""
PARSE_ERROR=""
PARSED_VCOLS=""
PARSED_TIMES=()

cleanup()
{
    if [[ -n "${CURRENT_OUTPUT}" && -f "${CURRENT_OUTPUT}" ]]; then
        rm -f -- "${CURRENT_OUTPUT}"
    fi
}
trap cleanup EXIT

parse_results()
{
    local output_file=$1
    local expected_size=$2
    local expected_block_size=$3
    local expected_iteration=1
    local row
    local -a rows=()
    local -a fields=()

    PARSE_ERROR=""
    PARSED_VCOLS=""
    PARSED_TIMES=()

    mapfile -t rows < <(
        awk -F, '$1 == "SGEQRR_RESULT" { print }' "${output_file}"
    )
    if [[ ${#rows[@]} -ne ${ITERATIONS} ]]; then
        PARSE_ERROR="expected ${ITERATIONS} result rows, found ${#rows[@]}"
        return 1
    fi

    for row in "${rows[@]}"; do
        IFS=',' read -r -a fields <<<"${row}"
        if [[ ${#fields[@]} -ne 9 ]]; then
            PARSE_ERROR="expected 9 CSV fields: ${row}"
            return 1
        fi
        if [[ "${fields[0]}" != "SGEQRR_RESULT" ||
              "${fields[1]}" != "${expected_size}" ||
              "${fields[2]}" != "${expected_size}" ||
              "${fields[3]}" != "${expected_block_size}" ||
              "${fields[4]}" != "${LR_TOL}" ||
              "${fields[5]}" != "${expected_iteration}" ||
              "${fields[6]}" != "${expected_size}" ||
              "${fields[8]}" != "ok" ]]; then
            PARSE_ERROR="unexpected result values: ${row}"
            return 1
        fi
        if [[ ! "${fields[7]}" =~ ^[0-9]+\.[0-9]{6}$ ]] ||
           ! awk -v seconds="${fields[7]}" 'BEGIN { exit !(seconds > 0) }'; then
            PARSE_ERROR="invalid runtime: ${fields[7]}"
            return 1
        fi
        PARSED_TIMES+=("${fields[7]}")
        expected_iteration=$((expected_iteration + 1))
    done

    PARSED_VCOLS="${expected_size}"
}

median_of_five()
{
    [[ $# -eq 5 ]] || return 1
    printf '%s\n' "$@" | sort -g | sed -n '3p'
}

write_header()
{
    printf '%s\n' \
        'matrix_size,block_size,lr_tol,warmup_seconds,sample_1_seconds,sample_2_seconds,sample_3_seconds,sample_4_seconds,sample_5_seconds,median_seconds,vcols,status' \
        >"$1"
}

write_row()
{
    local csv_file=$1
    shift
    printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' "$@" \
        >>"${csv_file}"
}

write_failure_row()
{
    local csv_file=$1
    local matrix_size=$2
    local block_size=$3
    local failure_status=$4

    write_row "${csv_file}" \
        "${matrix_size}" "${block_size}" "${LR_TOL}" \
        "" "" "" "" "" "" "" "" "${failure_status}"
}

run_configuration()
{
    local binary=$1
    local matrix_size=$2
    local block_size=$3
    local csv_file=$4
    local exit_code median
    local -a command=(
        "${binary}" -N "${matrix_size}" --nb "${block_size}"
        --lr-tol 0 --niter "${ITERATIONS}"
    )

    CURRENT_OUTPUT="$(mktemp)"
    "${command[@]}" >"${CURRENT_OUTPUT}" 2>&1
    exit_code=$?
    if [[ ${exit_code} -ne 0 ]]; then
        write_failure_row "${csv_file}" "${matrix_size}" \
            "${block_size}" "command_failed_${exit_code}"
        printf 'command failed for N=%s nb=%s (exit %d)\n' \
            "${matrix_size}" "${block_size}" "${exit_code}" >&2
        cat "${CURRENT_OUTPUT}" >&2
        rm -f -- "${CURRENT_OUTPUT}"
        CURRENT_OUTPUT=""
        return 1
    fi

    if ! parse_results "${CURRENT_OUTPUT}" \
            "${matrix_size}" "${block_size}"; then
        write_failure_row "${csv_file}" "${matrix_size}" \
            "${block_size}" parse_failed
        printf 'parse failed for N=%s nb=%s: %s\n' \
            "${matrix_size}" "${block_size}" "${PARSE_ERROR}" >&2
        cat "${CURRENT_OUTPUT}" >&2
        rm -f -- "${CURRENT_OUTPUT}"
        CURRENT_OUTPUT=""
        return 1
    fi

    median="$(median_of_five "${PARSED_TIMES[@]:1}")"
    write_row "${csv_file}" \
        "${matrix_size}" "${block_size}" "${LR_TOL}" \
        "${PARSED_TIMES[0]}" "${PARSED_TIMES[1]}" \
        "${PARSED_TIMES[2]}" "${PARSED_TIMES[3]}" \
        "${PARSED_TIMES[4]}" "${PARSED_TIMES[5]}" \
        "${median}" "${PARSED_VCOLS}" ok

    rm -f -- "${CURRENT_OUTPUT}"
    CURRENT_OUTPUT=""
}

main()
{
    local specification label executable binary csv_file
    local matrix_size block_size
    local failures=0

    if [[ $# -ne 0 ]]; then
        printf 'This benchmark script does not accept arguments.\n' >&2
        return 2
    fi
    for specification in "${IMPLEMENTATIONS[@]}"; do
        IFS='|' read -r label executable <<<"${specification}"
        binary="${BINARY_DIR}/${executable}"
        if [[ ! -x "${binary}" ]]; then
            printf 'Required executable not found: %s\n' "${binary}" >&2
            return 1
        fi
    done

    mkdir -p -- "${OUTPUT_DIR}"
    for specification in "${IMPLEMENTATIONS[@]}"; do
        IFS='|' read -r label executable <<<"${specification}"
        write_header "${OUTPUT_DIR}/${label}.csv"
    done

    for specification in "${IMPLEMENTATIONS[@]}"; do
        IFS='|' read -r label executable <<<"${specification}"
        binary="${BINARY_DIR}/${executable}"
        csv_file="${OUTPUT_DIR}/${label}.csv"
        for matrix_size in "${MATRIX_SIZES[@]}"; do
            for block_size in "${BLOCK_SIZES[@]}"; do
                printf '[%s] N=%s nb=%s\n' \
                    "${label}" "${matrix_size}" "${block_size}"
                if ! run_configuration "${binary}" \
                        "${matrix_size}" "${block_size}" "${csv_file}"; then
                    failures=$((failures + 1))
                fi
            done
        done
    done

    if [[ ${failures} -ne 0 ]]; then
        printf 'Completed with %d failed configurations.\n' \
            "${failures}" >&2
        return 1
    fi
    printf 'Benchmark results written to %s\n' "${OUTPUT_DIR}"
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    main "$@"
fi
