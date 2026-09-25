#!/usr/bin/env bash

set -uo pipefail
export LC_ALL=C

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly BINARY_DIR="${SCRIPT_DIR}/external/householder-block-adaptive-range-finder"
readonly OUTPUT_DIR="${SCRIPT_DIR}/benchmark_results/fixed_rank_range_finder"
readonly M=16384
readonly ITERATIONS=4
readonly ADAPTIVE_NB=128
readonly ADAPTIVE_LR_TOL="0.000000000e+00"
readonly ADAPTIVE_EXECUTABLE="testing_sgeqrr_gpu"
readonly -a K_SIZES=(128 256 512 1024 1536 2048 2560 3072 4096 5120 6144 7168 8192 9216 10240 11264 12288 13312 14336 15360 16384 )
readonly -a IMPLEMENTATIONS=(
    'cpu|testing_sgeqrr_nonadaptive_cpu|'
    'magma_gpu|testing_sgeqrr_nonadaptive_gpu|--nocheck'
    'cusolver_gpu|testing_sgeqrr_nonadaptive_gpu_cusolver|'
)

CURRENT_OUTPUT=""
PARSE_ERROR=""
PARSED_TIMES=()

cleanup()
{
    if [[ -n "${CURRENT_OUTPUT}" && -f "${CURRENT_OUTPUT}" ]]; then
        rm -f -- "${CURRENT_OUTPUT}"
    fi
}
trap cleanup EXIT

write_header()
{
    printf '%s\n' \
        'm,n,k,warmup_seconds,sample_1_seconds,sample_2_seconds,sample_3_seconds,mean_seconds,status' \
        >"$1"
}

write_result_row()
{
    local csv_file=$1
    local k=$2
    local mean=$3

    printf '%s,%s,%s,%s,%s,%s,%s,%s,ok\n' \
        "${M}" "${M}" "${k}" \
        "${PARSED_TIMES[0]}" "${PARSED_TIMES[1]}" \
        "${PARSED_TIMES[2]}" "${PARSED_TIMES[3]}" "${mean}" \
        >>"${csv_file}"
}

write_failure_row()
{
    local csv_file=$1
    local k=$2
    local status=$3

    printf '%s,%s,%s,,,,,,%s\n' \
        "${M}" "${M}" "${k}" "${status}" >>"${csv_file}"
}

write_adaptive_header()
{
    printf '%s\n' \
        'm,n,vcols,nb,lr_tol,warmup_seconds,sample_1_seconds,sample_2_seconds,sample_3_seconds,mean_seconds,status' \
        >"$1"
}

write_adaptive_result_row()
{
    local csv_file=$1
    local vcols=$2
    local mean=$3

    printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,ok\n' \
        "${M}" "${M}" "${vcols}" "${ADAPTIVE_NB}" \
        "${ADAPTIVE_LR_TOL}" \
        "${PARSED_TIMES[0]}" "${PARSED_TIMES[1]}" \
        "${PARSED_TIMES[2]}" "${PARSED_TIMES[3]}" "${mean}" \
        >>"${csv_file}"
}

write_adaptive_failure_row()
{
    local csv_file=$1
    local vcols=$2
    local status=$3

    printf '%s,%s,%s,%s,%s,,,,,,%s\n' \
        "${M}" "${M}" "${vcols}" "${ADAPTIVE_NB}" \
        "${ADAPTIVE_LR_TOL}" "${status}" >>"${csv_file}"
}

parse_results()
{
    local output_file=$1
    local expected_k=$2
    local expected_iteration=1
    local row
    local -a rows=()
    local -a fields=()

    PARSE_ERROR=""
    PARSED_TIMES=()

    mapfile -t rows < <(
        awk -F, '$1 == "SGEQRR_NONADAPTIVE_RESULT" { print }' \
            "${output_file}"
    )
    if [[ ${#rows[@]} -ne ${ITERATIONS} ]]; then
        PARSE_ERROR="expected ${ITERATIONS} result rows, found ${#rows[@]}"
        return 1
    fi

    for row in "${rows[@]}"; do
        IFS=',' read -r -a fields <<<"${row}"
        if [[ ${#fields[@]} -ne 7 ]]; then
            PARSE_ERROR="expected 7 CSV fields: ${row}"
            return 1
        fi
        if [[ "${fields[0]}" != "SGEQRR_NONADAPTIVE_RESULT" ||
              "${fields[1]}" != "${M}" ||
              "${fields[2]}" != "${M}" ||
              "${fields[3]}" != "${expected_k}" ||
              "${fields[4]}" != "${expected_iteration}" ||
              "${fields[6]}" != "ok" ]]; then
            PARSE_ERROR="unexpected result values: ${row}"
            return 1
        fi
        if [[ ! "${fields[5]}" =~ ^[0-9]+([.][0-9]+)?$ ]] ||
           ! awk -v seconds="${fields[5]}" \
               'BEGIN { exit !(seconds > 0) }'; then
            PARSE_ERROR="invalid runtime: ${fields[5]}"
            return 1
        fi
        PARSED_TIMES+=("${fields[5]}")
        expected_iteration=$((expected_iteration + 1))
    done
}

parse_adaptive_results()
{
    local output_file=$1
    local expected_vcols=$2
    local expected_iteration=1
    local row
    local -a rows=()
    local -a fields=()

    PARSE_ERROR=""
    PARSED_TIMES=()

    mapfile -t rows < <(
        awk -F, '$1 == "SGEQRR_RESULT" { print }' "${output_file}"
    )
    if [[ ${#rows[@]} -ne ${ITERATIONS} ]]; then
        PARSE_ERROR="expected ${ITERATIONS} adaptive result rows, found ${#rows[@]}"
        return 1
    fi

    for row in "${rows[@]}"; do
        IFS=',' read -r -a fields <<<"${row}"
        if [[ ${#fields[@]} -ne 9 ]]; then
            PARSE_ERROR="expected 9 adaptive CSV fields: ${row}"
            return 1
        fi
        if [[ "${fields[0]}" != "SGEQRR_RESULT" ||
              "${fields[1]}" != "${M}" ||
              "${fields[2]}" != "${M}" ||
              "${fields[3]}" != "${ADAPTIVE_NB}" ||
              "${fields[4]}" != "${ADAPTIVE_LR_TOL}" ||
              "${fields[5]}" != "${expected_iteration}" ||
              "${fields[6]}" != "${expected_vcols}" ||
              "${fields[8]}" != "ok" ]]; then
            PARSE_ERROR="unexpected adaptive result values: ${row}"
            return 1
        fi
        if [[ ! "${fields[7]}" =~ ^[0-9]+([.][0-9]+)?$ ]] ||
           ! awk -v seconds="${fields[7]}" \
               'BEGIN { exit !(seconds > 0) }'; then
            PARSE_ERROR="invalid adaptive runtime: ${fields[7]}"
            return 1
        fi
        PARSED_TIMES+=("${fields[7]}")
        expected_iteration=$((expected_iteration + 1))
    done
}

mean_of_measured_samples()
{
    awk -v a="${PARSED_TIMES[1]}" \
        -v b="${PARSED_TIMES[2]}" \
        -v c="${PARSED_TIMES[3]}" \
        'BEGIN { printf "%.9f", (a + b + c) / 3.0 }'
}

run_configuration()
{
    local binary=$1
    local check_flag=$2
    local k=$3
    local csv_file=$4
    local exit_code mean
    local -a command=(
        "${binary}" -N "${M},${M}" --k "${k}"
        --niter "${ITERATIONS}" --rand
    )

    if [[ -n "${check_flag}" ]]; then
        command+=("${check_flag}")
    fi

    CURRENT_OUTPUT="$(mktemp)"
    "${command[@]}" >"${CURRENT_OUTPUT}" 2>&1
    exit_code=$?
    if [[ ${exit_code} -ne 0 ]]; then
        write_failure_row "${csv_file}" "${k}" "command_failed_${exit_code}"
        printf 'Command failed for M=%s N=%s K=%s (exit %d).\n' \
            "${M}" "${M}" "${k}" "${exit_code}" >&2
        cat "${CURRENT_OUTPUT}" >&2
        rm -f -- "${CURRENT_OUTPUT}"
        CURRENT_OUTPUT=""
        return 1
    fi

    if ! parse_results "${CURRENT_OUTPUT}" "${k}"; then
        write_failure_row "${csv_file}" "${k}" parse_failed
        printf 'Parse failed for M=%s N=%s K=%s: %s\n' \
            "${M}" "${M}" "${k}" "${PARSE_ERROR}" >&2
        cat "${CURRENT_OUTPUT}" >&2
        rm -f -- "${CURRENT_OUTPUT}"
        CURRENT_OUTPUT=""
        return 1
    fi

    mean="$(mean_of_measured_samples)"
    write_result_row "${csv_file}" "${k}" "${mean}"

    rm -f -- "${CURRENT_OUTPUT}"
    CURRENT_OUTPUT=""
}

run_adaptive_configuration()
{
    local binary=$1
    local vcols=$2
    local csv_file=$3
    local exit_code mean
    local -a command=(
        "${binary}" -N "${M},${M}" --vcols "${vcols}"
        --nb "${ADAPTIVE_NB}" --lr-tol 0
        --niter "${ITERATIONS}" --rand --nocheck
    )

    CURRENT_OUTPUT="$(mktemp)"
    "${command[@]}" >"${CURRENT_OUTPUT}" 2>&1
    exit_code=$?
    if [[ ${exit_code} -ne 0 ]]; then
        write_adaptive_failure_row \
            "${csv_file}" "${vcols}" "command_failed_${exit_code}"
        printf 'Adaptive command failed for M=%s N=%s vcols=%s (exit %d).\n' \
            "${M}" "${M}" "${vcols}" "${exit_code}" >&2
        cat "${CURRENT_OUTPUT}" >&2
        rm -f -- "${CURRENT_OUTPUT}"
        CURRENT_OUTPUT=""
        return 1
    fi

    if ! parse_adaptive_results "${CURRENT_OUTPUT}" "${vcols}"; then
        write_adaptive_failure_row "${csv_file}" "${vcols}" parse_failed
        printf 'Adaptive parse failed for M=%s N=%s vcols=%s: %s\n' \
            "${M}" "${M}" "${vcols}" "${PARSE_ERROR}" >&2
        cat "${CURRENT_OUTPUT}" >&2
        rm -f -- "${CURRENT_OUTPUT}"
        CURRENT_OUTPUT=""
        return 1
    fi

    mean="$(mean_of_measured_samples)"
    write_adaptive_result_row "${csv_file}" "${vcols}" "${mean}"

    rm -f -- "${CURRENT_OUTPUT}"
    CURRENT_OUTPUT=""
}

main()
{
    local specification label executable check_flag binary csv_file k
    local adaptive_binary adaptive_csv vcols
    local failures=0

    if [[ $# -ne 0 ]]; then
        printf 'This benchmark script does not accept arguments.\n' >&2
        return 2
    fi
    for specification in "${IMPLEMENTATIONS[@]}"; do
        IFS='|' read -r label executable check_flag <<<"${specification}"
        binary="${BINARY_DIR}/${executable}"
        if [[ ! -x "${binary}" ]]; then
            printf 'Required executable not found: %s\n' "${binary}" >&2
            return 1
        fi
    done
    adaptive_binary="${BINARY_DIR}/${ADAPTIVE_EXECUTABLE}"
    if [[ ! -x "${adaptive_binary}" ]]; then
        printf 'Required executable not found: %s\n' "${adaptive_binary}" >&2
        return 1
    fi

    mkdir -p -- "${OUTPUT_DIR}"
    for specification in "${IMPLEMENTATIONS[@]}"; do
        IFS='|' read -r label executable check_flag <<<"${specification}"
        write_header "${OUTPUT_DIR}/${label}.csv"
    done
    adaptive_csv="${OUTPUT_DIR}/adaptive_magma_gpu.csv"
    write_adaptive_header "${adaptive_csv}"

    for specification in "${IMPLEMENTATIONS[@]}"; do
        IFS='|' read -r label executable check_flag <<<"${specification}"
        binary="${BINARY_DIR}/${executable}"
        csv_file="${OUTPUT_DIR}/${label}.csv"
        for k in "${K_SIZES[@]}"; do
            printf '[%s] M=%s N=%s K=%s\n' "${label}" "${M}" "${M}" "${k}"
            if ! run_configuration \
                    "${binary}" "${check_flag}" "${k}" "${csv_file}"; then
                failures=$((failures + 1))
            fi
        done
    done

    for vcols in "${K_SIZES[@]}"; do
        printf '[adaptive_magma_gpu] M=%s N=%s vcols=%s nb=%s\n' \
            "${M}" "${M}" "${vcols}" "${ADAPTIVE_NB}"
        if ! run_adaptive_configuration \
                "${adaptive_binary}" "${vcols}" "${adaptive_csv}"; then
            failures=$((failures + 1))
        fi
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
