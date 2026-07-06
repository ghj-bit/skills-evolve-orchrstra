#!/usr/bin/env bash

terminalbench_sanitize_docker_name() {
    python - "$1" <<'PY'
import re
import sys

print(re.sub(r"[^a-z0-9_-]+", "-", sys.argv[1].lower()).strip("-"))
PY
}

terminalbench_monitor_verified_task_docker() {
    local out_dir="$1"
    local monitor_log="${2:-${out_dir}/docker_monitor.log}"
    local verify_file="${out_dir}/verification.jsonl"
    local seen_file="${out_dir}/.docker_monitor_seen_tasks"
    local interval="${TERMINALBENCH_DOCKER_MONITOR_INTERVAL:-20}"
    local image_prefix="${TERMINALBENCH_DOCKER_IMAGE_PREFIX:-alexgshaw}"
    local image_tag="${TERMINALBENCH_DOCKER_IMAGE_TAG:-20251031}"

    : > "${seen_file}"
    {
        echo "Docker cleanup monitor started."
        echo "verification_file: ${verify_file}"
        echo "interval_seconds: ${interval}"
        echo "image_prefix: ${image_prefix}"
        echo "image_tag: ${image_tag}"
    } >> "${monitor_log}"

    while true; do
        if [[ -f "${verify_file}" ]]; then
            while IFS= read -r task_id; do
                [[ -n "${task_id}" ]] || continue
                if grep -Fxq "${task_id}" "${seen_file}" 2>/dev/null; then
                    continue
                fi
                echo "${task_id}" >> "${seen_file}"

                local safe_task_id
                safe_task_id="$(terminalbench_sanitize_docker_name "${task_id}")"
                local project_prefix="tbench-${safe_task_id}-"
                local image_name="${image_prefix}/${task_id}:${image_tag}"

                echo "[$(date '+%Y-%m-%d %H:%M:%S')] verified task detected: ${task_id}; project_prefix=${project_prefix}; image=${image_name}" >> "${monitor_log}"

                while IFS= read -r container_name; do
                    [[ -n "${container_name}" ]] || continue
                    local project_name="${container_name%-main-1}"
                    echo "Stopping verified task container: ${container_name}" >> "${monitor_log}"
                    docker rm -f "${container_name}" >> "${monitor_log}" 2>&1 || true
                    if [[ -n "${project_name}" && "${project_name}" != "${container_name}" ]]; then
                        echo "Removing verified task network: ${project_name}_default" >> "${monitor_log}"
                        docker network rm "${project_name}_default" >> "${monitor_log}" 2>&1 || true
                    fi
                done < <(
                    docker ps -a --format '{{.Names}}' 2>/dev/null \
                        | grep -E "^${project_prefix}.*-main-1$" || true
                )

                if docker image inspect "${image_name}" >/dev/null 2>&1; then
                    echo "Removing verified task image: ${image_name}" >> "${monitor_log}"
                    docker rmi "${image_name}" >> "${monitor_log}" 2>&1 || true
                else
                    echo "Task image not present or already removed: ${image_name}" >> "${monitor_log}"
                fi
            done < <(
                python - "${verify_file}" <<'PY'
import json
import sys

path = sys.argv[1]
seen = set()
try:
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                task_id = json.loads(line).get("task_id")
            except json.JSONDecodeError:
                continue
            if task_id and task_id not in seen:
                seen.add(task_id)
                print(task_id)
except FileNotFoundError:
    pass
PY
            )
        fi
        sleep "${interval}"
    done
}

terminalbench_start_docker_cleanup_monitor() {
    local out_dir="$1"
    local log_file="${2:-${out_dir}/run.log}"
    TERMINALBENCH_DOCKER_MONITOR_PID=""
    if [[ "${TERMINALBENCH_DOCKER_MONITOR:-1}" != "1" ]]; then
        echo "Docker cleanup monitor disabled." >> "${log_file}"
        return 0
    fi
    if ! command -v docker >/dev/null 2>&1; then
        echo "Docker cleanup monitor disabled: docker command not found." >> "${log_file}"
        return 0
    fi
    terminalbench_monitor_verified_task_docker "${out_dir}" "${out_dir}/docker_monitor.log" &
    TERMINALBENCH_DOCKER_MONITOR_PID=$!
    echo "Docker cleanup monitor enabled with pid ${TERMINALBENCH_DOCKER_MONITOR_PID}." >> "${log_file}"
}

terminalbench_stop_docker_cleanup_monitor() {
    if [[ -n "${TERMINALBENCH_DOCKER_MONITOR_PID:-}" ]]; then
        kill "${TERMINALBENCH_DOCKER_MONITOR_PID}" 2>/dev/null || true
        wait "${TERMINALBENCH_DOCKER_MONITOR_PID}" 2>/dev/null || true
        TERMINALBENCH_DOCKER_MONITOR_PID=""
    fi
}
