#!/usr/bin/env bash
# Run one SkillsBench task with Qwen (qwen-code agent, ASU endpoint) and a given skills dir.
#   pipeline/run_task.sh <task_id> <skills_dir|none> <jobs_dir>
# Verified 2026-10-28: 3d-scan-calc + human skill -> reward 1.0 in 5.3 min.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
TASK=$1; SKILLS=$2; JOBS=$3
MODEL=${MODEL:-qwen36-35b-a3b}
set -a; . "$HOME/workspace/jarvis/.secrets/asu_llm.env"; set +a
args=(--tasks-dir "$ROOT/skillsbench/tasks/$TASK" --jobs-dir "$JOBS" --agent qwen-code
      --model "vllm/$MODEL" --agent-env "BENCHFLOW_PROVIDER_BASE_URL=$OPENAI_BASE_URL" --sandbox docker)
if [ "$SKILLS" = none ]; then args+=(--skill-mode no-skill); else args+=(--skill-mode with-skill --skills-dir "$SKILLS"); fi
exec bench eval run "${args[@]}"
