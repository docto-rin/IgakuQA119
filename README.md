# IgakuQA119: LLM Evaluation on the 119th Japanese Medical Licensing Examination

## Overview

IgakuQA119 is a repository designed to evaluate the performance of Large Language Models (LLMs) using the 119th Japanese Medical Licensing Examination (JMLE). This project, inspired by the [nmle-rta](https://github.com/iKora128/nmle-rta/tree/main) repository, assesses LLMs' comprehension and application abilities within the context of Japan's latest medical licensing exam.

## Leaderboard

<!-- LEADERBOARD_START -->

| Rank | Entry | Overall Score (Rate) | Overall Acc. | No-Img Score (Rate) | No-Img Acc. |
|------|------|---------------------|-------------|--------------------|------------|
| 1 | Gemini-2.5-Pro | 485/500 (97.00%) | 389/400 (97.25%) | 372/383 (97.13%) | 290/297 (97.64%) |
| 2 | Gemini-2.0-Flash | 436/500 (87.20%) | 352/400 (88.00%) | 333/383 (86.95%) | 263/297 (88.55%) |
| 3 | CA-DSR1-DQ32B-JP | 376/500 (75.20%) | 292/400 (73.00%) | 295/383 (77.02%) | 223/297 (75.08%) |
| 4 | Gemma-3-27B | 320/500 (64.00%) | 252/400 (63.00%) | 252/383 (65.80%) | 196/297 (65.99%) |
| 5 | PLaMo-1.0-Prime | 201/500 (40.20%) | 161/400 (40.25%) | 153/383 (39.95%) | 119/297 (40.07%) |

<!-- LEADERBOARD_END -->

## 1. Setup Instructions

**Note**: Requires Python 3.10 or higher.

### 1.1 Package Installation

Use `uv` to synchronize packages:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
```

### 1.2 Setting Environment Variables

Copy `.env.example` to `.env` and set required API keys:

```bash
cp .env.example .env
# Open .env and set necessary values (e.g., API keys)
```

## 2. Solving Questions with LLMs

Example script to solve exam questions using an LLM:

```bash
EXP="gemini-2.5-pro"
MODEL_NAME="gemini-2.5-pro-exp-03-25" # Actual model name used by the API

for suffix in A B C D E F; do
  uv run main.py "questions/119${suffix}_json.json" \
    --exp "119${suffix}_${EXP}" \
    --models "${MODEL_NAME}"
done
```

## 3. Grading Answers

Example script for grading answers:

```bash
EXP="gemini-2.5-pro"
ENTRY_NAME="Gemini-2.5-Pro" # Desired name for the Leaderboard entry

uv run grade_answers.py \
  --json_paths $(ls answers/119{A,B,C,D,E,F}_${EXP}.json) \
  --entry_name "${ENTRY_NAME}"
```

**This script automatically updates the Leaderboard section in this README file** with the consolidated results based on the processed JSON files.

## 4. Handling Skipped Questions

Occasionally, questions might be skipped during the initial run (e.g., due to API errors or timeouts). The following steps describe how to re-run only the skipped questions, merge the results with the original run, and then grade the complete set.

### 4.1 Example Variable Setup

```bash
EXP="gemini-2.0-flash"
MODEL_NAME="gemini-2.0-flash-exp"
ENTRY_NAME="Gemini-2.0-Flash"

# File listing skipped question IDs, typically generated during the initial run
SKIPPED_LIST="results/119_${EXP}_skipped.txt"
# Suffix for the retry experiment files
RERUN_EXP="${EXP}_retry"
# Suffix for the merged experiment files
MERGED_EXP="${EXP}_merged"
```

### 4.2 Workflow Commands for Skipped Questions

The following sequence of commands performs the complete workflow for handling skipped questions:
1.  **Re-run Skipped:** Process only the questions listed in the `skipped_list` using `rerun_skipped.py`. The results will be saved with the `RERUN_EXP` suffix.
2.  **Merge Results:** Combine the original answers (from the `EXP` run) and the re-run answers (from the `RERUN_EXP` run) into a new, complete set using `scripts/merge_results.py`. The merged results will have the `MERGED_EXP` suffix.
3.  **Grade Merged:** Grade the complete, merged answer set using `grade_answers.py` and update the leaderboard with the specified `ENTRY_NAME`.

```bash
# 1. Re-run Skipped Questions
uv run rerun_skipped.py \
  --skipped_list "${SKIPPED_LIST}" \
  --models "${MODEL_NAME}" \
  --questions_dir questions \
  --rerun_exp "${RERUN_EXP}"

# 2. Merge Results
uv run scripts/merge_results.py \
  --original_pattern "answers/119*_${EXP}.json" \
  --retry_pattern "answers/119*_${RERUN_EXP}.json" \
  --merged_exp "${MERGED_EXP}"

# 3. Grade Merged Results
uv run grade_answers.py \
  --json_paths $(ls answers/119{A,B,C,D,E,F}_${MERGED_EXP}.json) \
  --entry_name "${ENTRY_NAME}" \
  --output results
```

## License

This repository is licensed under the Apache License 2.0. For details, see the [LICENSE](LICENSE) file.

Original repository [nmle-rta](https://github.com/iKora128/nmle-rta/tree/main) is also licensed under Apache License 2.0 as authorized by the original author.
