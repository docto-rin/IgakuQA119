# IgakuQA119: LLM Evaluation on the 119th Japanese Medical Licensing Examination

## Overview

IgakuQA119 is a repository designed to evaluate the performance of Large Language Models (LLMs) using the 119th Japanese Medical Licensing Examination (JMLE). This project, inspired by the [nmle-rta](https://github.com/iKora128/nmle-rta/tree/main) repository, assesses LLMs' comprehension and application abilities within the context of Japan's latest medical licensing exam.

## Leaderboard

<!-- LEADERBOARD_START -->

| Rank | Entry | Overall Score (Rate) | Overall Acc. | No-Img Score (Rate) | No-Img Acc. |
|------|------|---------------------|-------------|--------------------|------------|
| 1 | Gemini 2.5 Pro | 485/500 (97.00%) | 389/400 (97.25%) | 372/383 (97.13%) | 290/297 (97.64%) |
| 2 | CA-DSR1-DQ32B-JP | 376/500 (75.20%) | 292/400 (73.00%) | 295/383 (77.02%) | 223/297 (75.08%) |
| 3 | Gemma-3-27B | 320/500 (64.00%) | 252/400 (63.00%) | 252/383 (65.80%) | 196/297 (65.99%) |

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
ENTRY_NAME="Gemini 2.5 Pro" # Desired name for the Leaderboard entry

uv run grade_answers.py \
  --json_paths $(ls answers/119{A,B,C,D,E,F}_${EXP}.json) \
  --entry_name "${ENTRY_NAME}"
```

**This script automatically updates the Leaderboard section in this README file** with the consolidated results based on the processed JSON files.

## License

This repository is licensed under the Apache License 2.0. For details, see the [LICENSE](LICENSE) file.

Original repository [nmle-rta](https://github.com/iKora128/nmle-rta/tree/main) is also licensed under Apache License 2.0 as authorized by the original author.
