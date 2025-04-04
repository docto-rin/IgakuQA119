# IgakuQA119: LLM Evaluation on the 119th Japanese Medical Licensing Examination

## Overview

IgakuQA119 is a repository designed to evaluate the performance of Large Language Models (LLMs) using the 119th Japanese Medical Licensing Examination (JMLE). This project, inspired by the [nmle-rta](https://github.com/iKora128/nmle-rta/tree/main) repository, assesses LLMs' comprehension and application abilities within the context of Japan's latest medical licensing exam.

**Note**: Requires Python 3.10 or higher.

## Setup Instructions

### Package Installation

Use `uv` to synchronize packages:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
```

### Setting Environment Variables

Copy `.env.example` to `.env` and set required API keys:

```bash
cp .env.example .env
# Open .env and set necessary values (e.g., API keys)
```

## Solving Questions with LLMs

Example script to solve exam questions using an LLM:

```bash
EXP="gemini-2.5-pro"
MODEL_NAME="gemini-2.5-pro"

for suffix in A B C D E F; do
  uv run main.py "question/119${suffix}_json.json" \
    --exp "119${suffix}_${EXP}" \
    --models "${MODEL_NAME}"
done
```

## Grading Answers

Example script for grading answers:

```bash
EXP="gemini-2.5-pro"
uv run grade_answers.py \
  --json_paths $(ls answer/json/119{A,B,C,D,E,F}_${EXP}.json) \
```

## License

This repository is licensed under the Apache License 2.0. For details, see the [LICENSE](LICENSE) file.

Original repository [nmle-rta](https://github.com/iKora128/nmle-rta/tree/main) is also licensed under Apache License 2.0 as authorized by the original author.
