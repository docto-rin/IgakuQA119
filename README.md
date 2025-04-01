# IgakuQA119: LLM Evaluation on the 119th Japanese Medical Licensing Examination

## Overview

IgakuQA119 is a repository designed to evaluate the performance of Large Language Models (LLMs) using the 119th Japanese Medical Licensing Examination (JMLE). This project is inspired by the [nmle-rta](https://github.com/iKora128/nmle-rta/tree/main) repository and aims to assess LLMs' comprehension and application abilities in the context of the latest medical licensing exam.

**Note**: This project requires Python 3.10 or higher.

## Setup Instructions

### Package Installation

Use `uv` to synchronize packages:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
```

### Setting Environment Variables

Copy `.env.example` to `.env` and set the required API keys:

```bash
cp .env.example .env
# Open .env in your editor and set the necessary values (e.g., API keys)
```

## Solving Questions with LLMs

Here is an example of solving exam questions using an LLM:

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

Below is an example of how to grade the answers:

```bash
EXP="gemini-2.5-pro"
uv run grade_answers.py \
  --json_paths $(ls answer/json/119{A,B,C,D,E,F}_${EXP}.json) \
  --answers_path results/correct_answers.csv
```

## License

The original [nmle-rta](https://github.com/iKora128/nmle-rta/tree/main) repository does not specify a license. In the absence of a license, the default copyright laws apply, meaning the author retains all rights, and others may not reproduce, distribute, or create derivative works without explicit permission. Therefore, before using or distributing any content from this repository, please ensure you have obtained permission from the original author. For more information, refer to GitHub's documentation on [licensing a repository](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/licensing-a-repository).

