# IgakuQA119: LLM Evaluation on the 119th Japanese Medical Licensing Examination

## Overview

IgakuQA119 is a repository designed to evaluate the performance of Large Language Models (LLMs) using the 119th Japanese Medical Licensing Examination (JMLE). This project, inspired by the [nmle-rta](https://github.com/iKora128/nmle-rta/tree/main) repository, assesses LLMs' comprehension and application abilities within the context of Japan's latest medical licensing exam. The dataset used for evaluation was obtained through a clean process, with details on acquisition provided [here](#dataset-acquisition). It supports evaluation using both cloud-based APIs (like OpenAI, Anthropic, Gemini) and local LLMs via Ollama.

## Leaderboard

<!-- LEADERBOARD_START -->

| Rank | Entry | Overall Score (Rate) | Overall Acc. | No-Img Score (Rate) | No-Img Acc. |
|------|------|---------------------|-------------|--------------------|------------|
| 1 | Gemini-2.5-Pro | 485/500 (97.00%) | 389/400 (97.25%) | 372/383 (97.13%) | 290/297 (97.64%) |
| 2 | Gemini-2.0-Flash | 436/500 (87.20%) | 352/400 (88.00%) | 333/383 (86.95%) | 263/297 (88.55%) |
| 3 | CA-DSR1-DQ32B-JP | 364/500 (72.80%) | 282/400 (70.50%) | 280/383 (73.11%) | 212/297 (71.38%) |
| 4 | CA-DSR1-DQ32B-JP-SFT | 362/500 (72.40%) | 284/400 (71.00%) | 279/383 (72.85%) | 213/297 (71.72%) |
| 5 | CA-DSR1-DQ32B-JP-CPT | 356/500 (71.20%) | 278/400 (69.50%) | 277/383 (72.32%) | 213/297 (71.72%) |
| 6 | Gemma-3-27B | 320/500 (64.00%) | 252/400 (63.00%) | 252/383 (65.80%) | 196/297 (65.99%) |
| 7 | PLaMo-1.0-Prime | 211/500 (42.20%) | 175/400 (43.75%) | 156/383 (40.73%) | 126/297 (42.42%) |

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

Copy `.env.example` to `.env` and set required API keys if you plan to use cloud-based LLMs:

```bash
cp .env.example .env
# Open .env and set necessary values (e.g., OPENAI_API_KEY, GEMINI_API_KEY)
```
*(Note: Environment variables are not strictly required for basic Ollama usage unless you need to configure a non-default Ollama host/port).*

### 1.3 Setting up Ollama (Optional, for Local LLMs)

If you want to use local LLMs (including models from Hugging Face):
1.  Install Ollama from [https://ollama.com/](https://ollama.com/).
2.  Ensure the Ollama service is running.
3.  Pull the desired model using the Ollama CLI (this might happen automatically when first referenced, depending on the model identifier used). For example:
    ```bash
    # Example for a specific Hugging Face GGUF model
    ollama pull huggingface.co/mmnga/cyberagent-DeepSeek-R1-Distill-Qwen-32B-Japanese-gguf:Q4_K_M
    # Or a standard Ollama model
    ollama pull llama3
    ```

## 2. Solving Questions with LLMs

You can solve the exam questions using either cloud-based LLM APIs or local models run via Ollama.

### 2.1 Example: Using a Cloud LLM (Gemini 2.5 Pro)

```bash
# Set variables for the experiment
EXP="gemini-2.5-pro"
# Use the model key defined in llm_solver.py or a dynamic key like gemini-*
MODEL_NAME="gemini-2.5-pro-exp-03-25"

for suffix in A B C D E F; do
  uv run main.py "questions/119${suffix}_json.json" \
    --exp "119${suffix}_${EXP}" \
    --models "${MODEL_NAME}"
done
```

### 2.2 Example: Using a Local LLM via Ollama (Hugging Face Model)

This example uses a specific GGUF model from Hugging Face, served locally via Ollama.

**Prerequisite:** Ensure Ollama is installed and the service is running (see step 1.3). You might want to run the target model in a separate terminal first to ensure it's downloaded and ready, although the script might trigger the download if Ollama is configured correctly.

```bash
# Optional: Run in a separate terminal to pre-load the model
# ollama run huggingface.co/mmnga/cyberagent-DeepSeek-R1-Distill-Qwen-32B-Japanese-gguf:Q4_K_M

# Set variables for the experiment
EXP="CA-DSR1-DQ32B-JP"
# Specify the model using its Hugging Face identifier recognized by Ollama
MODEL_NAME="huggingface.co/mmnga/cyberagent-DeepSeek-R1-Distill-Qwen-32B-Japanese-gguf:Q4_K_M"
# Alternatively, use a standard Ollama model name like "ollama-llama3"

# Run the evaluation script
for suffix in A B C D E F; do
  uv run main.py "questions/119${suffix}_json.json" \
    --exp "119${suffix}_${EXP}" \
    --models "${MODEL_NAME}"
done
```

**Note on Ollama Model Names:**
*   Use the full Hugging Face identifier (like `huggingface.co/user/repo:tag`) if Ollama supports it directly.
*   Alternatively, use the `ollama-<model_name>` prefix (e.g., `ollama-llama3`, `ollama-mistral`) for standard models pulled via Ollama. The script automatically routes requests to your local Ollama instance (`http://localhost:11434/v1` by default) for these identifiers.

## 3. Grading Answers

Example script for grading answers generated in step 2:

```bash
EXP="gemini-2.5-pro"
ENTRY_NAME="Gemini-2.5-Pro" # Desired name for the Leaderboard entry

uv run grade_answers.py \
  --json_paths $(ls answers/119{A,B,C,D,E,F}_${EXP}.json) \
  --entry_name "${ENTRY_NAME}"
```

**This script automatically updates the Leaderboard section in this README file** with the consolidated results based on the processed JSON files. It also saves detailed grading results (e.g., score summaries, wrong answer lists) into the specified output directory (default: `results/`).

**You can find example output files from a sample run in the `results/demo/` directory.** This can help you understand the structure and content of the files generated by the grading script.

## 4. Comparing LLM Answers and Analysis

Beyond basic grading, you can compare the incorrect answers between two different LLM experiments to analyze their differing strengths and weaknesses.

The `scripts/compare_wrong_answers.py` script facilitates this by:
- Identifying questions answered incorrectly by both models, or only by one.
- Generating detailed comparison reports in CSV and Markdown formats (split by comparison type).
- Optionally using a specified LLM (like Gemini or GPT-4o) to provide an analytical summary of the comparison.

For detailed usage instructions, see the [Scripts README](./scripts/README.md).

## 5. Handling Skipped Questions

Occasionally, questions might be skipped during the initial run (e.g., due to API errors, timeouts, or local model issues). The following steps describe how to re-run only the skipped questions, merge the results with the original run, and then grade the complete set.

### 5.1 Example Variable Setup

```bash
# Original experiment suffix (cloud or local)
EXP="gemini-2.0-flash"
# Corresponding model name used
MODEL_NAME="gemini-2.0-flash-exp"
# Leaderboard entry name
ENTRY_NAME="Gemini-2.0-Flash"

# File listing skipped question IDs, typically generated during the initial run
SKIPPED_LIST="results/119_${EXP}_skipped.txt"
# Suffix for the retry experiment files
RERUN_EXP="${EXP}_retry"
# Suffix for the merged experiment files
MERGED_EXP="${EXP}_merged"
```

### 5.2 Workflow Commands for Skipped Questions

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

## Dataset Acquisition

The question components of the dataset (question text, choices, images) were created by the author of the original repository ([nmle-rta](https://github.com/iKora128/nmle-rta/tree/main)) by processing PDFs of the actual exam questions using OCR. Direct permission was obtained from the original author to use and publish this data.

The grading logic, including correct answers and handling of special cases like excluded questions, was developed by the author of this repository based on official information published by the Ministry of Health, Labour and Welfare (MHLW) of Japan: [第１１９回医師国家試験の合格発表について](https://www.mhlw.go.jp/general/sikaku/successlist/2025/siken01/about.html).

All scripts used to create the complete dataset are publicly available in the `scripts/prepro_utils` directory of this repository for transparency.

## License

This repository is licensed under the Apache License 2.0. For details, see the [LICENSE](LICENSE) file.

The original repository [nmle-rta](https://github.com/iKora128/nmle-rta/tree/main) is also licensed under Apache License 2.0, as authorized by the original author.
