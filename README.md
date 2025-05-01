# IgakuQA119: LLM Evaluation on the 119th Japanese Medical Licensing Examination

## Overview

IgakuQA119 is a repository designed to **evaluate the performance of Large Language Models (LLMs) using the 119th Japanese Medical Licensing Examination (JMLE)**. This project, inspired by the [nmle-rta](https://github.com/iKora128/nmle-rta/tree/main) repository, assesses LLMs' comprehension and application abilities within the context of Japan's latest medical licensing exam.

**Key Features:**

*   **Comprehensive Evaluation:** Utilizes the complete 119th JMLE dataset.
*   **Flexible LLM Support:** Supports cloud-based APIs (OpenAI, Anthropic, Gemini, OpenRouter) and local LLMs via Ollama.
*   **Streamlined Workflow:** Manages experiments, grading, and comparisons using a configuration file (`experiments.yaml`) and a unified execution script (`run_exp.sh`).
*   **Automated Leaderboard:** Automatically updates the performance leaderboard upon grading.
*   **Transparent Data:** Provides details on dataset acquisition and preprocessing scripts.

**(Note:** For the previous manual execution instructions, please refer to `docs/README_legacy_manual.md`.)

## Leaderboard

<!-- LEADERBOARD_START -->

| Rank | Entry | Overall Score (Rate) | Overall Acc. | No-Img Score (Rate) | No-Img Acc. |
|------|------|---------------------|-------------|--------------------|------------|
| 1 | Gemini-2.5-Pro | 485/500 (97.00%) | 389/400 (97.25%) | 372/383 (97.13%) | 290/297 (97.64%) |
| 2 | Gemini-2.5-Flash | 478/500 (95.60%) | 382/400 (95.50%) | 371/383 (96.87%) | 287/297 (96.63%) |
| 3 | Qwen3-235B-A22B | 462/500 (92.40%) | 366/400 (91.50%) | 356/383 (92.95%) | 274/297 (92.26%) |
| 4 | Gemini-2.0-Flash | 436/500 (87.20%) | 352/400 (88.00%) | 333/383 (86.95%) | 263/297 (88.55%) |
| 5 | Qwen3-32B | 415/500 (83.00%) | 329/400 (82.25%) | 334/383 (87.21%) | 256/297 (86.20%) |
| 6 | Qwen3-30B-A3B | 412/500 (82.40%) | 328/400 (82.00%) | 323/383 (84.33%) | 251/297 (84.51%) |
| 7 | Cogito-32B-Think | 392/500 (78.40%) | 310/400 (77.50%) | 305/383 (79.63%) | 237/297 (79.80%) |
| 8 | CA-DSR1-DQ32B-JP-SFT | 374/500 (74.80%) | 294/400 (73.50%) | 290/383 (75.72%) | 222/297 (74.75%) |
| 9 | CA-DSR1-DQ32B-JP | 364/500 (72.80%) | 282/400 (70.50%) | 280/383 (73.11%) | 212/297 (71.38%) |
| 10 | CA-DSR1-DQ32B-JP-CPT | 356/500 (71.20%) | 278/400 (69.50%) | 277/383 (72.32%) | 213/297 (71.72%) |
| 11 | Cogito-32B-No-Think | 346/500 (69.20%) | 278/400 (69.50%) | 271/383 (70.76%) | 211/297 (71.04%) |
| 12 | Gemma-3-27B | 320/500 (64.00%) | 252/400 (63.00%) | 252/383 (65.80%) | 196/297 (65.99%) |
| 13 | PLaMo-1.0-Prime | 211/500 (42.20%) | 175/400 (43.75%) | 156/383 (40.73%) | 126/297 (42.42%) |

<!-- LEADERBOARD_END -->

## 1. Setup Instructions

**Requirements**:
*   Python 3.10 or higher.
*   `uv` package manager.
*   `yq` (Go version) for processing YAML configuration.

### 1.1 Install `uv` Dependencies

Use `uv` to install Python packages:

```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh
# Install packages defined in pyproject.toml
uv sync
```

### 1.2 Install `yq`

The `run_exp.sh` script requires the **Go version** of `yq` (from Mike Farah). Install it using your preferred method:

*   **Homebrew (macOS or Linux):**
    ```bash
    brew install yq
    ```
*   **Snap (Linux):**
    ```bash
    sudo snap install yq
    ```
*   **Other Methods:** See the official `yq` installation guide: [https://github.com/mikefarah/yq#install](https://github.com/mikefarah/yq#install)

Verify installation: `yq --version` should show output like `yq (https://github.com/mikefarah/yq/) version v4.x.x`.

### 1.3 Set Environment Variables (Optional, for Cloud LLMs)

If you plan to use cloud-based LLMs (OpenAI, Gemini, Anthropic, OpenRouter), copy the example environment file and add your API keys:

```bash
cp .env.example .env
# Open .env and set necessary values like:
# OPENAI_API_KEY="sk-..."
# GEMINI_API_KEY="..."
# OPENROUTER_API_KEY="..."
```
The scripts will automatically load these variables.

### 1.4 Set up Ollama (Optional, for Local LLMs)

To use local LLMs:
1.  Install Ollama from [https://ollama.com/](https://ollama.com/).
2.  Ensure the Ollama service is running (`ollama serve` or via the desktop application).
3.  You can pull models beforehand using `ollama pull <model_name>`, or let the system attempt to pull them when first used via the script (depending on the model identifier).

### 1.5 Configure Experiments (`experiments.yaml`)

All experiments, comparisons, and common settings are defined in the `experiments.yaml` file. Before running evaluations, review and potentially modify this file:

*   **`experiments`:** Define each LLM evaluation task. Set the `exp_suffix`, `model_name`, `entry_name` for the leaderboard, and optionally a `setup_command` (e.g., for Ollama) or `needs_rerun: true` if skipped question handling is anticipated.
*   **`comparisons`:** Define pairs of experiments to compare using the `compare_wrong_answers.py` script.
*   **`common_settings`:** Define shared parameters like question file suffixes and directory names.

See the comments within `experiments.yaml` for details on each field.

## 2. Workflow using `run_exp.sh`

The `run_exp.sh` script provides a unified interface for managing the evaluation workflow based on the `experiments.yaml` configuration.

**Basic Usage:**
*   `./run_exp.sh -e <experiment_key>`
*   `./run_exp.sh -t <task> -e <experiment_key>`
*   `./run_exp.sh -p <comparison_key>`

**Available Tasks (`-t` option):**

*   `all` (Default): Runs `setup`, `run`, `rerun` (if needed), and `grade` for the specified experiment.
*   `setup`: Executes the `setup_command` defined in `experiments.yaml` (e.g., start an Ollama model).
*   `run`: Runs the main evaluation loop (`main.py`) for the experiment.
*   `rerun`: Runs the skipped question handling process (`rerun_skipped.py` and `merge_results.py`) if `needs_rerun: true` is set for the experiment.
*   `grade`: Grades the results (`grade_answers.py`) and updates the leaderboard.
*   `compare`: Runs a comparison between two experiments defined in the `comparisons` section.
*   `list-exp`: Lists all available experiment keys defined in `experiments.yaml`.
*   `list-comp`: Lists all available comparison keys defined in `experiments.yaml`.

**Target Specification:**

*   `-e <experiment_key>`: Specifies the experiment key (from `experiments.yaml`) for tasks like `all`, `setup`, `run`, `rerun`, `grade`.
*   `-p <comparison_key>`: Specifies the comparison key (from `experiments.yaml`) for the `compare` task.

**Examples:**

```bash
# Make the script executable once
chmod +x run_exp.sh

# List available experiments
./run_exp.sh -t list-exp

# List available comparisons
./run_exp.sh -t list-comp

# Run the full workflow: [Setup -> Run -> Rerun/Merge (if needed) -> Grade] for an experiment
./run_exp.sh -e <experiment_key>
# Example:
./run_exp.sh -e gemini-2_5-pro
./run_exp.sh -e ca-dsr1-dq32b-jp

# Run only a specific task
./run_exp.sh -t <task_name> -e <experiment_key>
# Example Tasks: setup, run, rerun, grade

# Run a comparison
./run_exp.sh -p <comparison_key>
# Example:
./run_exp.sh -p base_vs_sft
```

### Workflow Steps:

#### Step 2.1: (Optional) Prepare Local Models

If using Ollama, you might want to ensure the model is running before starting the main evaluation.

```bash
# Example: Start the CA-DSR1 model using its setup command
./run_exp.sh -t setup -e ca-dsr1-dq32b-jp
```
This executes the `setup_command` defined for `ca-dsr1-dq32b-jp` in `experiments.yaml`. Alternatively, run `ollama run <model_name>` in a separate terminal.

#### Step 2.2: Run Experiments

Execute the evaluation for a specific experiment defined in `experiments.yaml`.

```bash
# Example: Run evaluation using Gemini 2.5 Pro
./run_exp.sh -e gemini-2_5-pro

# Example: Run evaluation using a local Ollama model
./run_exp.sh -e qwen3-32b
```
This will iterate through the question files (A-F) and run `main.py` with the `model_name` and `exp_suffix` specified in the YAML for the given experiment key. Answer files will be saved in the `answers/` directory.

#### Step 2.3: Grade Results

After an experiment completes, grade the generated answers.

```bash
# Example: Grade the results for Gemini 2.5 Pro
./run_exp.sh -t grade -e gemini-2_5-pro
```
This runs `grade_answers.py`, using the `exp_suffix` (or the merged suffix if re-run occurred) and `entry_name` from the YAML. **It automatically updates the Leaderboard in this README** and saves detailed results in the `results/` directory.

**Demo results** can be found in `results/demo/` to understand the output format.

#### Step 2.4: (Optional) Compare Results

Compare the performance of two different experiments.

```bash
# Example: Compare the base CA-DSR1 model vs its SFT variant
./run_exp.sh -p base_vs_sft
```
This uses the `compare_wrong_answers.py` script with the models and analyzer defined for the `base_vs_sft` key in the `comparisons` section of `experiments.yaml`. Comparison reports are saved in the `results/` directory. See `scripts/README.md` for more details on the comparison script.

## 3. Handling Skipped Questions

If questions were skipped during the initial run (e.g., due to errors), you can re-run them and merge the results.

### 3.1 Configuration

1.  Ensure the `grade_answers.py` script generated a `results/119_<exp_suffix>_skipped.txt` file during the initial grading attempt (or the `main.py` run created one).
2.  In `experiments.yaml`, find the entry for the experiment that had skipped questions.
3.  **Uncomment or add the line `needs_rerun: true`** within that experiment's definition.

```yaml
experiments:
  # ...
  gemini-2_0-flash: # Example experiment that had skips
    exp_suffix: "gemini-2_0-flash-2nd"
    model_name: "gemini-2.0-flash-exp"
    entry_name: "Gemini-2.0-Flash"
    needs_rerun: true # Enable the rerun workflow for this experiment
  # ...
```

### 3.2 Execution

You have two main options:

**Option A: Run the dedicated `rerun` task, then `grade`:**

```bash
# 1. Rerun skipped questions and merge results
./run_exp.sh -t rerun -e gemini-2_0-flash

# 2. Grade the merged results
./run_exp.sh -t grade -e gemini-2_0-flash
```
The `rerun` task executes `rerun_skipped.py` (creating `*_retry.json` files) and then `scripts/merge_results.py` (creating `*_merged.json` files). The subsequent `grade` task will automatically detect and use the `*_merged.json` files for grading.

**Option B: Run the `all` task:**

```bash
# Rerun everything, including skipped handling if needed
./run_exp.sh -e gemini-2_0-flash
```
If `needs_rerun: true` is set, the `all` task will automatically perform the `rerun` steps *after* the main `run` step (if any new answers were generated) and *before* the final `grade` step.

This streamlined process ensures that all available answers (original + retried) are considered for the final grading and leaderboard update.

## 4. Configuration Details (`experiments.yaml`)

The `experiments.yaml` file is central to managing evaluations.

*   **`experiments:`**: A dictionary where each key is a unique identifier for an experiment (e.g., `gemini-2_5-pro`).
    *   `exp_suffix`: String used in filenames for answers and results related to this experiment.
    *   `model_name`: The identifier passed to the solver. Naming conventions:
        *   **Cloud APIs (OpenAI, Anthropic, Gemini)**: Use names defined in `llm_solver.py` or dynamic prefixes like `gemini-*`, `gpt-*`, `claude-*`.
        *   **OpenRouter**: Use `openrouter-<provider>/<model_id>:<version>` (e.g., `openrouter-qwen/qwen3-235b-a22b:free`). Requires `OPENROUTER_API_KEY` in `.env`.
        *   **Ollama**: Use `ollama-<model_name>` (e.g., `ollama-llama3`) for standard Ollama models, or the full Hugging Face GGUF identifier if supported (e.g., `hf.co/user/repo:tag`). The script routes these to `http://localhost:11434/v1`.
    *   `entry_name`: The name displayed in the leaderboard for this experiment.
    *   `setup_command` (Optional): A shell command executed by the `setup` task (e.g., `ollama run ...`).
    *   `needs_rerun` (Optional): Set to `true` to enable the skipped question handling workflow for this experiment.
*   **`comparisons:`**: A dictionary defining comparison pairs.
    *   `model1_key`, `model2_key`: Experiment keys (from the `experiments` section) to compare.
    *   `analyzer`: The LLM model name used by `compare_wrong_answers.py` for analysis.
*   **`common_settings:`**: Shared parameters.
    *   `question_suffixes`: List of suffixes for question files (e.g., `["A", "B", "C", "D", "E", "F"]`).
    *   `questions_dir`, `answers_dir`, `results_dir`: Directory paths.
    *   `question_prefix`: Prefix for question filenames (e.g., `"119"`).

## Legacy Manual Workflow

For reference, the previous manual execution steps (without the `run_exp.sh` script) are archived in [docs/README_20250501_legacy_manual.md](docs/README_20250501_legacy_manual.md).

## Dataset Acquisition

The question components (text, choices, images) were processed from official exam PDFs using OCR by the author of the original [nmle-rta](https://github.com/iKora128/nmle-rta/tree/main) repository. Permission for use and publication was obtained.

The grading logic (correct answers, excluded questions handling) was developed based on official MHLW information: [第１１９回医師国家試験の合格発表について](https://www.mhlw.go.jp/general/sikaku/successlist/2025/siken01/about.html).

Preprocessing scripts are available in `scripts/prepro_utils` for transparency.

## License

This repository is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file.

The original repository [nmle-rta](https://github.com/iKora128/nmle-rta/tree/main) is also licensed under Apache License 2.0.
