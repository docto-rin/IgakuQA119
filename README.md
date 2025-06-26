# IgakuQA119: LLM Evaluation on the 119th Japanese Medical Licensing Examination

[![DOI](https://zenodo.org/badge/957662897.svg)](https://doi.org/10.5281/zenodo.15743221)

## Overview

IgakuQA119 is a repository designed to **evaluate the performance of Large Language Models (LLMs) using the 119th Japanese Medical Licensing Examination (JMLE)**. This project, inspired by the [nmle-rta](https://github.com/iKora128/nmle-rta/tree/main) repository, assesses LLMs' comprehension and application abilities within the context of Japan's latest medical licensing exam.

**Key Features:**

*   **Comprehensive Evaluation:** Utilizes the complete 119th JMLE dataset.
*   **Flexible LLM Support:** Supports cloud-based APIs (OpenAI, Anthropic, Gemini, OpenRouter) and local LLMs via Ollama.
*   **Streamlined Workflow:** Manages experiments, grading, and comparisons using a configuration file (`experiments.yaml`) and a unified execution script (`run_exp.sh`).
*   **Automated Leaderboard:** Automatically updates the performance leaderboard upon grading.
*   **Transparent Data:** Provides [details](#dataset-acquisition) on dataset acquisition and preprocessing scripts.

**(Note:** For reference, the previous manual execution steps (without the `run_exp.sh` script) are archived in [docs/README_20250501_legacy_manual.md](docs/README_20250501_legacy_manual.md).)

## Leaderboard

<!-- LEADERBOARD_START -->

| Rank | Entry | Overall Score | Overall Acc. | No-Img Score | No-Img Acc. |
|------|------|--------------|-------------|-------------|------------|
| 1 | Gemini-2.5-Pro | 485/500 (97.00%) | 389/400 (97.25%) | 372/383 (97.13%) | 290/297 (97.64%) |
| 2 | Gemini-2.5-Flash | 478/500 (95.60%) | 382/400 (95.50%) | 371/383 (96.87%) | 287/297 (96.63%) |
| 3 | Qwen3-235B-A22B | 462/500 (92.40%) | 366/400 (91.50%) | 356/383 (92.95%) | 274/297 (92.26%) |
| 4 | DeepSeek-R1-0528 | 461/500 (92.20%) | 367/400 (91.75%) | 364/383 (95.04%) | 282/297 (94.95%) |
| 5 | DeepSeek-R1 | 448/500 (89.60%) | 356/400 (89.00%) | 350/383 (91.38%) | 270/297 (90.91%) |
| 6 | Llama4-Maverick | 440/500 (88.00%) | 350/400 (87.50%) | 336/383 (87.73%) | 260/297 (87.54%) |
| 7 | Gemini-2.0-Flash | 436/500 (87.20%) | 352/400 (88.00%) | 333/383 (86.95%) | 263/297 (88.55%) |
| 8 | QwQ-32B | 430/500 (86.00%) | 334/400 (83.50%) | 344/383 (89.82%) | 260/297 (87.54%) |
| 9 | Qwen3-32B | 415/500 (83.00%) | 329/400 (82.25%) | 334/383 (87.21%) | 256/297 (86.20%) |
| 10 | Qwen3-30B-A3B | 412/500 (82.40%) | 328/400 (82.00%) | 323/383 (84.33%) | 251/297 (84.51%) |
| 11 | Qwen2.5-VL-72B | 403/500 (80.60%) | 325/400 (81.25%) | 309/383 (80.68%) | 245/297 (82.49%) |
| 12 | DeepSeek-V3-0324 | 399/500 (79.80%) | 311/400 (77.75%) | 312/383 (81.46%) | 236/297 (79.46%) |
| 13 | Qwen2.5-72B | 398/500 (79.60%) | 314/400 (78.50%) | 311/383 (81.20%) | 241/297 (81.14%) |
| 14 | Cogito-32B-Think | 392/500 (78.40%) | 310/400 (77.50%) | 305/383 (79.63%) | 237/297 (79.80%) |
| 15 | Llama4-Scout | 392/500 (78.40%) | 314/400 (78.50%) | 303/383 (79.11%) | 237/297 (79.80%) |
| 16 | CA-DSR1-DQ32B-JP-SFT | 374/500 (74.80%) | 294/400 (73.50%) | 290/383 (75.72%) | 222/297 (74.75%) |
| 17 | CA-DSR1-DQ32B-JP | 364/500 (72.80%) | 282/400 (70.50%) | 280/383 (73.11%) | 212/297 (71.38%) |
| 18 | CA-DSR1-DQ32B-JP-CPT | 356/500 (71.20%) | 278/400 (69.50%) | 277/383 (72.32%) | 213/297 (71.72%) |
| 19 | Cogito-32B-No-Think | 346/500 (69.20%) | 278/400 (69.50%) | 271/383 (70.76%) | 211/297 (71.04%) |
| 20 | GPT-4o-mini | 345/500 (69.00%) | 279/400 (69.75%) | 269/383 (70.23%) | 215/297 (72.39%) |
| 21 | Preferred-MedLLM-Qwen-72B | 332/500 (66.40%) | 272/400 (68.00%) | 261/383 (68.15%) | 209/297 (70.37%) |
| 22 | MedGemma-27B-Q6_K | 324/500 (64.80%) | 250/400 (62.50%) | 254/383 (66.32%) | 194/297 (65.32%) |
| 23 | Gemma-3-27B | 320/500 (64.00%) | 252/400 (63.00%) | 252/383 (65.80%) | 196/297 (65.99%) |
| 24 | PLaMo-2.0-Prime | 286/500 (57.20%) | 228/400 (57.00%) | 229/383 (59.79%) | 175/297 (58.92%) |
| 25 | PLaMo-1.0-Prime | 211/500 (42.20%) | 175/400 (43.75%) | 156/383 (40.73%) | 126/297 (42.42%) |

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

*   **`experiments`:** Define each LLM evaluation task. Set the `exp_suffix`, `model_name`, `entry_name`.
*   **`comparisons`:** Define pairs of experiments to compare.
*   **`common_settings`:** Define shared parameters like dataset paths.

For more details, see [3. Configuration Details (`experiments.yaml`)](#3-configuration-details-experimentsyaml).

## 2. Workflow using `run_exp.sh`

The `run_exp.sh` script provides a unified interface for managing the evaluation workflow based on the `experiments.yaml` configuration.

**Basic Usage:**

The script is executed using the following basic syntax:

```bash
./run_exp.sh [OPTIONS]
```

**Options:**

*   `-e <experiment_key>`: Specifies the target **experiment** defined in `experiments.yaml`. Used for tasks like running evaluations (`run`), grading (`grade`), or the full workflow (`all`).
*   `-p <comparison_key>`: Specifies the target **comparison** defined in `experiments.yaml`. Used only for the `compare` task.
*   `-t <task>`: Specifies the **task** to perform. If omitted, the default task is `all`.

**Available Tasks (`-t` option):**

*   `all` (Default): Runs the complete workflow for an experiment: `setup` -> `run` -> `rerun` (if `needs_rerun: true`) -> `grade`.
*   `setup`: Executes the optional `setup_command` defined for the experiment in `experiments.yaml` (e.g., `ollama run ...` to start a local model).
*   `run`: Runs the main evaluation script (`main.py`) for the specified experiment, generating answer files.
*   `rerun`: Handles skipped questions for an experiment (requires `needs_rerun: true` in YAML). It runs `rerun_skipped.py` and `scripts/merge_results.py`.
*   `grade`: Grades the answer files for the specified experiment using `grade_answers.py` and updates the leaderboard in this README. It automatically uses merged results if they exist.
*   `compare`: Runs a comparison between two experiments defined by the `<comparison_key>` using `compare_wrong_answers.py`. Requires the `-p` option.
*   `list-exp`: Lists all available experiment keys defined under `experiments:` in `experiments.yaml`.
*   `list-comp`: Lists all available comparison keys defined under `comparisons:` in `experiments.yaml`.

**Examples:**

```bash
# Make the script executable (only need to do this once)
chmod +x run_exp.sh

# --- Listing available configurations ---
# List all defined experiment keys
./run_exp.sh -t list-exp

# List all defined comparison keys
./run_exp.sh -t list-comp

# --- Running a full experiment workflow ---
# Run the default 'all' task (setup, run, rerun if needed, grade) for a specific experiment
./run_exp.sh -e gemini-2_5-pro
./run_exp.sh -e ca-dsr1-dq32b-jp # Example with a local model experiment

# --- Running specific tasks for an experiment ---
# Only run the setup command for a local model experiment
./run_exp.sh -t setup -e ca-dsr1-dq32b-jp

# Only run the evaluation (generate answers)
./run_exp.sh -t run -e gemini-2_5-pro

# Only handle skipped questions (rerun and merge)
./run_exp.sh -t rerun -e gemini-2_0-flash # Assumes needs_rerun: true is set

# Only grade the results (and update leaderboard)
./run_exp.sh -t grade -e gemini-2_5-pro

# --- Running a comparison ---
# Run a predefined comparison between two experiments
./run_exp.sh -p base_vs_sft # Uses the 'compare' task implicitly with -p
# Equivalent explicit command:
# ./run_exp.sh -t compare -p base_vs_sft
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
./run_exp.sh -t run -e gemini-2_5-pro

# Example: Run evaluation using a local Ollama model
./run_exp.sh -t run -e qwen3-32b
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

#### Step 2.4: (Optional) Handling Skipped Questions

If questions were skipped during the initial run (e.g., due to errors), the `run_exp.sh` script can help re-run them and merge the results. This functionality is primarily managed through the `rerun` task or as part of the `all` task when `needs_rerun: true` is configured for an experiment.

##### Configuration for Rerunning Skipped Questions

1.  Ensure the `grade_answers.py` script generated a `results/119_<exp_suffix>_skipped.txt` file during a previous grading attempt (or `main.py` created one).
2.  In `experiments.yaml`, find the entry for the experiment that had skipped questions.
3.  **Uncomment or add the line `needs_rerun: true`** within that experiment's definition.

```yaml
experiments:
  # ...
  gemini-2_0-flash: # Example experiment that had skips
    exp_suffix: "gemini-2_0-flash"
    model_name: "gemini-2.0-flash-exp"
    entry_name: "Gemini-2.0-Flash"
    needs_rerun: true # Enable the rerun workflow for this experiment
  # ...
```

##### Execution via `run_exp.sh`

You have two main options using `run_exp.sh`:

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
If `needs_rerun: true` is set for the `gemini-2_0-flash` experiment, the `all` task will automatically perform the `rerun` steps *after* the main `run` step (if any new answers were generated or if skipped files exist) and *before* the final `grade` step.

This streamlined process ensures that all available answers (original + retried) are considered for the final grading and leaderboard update.

#### Step 2.5: (Optional) Compare Results

Compare the performance of two different experiments.

```bash
# Example: Compare the base CA-DSR1 model vs its SFT variant
./run_exp.sh -p base_vs_sft
```
This uses the `scripts/compare_wrong_answers.py` script with the models and analyzer defined for the `base_vs_sft` key in the `comparisons` section of `experiments.yaml`. Comparison reports are saved in the `results/` directory. See `scripts/README.md` for more details on the comparison script.

**Demo results** can be found in `results/demo/` to understand the output format.

## 3. Configuration Details (`experiments.yaml`)

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
    *   `supports_vision` (Optional): Set to `true` or `false` to explicitly control whether the model should attempt to use vision capabilities for this experiment. If omitted, the default setting defined in `llm_solver.py` for the model type (e.g., `gemini-flexible`, `ollama-flexible`) is used.
*   **`comparisons:`**: A dictionary defining comparison pairs.
    *   `model1_key`, `model2_key`: Experiment keys (from the `experiments` section) to compare.
    *   `analyzer`: The LLM model name used by `compare_wrong_answers.py` for analysis.
*   **`common_settings:`**: Shared parameters.
    *   `question_suffixes`: List of suffixes for question files (e.g., `["A", "B", "C", "D", "E", "F"]`).
    *   `questions_dir`, `answers_dir`, `results_dir`: Directory paths.
    *   `question_prefix`: Prefix for question filenames (e.g., `"119"`).

## Dataset Acquisition

The question components (text, choices, images) were processed from official exam PDFs using OCR by the author of the original [nmle-rta](https://github.com/iKora128/nmle-rta/tree/main) repository. Permission for use and publication was obtained.

The grading logic (correct answers, excluded questions handling) was developed based on official MHLW information: [第１１９回医師国家試験の合格発表について](https://www.mhlw.go.jp/general/sikaku/successlist/2025/siken01/about.html).

Preprocessing scripts are available in `scripts/prepro_utils` for transparency.

## License

This repository is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file.

The original repository [nmle-rta](https://github.com/iKora128/nmle-rta/tree/main) is also licensed under Apache License 2.0.

## Citation

If you use IgakuQA119 in your research, we recommend using the "Cite this repository" button that has appeared on the right-hand sidebar of the repository page.

Alternatively, you can use the following DOI:

[![DOI](https://zenodo.org/badge/957662897.svg)](https://doi.org/10.5281/zenodo.15743221)

The above DOI corresponds to the latest versioned release as [published to Zenodo](https://zenodo.org/records/15743222), where you will find all earlier releases. To cite IgakuQA119 independent of version, use https://doi.org/10.5281/zenodo.15743221, which will always redirect to the latest release.