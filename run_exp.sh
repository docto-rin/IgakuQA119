#!/bin/bash

# --- Configuration ---
CONFIG_FILE="experiments.yaml"
DEFAULT_ANALYZER="gemini-2.5-pro-exp-03-25" # Default LLM for analysis if not specified
OLLAMA_PID="" # Variable to store Ollama process ID

# --- Helper Functions ---

# Check if yq is installed
check_yq() {
  if ! command -v yq &> /dev/null; then
    echo "Error: yq could not be found. Please install yq." >&2
    echo "See: https://github.com/mikefarah/yq#install" >&2
    exit 1
  fi
}

# Get a value from the YAML config, handling nulls gracefully
get_config_value() {
  local query=$1
  local default_val=${2:-""} # Optional default value
  local value
  value=$(yq e "$query" "$CONFIG_FILE")
  if [[ "$value" == "null" || -z "$value" ]]; then
    echo "$default_val"
  else
    echo "$value"
  fi
}

# Get a list (array) from the YAML config
get_config_list() {
    local query=$1
    local temp_file
    temp_file=$(mktemp)
    # Use yq to output YAML sequence format (compatible with bash read -a)
    yq e "$query | .[]" "$CONFIG_FILE" > "$temp_file"
    local -n _arr_ref=$2 # Use nameref for array assignment
    mapfile -t _arr_ref < "$temp_file"
    rm "$temp_file"
}


# --- Cleanup Function ---
cleanup() {
    echo "Cleaning up..."
    if [[ -n "$OLLAMA_PID" ]]; then
        echo "Attempting to stop Ollama process (PID: $OLLAMA_PID)..."
        # Check if the process exists before trying to kill it
        if kill -0 "$OLLAMA_PID" 2>/dev/null; then
            kill "$OLLAMA_PID"
            echo "Ollama process stopped."
        else
            echo "Ollama process (PID: $OLLAMA_PID) not found or already stopped."
        fi
        OLLAMA_PID="" # Clear the PID
    fi
}
# Trap signals to ensure cleanup runs on exit or interruption
trap cleanup EXIT SIGINT SIGTERM


# --- Task Functions ---

run_setup() {
  local exp_key=$1
  echo "--- Running Setup for: $exp_key ---"
  local setup_cmd
  setup_cmd=$(get_config_value ".experiments.${exp_key}.setup_command")
  if [[ -n "$setup_cmd" ]]; then
    # Check if the command seems to be related to ollama run/serve
    if [[ "$setup_cmd" == *"ollama run"* || "$setup_cmd" == *"ollama serve"* ]]; then
      echo "Executing setup command in background: $setup_cmd"
      eval "$setup_cmd" & # Run command in background
      OLLAMA_PID=$!     # Store the PID of the background process
      # Add a short delay to allow the background process (e.g., Ollama server) to initialize
      echo "Ollama process started in background (PID: $OLLAMA_PID)."
      echo "Waiting a few seconds for setup command to initialize..."
      sleep 5 # Adjust sleep duration if necessary
    else
      # Execute other setup commands normally (not in background)
      echo "Executing setup command: $setup_cmd"
      eval "$setup_cmd"
    fi
  else
    echo "No setup command defined for $exp_key."
  fi
  echo "--- Setup for $exp_key finished ---"
  echo
}

run_experiment() {
  local exp_key=$1
  echo "--- Running Experiment: $exp_key ---"

  local exp_suffix model_name questions_dir answers_dir question_prefix supports_vision
  exp_suffix=$(get_config_value ".experiments.${exp_key}.exp_suffix")
  model_name=$(get_config_value ".experiments.${exp_key}.model_name")
  # Read supports_vision, default to empty string if null or not present
  supports_vision=$(get_config_value ".experiments.${exp_key}.supports_vision")
  questions_dir=$(get_config_value ".common_settings.questions_dir" "questions")
  answers_dir=$(get_config_value ".common_settings.answers_dir" "answers")
  question_prefix=$(get_config_value ".common_settings.question_prefix" "119")

  local -a suffixes
  get_config_list ".common_settings.question_suffixes" suffixes

  if [[ -z "$exp_suffix" || -z "$model_name" ]]; then
    echo "Error: exp_suffix or model_name not defined for $exp_key" >&2
    return 1
  fi

  mkdir -p "$answers_dir" # Ensure answers directory exists

  # Base command template
  local cmd_template='uv run main.py "{q_path}" --exp "{exp_name}" --model_name "{model}"'
  # Add supports_vision argument only if it's defined in yaml
  local supports_vision_arg=""
  if [[ -n "$supports_vision" ]]; then
      supports_vision_arg=" --supports_vision \"${supports_vision}\""
  fi

  echo "Model: $model_name"
  echo "Experiment Suffix: $exp_suffix"
  # Display supports_vision status if set
  if [[ -n "$supports_vision" ]]; then
      echo "Supports Vision (Override): $supports_vision"
  fi
  echo "Question Files:"

  for suffix in "${suffixes[@]}"; do
    local q_file="${question_prefix}${suffix}_json.json"
    local q_path="${questions_dir}/${q_file}"
    local exp_name="${question_prefix}${suffix}_${exp_suffix}"
    local cmd
    # Build the command, substituting placeholders and adding the optional argument
    cmd=$(echo "$cmd_template" | sed "s|{q_path}|$q_path|" | sed "s|{exp_name}|$exp_name|" | sed "s|{model}|$model_name|")
    cmd+="$supports_vision_arg" # Append the vision argument if it exists

    if [[ ! -f "$q_path" ]]; then
        echo "Warning: Question file not found: $q_path. Skipping." >&2
        continue
    fi

    echo "  - Running for $q_file"
    echo "    Command: $cmd"
    eval "$cmd"
    local exit_code=$?
    if [[ $exit_code -ne 0 ]]; then
        echo "Error: Command failed for $q_file with exit code $exit_code" >&2
        # Decide whether to continue or exit (e.g., exit 1)
    fi
  done

  echo "--- Experiment $exp_key finished ---"
  echo
}

run_rerun_and_merge() {
    local exp_key=$1
    echo "--- Running Rerun/Merge for: $exp_key ---"

    local needs_rerun
    needs_rerun=$(get_config_value ".experiments.${exp_key}.needs_rerun" "false")

    if [[ "$needs_rerun" != "true" ]]; then
        echo "Skipping rerun/merge: 'needs_rerun' is not true for $exp_key."
        return 0
    fi

    local exp_suffix model_name results_dir questions_dir answers_dir question_prefix
    exp_suffix=$(get_config_value ".experiments.${exp_key}.exp_suffix")
    model_name=$(get_config_value ".experiments.${exp_key}.model_name")
    results_dir=$(get_config_value ".common_settings.results_dir" "results")
    questions_dir=$(get_config_value ".common_settings.questions_dir" "questions")
    answers_dir=$(get_config_value ".common_settings.answers_dir" "answers")
    question_prefix=$(get_config_value ".common_settings.question_prefix" "119")

    if [[ -z "$exp_suffix" || -z "$model_name" ]]; then
        echo "Error: exp_suffix or model_name not defined for $exp_key" >&2
        return 1
    fi

    local skipped_list="${results_dir}/${question_prefix}_${exp_suffix}_skipped.txt"
    local rerun_exp="${exp_suffix}_retry"
    local merged_exp="${exp_suffix}_merged"

    # 1. Rerun Skipped Questions
    if [[ -f "$skipped_list" ]]; then
        echo "Found skipped list: $skipped_list. Running rerun..."
        local rerun_cmd="uv run rerun_skipped.py \
          --skipped_list \"${skipped_list}\" \
          --model_name \"${model_name}\" \
          --questions_dir \"${questions_dir}\" \
          --rerun_exp \"${rerun_exp}\""
        echo "Executing: $rerun_cmd"
        eval "$rerun_cmd"
    else
        echo "Skipped list not found: $skipped_list. Skipping rerun step."
    fi

    # 2. Merge Results
    # Check if retry files were actually created before merging
    if ls "${answers_dir}/${question_prefix}"*_"${rerun_exp}".json 1> /dev/null 2>&1; then
        echo "Retry answer files found. Merging results..."
        local merge_cmd="uv run scripts/merge_results.py \
          --original_pattern \"${answers_dir}/${question_prefix}*_${exp_suffix}.json\" \
          --retry_pattern \"${answers_dir}/${question_prefix}*_${rerun_exp}.json\" \
          --merged_exp \"${merged_exp}\""
         echo "Executing: $merge_cmd"
         eval "$merge_cmd"
    else
         echo "No retry answer files found matching pattern. Skipping merge step."
    fi

    echo "--- Rerun/Merge for $exp_key finished ---"
    echo
}


run_grade() {
  local exp_key=$1
  echo "--- Grading Experiment: $exp_key ---"

  local exp_suffix entry_name answers_dir results_dir question_prefix needs_rerun merged_exp
  exp_suffix=$(get_config_value ".experiments.${exp_key}.exp_suffix")
  entry_name=$(get_config_value ".experiments.${exp_key}.entry_name")
  answers_dir=$(get_config_value ".common_settings.answers_dir" "answers")
  results_dir=$(get_config_value ".common_settings.results_dir" "results")
  question_prefix=$(get_config_value ".common_settings.question_prefix" "119")
  needs_rerun=$(get_config_value ".experiments.${exp_key}.needs_rerun" "false")
  merged_exp="${exp_suffix}_merged"

  local -a suffixes
  get_config_list '.common_settings.question_suffixes' suffixes

  if [[ -z "$exp_suffix" || -z "$entry_name" ]]; then
    echo "Error: exp_suffix or entry_name not defined for $exp_key" >&2
    return 1
  fi

  mkdir -p "$results_dir" # Ensure results directory exists

  local json_paths_arg=""
  local grading_suffix="$exp_suffix" # Default to original suffix

  # If rerun was possible, check if merged files exist and use them
  if [[ "$needs_rerun" == "true" ]]; then
      local first_merged_file="${answers_dir}/${question_prefix}${suffixes[0]}_${merged_exp}.json"
      if [[ -f "$first_merged_file" ]]; then
          echo "Merged results found, grading merged files with suffix: $merged_exp"
          grading_suffix="$merged_exp"
      else
          echo "Merged results not found (checked $first_merged_file). Grading original files with suffix: $exp_suffix"
      fi
  fi

  # Build the list of JSON paths for grading
  local paths_list=()
  for suffix in "${suffixes[@]}"; do
      paths_list+=("${answers_dir}/${question_prefix}${suffix}_${grading_suffix}.json")
  done
  # Join the array elements into a space-separated string
  json_paths_arg=$(printf "%s " "${paths_list[@]}")
  # Remove trailing space
  json_paths_arg=${json_paths_arg% }


  if [[ -z "$json_paths_arg" ]]; then
      echo "Error: Could not determine JSON paths for grading." >&2
      return 1
  fi

  local grade_cmd="uv run grade_answers.py --json_paths ${json_paths_arg} --entry_name \"${entry_name}\" --output \"${results_dir}\""

  echo "Grading Entry: $entry_name"
  echo "Using suffix: $grading_suffix"
  echo "Executing: $grade_cmd"
  eval "$grade_cmd"

  echo "--- Grading for $exp_key finished ---"
  echo
}

run_compare() {
    local comp_key=$1
    echo "--- Running Comparison: $comp_key ---"

    local model1_key model2_key analyzer
    model1_key=$(get_config_value ".comparisons.${comp_key}.model1_key")
    model2_key=$(get_config_value ".comparisons.${comp_key}.model2_key")
    analyzer=$(get_config_value ".comparisons.${comp_key}.analyzer" "$DEFAULT_ANALYZER")

    if [[ -z "$model1_key" || -z "$model2_key" ]]; then
        echo "Error: model1_key or model2_key not defined for comparison $comp_key" >&2
        return 1
    fi

    local model1_exp_suffix model1_entry_name model2_exp_suffix model2_entry_name
    model1_exp_suffix=$(get_config_value ".experiments.${model1_key}.exp_suffix")
    model1_entry_name=$(get_config_value ".experiments.${model1_key}.entry_name")
    model2_exp_suffix=$(get_config_value ".experiments.${model2_key}.exp_suffix")
    model2_entry_name=$(get_config_value ".experiments.${model2_key}.entry_name")

    if [[ -z "$model1_exp_suffix" || -z "$model1_entry_name" || -z "$model2_exp_suffix" || -z "$model2_entry_name" ]]; then
        echo "Error: Could not retrieve details for models in comparison $comp_key" >&2
        return 1
    fi

    local compare_cmd="uv run scripts/compare_wrong_answers.py \\
        \"${model1_exp_suffix}\" \\
        \"${model2_exp_suffix}\" \\
        --model1_name \"${model1_entry_name}\" \\
        --model2_name \"${model2_entry_name}\" \\
        --analyze_with_llm \"${analyzer}\""

    echo "Comparing: ${model1_entry_name} vs ${model2_entry_name}"
    echo "Analyzer: ${analyzer}"
    echo "Executing: $compare_cmd"
    eval "$compare_cmd"

    echo "--- Comparison $comp_key finished ---"
    echo
}


# --- Main Execution Logic ---

check_yq

# Default task
TASK="all"
TARGET="" # Experiment key or comparison key

# Parse command line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -t|--task) TASK="$2"; shift ;;
        -e|--experiment) TARGET="$2"; shift ;;
        -p|--comparison) TARGET="$2"; TASK="compare"; shift ;; # Set task to compare if -p is used
        -c|--config) CONFIG_FILE="$2"; shift ;;
        -h|--help)
            echo "Usage: $0 [-t <task>] [-e <experiment_key> | -p <comparison_key>] [-c <config_file>]"
            echo ""
            echo "Tasks:"
            echo "  all       (Default) Run setup, experiment, rerun/merge, and grade for the specified experiment."
            echo "  setup     Run only the setup command."
            echo "  run       Run only the main experiment loop."
            echo "  rerun     Run only the rerun/merge steps (if applicable)."
            echo "  grade     Run only the grading step."
            echo "  compare   Run the specified comparison (requires -p)."
            echo "  list-exp  List available experiment keys."
            echo "  list-comp List available comparison keys."
            echo ""
            echo "Arguments:"
            echo "  -e <experiment_key>  Specify the experiment to run tasks for."
            echo "  -p <comparison_key>  Specify the comparison to run (implies -t compare)."
            echo "  -c <config_file>     Specify the YAML configuration file (default: experiments.yaml)."
            echo "  -h, --help           Show this help message."
            exit 0
            ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# List tasks
if [[ "$TASK" == "list-exp" ]]; then
    echo "Available experiment keys:"
    yq e '.experiments | keys | .[]' "$CONFIG_FILE"
    exit 0
fi
if [[ "$TASK" == "list-comp" ]]; then
    echo "Available comparison keys:"
    yq e '.comparisons | keys | .[]' "$CONFIG_FILE"
    exit 0
fi


# Validate target based on task
if [[ "$TASK" == "compare" ]]; then
    if [[ -z "$TARGET" ]]; then
        echo "Error: Comparison key must be specified using -p for task 'compare'." >&2
        exit 1
    fi
    # Check if comparison key exists
     if ! yq e ".comparisons | has(\"${TARGET}\")" "$CONFIG_FILE" > /dev/null; then
         echo "Error: Comparison key '$TARGET' not found in $CONFIG_FILE" >&2
         exit 1
     fi

elif [[ "$TASK" != "list-exp" && "$TASK" != "list-comp" ]]; then # Tasks requiring an experiment key
     if [[ -z "$TARGET" ]]; then
        echo "Error: Experiment key must be specified using -e for task '$TASK'." >&2
        exit 1
    fi
     # Check if experiment key exists
     if ! yq e ".experiments | has(\"${TARGET}\")" "$CONFIG_FILE" > /dev/null; then
         echo "Error: Experiment key '$TARGET' not found in $CONFIG_FILE" >&2
         exit 1
     fi
fi


# Execute tasks
case "$TASK" in
    all)
        run_setup "$TARGET" && \
        run_experiment "$TARGET"

        # --- Stop Ollama Process After Experiment ---
        # This block is executed only in the 'all' task after run_experiment
        if [[ -n "$OLLAMA_PID" ]]; then
            echo "Experiment finished. Attempting to stop Ollama process (PID: $OLLAMA_PID)..."
            if kill -0 "$OLLAMA_PID" 2>/dev/null; then
                kill "$OLLAMA_PID"
                echo "Ollama process stopped."
            else
                echo "Ollama process (PID: $OLLAMA_PID) not found or already stopped."
            fi
            OLLAMA_PID="" # Clear the PID after attempting to stop
        fi
        # --- End Stop Ollama Process ---

        # Continue with the rest of the 'all' task workflow
        run_rerun_and_merge "$TARGET" && \
        run_grade "$TARGET"
        ;;
    setup)
        run_setup "$TARGET"
        # Note: Ollama process is intentionally left running for 'setup' task
        ;;
    run)
        # Assuming setup was run previously or Ollama is running independently
        run_experiment "$TARGET"
        # Note: Ollama process is not stopped here, as setup might be separate
        ;;
    rerun)
        # Assuming setup was run previously or Ollama is running independently
        run_rerun_and_merge "$TARGET"
        ;;
    grade)
        run_grade "$TARGET"
        ;;
    compare)
        run_compare "$TARGET"
        ;;
    *)
        echo "Error: Invalid task '$TASK'. Use -h for help." >&2
        exit 1
        ;;
esac

exit $?