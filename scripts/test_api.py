import os
import sys
import argparse
import traceback
from dotenv import load_dotenv
from src.llm_solver import LLMSolver

load_dotenv()

def test_model_api(model_key: str):
    """指定されたモデルキーの設定読み込みとAPI呼び出しをテストする"""
    print(f"\n--- Testing model: {model_key} ---")
    try:
        solver = LLMSolver()

        print("1. Getting config and client...")
        config, client = solver.get_config_and_client(model_key)
        print("   Config loaded:")
        config_display = {k: v for k, v in config.items() if k != 'api_key'}
        print(f"   {config_display}")
        print(f"   Client type: {type(client)}")
        print("   Config and client obtained successfully.")

        print("\n2. Performing a simple chat API call...")
        test_prompt = "こんにちは！自己紹介してください。"

        if config["client_type"] == "anthropic":
            response = client.messages.create(
                model=config["model_name"],
                messages=[{"role": "user", "content": test_prompt}],
                system=config["system_prompt"]
            )
            api_response = response.content[0].text
        else:
            messages = [
                {"role": config["system_role"], "content": config["system_prompt"]},
                {"role": "user", "content": test_prompt}
            ]
            response = client.chat.completions.create(
                model=config["model_name"],
                messages=messages,
                extra_body=config.get("extra_body", {})
            )
            api_response = response.choices[0].message.content

        print("   API call successful.")
        print("\n3. API Response:")
        print(api_response)
        print("-" * (len(f"--- Testing model: {model_key} ---")))
        return True

    except Exception as e:
        print(f"\nError testing model {model_key}:")
        print(traceback.format_exc())
        print("-" * (len(f"--- Testing model: {model_key} ---")))
        return False

def main():
    parser = argparse.ArgumentParser(description="Test LLMSolver config loading and basic API chat functionality.")
    parser.add_argument(
        "models",
        nargs='+',
        help="List of model keys to test (e.g., gpt-4o claude openrouter-google/gemini-2.5-flash-preview:free ollama-llama3)"
    )
    args = parser.parse_args()

    print("Starting API tests...")
    successful_tests = 0
    failed_tests = 0

    for model_key in args.models:
        if test_model_api(model_key):
            successful_tests += 1
        else:
            failed_tests += 1

    print("\n--- Test Summary ---")
    print(f"Total models tested: {len(args.models)}")
    print(f"Successful: {successful_tests}")
    print(f"Failed: {failed_tests}")
    print("--------------------")

    if failed_tests > 0:
        sys.exit(1)

if __name__ == "__main__":
    main()

# Usage:
# uv run python -m scripts.test_api <model_name1> <model_name2> ...
#
# Example (testing 3 models):
# uv run python -m scripts.test_api gemini-2.5-pro-exp-03-25 gemini-2.5-flash-preview-04-17 openrouter-qwen/qwen3-235b-a22b:free
