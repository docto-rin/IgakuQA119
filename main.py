import json
import argparse
from llm_solver import LLMSolver

def main():
    parser = argparse.ArgumentParser(description="医師国家試験の問題をLLMで解く")
    parser.add_argument(
        "input_json",
        help="入力JSONファイルのパス"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["o1", "gpt-4o", "o3-mini", "gemini", "deepseek", "claude"],
        default=["o1", "gpt-4o", "o3-mini", "gemini", "deepseek", "claude"],
        help="使用するモデル（複数指定可）"
    )
    
    args = parser.parse_args()

    try:
        # 問題データの読み込み
        with open(args.input_json, "r", encoding="utf-8") as f:
            questions = json.load(f)

        # ソルバーを初期化
        solver = LLMSolver()
        
        # 問題を解く
        results = solver.process_questions(questions, args.models)
        
        print("処理が完了しました。")
        print("結果は answer/ ディレクトリに保存されています。")

    except Exception as e:
        print(f"エラーが発生しました: {str(e)}")

if __name__ == "__main__":
    main() 