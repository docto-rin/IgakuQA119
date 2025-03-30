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
        default=["o1", "gpt-4o", "o3-mini", "gemini", "gemini-2.5-pro", "deepseek", "claude"],
        help="使用するモデル（複数指定可）。o1, gpt-4o, o3-mini, gemini, gemini-2.5-pro, deepseek, claude、または gemini- で始まる任意のモデル名を指定できます。"
    )
    parser.add_argument(
        "--questions",
        nargs="+",
        help="解く問題の番号（例：119A1 119A2）。指定しない場合は全ての問題を解きます。"
    )
    parser.add_argument(
        "--exp",
        default=None,
        help="結果ファイルの識別子。指定しない場合はタイムスタンプが使用されます。"
    )
    
    args = parser.parse_args()

    # モデル名の検証
    standard_models = ["o1", "gpt-4o", "o3-mini", "gemini", "gemini-2.5-pro", "deepseek", "claude"]
    for model in args.models:
        if model not in standard_models and not model.startswith("gemini-"):
            print(f"警告: '{model}' は未知のモデルです。o1, gpt-4o, o3-mini, gemini, gemini-2.5-pro, deepseek, claude、または gemini- で始まるモデル名を指定してください。")
            return

    try:
        # 問題データの読み込み
        with open(args.input_json, "r", encoding="utf-8") as f:
            questions = json.load(f)

        # 特定の問題番号が指定された場合、その問題のみをフィルタリング
        if args.questions:
            filtered_questions = [q for q in questions if q["number"] in args.questions]
            if not filtered_questions:
                print(f"指定された問題番号 {args.questions} は見つかりませんでした。")
                return
            questions = filtered_questions

        # ソルバーを初期化
        solver = LLMSolver()
        
        # 問題を解く
        results = solver.process_questions(questions, args.models, file_exp=args.exp)
        
        print("処理が完了しました。")
        print("結果は answer/ ディレクトリに保存されています。")

    except Exception as e:
        print(f"エラーが発生しました: {str(e)}")

if __name__ == "__main__":
    main()