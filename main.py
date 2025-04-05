import json
import argparse
from src.llm_solver import LLMSolver

def main():
    parser = argparse.ArgumentParser(description="医師国家試験の問題をLLMで解く")
    parser.add_argument(
        "input_json",
        help="入力JSONファイルのパス (例: questions/119A_json.json)"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        help="使用するモデル (例: gpt-4o, claude, gemini-2.5-pro)"
    )
    parser.add_argument(
        "--questions",
        nargs="+",
        help="解く問題の番号。指定しない場合は全ての問題を解きます (例: 119A1 119A2)"
    )
    parser.add_argument(
        "--exp",
        default=None,
        help="結果ファイルの識別子。指定しない場合はタイムスタンプが使用されます (例: baseline)"
    )
    
    args = parser.parse_args()

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
        print("結果は answers/ ディレクトリに保存されています。")

    except Exception as e:
        print(f"エラーが発生しました: {str(e)}")

if __name__ == "__main__":
    main()