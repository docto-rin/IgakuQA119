import re
import json
import argparse


def parse_medical_questions(text):
    questions = []
    current_question = None

    lines = text.split("\n")
    # 選択肢のパターン: アルファベット1文字 + ピリオド
    choice_pattern = re.compile(r"^([a-z])\.")

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # 問題番号の検出
        if line.startswith("=== "):
            if current_question:
                questions.append(current_question)
            question_number = line.strip("= ")
            current_question = {
                "number": question_number,
                "question": "",
                "choices": [],
                "has_image": False,
            }

        # 問題文の検出
        elif line.startswith("【問題】"):
            current_question["question"] = ""
            continue

        # 選択肢の開始検出
        elif line.startswith("【選択肢】"):
            continue

        # 選択肢の追加
        elif choice_pattern.match(line):
            choice = line.strip()
            current_question["choices"].append(choice)

        # 問題文の追加
        elif current_question and "【選択肢】" not in line and line.strip():
            if not current_question["choices"]:  # 選択肢が始まっていない場合
                current_question["question"] += line.strip() + " "
                # 画像問題かどうかのチェック
                if "別冊" in line:
                    current_question["has_image"] = True

    # 最後の問題を追加
    if current_question:
        questions.append(current_question)

    return questions


def main():
    parser = argparse.ArgumentParser(
        description="テキストファイルから問題データをJSONに変換"
    )
    parser.add_argument("input", help="入力テキストファイルのパス")
    parser.add_argument("output", help="出力JSONファイルのパス")

    args = parser.parse_args()

    try:
        # 入力ファイルを読み込み
        with open(args.input, "r", encoding="utf-8") as f:
            text = f.read()

        # テキストを解析
        questions = parse_medical_questions(text)

        # JSON形式で出力
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(questions, f, ensure_ascii=False, indent=2)

        print(f"変換が完了しました。出力ファイル: {args.output}")

    except Exception as e:
        print(f"エラーが発生しました: {str(e)}")


if __name__ == "__main__":
    main()
