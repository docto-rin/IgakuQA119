"""
このスクリプトは医師国家試験の問題を分割して解くためのものです
LLMに問題の整形もさせようとしました
ただしLLMに問題の整形をさせると、問題の内容が変わってしまう可能性がある
また正しく整形できないパターンが多いので本番では使用しない
"""

import json
from openai import OpenAI
import os
from datetime import datetime
from dotenv import load_dotenv
from tqdm import tqdm


# .envファイルを読み込む
load_dotenv()


class MedicalExamProcessor:
    def __init__(self, split_model: str = "gpt-4o"):
        """
        split_model: 問題分割に使用するモデル（"gpt-4o" または "deepseek"）
        """
        self.split_model = split_model
        self.setup_client(split_model)

    def setup_client(self, model: str) -> None:
        """
        指定されたモデルに応じてクライアントを設定
        """
        self.model = model
        match model:
            case "gpt-4o":
                self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
                self.model_name = "gpt-4o"
            case "deepseek":
                self.client = OpenAI(
                    api_key=os.getenv("DEEPSEEK_API_KEY"),
                    base_url="https://api.deepseek.com",
                )
                self.model_name = "deepseek-chat"
            case "gemini":
                self.client = OpenAI(
                    api_key=os.getenv("GEMINI_API_KEY"),
                    base_url="https://generativelanguage.googleapis.com/v1beta/",
                )
                self.model_name = "gemini-2.0-flash-exp"
            case _:
                raise ValueError(f"Unsupported model: {model}")

    def save_intermediate_results(self, results: list, timestamp: str) -> None:
        """
        途中結果を保存する
        """
        output_file = f"exam_results_{timestamp}_intermediate.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(
                {"timestamp": timestamp, "results": results},
                f,
                ensure_ascii=False,
                indent=2,
            )
        print(f"中間結果を保存しました: {output_file}")

    def solve_question(self, question: dict) -> dict:
        """
        各問題を解くための関数
        """
        system_prompt = """
        あなたは医師国家試験の問題を解く専門家です。
        与えられた問題に対して、最も適切な選択肢を選んでください。
        以下のJSON形式で出力してください：

        {
            "answer": "選択肢のアルファベット",
            "confidence": 0.85  # 0.0から1.0の間で解答の確信度を示す
        }
        """

        # 問題文の構築
        full_question = f"問題：{question['question_text']}\n\n選択肢：\n"
        for choice in question.get("answer_choices", []):
            full_question += f"{choice}\n"

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": full_question},
                ],
                response_format={"type": "json_object"},
            )

            result = json.loads(response.choices[0].message.content)
            result["model_used"] = self.model
            return result

        except Exception as e:
            print(f"エラーが発生しました: {str(e)}")
            return {
                "model_used": self.model,
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }

    def split_questions_by_page(self, text: str) -> list[dict]:
        """
        ページごとに問題を分割して処理
        """
        # === で区切られたページを分割
        pages = text.split("=== Page")
        if len(pages) == 1:  # === Page がない場合は === で分割を試みる
            pages = text.split("===")

        all_questions = []

        for i, page in enumerate(pages):
            if not page.strip():  # 空のページはスキップ
                continue

            print(f"\nページ {i} の処理中...")

            try:
                # 各ページの問題を分割
                page_questions = self.split_questions_single_page(page.strip())
                all_questions.extend(page_questions)
                print(f"ページ {i} から {len(page_questions)} 問の問題を抽出しました")
            except Exception as e:
                print(f"ページ {i} の処理中にエラーが発生: {str(e)}")
                # エラーが発生したページの内容を保存
                error_file = (
                    f"error_page_{i}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
                )
                with open(error_file, "w", encoding="utf-8") as f:
                    f.write(page)
                print(f"エラーページの内容を {error_file} に保存しました")

        return all_questions

    def split_questions_single_page(self, text: str) -> list[dict]:
        """
        1ページ分のテキストから問題を分割
        """
        system_prompt = """
        あなたは医師国家試験の問題を正確に分割するアシスタントです。
        与えられたテキストから問題を抽出し、以下の形式で出力してください。
        必ず指定された形式のJSONで出力し、余分なキーは含めないでください。
        また連問の場合は、question_textに同じ問題を記述してください。

        必須の出力形式:
        [
            {
                "question_number": "問題番号（例：118B1、118 + アルファベット + NN）",
                "question_text": "実際の問題文（もし連問の場合は、question_textに同じ問題を記述してください。）",
                "answer_choices": [
                    "a: 選択肢の内容",
                    "b: 選択肢の内容",
                    "c: 選択肢の内容",
                    "d: 選択肢の内容",
                    "e: 選択肢の内容",
                    これ以上の選択肢があれば追加してください
                ]
            }
        ]

        注意：
        - 配列を直接返してください
        - 必ず全ての選択肢（a〜e）を抽出してください
        - 問題文と選択肢は明確に分離してください
        """

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text},
                ],
                response_format={"type": "json_object"},
            )

            content = response.choices[0].message.content
            print(f"\n=== {self.model} APIレスポンス ===")
            print(content)
            print("==================\n")

            result = json.loads(content)
            if isinstance(result, dict):
                if "questions" in result:
                    return result["questions"]
                else:
                    return [result]

            return []

        except Exception as e:
            print(f"エラーが発生しました: {str(e)}")
            raise

    def split_questions_whole_text(self, text: str) -> list[dict]:
        """
        テキスト全体を一度に処理して問題を分割
        """
        system_prompt = """
        あなたは医師国家試験の問題を正確に分割するアシスタントです。
        与えられたテキスト全体から全ての問題を抽出し、以下の形式で出力してください。
        必ず指定された形式のJSONで出力し、余分なキーは含めないでください。
        また連問の場合は、question_textに同じ問題を記述してください。

        必須の出力形式:
        {
            "questions": [
                {
                    "question_number": "問題番号（例：118B1、118 + アルファベット + NN）",
                    "question_text": "実際の問題文（もし連問の場合は、question_textに同じ問題を記述してください。）",
                    "answer_choices": [
                        "a: 選択肢の内容",
                        "b: 選択肢の内容",
                        "c: 選択肢の内容",
                        "d: 選択肢の内容",
                        "e: 選択肢の内容"
                    ]
                }
            ]
        }

        注意：
        - 必ず全てのテキストを読み、全ての問題を抽出してください
        - 問題番号は必ず抽出してください（例：118B1、118B2など）
        - 必ず全ての選択肢（a〜e）を抽出してください
        - 問題文と選択肢は明確に分離してください
        - ページ区切りは無視して、テキスト全体から問題を抽出してください
        - 問題が複数ある場合は、全ての問題を配列に含めてください
        """

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text},
                ],
                response_format={"type": "json_object"},
                temperature=0.0,
            )

            content = response.choices[0].message.content
            print(f"\n=== {self.model} APIレスポンス ===")
            print(content)
            print("==================\n")

            result = json.loads(content)
            if isinstance(result, dict) and "questions" in result:
                questions = result["questions"]
                print(f"抽出された問題数: {len(questions)}")
                return questions
            elif isinstance(result, list):
                print(f"抽出された問題数: {len(result)}")
                return result
            else:
                print("問題が1つだけ抽出されました")
                return [result]

        except Exception as e:
            print(f"エラーが発生しました: {str(e)}")
            raise

    def process_exam_whole_text(
        self, text: str, models: list[str] = ["gpt-4o", "deepseek", "gemini"]
    ) -> None:
        """
        テキスト全体を一度に処理し、結果を保存する
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        try:
            # 問題の分割（ペキスト全体を一度に処理）
            print("問題を分割中...")
            questions = self.split_questions_whole_text(text)

            # 分割結果を保存
            with open(f"questions_whole_{timestamp}.json", "w", encoding="utf-8") as f:
                json.dump(questions, f, ensure_ascii=False, indent=2)
            print(f"問題分割結果を保存しました: questions_whole_{timestamp}.json")

            # 各問題を解く
            results = []
            for question in tqdm(questions, desc="問題を解析中"):
                question_results = {
                    "question": question,
                    "answers": [],
                }

                # 各モデルで解答
                for model in tqdm(
                    models,
                    desc=f"問題 {question['question_number']} をモデルで解析中",
                    leave=False,
                ):
                    try:
                        self.setup_client(model)
                        answer = self.solve_question(question)
                        question_results["answers"].append(answer)
                    except Exception as e:
                        error_msg = f"Error with model {model} for question {question['question_number']}: {str(e)}"
                        print(error_msg)
                        question_results["answers"].append(
                            {
                                "model_used": model,
                                "error": error_msg,
                                "timestamp": datetime.now().isoformat(),
                            }
                        )

                results.append(question_results)
                # 各問題が終わるごとに中間結果を保存
                self.save_intermediate_results(results, f"whole_{timestamp}")

            # 最終結果の保存
            output_file = f"exam_results_whole_{timestamp}_final.json"
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(
                    {"timestamp": timestamp, "results": results},
                    f,
                    ensure_ascii=False,
                    indent=2,
                )

            print(f"最終結果を保存しました: {output_file}")

        except Exception as e:
            print(f"エラーが発生しました: {str(e)}")
            # エラーが発生した場合でも、それまでの結果は保存
            if "results" in locals():
                self.save_intermediate_results(results, f"whole_{timestamp}_error")
            raise


# 使用例
if __name__ == "__main__":
    # DeepSeekを使って問題を分割し、3つのモデルで解答
    processor = MedicalExamProcessor(split_model="gpt-4o")

    # OCRテキストの読み込み
    with open("output/118B_ocr_text.txt", "r", encoding="utf-8") as f:
        exam_text = f.read()

    # 全体テキストとして処理を実行（3つのモデルで解答）
    processor.process_exam_whole_text(
        exam_text, models=["gpt-4o", "deepseek", "gemini"]
    )
