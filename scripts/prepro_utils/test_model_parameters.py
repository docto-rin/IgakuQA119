# test_model_parameters.py

"""
このスクリプトは、OpenAIのモデルのパラメータ対応をテストするためのものです。
パラメータ対応をテストするためには、OpenAIのAPIキーが必要です。
OpenAIのAPIキーは、.envファイルに記載してください。

o1およびo3-miniのパラメータ対応をテストします。
response_formatパラメータが機能するか確認
"""

import os
import sys
import json
from openai import OpenAI
import anthropic
from datetime import datetime
from dotenv import load_dotenv
import argparse

# プロジェクトのルートディレクトリをPYTHONPATHに追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# .envファイルを読み込む
load_dotenv()


class ModelParameterTester:
    def __init__(self):
        self.models = {
            "o1": {
                "api_key": os.getenv("OPENAI_API_KEY"),
                "base_url": None,
                "model_name": "o1",
                "supports_vision": True,
                "api_type": "openai",
                "parameters": {"response_format": {"type": "json_object"}},
            },
            "gpt-4o": {
                "api_key": os.getenv("OPENAI_API_KEY"),
                "base_url": None,
                "model_name": "gpt-4o",
                "supports_vision": True,
                "api_type": "openai",
                "parameters": {
                    "temperature": 0.3,
                    "response_format": {"type": "json_object"},
                },
            },
            "o3-mini": {
                "api_key": os.getenv("OPENAI_API_KEY"),
                "base_url": None,
                "model_name": "o3-mini",
                "supports_vision": False,
                "api_type": "openai",
                "parameters": {"response_format": {"type": "json_object"}},
            },
            "gemini": {
                "api_key": os.getenv("GEMINI_API_KEY"),
                "base_url": "https://generativelanguage.googleapis.com/v1beta/",
                "model_name": "gemini-2.0-flash-exp",
                "supports_vision": True,
                "api_type": "openai",
                "parameters": {
                    "temperature": 0.3,
                    "response_format": {"type": "json_object"},
                },
            },
            "deepseek": {
                "api_key": os.getenv("DEEPSEEK_API_KEY"),
                "base_url": "https://api.deepseek.com/v1",
                "model_name": "deepseek-reasoner",
                "supports_vision": False,
                "api_type": "openai",
                "parameters": {},  # パラメータなし
            },
            "claude": {
                "api_key": os.getenv("ANTHROPIC_API_KEY"),
                "base_url": None,
                "model_name": "claude-3-5-sonnet-20241022",
                "supports_vision": True,
                "api_type": "anthropic",  # api_typeを追加
                "parameters": {"max_tokens": 1000},
            },
        }

    def setup_client(self, model_key: str):
        """指定されたモデルのクライアントを設定"""
        model_config = self.models[model_key]
        if model_config["api_type"] == "anthropic":
            return anthropic.Anthropic(api_key=model_config["api_key"])
        else:
            if model_config["base_url"]:
                return OpenAI(
                    api_key=model_config["api_key"], base_url=model_config["base_url"]
                )
            return OpenAI(api_key=model_config["api_key"])

    def test_model_parameters(self, model_key: str):
        """
        指定されたモデルのパラメータ対応をテストする

        Args:
            model_key (str): テストするモデル名
        """
        client = self.setup_client(model_key)
        model_config = self.models[model_key]

        # テスト用の問題（単一選択と複数選択の両方をテスト）
        test_questions = [
            {
                "title": "単一選択のテスト",
                "question": """【問題】
褥瘡ができやすい部位はどこか。

【選択肢】
a. 耳介
b. 手掌
c. 臍
d. 外陰部
e. 踵部""",
            },
            {
                "title": "複数選択のテスト",
                "question": """【問題】
新生児マススクリーニングで正しいのはどれか。2つ選べ。

【選択肢】
a. 尿で検査される。
b. 患者家族の一部負担金がある。
c. 都道府県および指定都市が実施する。
d. 発見者数はフェニルケトン尿症が最多である。
e. 先天性代謝異常にはタンデムマス·スクリーニングが使用される。""",
            },
        ]

        system_prompt = """あなたは医師国家試験の問題を解く専門家です。与えられた問題に対して、最も適切な選択肢を選んでください。

以下のルールに従って回答してください：
1. 問題文に「2つ選べ」などの指示がない限り、必ず1つだけ選択してください
2. 問題文で複数選択が指示されている場合のみ、複数の選択肢を選んでください
3. 複数選択の場合は、選択肢をアルファベット順に並べて出力してください（例：ac, ce）
4. 必ず以下のJSON形式で出力してください：

{
    "answer": "選択肢のアルファベット（選択肢の中から）",
    "confidence": 0.85
}"""

        deepseek_prompt = """あなたは医師国家試験の問題を解く専門家です。与えられた問題に対して、最も適切な選択肢を選んでください。

以下のルールに従って回答してください：
1. 問題文に「2つ選べ」などの指示がない限り、必ず1つだけ選択してください
2. 問題文で複数選択が指示されている場合のみ、複数の選択肢を選んでください
3. 複数選択の場合は、選択肢をアルファベット順に並べて出力してください（例：ac, ce）
4. 回答は以下の形式で出力してください（余計な説明は不要です）：

answer: [選択したアルファベット]
confidence: [0.0から1.0の確信度]"""

        print(f"\n=== {model_key}のパラメータテスト ===")

        for test_case in test_questions:
            print(f"\n--- {test_case['title']} ---")
            try:
                if model_key == "claude":
                    response = client.messages.create(
                        model=model_config["model_name"],
                        messages=[{"role": "user", "content": test_case["question"]}],
                        system="""あなたは医師国家試験の問題を解く専門家です。与えられた問題に対して、最も適切な選択肢を選んでください。

以下のルールに従って回答してください：
1. 問題文に「2つ選べ」などの指示がない限り、必ず1つだけ選択してください
2. 問題文で複数選択が指示されている場合のみ、複数の選択肢を選んでください
3. 複数選択の場合は、選択肢をアルファベット順に並べて出力してください（例：ac, ce）
4. 回答は以下のJSON形式で出力してください。解説は含めないでください：
{"answer": "選択肢のアルファベット", "confidence": 0.95}""",
                        **model_config["parameters"],
                    )

                    # レスポンスから最初の有効なJSONを抽出
                    content = response.content[0].text.strip()
                    print(f"\nClaude APIレスポンス:")
                    print(f"Raw content: {content}")

                    try:
                        # 最初の{から最後の}までを抽出
                        json_start = content.find("{")
                        json_end = content.rfind("}") + 1
                        if json_start >= 0 and json_end > json_start:
                            json_str = content[json_start:json_end]
                            result = json.loads(json_str)
                            print(f"Extracted JSON: {json_str}")
                        else:
                            raise ValueError("No JSON object found in response")
                    except json.JSONDecodeError as e:
                        print(f"JSON parse error: {str(e)}")
                        print(f"Raw content: {content}")
                        raise
                elif model_key == "deepseek":
                    print("\nDeepSeek APIリクエスト:")
                    print("System prompt:", deepseek_prompt)
                    print("User prompt:", test_case["question"])

                    response = client.chat.completions.create(
                        model=model_config["model_name"],
                        messages=[
                            {"role": "system", "content": deepseek_prompt},
                            {"role": "user", "content": test_case["question"]},
                        ],
                    )

                    print("\nDeepSeek APIレスポンス:")
                    print("Full response:", response)
                    print("Message content:", response.choices[0].message.content)

                    # reasoning_contentがあれば表示
                    if hasattr(response.choices[0].message, "reasoning_content"):
                        print(
                            f"Reasoning content: {response.choices[0].message.reasoning_content}"
                        )

                    # 回答をパース
                    content = response.choices[0].message.content.strip()
                    print(
                        f"\nRaw response content: {repr(content)}"
                    )  # reprを使用して特殊文字を表示

                    # DeepSeekの出力を構造化
                    answer = None
                    confidence = 0.5

                    print("\nパース処理:")
                    for line in content.split("\n"):
                        line = line.strip().lower()  # 小文字に変換して比較
                        print(f"Processing line: {repr(line)}")  # 各行の処理を表示
                        if line.startswith("answer:"):
                            answer = line.split(":", 1)[1].strip()
                            print(f"Found answer: {answer}")
                        elif line.startswith("confidence:"):
                            try:
                                confidence = float(line.split(":", 1)[1].strip())
                                print(f"Found confidence: {confidence}")
                            except ValueError as e:
                                print(
                                    f"Warning: Could not parse confidence value from: {line}"
                                )
                                print(f"Error: {str(e)}")

                    result = {
                        "answer": answer if answer else content,
                        "confidence": confidence,
                    }
                    print(f"\n最終結果: {result}")
                else:
                    response = client.chat.completions.create(
                        model=model_config["model_name"],
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": test_case["question"]},
                        ],
                        **model_config["parameters"],
                    )
                    result = json.loads(response.choices[0].message.content)

                print(f"✅ パラメータテスト: 成功")
                print(f"レスポンス: {result}")

            except Exception as e:
                print(f"❌ パラメータテスト: 失敗")
                print(f"エラー: {str(e)}")


def main():
    """
    メイン関数
    """
    parser = argparse.ArgumentParser(
        description="LLMモデルのパラメータ対応をテストする"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["o1", "gpt-4o", "o3-mini", "gemini", "deepseek", "claude"],
        default=["deepseek", "claude"],
        help="テストするモデル（デフォルト: deepseek, claude）",
    )
    parser.add_argument(
        "--debug", action="store_true", help="デバッグモード（より詳細な出力）"
    )

    args = parser.parse_args()

    tester = ModelParameterTester()

    # テスト対象のモデルを指定（引数で指定されたもののみ）
    test_models = args.models

    print(f"テスト対象モデル: {', '.join(test_models)}")
    print("=" * 50)

    for model_key in test_models:
        try:
            if args.debug:
                print(f"\nモデル設定: {tester.models[model_key]}")
            tester.test_model_parameters(model_key)
        except Exception as e:
            print(f"❌ {model_key}のテスト中にエラーが発生: {str(e)}")
            if args.debug:
                import traceback

                print("詳細なエラー情報:")
                print(traceback.format_exc())


if __name__ == "__main__":
    main()
