import json
import os
import base64
import glob
from openai import OpenAI
import anthropic
from datetime import datetime
from dotenv import load_dotenv
from tqdm import tqdm
import argparse
from PIL import Image
import io

# .envファイルを読み込む
load_dotenv()

# 結果保存用のディレクトリを作成
os.makedirs("answer", exist_ok=True)


class MedicalExamSolver:
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
                "parameters": {"response_format": {"type": "json_object"},
                               "reasoning_effort": "high"},
            },
            "gemini": {
                "api_key": os.getenv("GEMINI_API_KEY"),
                "base_url": "https://generativelanguage.googleapis.com/v1beta/",
                "model_name": "gemini-2.0-flash-001",
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
                "parameters": {
                    "temperature": 0.3,
                    "response_format": {"type": "json_object"},
                },
            },
            "claude": {
                "api_key": os.getenv("ANTHROPIC_API_KEY"),
                "base_url": None,
                "model_name": "claude-3-5-sonnet-20241022",
                "supports_vision": True,
                "api_type": "anthropic",
                "parameters": {"temperature": 0.3, "max_tokens": 1000},
            },
        }

    def compress_image(self, image_path, max_size=1920):
        """画像を圧縮する"""
        with Image.open(image_path) as img:
            # 元の画像のアスペクト比を保持
            width, height = img.size
            if width > height:
                if width > max_size:
                    ratio = max_size / width
                    new_size = (max_size, int(height * ratio))
                    img = img.resize(new_size, Image.Resampling.LANCZOS)
            else:
                if height > max_size:
                    ratio = max_size / height
                    new_size = (int(width * ratio), max_size)
                    img = img.resize(new_size, Image.Resampling.LANCZOS)

            # 圧縮した画像をバイトストリームに保存
            buffer = io.BytesIO()
            img.save(buffer, format=img.format if img.format else "JPEG", quality=85)
            return buffer.getvalue()

    def encode_image(self, image_path):
        """画像を圧縮してbase64エンコードする"""
        compressed_image = self.compress_image(image_path)
        return base64.b64encode(compressed_image).decode("utf-8")

    def get_question_images(self, question_number):
        """問題番号に関連する画像を取得"""
        image_paths = []
        base_patterns = [
            f"images/{question_number}.jpg",
            f"images/{question_number}.png",
            f"images/{question_number}-*.jpg",
            f"images/{question_number}-*.png",
        ]

        for pattern in base_patterns:
            image_paths.extend(glob.glob(pattern))

        return sorted(image_paths)

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

    def solve_question(
        self, question: dict, model_key: str, include_explanation: bool = False
    ) -> dict:
        """各問題をLLMで解く"""
        client = self.setup_client(model_key)
        model_config = self.models[model_key]

        # 画像情報を取得
        has_images = question.get("has_image", False)
        image_count = 0
        if has_images:
            image_paths = self.get_question_images(question["number"])
            image_count = len(image_paths)

        system_prompt = """
        あなたは医師国家試験の問題を解く専門家です。
        与えられた問題に対して、最も適切な選択肢を選んでください。
        また複数回答がある場合は ac などスペース区切りなしの文字列で出力してください。
        以下のJSON形式で出力してください：

        {
            "answer": "選択肢のアルファベット（選択肢の中から）",
            "confidence": 0.85,  # 0.0から1.0の間で解答の確信度を示す
        }
        """

        if include_explanation:
            system_prompt = """
            あなたは医師国家試験の問題を解く専門家です。
            与えられた問題に対して、最も適切な選択肢を選んでください。
            また複数回答がある場合は ac などスペース区切りなしの文字列で出力してください。
            以下のJSON形式で出力してください：

            {
                "answer": "選択肢のアルファベット（選択肢の中から）",
                "confidence": 0.85,  # 0.0から1.0の間で解答の確信度を示す
                "explanation": "解答の根拠を簡潔に説明"
            }
            """

        # 問題文の構築
        question_text = f"""問題：{question["question"]}

選択肢：
{chr(10).join(question["choices"])}
"""

        try:
            if model_config["api_type"] == "anthropic":
                # Anthropic APIの場合
                content = []
                content.append({"type": "text", "text": question_text})

                if model_config["supports_vision"] and has_images:
                    for i, image_path in enumerate(image_paths, 1):
                        base64_image = self.encode_image(image_path)
                        content.extend(
                            [
                                {"type": "text", "text": f"画像{i}："},
                                {
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": "image/jpeg",
                                        "data": base64_image,
                                    },
                                },
                            ]
                        )

                # システムプロンプトを修正
                system_prompt_claude = """あなたは医師国家試験の問題を解く専門家です。
与えられた問題に対して、最も適切な選択肢を選んでください。
また複数回答がある場合は ac などスペース区切りなしの文字列で出力してください。

回答は以下のJSON形式で出力してください。解説は含めないでください："""

                if include_explanation:
                    system_prompt_claude += """
{"answer": "選択肢のアルファベット", "confidence": 0.95, "explanation": "解答の根拠を簡潔に説明"}"""
                else:
                    system_prompt_claude += """
{"answer": "選択肢のアルファベット", "confidence": 0.95}"""

                response = client.messages.create(
                    model=model_config["model_name"],
                    messages=[{"role": "user", "content": content}],
                    system=system_prompt_claude,
                    **model_config["parameters"],
                )

                # レスポンスから最初の有効なJSONを抽出
                content = response.content[0].text.strip()
                try:
                    # 最初の{から最後の}までを抽出
                    json_start = content.find("{")
                    json_end = content.rfind("}") + 1
                    if json_start >= 0 and json_end > json_start:
                        json_str = content[json_start:json_end]
                        result = json.loads(json_str)
                    else:
                        raise ValueError("No JSON object found in response")
                except json.JSONDecodeError as e:
                    print(f"JSON parse error: {str(e)}")
                    print(f"Raw content: {content}")
                    raise

            elif model_key == "deepseek":
                # DeepSeek APIの場合
                deepseek_prompt = """あなたは医師国家試験の問題を解く専門家です。
与えられた問題に対して、最も適切な選択肢を選んでください。
また複数回答がある場合は ac などスペース区切りなしの文字列で出力してください。

回答は以下の形式で出力してください（余計な説明は不要です）：

answer: [選択したアルファベット]
confidence: [0.0から1.0の確信度]"""

                if include_explanation:
                    deepseek_prompt += """
explanation: [解答の根拠を簡潔に説明]"""

                response = client.chat.completions.create(
                    model=model_config["model_name"],
                    messages=[
                        {"role": "system", "content": deepseek_prompt},
                        {"role": "user", "content": question_text},
                    ],
                )

                # 回答をパース
                content = response.choices[0].message.content.strip()

                # DeepSeekの出力を構造化
                answer = None
                confidence = 0.5
                explanation = None

                for line in content.split("\n"):
                    line = line.strip().lower()
                    if line.startswith("answer:"):
                        answer = line.split(":", 1)[1].strip()
                    elif line.startswith("confidence:"):
                        try:
                            confidence = float(line.split(":", 1)[1].strip())
                        except ValueError:
                            print(f"Warning: Could not parse confidence value from: {line}")
                    elif include_explanation and line.startswith("explanation:"):
                        explanation = line.split(":", 1)[1].strip()

                result = {
                    "answer": answer if answer else content,
                    "confidence": confidence,
                }

                if include_explanation and explanation:
                    result["explanation"] = explanation

            else:
                # OpenAI APIの場合
                messages = [{"role": "system", "content": system_prompt}]

                if model_config["supports_vision"] and has_images:
                    content = [{"type": "text", "text": question_text}]
                    for image_path in image_paths:
                        base64_image = self.encode_image(image_path)
                        content.append(
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                },
                            }
                        )
                    messages.append({"role": "user", "content": content})
                else:
                    messages.append({"role": "user", "content": question_text})

                response = client.chat.completions.create(
                    model=model_config["model_name"],
                    messages=messages,
                    **model_config["parameters"],
                )
                result = json.loads(response.choices[0].message.content)

            result["model_used"] = model_key
            result["timestamp"] = datetime.now().isoformat()
            result["image_info"] = {
                "has_images": has_images,
                "image_count": image_count,
            }
            return result

        except Exception as e:
            return {
                "model_used": model_key,
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
                "image_info": {"has_images": has_images, "image_count": image_count},
            }

    def save_results(self, results: list, filename: str) -> None:
        """結果をJSONファイルに保存"""
        filepath = os.path.join("answer", filename)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(
                {"timestamp": datetime.now().isoformat(), "results": results},
                f,
                ensure_ascii=False,
                indent=2,
            )

    def process_questions(
        self,
        questions: list,
        models: list[str] = None,
        include_explanation: bool = False,
    ) -> list:
        """全ての問題を処理"""
        if models is None:
            models = list(self.models.keys())

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results = []

        for question in tqdm(questions, desc="問題を解析中"):
            question_results = {"question": question, "answers": []}

            for model in tqdm(
                models, desc=f"問題 {question['number']} をモデルで解析中", leave=False
            ):
                try:
                    answer = self.solve_question(question, model, include_explanation)
                    question_results["answers"].append(answer)
                except Exception as e:
                    print(f"Error with model {model}: {str(e)}")
                    question_results["answers"].append(
                        {
                            "model_used": model,
                            "error": str(e),
                            "timestamp": datetime.now().isoformat(),
                        }
                    )

            results.append(question_results)

            # 中間結果を保存
            self.save_results(results, f"exam_results_{timestamp}_intermediate.json")

        # 最終結果を保存
        self.save_results(results, f"exam_results_{timestamp}_final.json")
        return results


def main():
    parser = argparse.ArgumentParser(description="医師国家試験の問題を複数のLLMで解く")
    parser.add_argument("input", help="入力JSONファイルのパス")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["o1", "gpt-4o", "o3-mini", "gemini", "deepseek", "claude"],
        default=["o1", "gpt-4o", "o3-mini", "gemini", "deepseek", "claude"],
        help="使用するモデル（複数指定可）",
    )
    parser.add_argument(
        "--explanation", action="store_true", help="解答の説明を含めるかどうか"
    )

    args = parser.parse_args()

    try:
        # 問題データの読み込み
        with open(args.input, "r", encoding="utf-8") as f:
            questions = json.load(f)

        # 問題を解く
        solver = MedicalExamSolver()
        solver.process_questions(questions, args.models, args.explanation)

        print("処理が完了しました。")

    except Exception as e:
        print(f"エラーが発生しました: {str(e)}")


if __name__ == "__main__":
    main()
