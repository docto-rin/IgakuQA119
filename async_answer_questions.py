import json
import os
import base64
import glob
from openai import AsyncOpenAI
import anthropic
from datetime import datetime
from dotenv import load_dotenv
from tqdm import tqdm
import argparse
from PIL import Image
import io
import asyncio
from typing import List, Dict, Any
import aiohttp

# .envファイルを読み込む
load_dotenv()

# 結果保存用のディレクトリを作成
os.makedirs("answer", exist_ok=True)


class AsyncMedicalExamSolver:
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
                    "temperature": 0.2,
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
                    "temperature": 0.2,
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
                    "temperature": 0.2,
                    "response_format": {"type": "json_object"},
                },
            },
            "claude": {
                "api_key": os.getenv("ANTHROPIC_API_KEY"),
                "base_url": None,
                "model_name": "claude-3-5-sonnet-20241022",
                "supports_vision": True,
                "api_type": "anthropic",
                "parameters": {"temperature": 0.2, "max_tokens": 1000},
            },
        }
        self.clients = {}

    def compress_image(self, image_path, max_size=1920):
        """画像を圧縮する"""
        with Image.open(image_path) as img:
            width, height = img.size
            if width > height and width > max_size:
                ratio = max_size / width
                new_size = (max_size, int(height * ratio))
                img = img.resize(new_size, Image.Resampling.LANCZOS)
            elif height > max_size:
                ratio = max_size / height
                new_size = (int(width * ratio), max_size)
                img = img.resize(new_size, Image.Resampling.LANCZOS)

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

    async def setup_client(self, model_key: str):
        """指定されたモデルのクライアントを設定（非同期）"""
        if model_key not in self.clients:
            model_config = self.models[model_key]
            if model_config["api_type"] == "anthropic":
                # Anthropicの非同期クライアントを設定
                self.clients[model_key] = anthropic.AsyncAnthropic(
                    api_key=model_config["api_key"]
                )
            else:
                if model_config["base_url"]:
                    self.clients[model_key] = AsyncOpenAI(
                        api_key=model_config["api_key"],
                        base_url=model_config["base_url"]
                    )
                else:
                    self.clients[model_key] = AsyncOpenAI(api_key=model_config["api_key"])
        
        return self.clients[model_key]

    async def solve_question(
        self, question: dict, model_key: str, include_explanation: bool = False
    ) -> dict:
        """各問題をLLMで解く（非同期）"""
        max_retries = 3
        retry_count = 0
        last_error = None
        retry_delay = 2  # 初期待機時間（秒）

        while retry_count < max_retries:
            try:
                client = await self.setup_client(model_key)
                model_config = self.models[model_key]

                # エラー発生時の待機時間を計算（指数バックオフ）
                if retry_count > 0:
                    wait_time = retry_delay * (2 ** (retry_count - 1))
                    print(f"{model_key}: {retry_count + 1}回目の試行を待機中... ({wait_time}秒)")
                    await asyncio.sleep(wait_time)

                has_images = question.get("has_image", False)
                image_count = 0
                if has_images:
                    image_paths = self.get_question_images(question["number"])
                    image_count = len(image_paths)

                system_prompt = """あなたは医師国家試験の問題を解く専門家です。与えられた問題に対して、最も適切な選択肢を選んでください。

以下のルールに従って回答してください：
1. 問題文に「2つ選べ」などの指示がない限り、必ず1つだけ選択してください
2. 問題文で複数選択が指示されている場合のみ、複数の選択肢を選んでください
3. 複数選択の場合は、選択肢をアルファベット順に並べて出力してください（例：ac, ce）
4. 必ず以下のJSON形式で出力してください：

{
    "answer": "選択肢のアルファベット",
    "confidence": 0.85
}"""

                if include_explanation:
                    system_prompt = system_prompt[:-1] + ',\n    "explanation": "解答の根拠を簡潔に説明"\n}'

                question_text = f"""問題：{question["question"]}

選択肢：
{chr(10).join(question["choices"])}
"""

                if model_config["api_type"] == "anthropic":
                    content = [{"type": "text", "text": question_text}]
                    if model_config["supports_vision"] and has_images:
                        for i, image_path in enumerate(image_paths, 1):
                            base64_image = self.encode_image(image_path)
                            content.extend([
                                {"type": "text", "text": f"画像{i}："},
                                {
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": "image/jpeg",
                                        "data": base64_image,
                                    },
                                },
                            ])

                    response = await client.messages.create(
                        model=model_config["model_name"],
                        messages=[{"role": "user", "content": content}],
                        system=system_prompt,
                        **model_config["parameters"],
                    )

                    content = response.content[0].text.strip()
                    try:
                        json_start = content.find("{")
                        json_end = content.rfind("}") + 1
                        if json_start >= 0 and json_end > json_start:
                            json_str = content[json_start:json_end]
                            result = json.loads(json_str)
                        else:
                            raise ValueError("No JSON object found in response")
                    except (json.JSONDecodeError, ValueError) as e:
                        print(f"試行 {retry_count + 1}/{max_retries} でJSONパースエラー: {str(e)}")
                        print(f"Raw content: {content}")
                        last_error = e
                        retry_count += 1
                        if retry_count < max_retries:
                            print("再試行中...")
                            continue
                        raise

                else:  # OpenAI APIの場合
                    messages = [{"role": "system", "content": system_prompt}]
                    if model_config["supports_vision"] and has_images:
                        content = [{"type": "text", "text": question_text}]
                        for image_path in image_paths:
                            base64_image = self.encode_image(image_path)
                            content.append({
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                },
                            })
                        messages.append({"role": "user", "content": content})
                    else:
                        messages.append({"role": "user", "content": question_text})

                    response = await client.chat.completions.create(
                        model=model_config["model_name"],
                        messages=messages,
                        **model_config["parameters"],
                    )

                    try:
                        result = json.loads(response.choices[0].message.content)
                    except (json.JSONDecodeError, AttributeError) as e:
                        print(f"試行 {retry_count + 1}/{max_retries} でJSONパースエラー: {str(e)}")
                        last_error = e
                        retry_count += 1
                        if retry_count < max_retries:
                            print("再試行中...")
                            continue
                        raise

                result["model_used"] = model_key
                result["timestamp"] = datetime.now().isoformat()
                result["image_info"] = {
                    "has_images": has_images,
                    "image_count": image_count,
                }
                return result

            except Exception as e:
                error_msg = str(e)
                print(f"{model_key}でエラーが発生: {error_msg}")
                
                # レート制限エラーの場合は長めに待機
                if "429" in error_msg or "quota" in error_msg.lower():
                    wait_time = retry_delay * (2 ** retry_count) * 2
                    print(f"{model_key}: レート制限により長めに待機します... ({wait_time}秒)")
                    await asyncio.sleep(wait_time)
                
                last_error = e
                retry_count += 1
                if retry_count < max_retries:
                    print(f"{model_key}: 試行 {retry_count}/{max_retries} で失敗。再試行準備中...")
                    continue

                return {
                    "model_used": model_key,
                    "error": f"全ての試行が失敗しました（{max_retries}回）: {str(last_error)}",
                    "timestamp": datetime.now().isoformat(),
                    "image_info": {"has_images": has_images, "image_count": image_count},
                }

    async def process_question_with_models(
        self, question: dict, models: List[str], include_explanation: bool
    ) -> Dict[str, Any]:
        """1つの問題を複数のモデルで並行処理"""
        tasks = []
        for model in models:
            task = self.solve_question(question, model, include_explanation)
            tasks.append(task)
        
        answers = await asyncio.gather(*tasks, return_exceptions=True)
        return {
            "question": question,
            "answers": [
                answer if not isinstance(answer, Exception) else {
                    "model_used": models[i],
                    "error": str(answer),
                    "timestamp": datetime.now().isoformat()
                }
                for i, answer in enumerate(answers)
            ]
        }

    async def process_questions(
        self,
        questions: list,
        models: List[str] = None,
        include_explanation: bool = False,
        batch_size: int = 5
    ) -> list:
        """全ての問題を処理（非同期）"""
        if models is None:
            models = list(self.models.keys())

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results = []
        failed_models = {}

        # バッチ処理
        for i in range(0, len(questions), batch_size):
            batch = questions[i:i + batch_size]
            tasks = [
                self.process_question_with_models(question, models, include_explanation)
                for question in batch
            ]
            
            batch_results = await asyncio.gather(*tasks)
            results.extend(batch_results)

            # 中間結果を保存
            self.save_results(results, f"exam_results_{timestamp}_intermediate.json")

            # エラー情報の収集
            for result in batch_results:
                for answer in result["answers"]:
                    if "error" in answer:
                        model = answer["model_used"]
                        if model not in failed_models:
                            failed_models[model] = {"count": 0, "last_error": None}
                        failed_models[model]["count"] += 1
                        failed_models[model]["last_error"] = answer["error"]

        # 失敗したモデルの情報を表示
        if failed_models:
            print("\n=== 失敗したモデルの情報 ===")
            for model, info in failed_models.items():
                print(f"\nモデル: {model}")
                print(f"失敗回数: {info['count']}")
                print(f"最後のエラー: {info['last_error']}")

        # 最終結果を保存
        self.save_results(results, f"exam_results_{timestamp}_final.json")
        return results

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


async def main():
    parser = argparse.ArgumentParser(description="医師国家試験の問題を複数のLLMで解く（非同期版）")
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
    parser.add_argument(
        "--batch-size",
        type=int,
        default=5,
        help="同時に処理する問題数（デフォルト: 5）"
    )

    args = parser.parse_args()

    try:
        with open(args.input, "r", encoding="utf-8") as f:
            questions = json.load(f)

        solver = AsyncMedicalExamSolver()
        await solver.process_questions(
            questions,
            args.models,
            args.explanation,
            args.batch_size
        )

        print("処理が完了しました。")

    except Exception as e:
        print(f"エラーが発生しました: {str(e)}")


if __name__ == "__main__":
    asyncio.run(main()) 