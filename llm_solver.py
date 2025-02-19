import os
from typing import Dict, List, Optional
import anthropic
from openai import OpenAI
from datetime import datetime
from dotenv import load_dotenv
from process_llm_output import OutputProcessor
from tqdm import tqdm
import json
import glob
import base64

load_dotenv()

class LLMSolver:
    def __init__(self):
        self.output_processor = OutputProcessor()
        
        # 共通のシステムプロンプトを定義
        self.system_prompt = """医師国家試験の問題について回答してください。
以下の形式で出力してください：

ルール：
1. 問題文に「2つ選べ」などの指示がない限り、必ず1つだけ選択してください
2. 複数選択の場合は、選択肢をアルファベット順に並べて出力してください（例：ac, ce）

answer: [選択した回答のアルファベット]
confidence: [0.0-1.0の確信度]
explanation: [回答の理由を簡潔に]"""

        self.models = {
            "o1": {
                "api_key": os.getenv("OPENAI_API_KEY"),
                "model_name": "o1",
                "client_type": "openai",
                "supports_vision": True,
                "system_prompt": self.system_prompt,
                "parameters": {}
            },
            "gpt-4o": {
                "api_key": os.getenv("OPENAI_API_KEY"),
                "model_name": "gpt-4o",
                "client_type": "openai",
                "supports_vision": True,
                "system_prompt": self.system_prompt,
                "parameters": {}
            },
            "o3-mini": {
                "api_key": os.getenv("OPENAI_API_KEY"),
                "model_name": "o3-mini",
                "client_type": "openai",
                "supports_vision": False,
                "system_prompt": self.system_prompt,
                "parameters": {"reasoning_effort": "high"}
            },
            "gemini": {
                "api_key": os.getenv("GEMINI_API_KEY"),
                "base_url": "https://generativelanguage.googleapis.com/v1beta/",
                "model_name": "gemini-2.0-flash-001",
                "client_type": "openai",
                "supports_vision": True,
                "system_prompt": self.system_prompt,
                "parameters": {}
            },
            "deepseek": {
                "api_key": os.getenv("DEEPSEEK_API_KEY"),
                "base_url": os.getenv("DEEPSEEK_ENDPOINT"),
                "model_name": "DeepSeek-R1",
                "client_type": "openai",
                "supports_vision": False,
                "system_prompt": self.system_prompt,
                "parameters": {}
            },
            "claude": {
                "api_key": os.getenv("ANTHROPIC_API_KEY"),
                "model_name": "claude-3-5-sonnet-20241022",
                "client_type": "anthropic",
                "supports_vision": True,
                "system_prompt": self.system_prompt,
                "parameters": {
                    "temperature": 0.2,
                    "max_tokens": 1000
                }
            }
        }

    def get_client(self, model_key: str):
        """モデルに応じたクライアントを返す"""
        config = self.models[model_key]
        
        if config["client_type"] == "anthropic":
            return anthropic.Anthropic(api_key=config["api_key"])
        else:
            if "base_url" in config:
                return OpenAI(api_key=config["api_key"], base_url=config["base_url"])
            return OpenAI(api_key=config["api_key"])

    def solve_question(self, question: Dict, model_key: str) -> Dict:
        """1つの問題を解く"""
        config = self.models[model_key]
        client = self.get_client(model_key)

        # 問題文を構築
        prompt = f"""問題：{question['question']}

選択肢：
{chr(10).join(question['choices'])}

回答を指定された形式で出力してください。"""

        try:
            if config["client_type"] == "anthropic":
                response = client.messages.create(
                    model=config["model_name"],
                    messages=[{"role": "user", "content": prompt}],
                    system=config["system_prompt"],
                    **config["parameters"]
                )
                raw_response = response.content[0].text
            else:
                # OpenAI APIの呼び出しを修正
                messages = [
                    {"role": "system", "content": config["system_prompt"]},
                    {"role": "user", "content": prompt}
                ]

                # 画像がある場合の処理を追加
                if config["supports_vision"] and question.get("has_image", False):
                    image_paths = self.get_question_images(question["number"])
                    content = [{"type": "text", "text": prompt}]
                    for image_path in image_paths:
                        base64_image = self.encode_image(image_path)
                        content.append({
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        })
                    messages = [
                        {"role": "system", "content": config["system_prompt"]},
                        {"role": "user", "content": content}
                    ]

                response = client.chat.completions.create(
                    model=config["model_name"],
                    messages=messages,
                    **config["parameters"]
                )
                raw_response = response.choices[0].message.content

            # レスポンスを受け取った直後にログ出力
            print(f"モデル {model_key} からの生の応答:")
            print(raw_response)
            
            # 応答を構造化データに変換
            parsed_response = self.output_processor.format_model_response(raw_response)
            
            return {
                "model_used": model_key,
                "raw_response": raw_response,
                "answer": parsed_response["answer"],
                "confidence": parsed_response["confidence"],
                "explanation": parsed_response["explanation"],
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            print(f"モデル {model_key} でエラーが発生: {str(e)}")
            return {
                "model_used": model_key,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    def process_questions(self, questions: List[Dict], models: Optional[List[str]] = None) -> List[Dict]:
        """全ての問題を処理"""
        if models is None:
            models = list(self.models.keys())

        results = []
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 進捗バーを追加
        for question in tqdm(questions, desc="問題を処理中"):
            question_result = {
                "question": question,
                "answers": []
            }

            for model in tqdm(models, desc=f"問題 {question['number']} をモデルで解析中", leave=False):
                answer = self.solve_question(question, model)
                question_result["answers"].append(answer)

            results.append(question_result)
            
            # 問題ごとに同じファイルに追記形式で保存
            try:
                self.output_processor.process_outputs(results, timestamp)
                print(f"問題 {question['number']} の結果を保存しました")
            except Exception as e:
                print(f"結果の保存中にエラーが発生: {str(e)}")

        return results 

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

    def encode_image(self, image_path):
        """画像をbase64エンコード"""
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8") 