import os
import copy
import traceback
from typing import Dict, List, Optional, Any
import time
import random
import anthropic
from openai import OpenAI
from datetime import datetime
from dotenv import load_dotenv
import textwrap
from tqdm import tqdm
import glob
import base64
from .process_llm_output import OutputProcessor

load_dotenv()

class LLMSolver:
    def __init__(self, system_prompt_type="default"):
        self.output_processor = OutputProcessor()

        if system_prompt_type == "default":
            self.system_prompt = textwrap.dedent("""\
                あなたは医師国家試験問題を解く優秀で論理的なアシスタントです。
                以下のルールを守って、問題文と選択肢（または数値入力の指示）を確認し、回答してください。

                【ルール】
                1. 明示的な指示がない場合は、単一選択肢のみ選ぶ（例: "a", "d"）。
                2. 「2つ選べ」「3つ選べ」などとあれば、その数だけ選択肢をアルファベット順で列挙する（例: "ac", "bd"）。
                3. 選択肢が存在せず数値入力が求められる場合は、指定がない限りそのままの数値を答える（例: answer: 42）。
                4. 画像（has_image=True）は参考情報とし、特別な形式は不要。
                5. 不要な装飾やMarkdown記法は含めず、以下の形式に従って厳密に出力してください：

                answer: [選んだ回答(単数/複数/数値)]
                confidence: [0.0～1.0の確信度]
                explanation: [選択理由や重要な根拠を簡潔に]

                【answerについて注意】
                - 問題は単数選択、複数選択、数値入力のいずれかであり、問題文からその形式を判断する。
                - 「どれか。」で終わる選択問題で数が明記されていない場合は、五者択一を意味するので選択肢を必ず1つだけ選び小文字のアルファベットで回答する。（単数選択）
                - 「2つ選べ」「3つ選べ」などと書いてある場合に限り、指定された数だけの複数選択肢を選び、小文字のアルファベット順（abcde順）に並び替えて列挙する。（複数選択）
                - 選択肢が存在しない場合は、小数や四捨五入など、問題文で特に指示があればそれに従い、選択肢記号ではなく数値を回答する。（数値入力）
                - 問題に関連しない余計な文は書かず、指定のキー(answer, confidence, explanation)を上記の出力に従って厳密に出力する。
            """)
        else:
            raise ValueError("無効なsystem_prompt_typeが指定されました。")

        self.models: Dict[str, Dict[str, Any]] = {
            "o1": {
                "api_key": os.getenv("OPENAI_API_KEY"),
                "model_name": "o1",
                "client_type": "openai",
                "supports_vision": True,
                "system_role": "system",
                "system_prompt": self.system_prompt,
                "parameters": {}
            },
            "gpt-4o": {
                "api_key": os.getenv("OPENAI_API_KEY"),
                "model_name": "gpt-4o",
                "client_type": "openai",
                "supports_vision": True,
                "system_role": "system",
                "system_prompt": self.system_prompt,
                "parameters": {}
            },
            "o3-mini": {
                "api_key": os.getenv("OPENAI_API_KEY"),
                "model_name": "o3-mini",
                "client_type": "openai",
                "supports_vision": False,
                "system_role": "system",
                "system_prompt": self.system_prompt,
                "parameters": {"reasoning_effort": "high"}
            },
            "claude-3.5-sonnet": {
                "api_key": os.getenv("ANTHROPIC_API_KEY"),
                "model_name": "claude-3-5-sonnet-20241022",
                "client_type": "anthropic",
                "supports_vision": True,
                "system_role": "system",
                "system_prompt": self.system_prompt,
                "parameters": {
                    "temperature": 0.2,
                    "max_tokens": 1000
                }
            },
            "gemma-3": {
                "api_key": os.getenv("GEMINI_API_KEY"),
                "base_url": "https://generativelanguage.googleapis.com/v1beta/",
                "model_name": "gemma-3-27b-it",
                "client_type": "openai",
                "supports_vision": False,
                "system_role": "user",
                "system_prompt": self.system_prompt,
                "parameters": {}
            },
            "plamo-1.0-prime": {
                "api_key": os.getenv("PLAMO_API_KEY"),
                "base_url": "https://platform.preferredai.jp/api/completion/v1",
                "model_name": "plamo-1.0-prime",
                "client_type": "openai",
                "supports_vision": False,
                "system_role": "user",
                "system_prompt": self.system_prompt,
                "parameters": {}
            },
            "gemini-flexible": {
                "api_key": os.getenv("GEMINI_API_KEY"),
                "base_url": "https://generativelanguage.googleapis.com/v1beta/",
                "model_name": None,
                "client_type": "openai",
                "supports_vision": True,
                "system_role": "system",
                "system_prompt": self.system_prompt,
                "parameters": {}
            },
            "openrouter-flexible": {
                "api_key": os.getenv("OPENROUTER_API_KEY"),
                "base_url": "https://openrouter.ai/api/v1",
                "model_name": None,
                "client_type": "openai",
                "supports_vision": False,
                "system_role": "system",
                "system_prompt": self.system_prompt,
                "parameters": {},
                "extra_headers": {
                    "HTTP-Referer": os.getenv("OPENROUTER_HTTP_REFERER", "https://yourapp.example"),
                    "X-Title": os.getenv("OPENROUTER_APP_TITLE", "LLMSolver")
                },
                "extra_body": {
                    "enable_thinking": True
                }
            },
            "ollama-flexible": {
                "api_key": "ollama",
                "base_url": "http://localhost:11434/v1",
                "model_name": None,
                "client_type": "openai",
                "supports_vision": False,
                "system_role": "system",
                "system_prompt": self.system_prompt,
                "parameters": {}
            }
        }

    def get_config_and_client(self, model_key: str):
        """モデルキーから設定とクライアントを返す。未定義キーは柔軟に解決"""

        if model_key in self.models:
            config = self.models[model_key]
        else:
            if model_key.startswith("gemini-"):
                config = copy.deepcopy(self.models["gemini-flexible"])
                config["model_name"] = model_key
            elif model_key.startswith("hf.co/") or model_key.startswith("huggingface.co/"):
                config = copy.deepcopy(self.models["ollama-flexible"])
                config["model_name"] = model_key
            elif model_key.startswith("ollama-"):
                config = copy.deepcopy(self.models["ollama-flexible"])
                config["model_name"] = model_key.split("ollama-", 1)[1]
            elif model_key.startswith("openrouter-"):
                config = copy.deepcopy(self.models["openrouter-flexible"])
                config["model_name"] = model_key.split("openrouter-", 1)[1]
            else:
                raise ValueError(f"未定義のモデルキーです: {model_key}")

        if config["client_type"] == "anthropic":
            client = anthropic.Anthropic(api_key=config["api_key"])
            return config, client
        else:
            client_args = {
                "api_key": config["api_key"]
            }
            if "base_url" in config and config["base_url"]:
                client_args["base_url"] = config["base_url"]
            if config.get("extra_headers"):
                client_args["default_headers"] = config["extra_headers"]

            client = OpenAI(**client_args)
            return config, client

    def solve_question(self, question: Dict, model_key: str, supports_vision_override: Optional[bool] = None) -> Dict:
        """1つの問題を解く"""
        config, client = self.get_config_and_client(model_key)

        if supports_vision_override is not None:
            print(f"Overriding supports_vision for {model_key} to {supports_vision_override}")
            config["supports_vision"] = supports_vision_override

        prompt = f"""問題：{question['question']}

選択肢：
{chr(10).join(question['choices'])}

回答を指定された形式で出力してください。"""

        max_retries = 5
        base_wait_time = 1

        for attempt in range(max_retries):
            try:
                if config["client_type"] == "anthropic":
                    response = client.messages.create(
                        model=config["model_name"],
                        messages=[{"role": "user", "content": prompt}],
                        system=config["system_prompt"],
                        **config["parameters"]
                    )
                    raw_response = response.content[0].text
                    cot = None
                else:
                    messages = [
                        {"role": config["system_role"], "content": config["system_prompt"]},
                        {"role": "user", "content": prompt}
                    ]

                    if config["supports_vision"] and question.get("has_image", False):
                        image_paths = self.get_question_images(question["number"])
                        if not image_paths:
                             print(f"Warning: Question {question['number']} has 'has_image=True' but no image files found.")
                             messages = [
                                {"role": config["system_role"], "content": config["system_prompt"]},
                                {"role": "user", "content": prompt}
                             ]
                        else:
                            content = [{"type": "text", "text": prompt}]
                            for image_path in image_paths:
                                try:
                                    base64_image = self.encode_image(image_path)
                                    content.append({
                                        "type": "image_url",
                                        "image_url": {
                                            "url": f"data:image/jpeg;base64,{base64_image}"
                                        }
                                    })
                                except FileNotFoundError:
                                    print(f"Warning: Image file not found: {image_path}")
                                    continue

                            if len(content) == 1:
                                print(f"Warning: No valid images found for question {question['number']} despite supports_vision=True and has_image=True.")
                                messages = [
                                    {"role": config["system_role"], "content": config["system_prompt"]},
                                    {"role": "user", "content": prompt}
                                ]
                            else:
                                messages = [
                                    {"role": config["system_role"], "content": config["system_prompt"]},
                                    {"role": "user", "content": content}
                                ]
                    else:
                         messages = [
                            {"role": config["system_role"], "content": config["system_prompt"]},
                            {"role": "user", "content": prompt}
                         ]


                    response = client.chat.completions.create(
                        model=config["model_name"],
                        messages=messages,
                        extra_body=config.get("extra_body", {}),
                        **config["parameters"]
                    )
                    raw_response = response.choices[0].message.content
                    cot = response.choices[0].message.reasoning if hasattr(response.choices[0].message, 'reasoning') else None

                print(f"モデル {model_key} からの応答:")
                if cot:
                    print(f"CoT: {cot}")
                print(f"生の応答: {raw_response}")

                formatted_response, success = self.output_processor.format_model_response(raw_response, cot)

                num_post_attempts = 0
                max_post_attempts = 3
                while not success and num_post_attempts < max_post_attempts:
                    print("ポストプロセスを開始します。")
                    fixed_response = self.perform_postprocessing(raw_response, config, client)
                    formatted_response, success = self.output_processor.format_model_response(fixed_response, cot)
                    num_post_attempts += 1
                    print(f"ポストプロセス試行回数: {num_post_attempts}")
                    if success:
                        print("ポストプロセス成功")
                        print(f"ポストプロセス後の応答: {fixed_response}")
                        raw_response = fixed_response
                    else:
                        raise Exception("ポストプロセスに失敗しました")

                return {
                    "model_used": model_key,
                    "raw_response": raw_response,
                    "answer": formatted_response["answer"],
                    "confidence": formatted_response["confidence"],
                    "explanation": formatted_response["explanation"],
                    "cot": formatted_response["cot"],
                    "timestamp": datetime.now().isoformat()
                }

            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = base_wait_time * (2 ** attempt) + random.uniform(0, 1)
                    print(f"モデル {model_key} でエラー発生: {e}. {wait_time:.2f}秒待機してリトライします ({attempt + 1}/{max_retries})")
                    time.sleep(wait_time)
                else:
                    print(f"モデル {model_key} で最大リトライ回数 ({max_retries}) に達しました。最終エラー: {e}")
                    traceback.print_exc()
                    return {
                        "model_used": model_key,
                        "error": repr(e),
                        "traceback": traceback.format_exc(),
                        "timestamp": datetime.now().isoformat()
                    }

    def process_questions(self, questions: List[Dict], model_name: str, file_exp: Optional[str] = None, supports_vision_override_str: Optional[str] = None) -> List[Dict]:
        """全ての問題を処理"""
        results = []
        if file_exp is None:
            file_exp = datetime.now().strftime("%Y%m%d_%H%M%S")

        supports_vision_override: Optional[bool] = None
        if supports_vision_override_str is not None:
            if supports_vision_override_str.lower() == 'true':
                supports_vision_override = True
            elif supports_vision_override_str.lower() == 'false':
                supports_vision_override = False
            else:
                print(f"Warning: Invalid value for --supports_vision '{supports_vision_override_str}'. Expected 'true' or 'false'. Ignoring override.")

        for question in tqdm(questions, desc="ブロックを処理中"):
            print(f"問題 {question['number']} を解答中")
            question_result = {
                "question": question,
                "answers": []
            }

            answer = self.solve_question(question, model_name, supports_vision_override=supports_vision_override)
            question_result["answers"].append(answer)

            results.append(question_result)

            try:
                self.output_processor.process_outputs(results, file_exp)
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

    def perform_postprocessing(self, raw_response: str, config: Dict[str, Any], client) -> str:
        """
        モデルに対してシステムプロンプトを再提示し、
        応答を指定フォーマットに従って再整形させる（ポストプロセス）。

        :param raw_response: モデルの初回の未整形応答
        :param config: モデル設定やシステムプロンプトを含む辞書
        :param client: APIを呼び出すためのクライアントインスタンス
        :return: モデルが再整形したテキスト応答（fixed_response）
        """
        system_role = config["system_role"]
        system_prompt = config["system_prompt"]

        retry_prompt = f"""
    以下の「整形前の応答」を、「整形方法の指示」にて指定された形式に厳密に整形してください。
    【整形前の応答】
    {raw_response}

    【整形方法の指示】
    {system_prompt}
    """

        messages = [
            {"role": system_role, "content": system_prompt},
            {"role": "user", "content": retry_prompt}
        ]

        max_retries = 3
        base_wait_time = 1

        for attempt in range(max_retries):
            try:
                if config["client_type"] == "anthropic":
                     response = client.messages.create(
                        model=config["model_name"],
                        messages=[{"role": "user", "content": retry_prompt}],
                        system=config["system_prompt"],
                        **config["parameters"]
                    )
                     fixed_response = response.content[0].text
                else:
                    response = client.chat.completions.create(
                        model=config["model_name"],
                        messages=messages,
                        extra_body=config.get("extra_body", {}),
                        **config["parameters"]
                    )
                    fixed_response = response.choices[0].message.content

                return fixed_response

            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = base_wait_time * (2 ** attempt) + random.uniform(0, 1)
                    print(f"ポストプロセス中にAPIエラー発生: {e}. {wait_time:.2f}秒待機してリトライします ({attempt + 1}/{max_retries})")
                    time.sleep(wait_time)
                else:
                    print(f"ポストプロセス中のAPI呼び出しで最大リトライ回数 ({max_retries}) に達しました。最終エラー: {e}")
                    return ""