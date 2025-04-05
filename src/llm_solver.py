import os
from typing import Dict, List, Optional, Any
import anthropic
from openai import OpenAI
from datetime import datetime
from dotenv import load_dotenv
from .process_llm_output import OutputProcessor
import textwrap
from tqdm import tqdm
import glob
import base64

load_dotenv()

class LLMSolver:
    def __init__(self, system_prompt_type="v2"):
        self.output_processor = OutputProcessor()
        
        # 共通のシステムプロンプトを定義
        if system_prompt_type == "v1":
            self.system_prompt = textwrap.dedent("""\
                あなたは医師国家試験の問題を解く非常に優秀で論理的なアシスタントです。
                以下のルールに従い、問題文と選択肢（または数値入力の指示）を確認した上で回答してください。

                【ルール】
                1. 問題文内に「2つ選べ」「3つ選べ」などの記載がある場合は、その数だけ選択肢を選び、アルファベットを昇順で列挙してください(例: 「2つ選べ」→ "ac")。
                2. 問題文に明示的な指示がない場合は、単一の選択肢(a, b, c, d, eなど)のみを選んでください。
                3. 数値入力が求められる問題では、選択肢がなく数値だけを回答する場合があります。その場合はanswerを数値で示してください(例: answer: 42)。
                4. 画像が提示されている場合（has_image=True）でも、特別な形式は必要ありません。問題文の内容に含まれる情報として適宜参照してください。
                5. 不要な装飾やMarkdown記法は含めず、以下の形式に従って厳密に出力してください：

                answer: [選んだ回答(単数 or 複数 or 数値)]
                confidence: [0.0～1.0の確信度(例: 0.85)]
                explanation: [その回答を選んだ理由を簡潔に、重要な医学的根拠や推論過程をまとめて記載]

                【注意】
                - 回答のうち、confidence はあなたの推定で構いませんが、0.0～1.0の範囲にしてください。
                - 複数選択の場合、アルファベット順に並べてください。(例: "ac", "bd")
                - 数値が小数になる場合などは、問題文の指示(四捨五入など)に従ってください。
                - 問題に関連しない余計な文は書かず、指定のキー(answer, confidence, explanation)のみ出力してください。
            """)
        elif system_prompt_type == "v2":
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
        
        self.models = {
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
            "deepseek": {
                "api_key": os.getenv("DEEPSEEK_API_KEY"),
                "base_url": os.getenv("DEEPSEEK_ENDPOINT"),
                "model_name": "DeepSeek-R1",
                "client_type": "openai",
                "supports_vision": False,
                "system_role": "system",
                "system_prompt": self.system_prompt,
                "parameters": {}
            },
            "claude": {
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
            "plamo-1.0-prime": {
                "api_key": os.getenv("PLAMO_API_KEY"), # Use PLAMO_API_KEY environment variable
                "base_url": "https://platform.preferredai.jp/api/completion/v1", # PLaMo API endpoint
                "model_name": "plamo-1.0-prime", # Specific PLaMo model
                "client_type": "openai", # OpenAI compatible API
                "supports_vision": False, # PLaMo currently does not support vision
                "system_role": "user", # Improved performance compared to 'system'
                "system_prompt": self.system_prompt,
                "parameters": {} # Add any specific PLaMo parameters here if needed
            },
            # Geminiの柔軟な指定用エントリー
            "gemini-flexible": {
                "api_key": os.getenv("GEMINI_API_KEY"),
                "base_url": "https://generativelanguage.googleapis.com/v1beta/",
                "model_name": None,  # 使用時に動的に設定
                "client_type": "openai",
                "supports_vision": True,
                "system_role": "system",
                "system_prompt": self.system_prompt,
                "parameters": {}
            },
            # Ollamaの柔軟な指定用エントリー
            "ollama-flexible": {
                "api_key": "ollama",  # ダミーAPIキー（実際は使用されない）
                "base_url": "http://localhost:11434/v1",
                "model_name": None,  # 使用時に動的に設定
                "client_type": "openai",
                "supports_vision": False,
                "system_role": "system",
                "system_prompt": self.system_prompt,
                "parameters": {}
            }
        }

    def get_config_and_client(self, model_key: str):
        # ハードコーディングされたモデルキーの場合
        if model_key in self.models:
            config = self.models[model_key]
            # もしクライアントタイプが "anthropic" ならAnthropicクライアントを使用
            if config["client_type"] == "anthropic":
                return config, anthropic.Anthropic(api_key=config["api_key"])
            # それ以外の場合はOpenAIクライアントを使用
            else:
                if "base_url" in config:
                    return config, OpenAI(api_key=config["api_key"], base_url=config["base_url"])
                return config, OpenAI(api_key=config["api_key"])

        # モデルキーが直接定義されていない場合の処理
        else:
            # もしモデルキーが "gemini-" で始まるならgemini-flexibleを使用
            if model_key.startswith("gemini-"):
                config = self.models["gemini-flexible"].copy()
                config["model_name"] = model_key
                return config, OpenAI(api_key=config["api_key"], base_url=config["base_url"])
            # もしモデルキーが "ollama-" で始まるならollama-flexibleを使用
            elif model_key.startswith("ollama-"):
                config = self.models["ollama-flexible"].copy()
                # "ollama-"以降の文字列を真のモデル名として設定
                config["model_name"] = model_key.split("ollama-", 1)[1]
                return config, OpenAI(api_key=config["api_key"], base_url=config["base_url"])
            # hf.co/ で始まる場合の対応（直接URL指定）
            elif model_key.startswith("huggingface.co/"):
                config = self.models["ollama-flexible"].copy()
                # モデル名にフルURLを渡す
                config["model_name"] = model_key
                return config, OpenAI(api_key=config["api_key"], base_url=config["base_url"])
            else:
                raise ValueError(f"未定義のモデルキーです: {model_key}")

    def solve_question(self, question: Dict, model_key: str) -> Dict:
        """1つの問題を解く"""
        config, client = self.get_config_and_client(model_key)

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
                    {"role": config["system_role"], "content": config["system_prompt"]},
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
                        {"role": config["system_role"], "content": config["system_prompt"]},
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
            formatted_response, success = self.output_processor.format_model_response(raw_response)

            # 解析が失敗した場合にポストプロセスを実施
            num_attempts = 0
            max_attempts = 3
            while not success and num_attempts < max_attempts:
                print("ポストプロセスを開始します。")
                fixed_response = self.perform_postprocessing(raw_response, config, client)
                formatted_response, success = self.output_processor.format_model_response(fixed_response)
                num_attempts += 1
                print(f"ポストプロセス試行回数: {num_attempts}")
                if success:
                    print("ポストプロセス成功")
                    print(f"ポストプロセス後の応答: {fixed_response}")
                else:
                    print("ポストプロセス失敗")

            return {
                "model_used": model_key,
                "raw_response": raw_response,
                "answer": formatted_response["answer"],
                "confidence": formatted_response["confidence"],
                "explanation": formatted_response["explanation"],
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            print(f"モデル {model_key} でエラーが発生: {str(e)}")
            return {
                "model_used": model_key,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    def process_questions(self, questions: List[Dict], models: Optional[List[str]] = None, file_exp: Optional[str] = None) -> List[Dict]:
        """全ての問題を処理"""
        if models is None:
            models = list(self.models.keys())

        results = []
        if file_exp is None:
            file_exp = datetime.now().strftime("%Y%m%d_%H%M%S")

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
                self.output_processor.process_outputs(results, file_exp)
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

        try:
            response = client.chat.completions.create(
                model=config["model_name"],
                messages=messages,
                **config["parameters"]
            )

            fixed_response = response.choices[0].message.content
            return fixed_response

        except Exception as e:
            print(f"ポストプロセス中に例外が発生しました: {e}")
            return ""
