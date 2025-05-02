import json
import os
import re
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

class OutputProcessor:
    def __init__(self):
        os.makedirs("./answers", exist_ok=True)
        self.cot_pattern = re.compile(
            r"<(think|thinking|thoughts)>(.*?)</\1>",
            re.DOTALL | re.IGNORECASE
        )

    def format_model_response(
        self,
        response: str,
        cot: Optional[str] = None
    ) -> Tuple[Dict[str, Any], bool]:
        """
        モデルの生の応答をパースして構造化データに変換し、解析が成功したかどうかを bool で返す。
        - cot が None の場合のみ、応答文字列から <think>〜</think> 等を抽出する
        - cot が与えられている場合はそれを優先し、タグ除去だけ行う
        """
        result: Dict[str, Any] = {
            "answer": None,
            "confidence": None,
            "explanation": None,
            "cot": cot
        }
        success = True

        if not response or not isinstance(response, str):
            print(f"警告: 無効な応答を受け取りました: {response}")
            return result, False

        cleaned_response = response

        if cot is None:
            cot_match = self.cot_pattern.search(response)
            if cot_match:
                print("情報: CoT を抽出しました。")
                result["cot"] = cot_match.group(2).strip()
            else:
                print("情報: CoT タグが見つかりませんでした。")
        cleaned_response = self.cot_pattern.sub("", response).strip()

        try:
            lines = cleaned_response.lower().split('\n')
            temp_expl_lines = []
            found_expl_key = False

            for line in lines:
                line = line.strip()
                if not line:
                    continue

                if line.startswith('answer:'):
                    result['answer'] = line.replace('answer:', '').strip()

                elif line.startswith('confidence:'):
                    conf_str = line.replace('confidence:', '').strip()
                    match = re.search(r"(\d\.?\d*)", conf_str)
                    if match:
                        val = float(match.group(1))
                        result['confidence'] = max(0.0, min(1.0, val))
                    else:
                        print(f"警告: 確信度の数値変換に失敗: {line}")
                        success = False

                elif line.startswith('explanation:'):
                    found_expl_key = True
                    first_line = line.replace('explanation:', '').strip()
                    if first_line:
                        temp_expl_lines.append(first_line)

                elif found_expl_key:
                    temp_expl_lines.append(line)

            if temp_expl_lines:
                result['explanation'] = "\n".join(temp_expl_lines).strip()
            elif found_expl_key:
                result['explanation'] = ""

            if not result['answer']:
                print("警告: answer が見つかりませんでした")
                success = False

            if not found_expl_key and result['explanation'] is None:
                result['explanation'] = cleaned_response

        except Exception as e:
            print(f"例外発生: 応答解析中にエラー: {e}")
            success = False

        return result, success

    def save_json_output(self, results: List[Dict], file_exp: str) -> None:
        """結果をJSONファイルとして保存（cotフィールドも含む）"""
        output_dir = "./answers"
        os.makedirs(output_dir, exist_ok=True)
        safe_file_exp = os.path.basename(file_exp)
        output_file = os.path.join(output_dir, f"{safe_file_exp}.json")

        formatted_results = []
        for result in results:
            if "question" not in result or not isinstance(result["question"], dict):
                print(f"警告: 'question' キーが無効な結果データをスキップします: {result}")
                continue

            formatted_result = {
                "question_number": result["question"].get("number", "Unknown"),
                "question_text": result["question"].get("question", ""),
                "choices": result["question"].get("choices", []),
                "has_image": result["question"].get("has_image", False),
                "answers": []
            }

            if "answers" not in result or not isinstance(result["answers"], list):
                 print(f"警告: 'answers' キーが無効な結果データをスキップします (Question: {formatted_result['question_number']})")
                 formatted_results.append(formatted_result)
                 continue


            for answer in result["answers"]:
                if not isinstance(answer, dict):
                    print(f"警告: 無効な answer 形式です。スキップします: {answer}")
                    continue

                formatted_answer = {
                    "model": answer.get("model_used", "Unknown"),
                    "timestamp": answer.get("timestamp", ""),
                }

                if "error" in answer:
                    formatted_answer["error"] = answer["error"]
                else:
                    formatted_answer["answer"] = answer.get("answer")
                    formatted_answer["confidence"] = answer.get("confidence")
                    formatted_answer["explanation"] = answer.get("explanation")
                    formatted_answer["cot"] = answer.get("cot")

                formatted_result["answers"].append(formatted_answer)

            formatted_results.append(formatted_result)

        try:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump({
                    "experiment_id": file_exp,
                    "results": formatted_results
                }, f, ensure_ascii=False, indent=2)
        except Exception as e:
             print(f"エラー: JSONファイルの書き込み中にエラーが発生しました ({output_file}): {e}")


    def process_outputs(self, results: List[Dict], file_exp: str = None) -> None:
        """全ての結果を処理"""
        if file_exp is None:
            file_exp = datetime.now().strftime("%Y%m%d_%H%M%S")

        self.save_json_output(results, file_exp)