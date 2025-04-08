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

    def format_model_response(self, response: str) -> Tuple[Dict[str, Any], bool]:
        """
        モデルの生の応答をパースして構造化データに変換し、
        解析が成功したかどうかをboolで返す。
        CoT部分も抽出する。
        """
        result = {
            "answer": None,
            "confidence": None,
            "explanation": None,
            "cot": None
        }

        success = True  # 解析が成功したかのフラグ

        # 応答が空または無効な場合
        if not response or not isinstance(response, str):
            print(f"警告: 無効な応答を受け取りました: {response}")
            return result, False

        cleaned_response = response # 解析用のレスポンス文字列を準備
        try:
            # --- CoTの抽出 ---
            cot_match = self.cot_pattern.search(response)
            if cot_match:
                print(f"情報: CoTを抽出しました。")
                extracted_cot = cot_match.group(2).strip()
                result['cot'] = extracted_cot
                cleaned_response = self.cot_pattern.sub('', response).strip()
            else:
                print(f"情報: CoTタグが見つかりませんでした。")
                result['cot'] = None

            # --- answer, confidence, explanation の抽出 ---
            # CoT除去後のcleaned_responseを解析対象とする
            lines = cleaned_response.lower().split('\n')
            temp_explanation_lines = [] # explanationが複数行にわたる場合を考慮
            found_explanation_key = False

            for line in lines:
                line = line.strip()
                if not line: # 空行はスキップ
                    continue

                if line.startswith('answer:'):
                    result['answer'] = line.replace('answer:', '').strip()
                elif line.startswith('confidence:'):
                    try:
                        confidence_str = line.replace('confidence:', '').strip()
                        # confidence値に余計な文字が含まれている場合への対応
                        confidence_match = re.search(r"(\d\.?\d*)", confidence_str)
                        if confidence_match:
                             confidence = float(confidence_match.group(1))
                             # 0.0から1.0の範囲にクリップ
                             result['confidence'] = max(0.0, min(1.0, confidence))
                        else:
                             print(f"警告: 確信度の数値変換に失敗 (数値が見つからない): {line}")
                             success = False # 確信度が必須ではない場合、これをコメントアウトしても良い
                    except ValueError:
                        print(f"警告: 確信度の数値変換に失敗: {line}")
                        success = False # 確信度が必須ではない場合、これをコメントアウトしても良い
                elif line.startswith('explanation:'):
                    # explanationキーが見つかったら、それ以降の行をexplanationとして扱う準備
                    found_explanation_key = True
                    # explanation: の後のテキストも最初の行として追加
                    expl_line = line.replace('explanation:', '').strip()
                    if expl_line:
                       temp_explanation_lines.append(expl_line)
                elif found_explanation_key:
                     # explanationキーが見つかった後の行はexplanationの一部として追加
                     temp_explanation_lines.append(line)

            if temp_explanation_lines:
                 result['explanation'] = "\n".join(temp_explanation_lines).strip()
            elif found_explanation_key: # キーは見つかったが内容がない場合
                 result['explanation'] = "" # 空文字列を設定

            # 必須フィールド（answer）がなければ失敗と判断
            if not result['answer']:
                print("警告: answerが見つかりませんでした")
                success = False

            # explanationが見つからなかった場合（キー自体が存在しない場合）
            if not found_explanation_key and 'explanation' not in result:
                 result['explanation'] = cleaned_response # explanationがなければ元の応答（CoT除く）を入れる（暫定対応）

        except Exception as e:
            print(f"例外発生: 応答解析中にエラー: {e}")
            success = False

        return result, success

    def save_json_output(self, results: List[Dict], file_exp: str) -> None:
        """結果をJSONファイルとして保存（cotフィールドも含む）"""
        # ファイルパスの生成（ディレクトリが存在しない場合は作成）
        output_dir = "./answers"
        os.makedirs(output_dir, exist_ok=True)
        # ファイル名からディレクトリ構造を除去 (例: "A/119A_exp" -> "119A_exp")
        safe_file_exp = os.path.basename(file_exp)
        output_file = os.path.join(output_dir, f"{safe_file_exp}.json")

        # JSON用にデータを整形
        formatted_results = []
        for result in results:
            # questionキーとその値が存在するかチェック
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

            # answersキーとその値が存在するかチェック
            if "answers" not in result or not isinstance(result["answers"], list):
                 print(f"警告: 'answers' キーが無効な結果データをスキップします (Question: {formatted_result['question_number']})")
                 formatted_results.append(formatted_result) # answersがなくてもquestion情報は記録
                 continue


            for answer in result["answers"]:
                 # answerが辞書形式かチェック
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

        # JSON形式で1つのファイルに保存
        self.save_json_output(results, file_exp)