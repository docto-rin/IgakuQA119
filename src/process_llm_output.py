import json
import os
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

class OutputProcessor:
    def __init__(self):
        os.makedirs("./answers", exist_ok=True)

    def format_model_response(self, response: str) -> Tuple[Dict[str, Any], bool]:
        """
        モデルの生の応答をパースして構造化データに変換し、
        解析が成功したかどうかをboolで返す。
        """
        result = {
            "answer": None,
            "confidence": None,
            "explanation": None
        }
        
        success = True  # 解析が成功したかのフラグ
        
        # 応答が空または無効な場合
        if not response or not isinstance(response, str):
            print(f"警告: 無効な応答を受け取りました: {response}")
            return result, False
        
        try:
            lines = response.lower().split('\n')
            for line in lines:
                line = line.strip()
                if line.startswith('answer:'):
                    result['answer'] = line.replace('answer:', '').strip()
                elif line.startswith('confidence:'):
                    try:
                        confidence = float(line.replace('confidence:', '').strip())
                        result['confidence'] = confidence
                    except ValueError:
                        print(f"警告: 確信度の変換に失敗: {line}")
                elif line.startswith('explanation:'):
                    result['explanation'] = line.replace('explanation:', '').strip()
                    
            # 必須フィールド（answer）がなければ失敗と判断
            if not result['answer']:
                print("警告: 回答が見つかりませんでした")
                success = False
                
        except Exception as e:
            print(f"例外発生: 応答解析中にエラー: {e}")
            success = False
        
        return result, success

    def save_json_output(self, results: List[Dict], file_exp: str) -> None:
        """結果をJSONファイルとして保存"""
        output_file = f"./answers/{file_exp}.json"
        
        # JSON用にデータを整形
        formatted_results = []
        for result in results:
            formatted_result = {
                "question_number": result["question"]["number"],
                "question_text": result["question"]["question"],
                "choices": result["question"]["choices"],
                "has_image": result["question"].get("has_image", False),
                "answers": []
            }
            
            for answer in result["answers"]:
                formatted_answer = {
                    "model": answer["model_used"],
                    "timestamp": answer.get("timestamp", ""),
                }
                
                if "error" in answer:
                    formatted_answer["error"] = answer["error"]
                else:
                    formatted_answer["answer"] = answer.get("answer")
                    formatted_answer["confidence"] = answer.get("confidence")
                    if "explanation" in answer:
                        formatted_answer["explanation"] = answer["explanation"]
                
                formatted_result["answers"].append(formatted_answer)
            
            formatted_results.append(formatted_result)

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump({
                "experiment_id": file_exp,
                "results": formatted_results
            }, f, ensure_ascii=False, indent=2)

    def process_outputs(self, results: List[Dict], file_exp: str = None) -> None:
        """全ての結果を処理"""
        if file_exp is None:
            file_exp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # JSON形式で1つのファイルに保存
        self.save_json_output(results, file_exp)