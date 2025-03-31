import json
import os
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

class OutputProcessor:
    def __init__(self):
        # 出力ディレクトリの作成
        os.makedirs("answer", exist_ok=True)
        os.makedirs("answer/raw", exist_ok=True)
        os.makedirs("answer/json", exist_ok=True)

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


    def save_model_output(self, results: List[Dict], model_name: str, file_exp: str) -> None:
        """モデルごとの出力を読みやすい形式でテキストファイルに保存"""
        output_file = f"answer/raw/{file_exp}.txt"

        with open(output_file, "a", encoding="utf-8") as f:
            for result in results:
                question = result["question"]
                answers = result["answers"]
                
                # 問題情報のフォーマット
                f.write(f"{'='*50}\n")
                f.write(f"問題 {question['number']}\n")
                f.write(f"{'='*50}\n\n")
                
                # 問題文
                f.write("【問題文】\n")
                f.write(f"{question['question']}\n\n")
                
                # 選択肢
                f.write("【選択肢】\n")
                for choice in question["choices"]:
                    f.write(f"{choice}\n")
                f.write("\n")
                
                # モデルの回答
                f.write("【回答】\n")
                for answer in answers:
                    if answer["model_used"] == model_name:
                        if "error" in answer:
                            f.write(f"エラー: {answer['error']}\n")
                        else:
                            f.write(f"選択した回答: {answer['answer']}\n")
                            if answer.get('confidence'):
                                f.write(f"確信度: {answer['confidence']}\n")
                            if answer.get('explanation'):
                                f.write(f"説明: {answer['explanation']}\n")
                
                f.write("\n")

    def save_json_output(self, results: List[Dict], file_exp: str) -> None:
        """結果をJSONファイルとしても保存"""
        output_file = f"answer/json/{file_exp}.json"
        
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
                "experiment_id": file_exp,  # 'timestamp'から'experiment_id'に変更
                "results": formatted_results
            }, f, ensure_ascii=False, indent=2)

    def clean_old_files(self, file_exp: str) -> None:
        """同じ実験名のファイルのみを削除（他の実験は保持）"""
        # rawディレクトリの同じ実験のファイルを削除
        raw_dir = "answer/raw"
        for file in os.listdir(raw_dir):
            if file.endswith(f"_{file_exp}.txt"):
                os.remove(os.path.join(raw_dir, file))

        # jsonディレクトリの同じ実験のファイルを削除
        json_dir = "answer/json"
        for file in os.listdir(json_dir):
            if file.endswith(f"_{file_exp}.json"):
                os.remove(os.path.join(json_dir, file))

    def process_outputs(self, results: List[Dict], file_exp: str = None) -> None:
        """全ての結果を処理"""
        if file_exp is None:
            file_exp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 削除機能の呼び出しをコメントアウト
        # self.clean_old_files(file_exp)
        
        # 使用されているすべてのモデルを特定
        models = set()
        for result in results:
            for answer in result["answers"]:
                models.add(answer["model_used"])
        
        # モデルごとに1つのファイルに出力
        for model in models:
            self.save_model_output(results, model, file_exp)
        
        # JSON形式で1つのファイルに保存
        self.save_json_output(results, file_exp)