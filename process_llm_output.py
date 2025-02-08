import json
import os
from datetime import datetime
from typing import List, Dict, Any, Optional

class OutputProcessor:
    def __init__(self):
        # 出力ディレクトリの作成
        os.makedirs("answer", exist_ok=True)
        os.makedirs("answer/raw", exist_ok=True)
        os.makedirs("answer/json", exist_ok=True)

    def format_model_response(self, response: str) -> Dict[str, Any]:
        """モデルの生の応答をパースして構造化データに変換"""
        result = {
            "answer": None,
            "confidence": None,
            "explanation": None
        }
        
        # 応答が空または無効な場合のチェック
        if not response or not isinstance(response, str):
            print(f"警告: 無効な応答を受け取りました: {response}")
            return result
        
        try:
            # 応答テキストを行ごとに処理
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
                
            # 必須フィールドのチェック
            if not result['answer']:
                print("警告: 回答が見つかりませんでした")
            
            return result
            
        except Exception as e:
            print(f"応答のパース中にエラーが発生: {str(e)}")
            print(f"問題の応答: {response}")
            return result

    def save_model_output(self, results: List[Dict], model_name: str, timestamp: str) -> None:
        """モデルごとの出力を読みやすい形式でテキストファイルに保存"""
        output_file = f"answer/raw/{model_name}_{timestamp}.txt"

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

    def save_json_output(self, results: List[Dict], timestamp: str) -> None:
        """結果をJSONファイルとしても保存"""
        output_file = f"answer/json/results_{timestamp}.json"
        
        # JSON用にデータを整形
        formatted_results = []
        for result in results:
            formatted_result = {
                "question_number": result["question"]["number"],
                "question_text": result["question"]["question"],
                "choices": result["question"]["choices"],
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
                "timestamp": timestamp,
                "results": formatted_results
            }, f, ensure_ascii=False, indent=2)

    def clean_old_files(self, timestamp: str) -> None:
        """古い出力ファイルを削除"""
        # rawディレクトリの古いファイルを削除
        raw_dir = "answer/raw"
        for file in os.listdir(raw_dir):
            if file.endswith(".txt") and not file.endswith(f"{timestamp}.txt"):
                os.remove(os.path.join(raw_dir, file))

        # jsonディレクトリの古いファイルを削除
        json_dir = "answer/json"
        for file in os.listdir(json_dir):
            if file.endswith(".json") and not file.endswith(f"{timestamp}.json"):
                os.remove(os.path.join(json_dir, file))

    def process_outputs(self, results: List[Dict], timestamp: str = None) -> None:
        """全ての結果を処理"""
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 古いファイルを削除
        self.clean_old_files(timestamp)
        
        # 使用されているすべてのモデルを特定
        models = set()
        for result in results:
            for answer in result["answers"]:
                models.add(answer["model_used"])
        
        # モデルごとに1つのファイルに出力
        for model in models:
            self.save_model_output(results, model, timestamp)
        
        # JSON形式で1つのファイルに保存
        self.save_json_output(results, timestamp) 