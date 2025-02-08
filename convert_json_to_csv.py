import json
import pandas as pd
import sys
from pathlib import Path

def process_json_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 結果を格納するリスト
    rows = []
    
    # 全てのモデル名を取得
    model_names = set()
    for result in data['results']:
        for answer in result['answers']:
            if 'model_used' in answer:
                model_names.add(answer['model_used'])
    model_names = sorted(list(model_names))
    
    # 各問題についてデータを整理
    for result in data['results']:
        row = {'number': result['question']['number']}
        
        # 画像の有無を追加
        row['has_image'] = result['question']['has_image']
        
        # 各モデルの回答を追加
        model_answers = {model: '' for model in model_names}
        for answer in result['answers']:
            if 'model_used' in answer and 'answer' in answer:
                model_answers[answer['model_used']] = answer['answer']
        
        row.update(model_answers)
        rows.append(row)
    
    return pd.DataFrame(rows)

def main():
    if len(sys.argv) != 2:
        print("使用方法: python convert_json_to_csv.py <JSONファイルパス>")
        sys.exit(1)
    
    json_path = sys.argv[1]
    if not Path(json_path).exists():
        print(f"ファイルが見つかりません: {json_path}")
        sys.exit(1)
    
    # JSONファイルを処理
    df = process_json_file(json_path)
    
    # 出力ディレクトリの作成
    output_dir = Path('csv')
    output_dir.mkdir(exist_ok=True)
    
    # 入力ファイル名から出力ファイル名を生成
    input_filename = Path(json_path).stem
    output_path = output_dir / f'{input_filename}.csv'
    
    # CSVとして保存
    df.to_csv(output_path, index=False)
    print(f"CSVファイルを保存しました: {output_path}")

if __name__ == "__main__":
    main() 