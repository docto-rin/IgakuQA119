import json
import pandas as pd
import sys
import os
from pathlib import Path

def process_json_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 結果を格納するリスト
    rows = []
    
    # 各問題についてデータを整理
    for result in data['results']:
        row = {
            'question_number': result['question_number'],
            'question_text': result['question_text'],
            'choices': ' | '.join(result['choices']) if result['choices'] else ''
        }
        
        # 各モデルの回答とexplanationを追加
        for answer in result['answers']:
            model = answer['model']
            if 'error' in answer:
                row[f'{model}_answer'] = 'ERROR'
                row[f'{model}_explanation'] = answer['error']
            else:
                row[f'{model}_answer'] = answer.get('answer', '')
                row[f'{model}_explanation'] = answer.get('explanation', '')
                row[f'{model}_confidence'] = answer.get('confidence', '')
        
        rows.append(row)
    
    return pd.DataFrame(rows)

def main():
    if len(sys.argv) != 2:
        print("使用方法: python json_to_csv_converter.py <JSONファイルパス>")
        sys.exit(1)
    
    # 入力パスを処理
    input_path = sys.argv[1]
    
    # パスが "dev/nmle-rta" で始まる場合、先頭の "dev/nmle-rta" を削除
    if input_path.startswith('dev/nmle-rta/'):
        input_path = input_path[len('dev/nmle-rta/'):]
    
    # カレントディレクトリを取得
    current_dir = Path.cwd()
    
    # 相対パスを絶対パスに変換
    json_path = (current_dir / input_path).resolve()
    
    if not json_path.exists():
        print(f"ファイルが見つかりません: {json_path}")
        sys.exit(1)
    
    # JSONファイルを処理
    df = process_json_file(json_path)
    
    # スクリプトのディレクトリを取得
    script_dir = Path(__file__).parent.absolute()
    
    # 出力ディレクトリの作成（スクリプトと同じディレクトリ内）
    output_dir = script_dir / 'csv'
    output_dir.mkdir(exist_ok=True)
    
    # 入力ファイル名から出力ファイル名を生成
    input_filename = json_path.stem
    output_path = output_dir / f'{input_filename}.csv'
    
    # CSVとして保存
    df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"CSVファイルを保存しました: {output_path}")

if __name__ == "__main__":
    main() 