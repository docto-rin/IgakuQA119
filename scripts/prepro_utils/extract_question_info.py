import json
import csv
import glob
from pathlib import Path

def extract_question_info(json_file_path):
    # JSONファイルを読み込む
    with open(json_file_path, 'r', encoding='utf-8') as f:
        questions = json.load(f)
    
    # 出力用のデータを準備
    question_info = []
    for question in questions:
        question_info.append({
            'question_number': question['number'],
            'has_image': question['has_image']
        })
    
    return question_info

def save_to_csv(question_info, output_path):
    # CSVファイルに書き出し
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['question_number', 'has_image'])
        writer.writeheader()
        writer.writerows(question_info)

def main():
    # questionディレクトリ内の全てのJSONファイルを処理
    json_files = glob.glob('question/*_json.json')
    
    for json_file in json_files:
        # 出力ファイル名を生成
        base_name = Path(json_file).stem.replace('_json', '')
        output_path = f'csv/{base_name}_question_info.csv'
        
        # 処理を実行
        question_info = extract_question_info(json_file)
        save_to_csv(question_info, output_path)
        print(f'Processed {json_file} -> {output_path}')

if __name__ == '__main__':
    main() 