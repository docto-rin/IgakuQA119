import re
import sys
import logging
from datetime import datetime
from pathlib import Path

# パターン定義
PREFIX_NUM = "118"      # 問題番号の接頭辞
PREFIX_ALPHA = "B"      # アルファベット部分

def setup_logging(input_filename: str):
    """ロギングの設定"""
    base_name = Path(input_filename).stem
    log_filename = f'error_log_{base_name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    logging.basicConfig(
        filename=log_filename,
        level=logging.ERROR,
        format='%(asctime)s - %(levelname)s - [%(filename)s] - %(message)s'
    )

def split_choices_by_sequence(choices_text: str, question_num: str) -> list[str]:
    """選択肢を分割"""
    choices = []
    choice_letters = ['a', 'b', '[cC]', 'd', 'e']
    
    # 各選択肢の開始位置を見つける
    start_positions = []
    letter_found = []
    
    for letter in choice_letters:
        match = re.search(f"{letter}", choices_text)
        if match:
            start_positions.append(match.start())
            letter_found.append(letter)
    
    if len(start_positions) < 5:
        logging.warning(f"問題 {question_num}: 一部の選択肢が見つかりません。見つかった選択肢: {letter_found}")
    
    # 開始位置を使って選択肢を分割
    for i in range(len(start_positions)):
        current_start = start_positions[i]
        letter = letter_found[i]
        
        if i < len(start_positions) - 1:
            current_end = start_positions[i + 1]
            content = choices_text[current_start+1:current_end].strip()
        else:
            content = choices_text[current_start+1:].strip()
        
        # 選択肢の記号を小文字に統一
        display_letter = letter.replace('[cC]', 'c').strip('[]')
        choices.append(f"{display_letter}. {content}")
    
    return choices

def split_question_and_choices(text: str, question_num: str, input_file: str) -> tuple[str, list[str]]:
    """問題文と選択肢を分割"""
    try:
        parts = text.split('。a')
        if len(parts) != 2:
            logging.error(f"Input: {input_file} - 問題 {question_num}: 分割エラー - '。a'が{len(parts)-1}個見つかりました")
            return text.strip(), []
        
        question = parts[0] + '。'
        choices_text = 'a' + parts[1]
        choices = split_choices_by_sequence(choices_text, question_num)
        
        if not choices:
            logging.error(f"Input: {input_file} - 問題 {question_num}: 選択肢の分割に失敗しました")
        
        return question.strip(), choices

    except Exception as e:
        logging.error(f"Input: {input_file} - 問題 {question_num}: 予期せぬエラー - {str(e)}")
        return text.strip(), []

def split_questions(text: str, input_file: str, start: int = 1, end: int = 75) -> str:
    clean_text = re.sub(r'=== Page \d+ ===\n', '', text)
    formatted = ''
    
    for i in range(start, end + 1):
        current_pattern = f'{PREFIX_NUM}{PREFIX_ALPHA}{i}'
        next_pattern = f'{PREFIX_NUM}{PREFIX_ALPHA}{i + 1}'
        
        pattern = f'({current_pattern}.*?)(?={PREFIX_NUM}{PREFIX_ALPHA}\\d+|$)'
        match = re.search(pattern, clean_text, re.DOTALL)
        
        if match:
            content = match.group(1).replace(current_pattern, '').strip()
            question, choices = split_question_and_choices(content, current_pattern, input_file)
            
            formatted += f'=== {current_pattern} ===\n'
            formatted += '【問題】\n'
            formatted += question + '\n'
            if choices:
                formatted += '【選択肢】\n'
                formatted += '\n'.join(choices) + '\n'
            formatted += '\n'
    
    return formatted

def main():
    if len(sys.argv) != 3:
        print("Usage: python script.py input.txt output.txt")
        sys.exit(1)
        
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    try:
        input_filename = Path(input_file).name
        setup_logging(input_filename)
        
        with open(input_file, 'r', encoding='utf-8') as f:
            text = f.read()
        
        result = split_questions(text, input_filename)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(result)
            
        print(f"処理が完了しました: {output_file}")
        
    except FileNotFoundError:
        logging.error(f"Input: {input_file} - ファイルが見つかりません")
        print(f"エラー: ファイルが見つかりません: {input_file}")
    except Exception as e:
        logging.error(f"Input: {input_file} - エラーが発生しました: {e}")
        print(f"エラーが発生しました。ログを確認してください。")

if __name__ == '__main__':
    main()