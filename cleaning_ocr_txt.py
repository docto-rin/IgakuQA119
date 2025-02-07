import re
import sys
import logging
from datetime import datetime
from pathlib import Path


def extract_pattern_from_filename(filename: str) -> tuple[str, str]:
    """ファイル名からパターンを抽出"""
    match = re.match(r"(\d+)([A-Z])", Path(filename).stem)
    if not match:
        raise ValueError(f"ファイル名が正しい形式ではありません: {filename}")
    return match.group(1), match.group(2)


def setup_logging(input_filename: str):
    """ロギングの設定"""
    base_name = Path(input_filename).stem
    log_filename = (
        f"error_log_{base_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )
    logging.basicConfig(
        filename=log_filename,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - [%(filename)s] - %(message)s",
    )


def is_english_text(text: str) -> bool:
    """テキストが主に英語かどうかを判定"""
    english_chars = sum(1 for c in text if c.isascii() and (c.isalnum() or c.isspace()))
    total_chars = len(text.strip())
    return english_chars / total_chars > 0.7 if total_chars > 0 else False


def tokenize_english(text: str) -> list[str]:
    """英語テキストをトークン化"""
    # 基本的な単語の区切りパターン
    # 1. 選択肢のマーカー(a-e)の前後にスペースを追加
    text = re.sub(r"([a-zA-Z])([abcdeABCDE])([a-zA-Z])", r"\1 \2 \3", text)

    # 2. 大文字の前にスペースを追加（例：WhichisNOT -> Which is NOT）
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)

    # 3. 記号の前後にスペースを追加
    text = re.sub(r"([.,?!])", r" \1 ", text)

    # 4. 連続する文字を単語として扱う
    words = []
    current_word = ""

    for char in text:
        if char.isalpha():
            current_word += char
        else:
            if current_word:
                words.append(current_word)
                current_word = ""
            if not char.isspace():
                words.append(char)

    if current_word:
        words.append(current_word)

    return words


def split_english_text(text: str) -> tuple[str, list[str]]:
    """英語テキストを問題文と選択肢に分割"""
    # まず選択肢の開始位置を探す（最初のa-eで始まる部分）
    match = re.search(r"(?:^|\s)([abcdeABCDE])[\s\.]", text)
    if not match:
        return text, []

    # 問題文と選択肢部分を分割
    question = text[: match.start()].strip()
    choices_text = text[match.start() :].strip()

    # 選択肢を分割
    choices = []
    current_letter = None
    current_text = ""

    # 選択肢のパターンで分割
    choice_matches = re.finditer(
        r"([abcdeABCDE])[.\s]([^abcdeABCDE]*?)(?=[abcdeABCDE][.\s]|$)",
        choices_text,
        re.IGNORECASE,
    )

    for match in choice_matches:
        letter = match.group(1).lower()
        content = match.group(2).strip()
        if content:
            choices.append(f"{letter}. {content}")

    return question.strip(), choices


def split_choices_by_sequence(choices_text: str, question_num: str) -> list[str]:
    """選択肢を分割"""
    choices = []
    choice_letters = ["a", "b", "[cC]", "d", "e"]

    logging.info(f"問題 {question_num}: 選択肢テキスト: {choices_text}")

    start_positions = []
    letter_found = []

    for letter in choice_letters:
        match = re.search(f"{letter}", choices_text)
        if match:
            start_positions.append(match.start())
            letter_found.append(letter)
            logging.info(
                f"問題 {question_num}: 選択肢 {letter} が位置 {match.start()} で見つかりました"
            )

    if len(start_positions) < 5:
        logging.warning(
            f"問題 {question_num}: 一部の選択肢が見つかりません。見つかった選択肢: {letter_found}"
        )

    for i in range(len(start_positions)):
        current_start = start_positions[i]
        letter = letter_found[i]

        if i < len(start_positions) - 1:
            current_end = start_positions[i + 1]
            content = choices_text[current_start + 1 : current_end].strip()
        else:
            content = choices_text[current_start + 1 :].strip()

        display_letter = letter.replace("[cC]", "c").strip("[]")
        choices.append(f"{display_letter}. {content}")
        logging.info(f"問題 {question_num}: 選択肢 {display_letter} の内容: {content}")

    return choices


def split_question_and_choices(
    text: str, question_num: str, input_file: str
) -> tuple[str, list[str]]:
    """問題文と選択肢を分割"""
    try:
        # デバッグ用ログ
        logging.info(f"問題 {question_num}: 入力テキスト: {text}")

        # 英語テキストかどうかを判定
        is_english = is_english_text(text)
        logging.info(f"問題 {question_num}: 英語テキスト判定: {is_english}")

        if is_english:
            return split_english_text(text)

        # 日本語テキストの処理（従来の方法）
        parts = text.split("。a")
        if len(parts) != 2:
            logging.error(
                f"Input: {input_file} - 問題 {question_num}: 分割エラー - '。a'が{len(parts) - 1}個見つかりました"
            )
            return text.strip(), []

        question = parts[0] + "。"
        choices_text = "a" + parts[1]

        logging.info(f"問題 {question_num}: 問題文: {question}")
        logging.info(f"問題 {question_num}: 選択肢テキスト: {choices_text}")

        choices = split_choices_by_sequence(choices_text, question_num)

        if not choices:
            logging.error(
                f"Input: {input_file} - 問題 {question_num}: 選択肢の分割に失敗しました"
            )

        return question.strip(), choices

    except Exception as e:
        logging.error(
            f"Input: {input_file} - 問題 {question_num}: 予期せぬエラー - {str(e)}"
        )
        return text.strip(), []


def split_questions(text: str, input_file: str, start: int = 1, end: int = 50) -> str:
    clean_text = re.sub(r"=== Page \d+ ===\n", "", text)
    formatted = ""
    shared_text = {}  # 連問で共有するテキストを保存

    # ファイル名からパターンを抽出
    prefix_num, prefix_alpha = extract_pattern_from_filename(input_file)

    # デバッグ用ログ
    logging.info(f"クリーニング後のテキスト: {clean_text[:200]}...")

    # まず連問のパターンを探す
    shared_text_pattern = (
        r"(次の文を読み、\d+、\d+の問いに答えよ。)(.*?)(?=\d+[A-Z]\d+|$)"
    )
    for match in re.finditer(shared_text_pattern, clean_text, re.DOTALL):
        intro, content = match.groups()
        # 問題番号を抽出
        nums = re.findall(r"\d+", intro)
        if len(nums) >= 2:
            q1, q2 = nums[:2]
            q1_full = f"{prefix_num}{prefix_alpha}{q1}"
            q2_full = f"{prefix_num}{prefix_alpha}{q2}"
            # 導入文（次の文を読み...）を含めた完全なテキストを保存
            shared_text[q1_full] = (intro + content).strip()
            shared_text[q2_full] = (intro + content).strip()
            # 元のテキストから導入文のみを削除するために、導入文の位置を記録
            shared_text[f"{q1_full}_intro"] = intro
            shared_text[f"{q2_full}_intro"] = intro
            logging.info(
                f"連問を検出: {q1_full}と{q2_full}が共有テキスト「{content[:50]}...」を持ちます"
            )

    for i in range(start, end + 1):
        current_pattern = f"{prefix_num}{prefix_alpha}{i}"
        pattern = f"({current_pattern}.*?)(?={prefix_num}{prefix_alpha}\\d+|$)"
        match = re.search(pattern, clean_text, re.DOTALL)

        if match:
            content = match.group(1).replace(current_pattern, "").strip()
            logging.info(f"問題 {current_pattern} が見つかりました")

            # 共有テキストがある場合は、元の問題文から導入文のみを削除
            if current_pattern in shared_text:
                intro = shared_text[f"{current_pattern}_intro"]
                # 導入文のみを削除し、残りのテキストは保持
                content = content.replace(intro, "", 1).strip()
                # 共有テキスト全体を問題文の前に追加
                content = shared_text[current_pattern] + "\n\n" + content

            question, choices = split_question_and_choices(
                content, current_pattern, input_file
            )

            formatted += f"=== {current_pattern} ===\n"
            formatted += "【問題】\n"
            formatted += question + "\n"
            if choices:
                formatted += "【選択肢】\n"
                formatted += "\n".join(choices) + "\n"
            formatted += "\n"
        else:
            logging.warning(f"問題 {current_pattern} が見つかりません")

    return formatted


def main():
    if len(sys.argv) != 3:
        print("""
使用方法: python cleaning_ocr_txt.py 入力ファイル 出力ファイル

引数:
    入力ファイル: OCRの結果のテキストファイル（例: 118E.txt）
    出力ファイル: 整形後の出力先ファイル（例: 118E_cleaned.txt）

例:
    python cleaning_ocr_txt.py input/118E.txt output/118E_cleaned.txt
    python cleaning_ocr_txt.py input/119B.txt output/119B_cleaned.txt

注意:
    - 入力ファイル名は "数字+アルファベット" の形式である必要があります（例: 118E, 119B）
    - 処理の詳細なログは error_log_[ファイル名]_[日時].log に出力されます
        """)
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    try:
        input_filename = Path(input_file).name
        setup_logging(input_filename)

        with open(input_file, "r", encoding="utf-8") as f:
            text = f.read()

        result = split_questions(text, input_file)

        if not result.strip():
            logging.error("出力結果が空です")
            print("エラー: 出力結果が空です。ログを確認してください。")
            sys.exit(1)

        with open(output_file, "w", encoding="utf-8") as f:
            f.write(result)

        print(f"処理が完了しました: {output_file}")

    except FileNotFoundError:
        logging.error(f"Input: {input_file} - ファイルが見つかりません")
        print(f"エラー: ファイルが見つかりません: {input_file}")
    except Exception as e:
        logging.error(f"Input: {input_file} - エラーが発生しました: {e}")
        print(f"エラーが発生しました。ログを確認してください。")


if __name__ == "__main__":
    main()
