import json
import argparse
from pathlib import Path
import glob
import sys

def load_json(filepath):
    """JSONファイルを読み込む"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"情報: ファイルが見つかりません: {filepath}。スキップします。")
        return None
    except json.JSONDecodeError:
        print(f"エラー: JSONファイルが不正な形式です: {filepath}。スキップします。", file=sys.stderr)
        return None
    except Exception as e:
        print(f"エラー: JSONファイルの読み込み中に予期せぬエラー ({filepath}): {e}", file=sys.stderr)
        return None

def merge_results_data(original_data, retry_data):
    """2つの結果データ(辞書)をマージする"""
    if not original_data:
        print("警告: オリジナルデータが存在しないため、リトライデータをそのまま使用します。")
        return retry_data
    if not retry_data:
        return original_data

    if 'results' not in original_data or not isinstance(original_data['results'], list):
        print(f"警告: オリジナルデータに有効な 'results' リストがありません。リトライデータを返します。", file=sys.stderr)
        return retry_data
    if 'results' not in retry_data or not isinstance(retry_data['results'], list):
        print(f"警告: リトライデータに有効な 'results' リストがありません。オリジナルデータを返します。", file=sys.stderr)
        return retry_data

    original_results_map = {}
    malformed_original = 0
    duplicate_original = 0
    for q_data in original_data['results']:
        if isinstance(q_data, dict) and 'question_number' in q_data:
            q_num = q_data['question_number']
            if q_num in original_results_map:
                 duplicate_original += 1
            original_results_map[q_num] = q_data
        else:
            malformed_original += 1
    if malformed_original > 0: print(f"警告(オリジナル): 不正な形式または問題番号のないデータが {malformed_original} 件ありました。")
    if duplicate_original > 0: print(f"警告(オリジナル): 問題番号が重複しているデータが {duplicate_original} 件ありました。")


    merged_count = 0
    malformed_retry = 0
    for q_data in retry_data['results']:
         if isinstance(q_data, dict) and 'question_number' in q_data:
            q_num = q_data['question_number']
            original_results_map[q_num] = q_data
            merged_count += 1
         else:
             malformed_retry += 1
    if malformed_retry > 0: print(f"警告(リトライ): 不正な形式または問題番号のないデータが {malformed_retry} 件ありました。")


    print(f"情報: {merged_count} 件のリトライ結果でマージ（上書き/追加）しました。")

    merged_results_list = list(original_results_map.values())

    final_merged_data = original_data.copy()
    final_merged_data['results'] = merged_results_list

    return final_merged_data

def save_json(data, filepath):
    """JSONデータをファイルに保存する"""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"マージ結果を {filepath} に保存しました。")
        return True
    except TypeError as e:
        if 'Out of range float values are not JSON compliant' in str(e) or 'NaN' in str(e) or 'Infinity' in str(e):
            print(f"エラー: JSONシリアライズ中に非準拠の値(NaN/Infinity)が検出されました ({filepath})。保存できませんでした。", file=sys.stderr)
            return False
        else:
             print(f"エラー: JSONデータのシリアライズ中に型エラー ({filepath}): {e}", file=sys.stderr)
             return False
    except Exception as e:
        print(f"エラー: JSONファイルの書き込み中に予期せぬエラー ({filepath}): {e}", file=sys.stderr)
        return False

def main():
    parser = argparse.ArgumentParser(
        description='初回実行結果と再実行結果のJSONファイルをマージするスクリプト',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--original_pattern',
        required=True,
        help='初回実行の結果JSONファイルのパターン (例: "answers/119*_{EXP}.json") ※GLOBパターン'
    )
    parser.add_argument(
        '--retry_pattern',
        required=True,
        help='再実行の結果JSONファイルのパターン (例: "answers/119*_{RERUN_EXP_BASE}.json") ※GLOBパターン'
    )
    parser.add_argument(
        '--merged_exp',
        required=True,
        help='マージ結果のファイル名に使用する識別子 (例: {EXP}_merged)'
    )
    parser.add_argument(
        '--output_dir',
        default='answers/',
        help='マージ結果のJSONファイルを出力するディレクトリ (例: answers/)'
    )

    args = parser.parse_args()

    original_files = sorted(glob.glob(args.original_pattern))
    retry_files = sorted(glob.glob(args.retry_pattern))

    if not original_files:
        print(f"エラー: オリジナルファイルが見つかりません。パターンを確認してください: {args.original_pattern}", file=sys.stderr)
        sys.exit(1)
    if not retry_files:
         print(f"警告: リトライファイルが見つかりません。パターン: {args.retry_pattern}。オリジナルファイルのみが出力されます。")

    original_files_dict = {Path(f).name: f for f in original_files}
    retry_files_dict = {Path(f).name: f for f in retry_files}

    # ファイル名のベース部分（例: 119A）でマッチングを試みる
    # この抽出ロジックはファイル名の命名規則に強く依存する

    output_dir_path = Path(args.output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    merged_count = 0
    failed_count = 0

    for original_filename, original_filepath in original_files_dict.items():
        print(f"\n--- ファイル {original_filename} を処理中 ---")

        # オリジナルファイル名からベース部分を推測
        parts = Path(original_filename).stem.split('_', 1)
        if len(parts) < 2:
             print(f"警告: オリジナルファイル名 {original_filename} からベース部分とEXPを分割できません。スキップします。")
             failed_count += 1
             continue
        file_base = parts[0] # 例: "119A"

        # 対応するリトライファイル名を推測
        corresponding_retry_filepath = None
        for retry_filename, retry_filepath in retry_files_dict.items():
            if Path(retry_filename).stem.startswith(file_base + '_'):
                 corresponding_retry_filepath = retry_filepath
                 print(f"情報: 対応するリトライファイルとして {retry_filename} を使用します。")
                 break

        original_data = load_json(original_filepath)
        retry_data = None
        if corresponding_retry_filepath:
             retry_data = load_json(corresponding_retry_filepath)
        elif retry_files:
             print(f"警告: {file_base} に対応するリトライファイルが見つかりませんでした。")


        if original_data is None:
             print(f"エラー: オリジナルデータ {original_filename} を読み込めませんでした。マージをスキップします。", file=sys.stderr)
             failed_count += 1
             continue

        merged_data = merge_results_data(original_data, retry_data)

        if merged_data:
            output_filename = f"{file_base}_{args.merged_exp}.json"
            output_filepath = output_dir_path / output_filename
            if save_json(merged_data, output_filepath):
                merged_count += 1
            else:
                failed_count += 1
        else:
            print(f"エラー: {original_filename} のマージ処理で有効なデータが得られませんでした。", file=sys.stderr)
            failed_count += 1


    print("\n--- マージ処理結果 ---")
    print(f"正常にマージされたファイル数: {merged_count}")
    if failed_count > 0:
        print(f"エラーまたはスキップされたファイル数: {failed_count}")

if __name__ == "__main__":
    main()