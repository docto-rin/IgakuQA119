import json
import pandas as pd
from pathlib import Path

LEADERBOARD_START_MARKER = "<!-- LEADERBOARD_START -->"
LEADERBOARD_END_MARKER = "<!-- LEADERBOARD_END -->"

def load_model_answers(json_path):
    """JSONファイルからモデルの解答データを読み込む"""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        print(f"エラー: JSONファイルが見つかりません: {json_path}")
        raise
    except json.JSONDecodeError:
        print(f"エラー: JSONファイルが不正な形式です: {json_path}")
        raise
    except Exception as e:
        print(f"エラー: JSONファイルの読み込み中に予期せぬエラー: {e}")
        raise

def load_correct_answers(answers_path):
    """正解データを読み込む（CSVまたはDataFrame/Pickle）"""
    path = Path(answers_path)
    if not path.exists():
        raise FileNotFoundError(f"正解ファイルが見つかりません: {answers_path}")

    if path.suffix.lower() == '.csv':
        try:
            df = pd.read_csv(answers_path)
        except Exception as e:
             raise ValueError(f"CSV正解ファイルの読み込みに失敗しました: {e}")
    elif path.suffix.lower() in ['.pkl', '.pickle']:
         try:
            df = pd.read_pickle(answers_path)
         except Exception as e:
            raise ValueError(f"Pickle正解ファイルの読み込みに失敗しました: {e}")
    else:
        try:
            df = pd.read_pickle(answers_path)
            print(f"情報: 拡張子不明のためPickleとして読み込みました: {answers_path}")
        except Exception as e:
            raise ValueError(f"拡張子がCSV/Pickleでないため読み込めません ({path.suffix}): {e}")

    required_cols = ['問題番号', '解答']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"正解ファイルには '{required_cols}' の列が必要です。")

    try:
        df['問題番号'] = df['問題番号'].astype(str)
    except Exception as e:
        raise ValueError(f"'問題番号'列を文字列に変換できませんでした: {e}")

    return df

def parse_input_paths(json_path_strs):
    """
    入力されたJSONパスのリストを検証し、ファイル識別子、プレフィックス等を抽出する。

    Args:
        json_path_strs (list[str]): JSONファイルへのパス文字列のリスト。

    Returns:
        tuple: (valid_json_paths, file_identifier, consolidated_prefix, processed_blocks, base_numbers_from_files)
               - valid_json_paths (list[str]): 検証済みの有効なJSONファイルパスのリスト。
               - file_identifier (str): ファイル名から推測された共通の識別子 (例: "my-exp")。
               - consolidated_prefix (str): レポートファイル名のプレフィックス (例: "119_")。
               - processed_blocks (set[str]): 処理されたブロック名のセット (例: {"119A", "119B"})。
               - base_numbers_from_files (set[str]): ファイル名から抽出された試験番号のセット (例: {"119"})。

    Raises:
        ValueError: 処理可能なJSONファイルが1つも見つからない場合。
    """
    valid_json_paths = []
    processed_blocks = set()
    base_numbers_from_files = set()
    first_valid_json_path = None
    first_file_identifier = None
    first_base_number = None

    for path_str in json_path_strs:
        path = Path(path_str)
        if not path.exists():
            print(f"警告: JSONファイルが見つかりません: {path}。スキップします。")
            continue
        if not path.is_file():
            print(f"警告: {path} はファイルではありません。スキップします。")
            continue

        valid_json_paths.append(str(path))
        filename = path.stem # 拡張子なしのファイル名 (例: "119A_my-exp")
        parts = filename.split('_', 1) # 最初の '_' で分割

        # 最初の有効なJSONパスとファイル識別子、試験番号を記録
        if first_valid_json_path is None:
            first_valid_json_path = path
            if len(parts) > 1:
                first_file_identifier = parts[1] # 例: "my-exp"

        base_number_with_block = parts[0] if parts else "" # 例: "119A"

        block_id = ""
        base_number = ""
        # ファイル名の最初の部分が "数字 + 英字" の形式かチェック (例: "119A")
        if base_number_with_block and len(base_number_with_block) > 1 and base_number_with_block[-1].isalpha():
            block_id = base_number_with_block[-1].upper() # 例: "A"
            base_number = base_number_with_block[:-1] # 例: "119"
            if base_number.isdigit():
                 if first_base_number is None: first_base_number = base_number
                 # 同じブロックが複数指定されていないかチェック
                 block_key = f"{base_number}{block_id}"
                 if block_key in processed_blocks:
                      print(f"警告: ブロック {block_id} (試験番号 {base_number}) が複数回指定されています ({path.name})。")
                 processed_blocks.add(block_key)
                 base_numbers_from_files.add(base_number) # 試験番号をセットに追加
            else:
                print(f"警告: ファイル名 {path.name} から有効な試験番号を特定できません ({base_number_with_block})。")
        # ファイル名の最初の部分が数字のみかチェック (例: "119")
        elif base_number_with_block.isdigit():
             base_number = base_number_with_block
             if first_base_number is None: first_base_number = base_number
             base_numbers_from_files.add(base_number)
             print(f"情報: ファイル名 {path.name} にはブロックIDが含まれていません。試験番号 {base_number} として扱います。")
        else:
             print(f"警告: ファイル名 {path.name} から試験番号やブロックIDを特定できません。")


    if not valid_json_paths:
        raise ValueError("処理可能なJSONファイルが1つも見つかりませんでした。")

    # レポートファイル名に使用する識別子を決定 (最初のファイルから取得、なければ "UnknownExp")
    file_identifier = first_file_identifier or "UnknownExp"
    print(f"情報: レポートファイル名に使用する識別子: '{file_identifier}'")

    # 複数の試験番号が混在している場合に警告
    if len(base_numbers_from_files) > 1:
        print(f"警告: 複数の試験番号がファイル名に含まれています: {sorted(list(base_numbers_from_files))}。")
        print(f"       レポートのプレフィックスには、最初のファイルから取得した '{first_base_number}' を使用します。")
    elif not base_numbers_from_files:
        print(f"警告: ファイル名から試験番号を特定できませんでした。レポートプレフィックスは 'UnknownNumber_' になります。")

    # レポートファイルのプレフィックスを決定 (例: "119_")
    report_base_number = first_base_number if first_base_number else "UnknownNumber"
    consolidated_prefix = f"{report_base_number}_"

    return valid_json_paths, file_identifier, consolidated_prefix, processed_blocks, base_numbers_from_files

def save_skipped_list(unique_skipped_questions, output_dir, prefix, file_identifier, skipped_output_file=None):
    """
    スキップされた問題番号のリストを指定されたファイルに保存する。

    Args:
        unique_skipped_questions (list[str]): スキップされた一意の問題番号のリスト (ソート済み)。
        output_dir (str): 出力先ディレクトリのパス。
        prefix (str): ファイル名のプレフィックス (例: "119_")。
        file_identifier (str): ファイル名の識別子 (例: "my-exp")。
        skipped_output_file (str, optional): 出力ファイル名を直接指定する場合のパス。 Defaults to None.
    """
    if not unique_skipped_questions:
        # スキップがなかった場合でも、メッセージは main 側で出すことが多いのでここでは出さない
        # print("\nスキップされた問題はありませんでした。")
        return

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    # スキップリストの出力ファイルパスを決定
    if skipped_output_file:
        skipped_file_path = Path(skipped_output_file)
        # 指定されたパスの親ディレクトリも念のため作成
        try:
            skipped_file_path.parent.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            print(f"警告: スキップリスト出力先の親ディレクトリ作成中にエラー: {e}。出力ディレクトリ '{output_dir}' にフォールバックします。")
            # フォールバック先を決定
            skipped_file_name = f"{prefix}{file_identifier}_skipped.txt"
            skipped_file_path = output_path / skipped_file_name
    else:
        # デフォルトのファイル名 (例: results/119_my-exp_skipped.txt)
        skipped_file_name = f"{prefix}{file_identifier}_skipped.txt"
        skipped_file_path = output_path / skipped_file_name

    print(f"\n--- スキップされた問題番号 ({len(unique_skipped_questions)} 件) をファイルに出力 ---")
    try:
        with open(skipped_file_path, 'w', encoding='utf-8') as f:
            for q_num in unique_skipped_questions:
                f.write(f"{q_num}\n")
        print(f"スキップされた問題番号を {skipped_file_path} に保存しました。")
        # 再実行用のコマンド例を表示
        print("これらの問題を再実行するには、以下のコマンドを使用します（パスやモデル名、exp名を適宜変更してください）:")
        # rerun_skipped.py を使うコマンド例
        print(f"  uv run python rerun_skipped.py --skipped_list '{skipped_file_path}' --model_name <YOUR_MODEL_NAME> --questions_dir questions --rerun_exp {file_identifier}_retry")
        print(f"  注意: <YOUR_MODEL_NAME> は再実行に使用するモデル名に置き換えてください。")
        print(f"        再実行後、結果をマージするには scripts/merge_results.py を使用します。")
        print(f"        例: uv run python scripts/merge_results.py --original_pattern \"answers/{prefix}*_{file_identifier}.json\" --retry_pattern \"answers/{prefix}*_{file_identifier}_retry.json\" --merged_exp {file_identifier}_merged")

    except Exception as e:
        print(f"エラー: スキップされた問題番号のファイル '{skipped_file_path}' の書き込み中にエラー: {e}")


def update_leaderboard_data(data_file, entry_name, stats):
    """
    LeaderboardデータをJSONファイルで更新または新規作成する。
    """
    data_path = Path(data_file)
    leaderboard_data = {}

    if data_path.exists() and data_path.is_file():
        try:
            with open(data_path, 'r', encoding='utf-8') as f:
                leaderboard_data = json.load(f)
            if not isinstance(leaderboard_data, dict):
                 print(f"警告: {data_file}の内容が辞書形式ではありません。新しいデータで上書きします。")
                 leaderboard_data = {}
        except json.JSONDecodeError:
            print(f"警告: {data_file} が不正なJSON形式です。新しいデータで上書きします。")
            leaderboard_data = {}
        except Exception as e:
            print(f"エラー: {data_file} の読み込み中にエラーが発生しました: {e}。新規作成します。")
            leaderboard_data = {}

    def get_stat_values(category_stats):
        if not category_stats or not isinstance(category_stats, dict):
            return { 'score': 0.0, 'possible_score': 0.0, 'score_rate': 0.0, 'correct': 0, 'total': 0, 'accuracy': 0.0 }
        return {
            'score': category_stats.get('score', 0.0),
            'possible_score': category_stats.get('possible_score', 0.0),
            'score_rate': category_stats.get('score_rate', 0.0),
            'correct': category_stats.get('correct', 0),
            'total': category_stats.get('total', 0),
            'accuracy': category_stats.get('accuracy', 0.0)
        }

    overall_stats = get_stat_values(stats.get('total'))
    no_image_top_level = stats.get('no_image')
    no_image_stats = get_stat_values(no_image_top_level) if isinstance(no_image_top_level, dict) else get_stat_values({})


    model_entry = {
        "overall_score": overall_stats['score'],
        "overall_possible_score": overall_stats['possible_score'],
        "overall_score_rate": overall_stats['score_rate'],
        "overall_correct": overall_stats['correct'],
        "overall_total": overall_stats['total'],
        "overall_accuracy": overall_stats['accuracy'],
        "no_image_score": no_image_stats['score'],
        "no_image_possible_score": no_image_stats['possible_score'],
        "no_image_score_rate": no_image_stats['score_rate'],
        "no_image_correct": no_image_stats['correct'],
        "no_image_total": no_image_stats['total'],
        "no_image_accuracy": no_image_stats['accuracy'],
    }

    for key, value in model_entry.items():
        if pd.isna(value):
            model_entry[key] = 0.0 if isinstance(value, float) else 0

    leaderboard_data[entry_name] = model_entry

    try:
        data_path.parent.mkdir(parents=True, exist_ok=True)
        with open(data_path, 'w', encoding='utf-8') as f:
            json.dump(leaderboard_data, f, indent=4, ensure_ascii=False, allow_nan=False)
        print(f"Leaderboardデータを {data_file} に保存/更新しました。")
    except TypeError as e:
        if 'Out of range float values are not JSON compliant' in str(e) or 'NaN' in str(e) or 'Infinity' in str(e):
            print(f"エラー: JSONシリアライズ中に非準拠の浮動小数点値(NaN/Infinity)が検出されました。0に置換して再試行します。")
            def replace_nan_inf(obj):
                if isinstance(obj, dict):
                    return {k: replace_nan_inf(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [replace_nan_inf(elem) for elem in obj]
                elif isinstance(obj, float) and (pd.isna(obj) or abs(obj) == float('inf')):
                    return 0.0
                return obj
            try:
                cleaned_data = replace_nan_inf(leaderboard_data)
                with open(data_path, 'w', encoding='utf-8') as f:
                    json.dump(cleaned_data, f, indent=4, ensure_ascii=False, allow_nan=False)
                print(f"Leaderboardデータを {data_file} に保存/更新しました (非準拠値を0に置換)。")
            except Exception as inner_e:
                 print(f"エラー: 非準拠値置換後のJSON書き込み中に再度エラーが発生しました: {inner_e}")
        else:
             print(f"エラー: JSONシリアライズ中に予期せぬ型エラーが発生しました: {e}")
    except Exception as e:
        print(f"エラー: {data_file} への書き込み中にエラーが発生しました: {e}")

    return leaderboard_data

def update_readme(readme_path_str, markdown_table):
    """READMEファイルを読み込み、マーカー間にMarkdownテーブルを挿入/更新する"""
    readme_path = Path(readme_path_str)
    if not readme_path.exists() or not readme_path.is_file():
        print(f"エラー: READMEファイルが見つからないか、ファイルではありません: {readme_path}")
        return

    try:
        content = readme_path.read_text(encoding='utf-8')
        lines = content.splitlines()
        start_index, end_index = -1, -1
        for i, line in enumerate(lines):
            if LEADERBOARD_START_MARKER in line: start_index = i
            elif LEADERBOARD_END_MARKER in line:
                end_index = i
                if start_index != -1 and start_index < end_index:
                     break

        if start_index != -1 and end_index != -1 and start_index < end_index:
            new_lines = lines[:start_index + 1]
            new_lines.append("")
            new_lines.append(markdown_table)
            new_lines.append("")
            new_lines.extend(lines[end_index:])

            new_content = "\n".join(new_lines)
            if not new_content.endswith('\n'): new_content += '\n'
            readme_path.write_text(new_content, encoding='utf-8')
            print(f"READMEファイル ({readme_path}) のLeaderboardを更新しました。")
        else:
             missing_markers = []
             if start_index == -1: missing_markers.append(LEADERBOARD_START_MARKER)
             if end_index == -1: missing_markers.append(LEADERBOARD_END_MARKER)
             if start_index != -1 and end_index != -1 and start_index >= end_index:
                 print(f"警告: READMEファイル ({readme_path}) のLeaderboardマーカーの順序が不正です。")
             elif missing_markers:
                 print(f"警告: READMEファイル ({readme_path}) にLeaderboardマーカーが見つかりません: {', '.join(missing_markers)}")
             print(f"       必要なマーカー: {LEADERBOARD_START_MARKER} と {LEADERBOARD_END_MARKER} (この順序で別々の行に)")
             print(f"       READMEは更新されませんでした。")

    except Exception as e:
        print(f"エラー: READMEファイル '{readme_path}' の更新中にエラーが発生しました: {e}")