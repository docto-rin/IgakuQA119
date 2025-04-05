# grade_answers.py
import json
import pandas as pd
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import glob # ファイル検索用にglobを追加

# --- 定数定義 ---
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
        # 拡張子がない場合やその他の場合はPickleとして試行
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

def grade_answers(model_data, correct_df):
    """
    モデルの解答を採点し、スキップされた問題番号も返す。
    戻り値: (results_df, accuracy, model_name_in_json, skipped_questions)
    """
    results = []
    correct_count = 0
    total_questions = 0 # 正解が存在する問題の数
    model_name_in_json = None # JSONデータからモデル名を取得
    skipped_questions = [] # スキップされた問題番号を記録

    try:
        correct_answers = dict(zip(correct_df['問題番号'].astype(str), correct_df['解答'].astype(str)))
    except KeyError:
        raise ValueError("正解データに '問題番号' または '解答' 列がありません。")
    except Exception as e:
        raise ValueError(f"正解データのマッピング作成中にエラー: {e}")


    if 'results' not in model_data or not isinstance(model_data['results'], list):
        print(f"警告: JSONデータに 'results' キーがないか、リスト形式ではありません。")
        return pd.DataFrame(results), 0.0, None, []

    for i, question in enumerate(model_data['results']):
        if not isinstance(question, dict):
            print(f"警告: resultsリストの要素 {i} が辞書形式ではありません。スキップします: {question}")
            continue

        question_number_raw = question.get('question_number')
        question_number = str(question_number_raw) if question_number_raw is not None else 'N/A'

        required_keys = ['question_number', 'answers', 'has_image']
        if not all(key in question for key in required_keys):
            missing_keys = [key for key in required_keys if key not in question]
            print(f"警告: questionデータに必要なキーがありません ({missing_keys})。スキップします: {question_number}")
            if question_number != 'N/A':
                skipped_questions.append(question_number)
            continue

        if not isinstance(question['answers'], list) or not question['answers']:
            print(f"警告: question {question_number} の 'answers' が空またはリスト形式ではありません。スキップします。")
            if question_number != 'N/A':
                 skipped_questions.append(question_number)
            continue

        answer_data = question['answers'][0]
        if not isinstance(answer_data, dict):
            print(f"警告: question {question_number} の最初のanswerが辞書形式ではありません。スキップします。")
            if question_number != 'N/A':
                 skipped_questions.append(question_number)
            continue

        required_answer_keys = ['answer', 'model', 'confidence']
        missing_keys = [key for key in required_answer_keys if key not in answer_data]
        if missing_keys:
             # 'answer' や 'confidence' が欠けている場合にスキップ
             if 'answer' in missing_keys or 'confidence' in missing_keys:
                 print(f"警告: answerデータに必要なキーがありません ({missing_keys})。スキップします: {question_number}")
                 if question_number != 'N/A':
                     skipped_questions.append(question_number)
                 continue
             else: # modelキーがないなどは警告のみで続行する場合（今回はスキップ）
                 print(f"警告: answerデータに必要なキーがありません ({missing_keys})。スキップします: {question_number}")
                 if question_number != 'N/A':
                     skipped_questions.append(question_number)
                 continue


        model_answer = answer_data.get('answer')
        model_name = answer_data.get('model', 'UnknownModel')
        confidence = answer_data.get('confidence')
        has_image = question.get('has_image', False)

        if model_name_in_json is None and model_name != 'UnknownModel':
            model_name_in_json = model_name

        correct_answer = correct_answers.get(question_number)
        is_correct = False

        if correct_answer is not None:
            total_questions += 1
            try:
                model_answer_str = str(model_answer).strip().lower() if model_answer is not None else ""
                correct_answer_str = str(correct_answer).strip().lower()
                model_answer_str = model_answer_str.replace('[', '').replace(']', '').replace(',', '').replace(' ', '')

                if not model_answer_str:
                    is_correct = False
                elif question_number == '119E28':
                    is_correct = model_answer_str in ['a', 'c']
                else:
                    is_correct = model_answer_str == correct_answer_str

            except Exception as e:
                print(f"エラー: 問題 {question_number} の解答比較中にエラーが発生しました。 Model: '{model_answer}', Correct: '{correct_answer}'. Error: {e}")

            if is_correct:
                correct_count += 1
        else:
             pass

        results.append({
            'question_number': question_number,
            'model': model_name,
            'model_answer': model_answer,
            'correct_answer': correct_answer,
            'is_correct': is_correct,
            'confidence': confidence,
            'has_image': has_image
        })

    accuracy = correct_count / total_questions if total_questions > 0 else 0.0
    return pd.DataFrame(results), accuracy, model_name_in_json, sorted(list(set(skipped_questions)))

def print_individual_report_summary(results_df, accuracy, json_filename):
    """
    個別のJSONファイル処理結果の要約をコンソールに出力する。
    ファイル出力は行わない。
    """
    if results_df.empty:
        print(f"情報: {json_filename} の結果データフレームが空のため、要約表示をスキップします。")
        return

    report_model_name = "UnknownModel"
    if not results_df.empty and 'model' in results_df.columns:
        valid_models = results_df['model'].dropna()
        if not valid_models.empty:
            report_model_name = valid_models.iloc[0]

    print(f"ファイル: {json_filename}")
    print(f"モデル (JSON内): {report_model_name}")
    print(f"正答率: {accuracy:.2%}")

    valid_results = results_df.dropna(subset=['correct_answer'])
    correct_sum = valid_results['is_correct'].sum()
    total_len_graded = len(valid_results)
    print(f"正解数 (採点対象): {correct_sum} / {total_len_graded}")
    if len(results_df) != total_len_graded:
        print(f"(全問題数: {len(results_df)}, うち正解不明: {len(results_df) - total_len_graded})")

    # 信頼度分析 (省略)
    print("-" * 20)

# --- 統合レポートファイル生成 (ファイル名に file_identifier を使用) ---
def generate_consolidated_files(consolidated_df, output_dir, prefix, file_identifier):
    """統合された結果からレポートファイル（プロット、CSV）を生成する"""
    if consolidated_df.empty:
        print("情報: 統合結果データフレームが空のため、ファイル生成をスキップします。")
        return

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    # ファイル名プレフィックスに file_identifier を使用
    file_prefix = f"{prefix}{file_identifier}_" # 例: 119_my-exp_

    try:
        print("\n--- 統合レポートファイルの作成 ---")
        # --- 信頼度分布プロット (統合版) ---
        if 'confidence' in consolidated_df.columns and pd.api.types.is_numeric_dtype(consolidated_df['confidence']):
            plt.figure(figsize=(10, 6))
            plot_data = consolidated_df.dropna(subset=['confidence', 'is_correct'])
            if not plot_data.empty:
                 sns.histplot(data=plot_data, x='confidence', hue='is_correct', bins=20, multiple='stack')
                 # タイトルには file_identifier を使用 (どちらでも良いが一貫性のため)
                 plt.title(f'Confidence Distribution and Correctness ({file_identifier} - Consolidated)')
                 plt.xlabel('Confidence')
                 plt.ylabel('Count (Graded Questions)')
                 plot_path = output_path / f"{file_prefix}confidence_distribution.png"
                 plt.savefig(plot_path)
                 plt.close()
                 print(f"信頼度分布グラフを {plot_path} に保存しました。")
            else:
                 print("情報: 信頼度プロット用の有効なデータがありません。")
        else:
            print("情報: 統合結果に'confidence'列がないか数値型でないため、信頼度グラフは生成されません。")

        # --- 間違えた問題のリスト (統合版) ---
        wrong_answers = consolidated_df[consolidated_df['is_correct'] == False]
        if not wrong_answers.empty:
            wrong_csv_path = output_path / f"{file_prefix}wrong_answers.csv"
            wrong_answers.to_csv(wrong_csv_path, index=False, encoding='utf-8-sig')
            print(f"誤答問題リストを {wrong_csv_path} に保存しました。")
        else:
             print("情報: 統合結果で間違えた問題はありませんでした (採点対象内で)。")

        # --- 全結果のCSV (統合版) ---
        results_csv_path = output_path / f"{file_prefix}grading_results.csv"
        consolidated_df.to_csv(results_csv_path, index=False, encoding='utf-8-sig')
        print(f"全採点結果CSVを {results_csv_path} に保存しました。")

    except Exception as e:
        print(f"エラー: 統合レポートファイルの保存中にエラーが発生しました: {e}")

# --- 結果統合関数 ---
def consolidate_results(all_results):
    """複数ブロックの結果を統合し、各種統計情報を計算する"""
    if not all_results:
        print("警告: 統合する結果がありません。")
        empty_stats = {
            'total': {'df': pd.DataFrame(), 'correct': 0, 'total': 0, 'accuracy': 0.0, 'score': 0.0, 'possible_score': 0.0, 'score_rate': 0.0, 'block_accuracies': {}, 'block_score_rates': {}},
            'general': {'df': pd.DataFrame(), 'correct': 0, 'total': 0, 'accuracy': 0.0, 'score': 0.0, 'possible_score': 0.0, 'score_rate': 0.0},
            'required': {'df': pd.DataFrame(), 'correct': 0, 'total': 0, 'accuracy': 0.0, 'score': 0.0, 'possible_score': 0.0, 'score_rate': 0.0},
            'block_accuracies': {}, 'block_score_rates': {},
            'no_image': {
                'df': pd.DataFrame(), 'correct': 0, 'total': 0, 'accuracy': 0.0, 'score': 0.0, 'possible_score': 0.0, 'score_rate': 0.0,
                'general': {'df': pd.DataFrame(), 'correct': 0, 'total': 0, 'accuracy': 0.0, 'score': 0.0, 'possible_score': 0.0, 'score_rate': 0.0},
                'required': {'df': pd.DataFrame(), 'correct': 0, 'total': 0, 'accuracy': 0.0, 'score': 0.0, 'possible_score': 0.0, 'score_rate': 0.0},
                'block_accuracies': {}, 'block_score_rates': {}
            },
        }
        return empty_stats

    combined_df = pd.concat(all_results, ignore_index=True)

    graded_df = combined_df.dropna(subset=['correct_answer']).copy()
    if len(graded_df) != len(combined_df):
        print(f"情報: 統合された全 {len(combined_df)} 件のうち、正解データが存在しない {len(combined_df) - len(graded_df)} 件を集計から除外します。")

    if graded_df.empty:
        print("警告: 正解データが存在する有効な結果が統合後に見つかりませんでした。")
        empty_stats['total']['df'] = pd.DataFrame()
        return empty_stats

    try:
        graded_df['question_number'] = graded_df['question_number'].astype(str)
        graded_df['block'] = graded_df['question_number'].apply(
            lambda x: x[3].upper() if isinstance(x, str) and len(x) > 3 and x[3].isalpha() else 'Unknown'
        )
        graded_df['has_image'] = graded_df['has_image'].fillna(False).astype(bool)
        graded_df['is_correct'] = graded_df['is_correct'].fillna(False).astype(bool)
    except Exception as e:
        print(f"エラー: 統合データの前処理中にエラーが発生しました: {e}")
        empty_stats['total']['df'] = pd.DataFrame()
        return empty_stats


    general_blocks = ['A', 'C', 'D', 'F']
    required_blocks = ['B', 'E']
    valid_blocks = general_blocks + required_blocks

    invalid_block_mask = ~graded_df['block'].isin(valid_blocks)
    if invalid_block_mask.any():
        invalid_count = invalid_block_mask.sum()
        print(f"警告: 不正または不明なブロックID ('Unknown'など) を持つ問題が {invalid_count} 件あります（採点対象内）。")
        print("これらは一般/必修の集計に含まれません。例:")
        print(graded_df.loc[invalid_block_mask, ['question_number', 'block']].head())

    graded_df['is_general'] = graded_df['block'].isin(general_blocks)
    graded_df['is_required'] = graded_df['block'].isin(required_blocks)

    # --- 点数計算関数 ---
    def required_point(question_number):
        try:
            block = question_number[3].upper() if len(question_number) > 3 else ''
            if block in ['B', 'E']:
                num_part = ''.join(filter(str.isdigit, question_number[4:]))
                num = int(num_part) if num_part else 0
                return 3 if 26 <= num <= 50 else 1
            else: return 1
        except (IndexError, ValueError, TypeError):
            print(f"警告: 必修問題の点数計算で問題番号形式エラー: {question_number}。デフォルト1点とします。")
            return 1

    def calculate_points(row):
        block = row['block']
        is_correct = row['is_correct']
        question_number = row['question_number']
        possible = 0
        score = 0
        if block in required_blocks:
            possible = required_point(question_number)
            score = possible if is_correct else 0
        elif block in general_blocks:
            possible = 1
            score = 1 if is_correct else 0
        return pd.Series({'score': score, 'possible': possible})

    try:
        points_df = graded_df.apply(calculate_points, axis=1)
        graded_df[['score', 'possible']] = points_df
    except Exception as e:
        print(f"エラー: 点数計算中にエラーが発生しました: {e}。点数は0として扱われます。")
        graded_df['score'] = 0
        graded_df['possible'] = 0

    no_image_df = graded_df[~graded_df['has_image']].copy()

    print(f"\n統合結果概要 (採点対象):")
    print(f"  総問題数: {len(graded_df)}")
    processed_blocks_list = sorted([b for b in graded_df['block'].unique() if b != 'Unknown'])
    unknown_block_count = (graded_df['block'] == 'Unknown').sum()
    print(f"  処理したブロック: {processed_blocks_list}" + (f" (不明ブロック: {unknown_block_count}件)" if unknown_block_count > 0 else ""))
    print(f"  一般問題 (A,C,D,F): {graded_df['is_general'].sum()} 件")
    print(f"  必修問題 (B,E): {graded_df['is_required'].sum()} 件")
    print(f"  画像あり問題: {graded_df['has_image'].sum()} 件")
    print(f"  画像なし問題: {len(no_image_df)} 件")

    # --- 集計関数 ---
    def calculate_category_stats(df):
        if df.empty:
            return {'correct': 0, 'total': 0, 'accuracy': 0.0, 'score': 0.0, 'possible_score': 0.0, 'score_rate': 0.0}
        correct = int(df['is_correct'].sum())
        total = len(df)
        accuracy = correct / total if total > 0 else 0.0
        score = float(df['score'].sum())
        possible_score = float(df['possible'].sum())
        score_rate = score / possible_score if possible_score > 0 else 0.0
        return {'correct': correct, 'total': total, 'accuracy': accuracy,
                'score': score, 'possible_score': possible_score, 'score_rate': score_rate}

    def calculate_block_stats(df):
         if df.empty or 'block' not in df.columns: return {}, {}
         df_filtered = df[df['block'] != 'Unknown']
         if df_filtered.empty: return {}, {}
         block_agg = df_filtered.groupby('block').agg(
             correct=('is_correct', 'sum'),
             total=('is_correct', 'size'),
             score=('score', 'sum'),
             possible=('possible', 'sum')
         ).astype(float)
         block_agg['accuracy'] = (block_agg['correct'] / block_agg['total']).fillna(0.0)
         block_agg['score_rate'] = (block_agg['score'] / block_agg['possible']).fillna(0.0)
         return block_agg['accuracy'].to_dict(), block_agg['score_rate'].to_dict()

    total_stats = calculate_category_stats(graded_df)
    general_stats = calculate_category_stats(graded_df[graded_df['is_general']])
    required_stats = calculate_category_stats(graded_df[graded_df['is_required']])
    no_image_total_stats = calculate_category_stats(no_image_df)
    no_image_general_stats = calculate_category_stats(no_image_df[no_image_df['is_general']])
    no_image_required_stats = calculate_category_stats(no_image_df[no_image_df['is_required']])
    block_accuracies, block_score_rates = calculate_block_stats(graded_df)
    no_image_block_accuracies, no_image_block_score_rates = calculate_block_stats(no_image_df)

    stats = {
        'total': {**total_stats, 'df': graded_df, 'block_accuracies': block_accuracies, 'block_score_rates': block_score_rates},
        'general': {**general_stats, 'df': graded_df[graded_df['is_general']]},
        'required': {**required_stats, 'df': graded_df[graded_df['is_required']]},
        'block_accuracies': block_accuracies,
        'block_score_rates': block_score_rates,
        'no_image': {
            **no_image_total_stats, 'df': no_image_df,
            'general': {**no_image_general_stats, 'df': no_image_df[no_image_df['is_general']]},
            'required': {**no_image_required_stats, 'df': no_image_df[no_image_df['is_required']]},
            'block_accuracies': no_image_block_accuracies,
            'block_score_rates': no_image_block_score_rates
        },
    }
    return stats

# --- 統合レポート生成 (タイトルとファイル名識別子を区別) ---
def generate_consolidated_report(stats, entry_name_for_title, file_identifier, output_dir=None, prefix=""):
    """
    統合された結果のレポートを生成し、サマリテキストファイルとして保存する。
    レポート内のタイトルには entry_name_for_title を、ファイル名には file_identifier を使用する。
    """
    summary_lines = []

    def format_rate(rate): return f"{rate:.2%}" if pd.notna(rate) and isinstance(rate, (int, float)) else "N/A"
    def format_count(correct, total): return f"{correct} / {total}" if pd.notna(correct) and pd.notna(total) and total >= 0 else "N/A"
    def format_score(score, possible):
         if not (pd.notna(score) and pd.notna(possible)): return "N/A"
         score_str = f"{int(score)}" if score == int(score) else f"{score:.1f}"
         possible_str = f"{int(possible)}" if possible == int(possible) else f"{possible:.1f}"
         if possible == 0: return f"{score_str} / {possible_str} (0点満点)"
         return f"{score_str} / {possible_str}"

    # レポートタイトルには entry_name_for_title を使用
    summary_lines.append(f"===== モデルエントリー: {entry_name_for_title} の結果 =====")
    summary_lines.append(f"(ファイル識別子: {file_identifier})") # 参考情報としてファイル識別子も追記
    summary_lines.append("")

    total_data = stats.get('total', {})
    general_data = stats.get('general', {})
    required_data = stats.get('required', {})
    no_image_data = stats.get('no_image', {})
    no_image_general_data = no_image_data.get('general', {}) if isinstance(no_image_data, dict) else {}
    no_image_required_data = no_image_data.get('required', {}) if isinstance(no_image_data, dict) else {}

    sections = {
        "全体": total_data,
        "一般問題 (A,C,D,F)": general_data,
        "必修問題 (B,E)": required_data,
        "画像なし - 全体": no_image_data,
        "画像なし - 一般問題 (A,C,D,F)": no_image_general_data,
        "画像なし - 必修問題 (B,E)": no_image_required_data,
    }

    for title, data in sections.items():
        if data and isinstance(data, dict):
            summary_lines.append(f"--- {title} ---")
            summary_lines.append(f"  正解数/問題数: {format_count(data.get('correct'), data.get('total'))}")
            summary_lines.append(f"  正答率: {format_rate(data.get('accuracy'))}")
            summary_lines.append(f"  獲得点/満点: {format_score(data.get('score'), data.get('possible_score'))}")
            summary_lines.append(f"  得点率: {format_rate(data.get('score_rate'))}")
            summary_lines.append("")
        else:
            summary_lines.append(f"--- {title} ---")
            summary_lines.append("  (データなし)")
            summary_lines.append("")

    def add_block_results(title, acc_dict, sr_dict):
        summary_lines.append(title)
        acc_dict = acc_dict if isinstance(acc_dict, dict) else {}
        sr_dict = sr_dict if isinstance(sr_dict, dict) else {}
        all_blocks = sorted(list(set(acc_dict.keys()) | set(sr_dict.keys())))
        if not all_blocks:
             summary_lines.append("  (ブロックデータなし)")
        else:
            for block in all_blocks:
                acc = acc_dict.get(block)
                sr = sr_dict.get(block)
                summary_lines.append(f"  ブロック {block}: 正答率 {format_rate(acc)}, 得点率 {format_rate(sr)}")
        summary_lines.append("")

    add_block_results("--- ブロック別結果 (全体) ---", stats.get('block_accuracies', {}), stats.get('block_score_rates', {}))
    no_image_block_acc = no_image_data.get('block_accuracies', {}) if isinstance(no_image_data, dict) else {}
    no_image_block_sr = no_image_data.get('block_score_rates', {}) if isinstance(no_image_data, dict) else {}
    add_block_results("--- 画像なしのブロック別結果 ---", no_image_block_acc, no_image_block_sr)

    summary = "\n".join(summary_lines)
    print("\n--- 統合サマリ ---") # コンソール出力の見出しを修正
    print(summary)

    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)
        # ファイル名には file_identifier を使用
        summary_file = output_path / f"{prefix}{file_identifier}_summary.txt"
        try:
            with open(summary_file, "w", encoding="utf-8") as f:
                f.write(summary)
            print(f"サマリを {summary_file} に保存しました。")
        except Exception as e:
            print(f"エラー: サマリファイル '{summary_file}' の書き込み中にエラーが発生しました: {e}")


# --- Leaderboard 機能 ---
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

def generate_leaderboard_markdown(leaderboard_data):
    """
    Leaderboardデータ(辞書)からMarkdownテーブル文字列を生成する。
    """
    if not leaderboard_data or not isinstance(leaderboard_data, dict):
        return "_(Leaderboard data is empty or invalid)_"

    def sort_key(item):
        scores_dict = item[1]
        if not isinstance(scores_dict, dict): return -float('inf')
        score_rate = scores_dict.get('overall_score_rate')
        if isinstance(score_rate, (int, float)) and pd.notna(score_rate):
            return score_rate
        else:
             return -float('inf')

    try:
        sorted_entries = sorted(leaderboard_data.items(), key=sort_key, reverse=True)
    except Exception as e:
        print(f"エラー: Leaderboardデータのソート中にエラー: {e}")
        return "_(Error generating leaderboard during sorting)_"

    def format_lb_cell(value, total, rate):
        value_valid = pd.notna(value) and isinstance(value, (int, float))
        total_valid = pd.notna(total) and isinstance(total, (int, float))
        rate_valid = pd.notna(rate) and isinstance(rate, (int, float))

        if not total_valid or total == 0: return "N/A"
        if not value_valid: value_str = "?"
        else: value_str = f"{int(value)}" if value == int(value) else f"{value:.1f}"
        total_str = f"{int(total)}" if total == int(total) else f"{total:.1f}"
        if not rate_valid: rate_str = "?%"
        else: rate_str = f"{rate:.2%}"
        return f"{value_str}/{total_str} ({rate_str})"

    headers = ["Rank", "Entry", "Overall Score (Rate)", "Overall Acc.", "No-Img Score (Rate)", "No-Img Acc."]
    header_line = "| " + " | ".join(headers) + " |"
    separator_line = "|-" + "-|".join(['-' * len(h) for h in headers]) + "-|"

    table_lines = [header_line, separator_line]
    for rank, (entry_name, scores) in enumerate(sorted_entries, 1):
         if not isinstance(scores, dict):
             print(f"警告: エントリー '{entry_name}' のデータが不正です。Leaderboard表示をスキップします。")
             row_values = ["Error"] * (len(headers) - 2)
         else:
             row_values = [
                 format_lb_cell(scores.get('overall_score'), scores.get('overall_possible_score'), scores.get('overall_score_rate')),
                 format_lb_cell(scores.get('overall_correct'), scores.get('overall_total'), scores.get('overall_accuracy')),
                 format_lb_cell(scores.get('no_image_score'), scores.get('no_image_possible_score'), scores.get('no_image_score_rate')),
                 format_lb_cell(scores.get('no_image_correct'), scores.get('no_image_total'), scores.get('no_image_accuracy')),
             ]
         safe_entry_name = str(entry_name).replace("|", "\\|")
         row = [str(rank), safe_entry_name] + row_values
         table_lines.append("| " + " | ".join(row) + " |")

    return "\n".join(table_lines)

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


# --- main関数 (ファイル名生成部分を修正) ---

def main():
    parser = argparse.ArgumentParser(
        description='モデルの解答を採点し、結果をLeaderboardに反映するスクリプト',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--answers_path', '-a', default='results/correct_answers.csv',
                        help='正解データのパス（CSVまたはPickle）')
    parser.add_argument('--output', '-o', default='./results',
                        help='統合レポートファイル（CSV, PNG, TXT, スキップリスト）の出力先ディレクトリ')
    parser.add_argument('--readme', default='README.md',
                        help='更新するREADMEファイルのパス')
    parser.add_argument('--leaderboard_data', default='results/leaderboard.json',
                        help='Leaderboardデータを保存/更新するJSONファイルのパス')
    parser.add_argument('--entry_name', '-e', default=None,
                        help='Leaderboardやレポートタイトルで使用するエントリー名。指定されない場合はJSONファイル名や内容から推測。')
    parser.add_argument('--json_paths', '-j', nargs='+', required=True,
                        help='モデル解答のJSONファイルへのパス（複数指定可能、例: results/119A_my-exp.json ...）')
    parser.add_argument('--skipped_output_file', default=None,
                        help='スキップされた問題番号を出力するファイル名。指定しない場合は、outputディレクトリに"{prefix}{file_identifier}_skipped.txt"として自動生成。')


    args = parser.parse_args()

    valid_json_paths = []
    processed_blocks = set()
    base_numbers_from_files = set()
    first_valid_json_path = None
    first_file_identifier = None # ファイル名から推測する識別子 (--exp 相当)

    # JSONパスの検証と情報抽出
    for path_str in args.json_paths:
        path = Path(path_str)
        if not path.exists():
            print(f"警告: JSONファイルが見つかりません: {path}。スキップします。")
            continue
        if not path.is_file():
            print(f"警告: {path} はファイルではありません。スキップします。")
            continue

        valid_json_paths.append(str(path))
        filename = path.stem # 拡張子なしのファイル名 (例: 119A_my-exp)
        parts = filename.split('_', 1) # 最初の '_' で分割

        if first_valid_json_path is None:
            first_valid_json_path = path
            # 最初の有効なファイルからファイル識別子を推測
            if len(parts) > 1:
                first_file_identifier = parts[1] # 例: my-exp

        base_number_with_block = parts[0] if parts else ""

        block_id = ""
        base_number = ""
        if base_number_with_block and len(base_number_with_block) > 1 and base_number_with_block[-1].isalpha():
            block_id = base_number_with_block[-1].upper()
            base_number = base_number_with_block[:-1]
            if base_number.isdigit():
                 if block_id in processed_blocks:
                      print(f"警告: ブロック {block_id} (試験番号 {base_number}) が複数回指定されています ({path.name})。")
                 processed_blocks.add(block_id)
                 base_numbers_from_files.add(base_number)
            else:
                print(f"警告: ファイル名 {path.name} から有効な試験番号を特定できません ({base_number_with_block})。")

        elif base_number_with_block.isdigit():
             base_number = base_number_with_block
             base_numbers_from_files.add(base_number)
             print(f"情報: ファイル名 {path.name} にはブロックIDが含まれていません。試験番号 {base_number} として扱います。")
        else:
             print(f"警告: ファイル名 {path.name} から試験番号やブロックIDを特定できません。")


    if not valid_json_paths:
        parser.error("処理可能なJSONファイルが1つも見つかりませんでした。")

    # ファイル名から取得した識別子、なければデフォルト値
    file_identifier = first_file_identifier or "UnknownExp"
    print(f"情報: レポートファイル名に使用する識別子: '{file_identifier}'")


    if len(base_numbers_from_files) > 1:
        print(f"警告: 複数の試験番号がファイル名に含まれています: {sorted(list(base_numbers_from_files))}。")
        print(f"       レポートのプレフィックスには、最初のファイルから取得した '{list(base_numbers_from_files)[0]}' を使用します。")
    elif not base_numbers_from_files:
        print(f"警告: ファイル名から試験番号を特定できませんでした。レポートプレフィックスは 'UnknownNumber_' になります。")

    report_base_number = list(base_numbers_from_files)[0] if base_numbers_from_files else "UnknownNumber"
    consolidated_prefix = f"{report_base_number}_" # 例: 119_


    try:
        correct_df = load_correct_answers(args.answers_path)
    except (FileNotFoundError, ValueError) as e:
        parser.error(f"正解ファイルの読み込みエラー: {e}")
    except Exception as e:
         parser.error(f"正解ファイルの読み込み中に予期せぬエラー: {e}")


    all_results_dfs = []
    first_model_name_from_json = None
    all_skipped_questions = []

    for i, json_path_str in enumerate(valid_json_paths):
        json_path = Path(json_path_str)
        print(f"\n--- [{i+1}/{len(valid_json_paths)}] {json_path.name} を処理中 ---")

        try:
            model_data = load_model_answers(json_path_str)
            results_df, accuracy, model_name_in_current_json, skipped_q = grade_answers(model_data, correct_df)

            has_results = not results_df.empty
            has_skipped = bool(skipped_q)

            if not has_results and not has_skipped:
                print(f"情報: {json_path.name} から有効な採点結果もスキップ情報も得られませんでした。")
                continue
            elif not has_results and has_skipped:
                print(f"情報: {json_path.name} から有効な採点結果はありませんでしたが、スキップされた問題がありました: {skipped_q}")
            elif has_results and has_skipped:
                 print(f"情報: {json_path.name} の処理中にスキップされた問題がありました: {skipped_q}")

            if first_model_name_from_json is None and model_name_in_current_json and model_name_in_current_json != 'UnknownModel':
                 first_model_name_from_json = model_name_in_current_json

            if has_results:
                print_individual_report_summary(results_df, accuracy, json_path.name)
                all_results_dfs.append(results_df)

            if has_skipped:
                all_skipped_questions.extend(skipped_q)

        except (FileNotFoundError, json.JSONDecodeError, ValueError) as e:
            print(f"エラー: {json_path.name} の処理中にエラーが発生しました: {e}。このファイルはスキップします。")
            continue
        except Exception as e:
             print(f"予期せぬエラー: {json_path.name} の処理中にエラー: {e}。このファイルはスキップします。")
             continue


    # --- Leaderboard用エントリー名の決定 ---
    effective_entry_name = "UnknownEntry" # デフォルト値
    if args.entry_name:
        effective_entry_name = args.entry_name
        print(f"\n情報: コマンドライン引数からLeaderboard/レポートタイトル用エントリー名 '{effective_entry_name}' を使用します。")
    elif first_file_identifier: # ファイル識別子をエントリー名の候補とする (以前の実装に近い挙動)
        effective_entry_name = file_identifier
        print(f"\n情報: ファイル名から推測した識別子 '{effective_entry_name}' をLeaderboard/レポートタイトル用エントリー名として使用します。")
    elif first_model_name_from_json:
        effective_entry_name = first_model_name_from_json
        print(f"\n情報: 最初のJSONファイルの 'model' フィールドからエントリー名 '{effective_entry_name}' をLeaderboard/レポートタイトル用に使用します。")
    else:
        print("\n警告: Leaderboard/レポートタイトルで使用するエントリー名を特定できませんでした。'UnknownEntry' として扱います。")


    # --- スキップされた問題番号のファイル出力 ---
    unique_skipped_questions = sorted(list(set(all_skipped_questions)))

    if unique_skipped_questions:
        output_path = Path(args.output)
        output_path.mkdir(exist_ok=True, parents=True)

        if args.skipped_output_file:
            skipped_file_path = Path(args.skipped_output_file)
            skipped_file_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            # ファイル名には file_identifier を使用
            skipped_file_name = f"{consolidated_prefix}{file_identifier}_skipped.txt"
            skipped_file_path = output_path / skipped_file_name

        print(f"\n--- スキップされた問題番号 ({len(unique_skipped_questions)} 件) をファイルに出力 ---")
        try:
            with open(skipped_file_path, 'w', encoding='utf-8') as f:
                for q_num in unique_skipped_questions:
                    f.write(f"{q_num}\n")
            print(f"スキップされた問題番号を {skipped_file_path} に保存しました。")
            print("これらの問題を再実行するには、以下のコマンドを使用します（パスやモデル名、exp名を適宜変更してください）:")
            # 再実行コマンド例では file_identifier を --exp に使うことを示唆
            print(f"  uv run main.py <INPUT_JSON_PATH> --models <YOUR_MODEL_NAME> --questions $(cat '{skipped_file_path}') --exp {file_identifier}_retry")
            print(f"  注意: <INPUT_JSON_PATH> には、スキップされた問題が含まれるJSONファイル（例: questions/{report_base_number}A_json.json など）を指定してください。")
            print(f"        必要に応じて、問題をブロックごとに分けて再実行してください。")

        except Exception as e:
            print(f"エラー: スキップされた問題番号のファイル '{skipped_file_path}' の書き込み中にエラー: {e}")
    else:
        if all_results_dfs:
             print("\nスキップされた問題はありませんでした。")


    # --- 統合処理とレポート生成 ---
    if not all_results_dfs:
        print("\n有効な採点結果がなかったため、統合レポートとLeaderboard更新をスキップします。")
        return

    print("\n--- 全ブロックの結果を統合中 ---")
    try:
        consolidated_stats = consolidate_results(all_results_dfs)
        consolidated_df = consolidated_stats.get('total', {}).get('df', pd.DataFrame())
    except Exception as e:
         print(f"エラー: 結果の統合処理中にエラーが発生しました: {e}。統合レポートとLeaderboard更新をスキップします。")
         return

    # 統合サマリ: タイトルに effective_entry_name、ファイル名に file_identifier を使用
    generate_consolidated_report(
        stats=consolidated_stats,
        entry_name_for_title=effective_entry_name,
        file_identifier=file_identifier,
        output_dir=args.output,
        prefix=consolidated_prefix
    )

    # 統合ファイル（プロット、CSV）: ファイル名に file_identifier を使用
    if not consolidated_df.empty:
        generate_consolidated_files(
            consolidated_df=consolidated_df,
            output_dir=args.output,
            prefix=consolidated_prefix,
            file_identifier=file_identifier
        )
    else:
        print("情報: 統合DataFrameが空のため、統合ファイル (PNG, CSV) の生成をスキップします。")


    # --- Leaderboard 更新処理 ---
    # Leaderboardのエントリー名には effective_entry_name を使用
    print("\n--- Leaderboard 更新 ---")
    if effective_entry_name == "UnknownEntry":
        print("警告: Leaderboard用エントリー名を特定できなかったため、Leaderboard は更新されません。")
    elif not consolidated_stats:
         print("警告: 統合結果がないため、Leaderboard は更新されません。")
    else:
        try:
            leaderboard_data = update_leaderboard_data(args.leaderboard_data, effective_entry_name, consolidated_stats)
            markdown_table = generate_leaderboard_markdown(leaderboard_data)
            update_readme(args.readme, markdown_table)
        except Exception as e:
            print(f"エラー: Leaderboardの更新処理中に予期せぬエラーが発生しました: {e}")

    print("\n処理が完了しました。")


if __name__ == "__main__":
    main()