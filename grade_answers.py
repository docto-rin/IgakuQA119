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

# --- 既存関数の修正・改善 ---

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
    """モデルの解答を採点する"""
    results = []
    correct_count = 0
    total_questions = 0 # 正解が存在する問題の数
    model_name_in_json = None # JSONデータからモデル名を取得

    try:
        correct_answers = dict(zip(correct_df['問題番号'].astype(str), correct_df['解答'].astype(str)))
    except KeyError:
        raise ValueError("正解データに '問題番号' または '解答' 列がありません。")
    except Exception as e:
        raise ValueError(f"正解データのマッピング作成中にエラー: {e}")


    if 'results' not in model_data or not isinstance(model_data['results'], list):
        print(f"警告: JSONデータに 'results' キーがないか、リスト形式ではありません。")
        return pd.DataFrame(results), 0.0, None

    for i, question in enumerate(model_data['results']):
        if not isinstance(question, dict):
            print(f"警告: resultsリストの要素 {i} が辞書形式ではありません。スキップします: {question}")
            continue

        required_keys = ['question_number', 'answers', 'has_image']
        if not all(key in question for key in required_keys):
            missing_keys = [key for key in required_keys if key not in question]
            print(f"警告: questionデータに必要なキーがありません ({missing_keys})。スキップします: {question.get('question_number', 'N/A')}")
            continue

        if not isinstance(question['answers'], list) or not question['answers']:
            print(f"警告: question {question.get('question_number', 'N/A')} の 'answers' が空またはリスト形式ではありません。スキップします。")
            continue

        answer_data = question['answers'][0] # 最初の解答を使用
        if not isinstance(answer_data, dict):
            print(f"警告: question {question.get('question_number', 'N/A')} の最初のanswerが辞書形式ではありません。スキップします。")
            continue

        required_answer_keys = ['answer', 'model', 'confidence']
        if not all(key in answer_data for key in required_answer_keys):
            missing_keys = [key for key in required_answer_keys if key not in answer_data]
            print(f"警告: answerデータに必要なキーがありません ({missing_keys})。スキップします: {question.get('question_number', 'N/A')}")
            continue

        question_number = str(question.get('question_number', 'Unknown')) # 文字列に統一
        model_answer = answer_data.get('answer')
        model_name = answer_data.get('model', 'UnknownModel') # JSON内のモデル名
        confidence = answer_data.get('confidence')
        has_image = question.get('has_image', False) # デフォルト値を設定

        if model_name_in_json is None and model_name != 'UnknownModel':
            model_name_in_json = model_name

        correct_answer = correct_answers.get(question_number)
        is_correct = False

        if correct_answer is not None:
            total_questions += 1
            try:
                model_answer_str = str(model_answer).strip().lower() if model_answer is not None else ""
                correct_answer_str = str(correct_answer).strip().lower()

                if not model_answer_str:
                    is_correct = False
                elif question_number == '119E28': # 特定問題の複数正解処理
                    is_correct = model_answer_str in ['a', 'c']
                else:
                    is_correct = model_answer_str == correct_answer_str

            except Exception as e:
                print(f"エラー: 問題 {question_number} の解答比較中にエラーが発生しました。 Model: '{model_answer}', Correct: '{correct_answer}'. Error: {e}")

            if is_correct:
                correct_count += 1
        else:
             # print(f"警告: 問題番号 {question_number} の正解が解答辞書に見つかりません。採点対象外とします。")
             pass # 正解がない場合は採点しないだけなので、警告は抑制してもよい

        results.append({
            'question_number': question_number,
            'model': model_name, # JSONから取得したモデル名
            'model_answer': model_answer,
            'correct_answer': correct_answer,
            'is_correct': is_correct,
            'confidence': confidence,
            'has_image': has_image
        })

    accuracy = correct_count / total_questions if total_questions > 0 else 0.0
    return pd.DataFrame(results), accuracy, model_name_in_json


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
        first_valid_model = results_df['model'].dropna().iloc[0] if not results_df['model'].dropna().empty else "UnknownModel"
        report_model_name = first_valid_model

    print(f"ファイル: {json_filename}")
    print(f"モデル (JSON内): {report_model_name}")
    print(f"正答率: {accuracy:.2%}")

    valid_results = results_df.dropna(subset=['correct_answer'])
    correct_sum = valid_results['is_correct'].sum()
    total_len_graded = len(valid_results)
    print(f"正解数 (採点対象): {correct_sum} / {total_len_graded}")
    if len(results_df) != total_len_graded:
        print(f"(全問題数: {len(results_df)}, うち正解不明: {len(results_df) - total_len_graded})")

    # 信頼度と正誤の関係を分析 (NaNを除外)
    print("\n信頼度と正誤の関係 (コンソール表示のみ):")
    try:
        if 'confidence' in results_df.columns and pd.api.types.is_numeric_dtype(results_df['confidence']):
            confidence_accuracy = valid_results.dropna(subset=['confidence', 'is_correct']).groupby('confidence')['is_correct'].mean()
            if not confidence_accuracy.empty:
                # print(confidence_accuracy) # 詳細すぎるのでコメントアウトしてもよい
                pass # 必要なら表示
            else:
                print("  情報: 信頼度と正誤の組み合わせデータがありません。")
        else:
            print("  情報: 'confidence'列が存在しないか数値型でないため、信頼度分析はスキップします。")
    except Exception as e:
        print(f"  エラー: 信頼度分析中にエラーが発生しました: {e}")

    print("-" * 20) # 区切り線

# --- 新しい関数: 統合レポートファイル生成 ---
def generate_consolidated_files(consolidated_df, output_dir, prefix, entry_name):
    """統合された結果からレポートファイル（プロット、CSV）を生成する"""
    if consolidated_df.empty:
        print("情報: 統合結果データフレームが空のため、ファイル生成をスキップします。")
        return

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    file_prefix = f"{prefix}{entry_name}_" # 例: 119_all_Gemma_3_

    try:
        print("\n--- 統合レポートファイルの作成 ---")
        # --- 信頼度分布プロット (統合版) ---
        if 'confidence' in consolidated_df.columns and pd.api.types.is_numeric_dtype(consolidated_df['confidence']):
            plt.figure(figsize=(10, 6))
            # dropna()でNaNを含む行を除外 (confidence と is_correct 両方)
            # 正解が存在する問題のみでプロット (consolidated_df は既に filter 済みのはずだが念のため)
            plot_data = consolidated_df.dropna(subset=['confidence', 'is_correct'])
            if not plot_data.empty:
                 sns.histplot(data=plot_data, x='confidence', hue='is_correct', bins=20, multiple='stack')
                 plt.title(f'Confidence Distribution and Correctness ({entry_name} - Consolidated)')
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
        wrong_answers = consolidated_df[~consolidated_df['is_correct']]
        if not wrong_answers.empty:
            wrong_csv_path = output_path / f"{file_prefix}wrong_answers.csv"
            wrong_answers.to_csv(wrong_csv_path, index=False, encoding='utf-8-sig')
            print(f"誤答問題リストを {wrong_csv_path} に保存しました。")
        else:
             print("情報: 統合結果で間違えた問題はありませんでした (採点対象内で)。")

        # --- 全結果のCSV (統合版) ---
        # consolidate_results で返される df は graded_df (正解が存在するもの) なので注意
        # もし正解不明も含めた全データが必要なら consolidate_results から original_combined_df を返すように修正が必要
        results_csv_path = output_path / f"{file_prefix}grading_results.csv"
        consolidated_df.to_csv(results_csv_path, index=False, encoding='utf-8-sig')
        print(f"全採点結果CSVを {results_csv_path} に保存しました。")

    except Exception as e:
        print(f"エラー: 統合レポートファイルの保存中にエラーが発生しました: {e}")


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
            # 'original_combined_df': pd.DataFrame() # 必要なら追加
        }
        return empty_stats

    combined_df = pd.concat(all_results, ignore_index=True)
    # original_combined_df = combined_df.copy() # 元の全データを保持する場合

    graded_df = combined_df.dropna(subset=['correct_answer']).copy()
    if len(graded_df) != len(combined_df):
        print(f"情報: 統合された全 {len(combined_df)} 件のうち、正解データが存在しない {len(combined_df) - len(graded_df)} 件を集計から除外します。")

    if graded_df.empty:
        print("警告: 正解データが存在する有効な結果が統合後に見つかりませんでした。")
        empty_stats['original_combined_df'] = combined_df # 元データだけ返す
        return empty_stats

    try:
        graded_df['question_number'] = graded_df['question_number'].astype(str)
        graded_df['block'] = graded_df['question_number'].str[3].str.upper()
        graded_df['has_image'] = graded_df['has_image'].fillna(False).astype(bool)
        graded_df['is_correct'] = graded_df['is_correct'].fillna(False).astype(bool)
    except Exception as e:
        print(f"エラー: 統合データの前処理中にエラーが発生しました: {e}")
        # エラーが発生した場合でも処理を続けられるように、空のDataFrameを返す
        return {
            'total': {'df': pd.DataFrame(), 'correct': 0, 'total': 0, 'accuracy': 0.0, 'score': 0.0, 'possible_score': 0.0, 'score_rate': 0.0, 'block_accuracies': {}, 'block_score_rates': {}},
            # 他のカテゴリも同様に空にする
            'general': {'df': pd.DataFrame(), 'correct': 0, 'total': 0, 'accuracy': 0.0, 'score': 0.0, 'possible_score': 0.0, 'score_rate': 0.0},
            'required': {'df': pd.DataFrame(), 'correct': 0, 'total': 0, 'accuracy': 0.0, 'score': 0.0, 'possible_score': 0.0, 'score_rate': 0.0},
            'block_accuracies': {}, 'block_score_rates': {},
            'no_image': { 'df': pd.DataFrame(), 'correct': 0, 'total': 0, 'accuracy': 0.0, 'score': 0.0, 'possible_score': 0.0, 'score_rate': 0.0, 'general': {}, 'required': {}, 'block_accuracies': {}, 'block_score_rates': {}},
            # 'original_combined_df': original_combined_df
        }


    general_blocks = ['A', 'C', 'D', 'F']
    required_blocks = ['B', 'E']
    valid_blocks = general_blocks + required_blocks

    invalid_block_mask = ~graded_df['block'].isin(valid_blocks)
    if invalid_block_mask.any():
        print(f"警告: 不正または不明なブロックIDを持つ問題が {invalid_block_mask.sum()} 件あります（採点対象内）。これらは集計から除外される可能性があります。")
        print(graded_df.loc[invalid_block_mask, ['question_number', 'block']].head())
        # graded_df = graded_df[graded_df['block'].isin(valid_blocks)] # 必要なら除外

    graded_df['is_general'] = graded_df['block'].isin(general_blocks)
    graded_df['is_required'] = graded_df['block'].isin(required_blocks)

    # --- 点数計算関数 (変更なし) ---
    def required_point(question_number):
        try:
            block = question_number[3].upper()
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
        if block in required_blocks:
            possible = required_point(question_number)
            score = possible if is_correct else 0
        elif block in general_blocks:
            possible = 1
            score = 1 if is_correct else 0
        else:
            possible = 0
            score = 0
        return pd.Series({'score': score, 'possible': possible})

    try:
        points_df = graded_df.apply(calculate_points, axis=1)
        graded_df[['score', 'possible']] = points_df
    except Exception as e:
        print(f"エラー: 点数計算中にエラーが発生しました: {e}。点数は0として扱われます。")
        graded_df['score'] = 0
        graded_df['possible'] = 0

    # 画像なしDFは点数計算後に作成
    no_image_df = graded_df[~graded_df['has_image']].copy()

    print(f"\n統合結果概要 (採点対象):")
    print(f"  総問題数: {len(graded_df)}")
    print(f"  処理したブロック: {sorted(list(graded_df['block'].unique()))}") # uniqueの結果をリストに変換
    print(f"  一般問題 (A,C,D,F): {graded_df['is_general'].sum()} 件")
    print(f"  必修問題 (B,E): {graded_df['is_required'].sum()} 件")
    print(f"  画像あり問題: {graded_df['has_image'].sum()} 件")
    print(f"  画像なし問題: {len(no_image_df)} 件")

    # --- 集計関数 (変更なし) ---
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
         block_agg = df.groupby('block').agg(
             correct=('is_correct', 'sum'),
             total=('is_correct', 'size'),
             score=('score', 'sum'),
             possible=('possible', 'sum')
         )
         block_agg['accuracy'] = (block_agg['correct'] / block_agg['total'].replace(0, pd.NA)).astype(float)
         block_agg['score_rate'] = (block_agg['score'] / block_agg['possible'].replace(0, pd.NA)).astype(float)
         # NaN を 0.0 に置換しない方が実態に近い場合もある
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
        # 'original_combined_df': original_combined_df # 必要なら追加
    }
    return stats


def generate_consolidated_report(stats, entry_name, output_dir=None, prefix=""):
    """統合された結果のレポートを生成し、サマリテキストファイルとして保存する"""
    summary_lines = []

    def format_rate(rate): return f"{rate:.2%}" if pd.notna(rate) and isinstance(rate, (int, float)) else "N/A"
    def format_count(correct, total): return f"{correct} / {total}" if pd.notna(correct) and pd.notna(total) and total > 0 else "N/A" if total == 0 else f"{correct} / {total}"
    def format_score(score, possible):
         if not (pd.notna(score) and pd.notna(possible)): return "N/A"
         if possible == 0: return "N/A (0点満点)"
         score_str = f"{int(score)}" if score == int(score) else f"{score:.1f}"
         possible_str = f"{int(possible)}" if possible == int(possible) else f"{possible:.1f}"
         return f"{score_str} / {possible_str}"

    summary_lines.append(f"===== モデルエントリー: {entry_name} の結果 =====") # モデル名 -> モデルエントリー名
    summary_lines.append("")

    sections = {
        "全体": stats['total'],
        "一般問題 (A,C,D,F)": stats['general'],
        "必修問題 (B,E)": stats['required'],
        "画像なし - 全体": stats['no_image'],
        "画像なし - 一般問題 (A,C,D,F)": stats['no_image']['general'],
        "画像なし - 必修問題 (B,E)": stats['no_image']['required'],
    }

    for title, data in sections.items():
        # data が None または 辞書でない場合のチェックを追加
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
        if not isinstance(acc_dict, dict): acc_dict = {}
        if not isinstance(sr_dict, dict): sr_dict = {}
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
    no_image_stats = stats.get('no_image', {})
    add_block_results("--- 画像なしのブロック別結果 ---", no_image_stats.get('block_accuracies', {}) if isinstance(no_image_stats, dict) else {}, no_image_stats.get('block_score_rates', {}) if isinstance(no_image_stats, dict) else {})

    summary = "\n".join(summary_lines)
    print(summary)

    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)
        # サマリファイル名に entry_name を含める
        summary_file = output_path / f"{prefix}{entry_name}_summary.txt"
        try:
            with open(summary_file, "w", encoding="utf-8") as f:
                f.write(summary)
            print(f"サマリを {summary_file} に保存しました。")
        except Exception as e:
            print(f"エラー: サマリファイル '{summary_file}' の書き込み中にエラーが発生しました: {e}")


# --- Leaderboard 機能 (変更なし、model_nameをentry_nameとして受け取るようにする) ---

def update_leaderboard_data(data_file, entry_name, stats): # model_name -> entry_name
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
            return { 'score': 0.0, 'possible_score': 0.0, 'score_rate': 0.0, 'correct': 0, 'total': 0, 'accuracy': 0.0 } # デフォルト値変更
        # .get() のデフォルト値を設定し、Noneの可能性を減らす
        return {
            'score': category_stats.get('score', 0.0),
            'possible_score': category_stats.get('possible_score', 0.0),
            'score_rate': category_stats.get('score_rate', 0.0),
            'correct': category_stats.get('correct', 0),
            'total': category_stats.get('total', 0),
            'accuracy': category_stats.get('accuracy', 0.0)
        }

    overall_stats = get_stat_values(stats.get('total'))
    no_image_stats = get_stat_values(stats.get('no_image')) # no_image 自体が存在しない可能性も考慮

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

    # NaN値の処理 (get_stat_valuesでデフォルト値設定したので基本不要だが念のため)
    for key, value in model_entry.items():
        if pd.isna(value):
            model_entry[key] = 0.0 if 'rate' in key or 'accuracy' in key else 0

    leaderboard_data[entry_name] = model_entry # model_name -> entry_name

    try:
        data_path.parent.mkdir(parents=True, exist_ok=True)
        # NaNを許容しない設定でダンプ
        with open(data_path, 'w', encoding='utf-8') as f:
            json.dump(leaderboard_data, f, indent=4, ensure_ascii=False, allow_nan=False)
        print(f"Leaderboardデータを {data_file} に保存/更新しました。")
    except TypeError as e:
         # NaNが原因の場合のエラーメッセージを確認
        if 'Out of range float values are not JSON compliant' in str(e) or 'NaN' in str(e):
            print(f"エラー: JSONシリアライズ中に非準拠の浮動小数点値(NaN/Infinity)が検出されました。0に置換して再試行します。")
            # NaN/InfinityをNoneに置換してからNoneを0に置換するなどの処理が必要
            def replace_nan_inf(obj):
                if isinstance(obj, dict):
                    return {k: replace_nan_inf(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [replace_nan_inf(elem) for elem in obj]
                elif isinstance(obj, float) and (pd.isna(obj) or obj == float('inf') or obj == float('-inf')):
                    return 0.0 # NaNやInfinityを0.0に置換
                return obj
            try:
                cleaned_data = replace_nan_inf(leaderboard_data)
                with open(data_path, 'w', encoding='utf-8') as f:
                    json.dump(cleaned_data, f, indent=4, ensure_ascii=False, allow_nan=False)
                print(f"Leaderboardデータを {data_file} に保存/更新しました (非準拠値を0に置換)。")
            except Exception as e:
                 print(f"エラー: 非準拠値置換後のJSON書き込み中に再度エラーが発生しました: {e}")

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
        # score_rateが有効な数値か確認 (Noneや非数値を弾く)
        if isinstance(score_rate, (int, float)) and pd.notna(score_rate):
            return score_rate
        else:
            # NoneやNaNの場合は得点率0点として扱うか、別の指標(Accuracyなど)でソートするか検討
            # ここでは最低値扱い
             return -float('inf')

    try:
        if isinstance(leaderboard_data, dict):
            sorted_entries = sorted(leaderboard_data.items(), key=sort_key, reverse=True) # sorted_models -> sorted_entries
        else:
            print("エラー: Leaderboardデータが辞書形式ではありません。")
            return "_(Error: Leaderboard data is not a dictionary)_"
    except Exception as e:
        print(f"エラー: Leaderboardデータのソート中にエラー: {e}")
        return "_(Error generating leaderboard during sorting)_"

    def format_lb_cell(value, total, rate):
        value_valid = isinstance(value, (int, float)) and pd.notna(value)
        total_valid = isinstance(total, (int, float)) and pd.notna(total)
        rate_valid = isinstance(rate, (int, float)) and pd.notna(rate)

        if not total_valid or total == 0: return "N/A"
        if not value_valid: value_str = "?"
        else: value_str = f"{int(value)}" if value == int(value) else f"{value:.1f}"
        total_str = f"{int(total)}" if total == int(total) else f"{total:.1f}"
        if not rate_valid: rate_str = "?%"
        else: rate_str = f"{rate:.2%}"
        return f"{value_str}/{total_str} ({rate_str})"

    # ヘッダー修正: Model -> Entry
    headers = ["Rank", "Entry", "Overall Score (Rate)", "Overall Acc.", "No-Img Score (Rate)", "No-Img Acc."]
    header_line = "| " + " | ".join(headers) + " |"
    separator_line = "|-" + "-|".join(['-' * len(h) for h in headers]) + "-|"

    table_lines = [header_line, separator_line]
    for rank, (entry_name, scores) in enumerate(sorted_entries, 1): # model_name -> entry_name
         if not isinstance(scores, dict):
             print(f"警告: エントリー '{entry_name}' のデータが不正です。スキップします。")
             row_values = ["N/A"] * (len(headers) - 2)
         else:
             row_values = [
                 format_lb_cell(scores.get('overall_score'), scores.get('overall_possible_score'), scores.get('overall_score_rate')),
                 format_lb_cell(scores.get('overall_correct'), scores.get('overall_total'), scores.get('overall_accuracy')),
                 format_lb_cell(scores.get('no_image_score'), scores.get('no_image_possible_score'), scores.get('no_image_score_rate')),
                 format_lb_cell(scores.get('no_image_correct'), scores.get('no_image_total'), scores.get('no_image_accuracy')),
             ]
         safe_entry_name = str(entry_name).replace("|", "\\|") # entry_nameをエスケープ
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
                if start_index != -1 and start_index < end_index: break

        if start_index != -1 and end_index != -1 and start_index < end_index:
            new_lines = lines[:start_index + 1] + [""] + [markdown_table] + [""] + lines[end_index:]
            new_content = "\n".join(new_lines)
            if not new_content.endswith('\n'): new_content += '\n'
            readme_path.write_text(new_content, encoding='utf-8')
            print(f"READMEファイル ({readme_path}) のLeaderboardを更新しました。")
        else:
             missing = [m for m, i in [(LEADERBOARD_START_MARKER, start_index), (LEADERBOARD_END_MARKER, end_index)] if i == -1 or not (start_index != -1 and start_index < end_index)]
             print(f"警告: READMEファイル ({readme_path}) に有効なLeaderboardマーカーペアが見つかりません。")
             print(f"       必要なマーカー: {LEADERBOARD_START_MARKER} と {LEADERBOARD_END_MARKER} (この順序で)")
             if missing: print(f"       見つからなかった/順序不正マーカー: {missing}")
             print(f"       READMEは更新されませんでした。")

    except Exception as e:
        print(f"エラー: READMEファイル '{readme_path}' の更新中にエラーが発生しました: {e}")


# --- main関数 (修正版) ---

def main():
    parser = argparse.ArgumentParser(
        description='モデルの解答を採点し、結果をLeaderboardに反映するスクリプト',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--answers_path', '-a', default='results/correct_answers.csv',
                        help='正解データのパス（CSVまたはPickle）')
    parser.add_argument('--output', '-o', default='./results',
                        help='統合レポートファイル（CSV, PNG, TXT）の出力先ディレクトリ')
    parser.add_argument('--readme', default='README.md',
                        help='更新するREADMEファイルのパス')
    parser.add_argument('--leaderboard_data', default='results/leaderboard.json',
                        help='Leaderboardデータを保存/更新するJSONファイルのパス')
    parser.add_argument('--entry_name', '-e', default=None,
                        help='Leaderboardで使用するエントリー名。指定されない場合はJSONファイル名や内容から推測します。シェル変数 ENTRY_NAME の値などを渡すことを想定。')
    parser.add_argument('--json_paths', '-j', nargs='+', required=True,
                        help='モデル解答のJSONファイルへのパス（複数指定可能、例: results/119A_modelX.json results/119B_modelX.json ...）')

    args = parser.parse_args()

    valid_json_paths = []
    processed_blocks = set()
    base_numbers_from_files = set()
    first_valid_json_path = None

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
        if first_valid_json_path is None:
            first_valid_json_path = path

        filename = path.stem
        parts = filename.split('_', 1)
        base_number_with_block = parts[0] if parts else ""

        block_id = ""
        base_number = ""
        if base_number_with_block and len(base_number_with_block) > 1 and base_number_with_block[-1].isalpha():
            block_id = base_number_with_block[-1].upper()
            base_number = base_number_with_block[:-1]
            if block_id in processed_blocks:
                 print(f"警告: ブロック {block_id} が複数回指定されています ({path.name})。")
            processed_blocks.add(block_id)
        else:
             print(f"警告: ファイル名 {path.name} からブロックIDを特定できません。")
        if base_number: base_numbers_from_files.add(base_number)

    if not valid_json_paths:
        parser.error("処理可能なJSONファイルが1つも見つかりませんでした。")

    first_entry_name_from_file = None
    if first_valid_json_path:
        first_filename_parts = first_valid_json_path.stem.split('_', 1)
        if len(first_filename_parts) > 1:
            first_entry_name_from_file = first_filename_parts[1] # ファイル名からのエントリー名候補

    if len(base_numbers_from_files) > 1:
        print(f"警告: 複数の試験番号がファイル名に含まれています: {base_numbers_from_files}。レポートのプレフィックスには最初のファイル ({first_valid_json_path.name if first_valid_json_path else 'N/A'}) の値を使用します。")
    elif not base_numbers_from_files:
        print(f"警告: ファイル名から試験番号を特定できませんでした。")

    try:
        correct_df = load_correct_answers(args.answers_path)
    except (FileNotFoundError, ValueError) as e:
        parser.error(f"正解ファイルの読み込みエラー: {e}")
    except Exception as e:
         parser.error(f"正解ファイルの読み込み中に予期せぬエラー: {e}")


    all_results_dfs = []
    first_model_name_from_json = None

    for i, json_path_str in enumerate(valid_json_paths):
        json_path = Path(json_path_str)
        print(f"\n--- [{i+1}/{len(valid_json_paths)}] {json_path.name} を処理中 ---")

        try:
            model_data = load_model_answers(json_path_str)
            results_df, accuracy, model_name_in_current_json = grade_answers(model_data, correct_df)

            if results_df.empty:
                print(f"情報: {json_path.name} から有効な採点結果が得られませんでした。スキップします。")
                continue

            if first_model_name_from_json is None and model_name_in_current_json:
                 first_model_name_from_json = model_name_in_current_json

            # 個別レポートの要約をコンソールに表示 (ファイル出力はしない)
            print_individual_report_summary(results_df, accuracy, json_path.name)

            all_results_dfs.append(results_df)

        except (FileNotFoundError, json.JSONDecodeError, ValueError) as e:
            print(f"エラー: {json_path.name} の処理中にエラーが発生しました: {e}。このファイルはスキップします。")
            continue
        except Exception as e:
             print(f"予期せぬエラー: {json_path.name} の処理中にエラー: {e}。このファイルはスキップします。")
             continue


    if not all_results_dfs:
        print("\n有効な処理結果がなかったため、統合レポートとLeaderboard更新をスキップします。")
        return

    print("\n--- 全ブロックの結果を統合中 ---")
    try:
        consolidated_stats = consolidate_results(all_results_dfs)
        # 統合結果のDataFrameを取得 (ファイル出力用)
        consolidated_df = consolidated_stats.get('total', {}).get('df', pd.DataFrame())
    except Exception as e:
         print(f"エラー: 結果の統合処理中にエラーが発生しました: {e}。統合レポートとLeaderboard更新をスキップします。")
         return

    # --- Leaderboard用エントリー名の決定 ---
    effective_entry_name = None
    
    if first_entry_name_from_file:
        effective_entry_name = first_entry_name_from_file
        print(f"情報: 最初のJSONファイル名 ({first_valid_json_path.name if first_valid_json_path else 'N/A'}) から推測したエントリー名 '{effective_entry_name}' をLeaderboardに使用します。")
    elif first_model_name_from_json: # JSON内のモデル名をフォールバックとして使用
        effective_entry_name = first_model_name_from_json
        print(f"情報: 最初のJSONファイルの 'model' フィールドからエントリー名 '{effective_entry_name}' をLeaderboardに使用します。")
    else:
        effective_entry_name = "UnknownEntry"
        print("警告: Leaderboardで使用するエントリー名を特定できませんでした。'UnknownEntry' として扱います。")


    # --- 統合レポート生成 ---
    first_base_number = list(base_numbers_from_files)[0] if base_numbers_from_files else "UnknownNumber"
    # 統合レポートのプレフィックス (例: 119_)
    consolidated_prefix = f"{first_base_number}_"

    # 統合サマリをコンソール出力 & テキストファイル保存
    generate_consolidated_report(consolidated_stats, effective_entry_name, args.output, consolidated_prefix)

    # 統合ファイル（プロット、CSV）を生成
    if not consolidated_df.empty:
        generate_consolidated_files(consolidated_df, args.output, consolidated_prefix, effective_entry_name)
    else:
        print("情報: 統合DataFrameが空のため、統合ファイル (PNG, CSV) の生成をスキップします。")


    # --- Leaderboard 更新処理 ---
    print("\n--- Leaderboard 更新 ---")
    if args.entry_name:
        effective_entry_name = args.entry_name
        print(f"情報: コマンドライン引数から指定されたエントリー名 '{effective_entry_name}' をLeaderboardに使用します。")
    elif effective_entry_name != "UnknownEntry":
        print(f"情報: エントリー名 '{effective_entry_name}' をLeaderboardに使用します。")

    if effective_entry_name == "UnknownEntry":
        print("警告: エントリー名を特定できなかったため、Leaderboard は更新されません。")
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