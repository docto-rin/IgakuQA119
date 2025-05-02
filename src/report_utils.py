import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

def format_rate(rate): return f"{rate:.2%}" if pd.notna(rate) and isinstance(rate, (int, float)) else "N/A"
def format_count(correct, total): return f"{correct} / {total}" if pd.notna(correct) and pd.notna(total) and total >= 0 else "N/A"
def format_score(score, possible):
     if not (pd.notna(score) and pd.notna(possible)): return "N/A"
     score_str = f"{int(score)}" if score == int(score) else f"{score:.1f}"
     possible_str = f"{int(possible)}" if possible == int(possible) else f"{possible:.1f}"
     if possible == 0: return f"{score_str} / {possible_str} (0点満点)"
     return f"{score_str} / {possible_str}"

def generate_consolidated_report(stats, entry_name_for_title, file_identifier, output_dir=None, prefix=""):
    """
    統合された結果のレポートを生成し、サマリテキストファイルとして保存する。
    レポート内のタイトルには entry_name_for_title を、ファイル名には file_identifier を使用する。
    """
    summary_lines = []
    summary_lines.append(f"===== モデルエントリー: {entry_name_for_title} の結果 =====")
    summary_lines.append(f"(ファイル識別子: {file_identifier})")
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
    print("\n--- 統合サマリ ---")
    print(summary)

    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)
        summary_file = output_path / f"{prefix}{file_identifier}_summary.txt"
        try:
            with open(summary_file, "w", encoding="utf-8") as f:
                f.write(summary)
            print(f"サマリを {summary_file} に保存しました。")
        except Exception as e:
            print(f"エラー: サマリファイル '{summary_file}' の書き込み中にエラーが発生しました: {e}")

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

    headers = ["Rank", "Entry", "Overall Score", "Overall Acc.", "No-Img Score", "No-Img Acc."]
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

    print("-" * 20)

def generate_consolidated_files(consolidated_df, output_dir, prefix, file_identifier):
    """統合された結果からレポートファイル（プロット、CSV）を生成する"""
    if consolidated_df.empty:
        print("情報: 統合結果データフレームが空のため、ファイル生成をスキップします。")
        return

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    file_prefix = f"{prefix}{file_identifier}_"

    try:
        print("\n--- 統合レポートファイルの作成 ---")
        # confidence 列が存在し、かつ数値型の場合のみプロットを試みる
        if 'confidence' in consolidated_df.columns and pd.api.types.is_numeric_dtype(consolidated_df['confidence']):
            plt.figure(figsize=(10, 6))
            # confidence と is_correct の両方が非NAのデータのみを使用
            plot_data = consolidated_df.dropna(subset=['confidence', 'is_correct'])
            if not plot_data.empty:
                 # is_correct を bool 型に変換 (NA除去後なので安全)
                 plot_data['is_correct'] = plot_data['is_correct'].astype(bool)
                 sns.histplot(data=plot_data, x='confidence', hue='is_correct', bins=20, multiple='stack')
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

        # is_correct が False の行を抽出 (NA は False とみなさない)
        wrong_answers = consolidated_df[consolidated_df['is_correct'] == False]
        if not wrong_answers.empty:
            wrong_csv_path = output_path / f"{file_prefix}wrong_answers.csv"
            wrong_answers.to_csv(wrong_csv_path, index=False, encoding='utf-8-sig')
            print(f"誤答問題リストを {wrong_csv_path} に保存しました。")
        else:
             # is_correct が True または None の場合
             if consolidated_df['is_correct'].isnull().all():
                 print("情報: 採点対象の問題がなかったため、誤答リストは生成されません。")
             else:
                 print("情報: 統合結果で間違えた問題はありませんでした (採点対象内で)。")


        results_csv_path = output_path / f"{file_prefix}grading_results.csv"
        consolidated_df.to_csv(results_csv_path, index=False, encoding='utf-8-sig')
        print(f"全採点結果CSVを {results_csv_path} に保存しました。")

    except Exception as e:
        print(f"エラー: 統合レポートファイルの保存中にエラーが発生しました: {e}")