import json
import pandas as pd
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns


def load_model_answers(json_path):
    """JSONファイルからモデルの解答データを読み込む"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    return data


def load_correct_answers(answers_path):
    """正解データを読み込む（CSVまたはDataFrame）"""
    # パスの拡張子に応じて読み込み方法を変える
    if Path(answers_path).suffix.lower() == '.csv':
        df = pd.read_csv(answers_path)
    else:
        # 拡張子がない場合やその他の場合はpickleとして読み込む
        df = pd.read_pickle(answers_path)
    
    return df


def grade_answers(model_data, correct_df):
    """モデルの解答を採点する"""
    results = []
    correct_count = 0
    total_questions = 0
    
    # 問題番号と正解のマッピングを作成
    correct_answers = dict(zip(correct_df['問題番号'], correct_df['解答']))
    
    for question in model_data['results']:
        question_number = question['question_number']
        
        # このモデルの最初の解答を使用（複数モデルがある場合は最初のものだけ）
        model_answer = question['answers'][0]['answer']
        model_name = question['answers'][0]['model']
        confidence = question['answers'][0]['confidence']
        has_image = question['has_image']
        
        # 正解と照合
        correct_answer = correct_answers.get(question_number)
        is_correct = False
        
        if correct_answer is not None:
            try:
                if question_number == '119E28':
                    is_correct = model_answer.lower() in ['a', 'c']
                else:
                    is_correct = model_answer.lower() == correct_answer.lower()
            except:
                print(f"Error comparing answers for question {question_number}: {model_answer} vs {correct_answer}")
            if is_correct:
                correct_count += 1
            total_questions += 1
        
        results.append({
            'question_number': question_number,
            'model': model_name,
            'model_answer': model_answer,
            'correct_answer': correct_answer,
            'is_correct': is_correct,
            'confidence': confidence,
            'has_image': has_image  # has_image情報を追加
        })
    
    accuracy = correct_count / total_questions if total_questions > 0 else 0
    
    return pd.DataFrame(results), accuracy


def generate_report(results_df, accuracy, output_dir=None, prefix=""):
    """結果のレポートを生成する"""
    print(f"全体の正答率: {accuracy:.2%}")
    print(f"正解数: {results_df['is_correct'].sum()} / {len(results_df)}")
    
    # 信頼度と正誤の関係を分析
    print("\n信頼度と正誤の関係:")
    # カテゴリ化せずに直接の値を使用
    confidence_accuracy = results_df.groupby('confidence')['is_correct'].mean()
    print(confidence_accuracy)
    
    # 信頼度の分布をプロット
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)
        
        plt.figure(figsize=(10, 6))
        sns.histplot(data=results_df, x='confidence', hue='is_correct', bins=20, multiple='stack')
        plt.title('Confidence Distribution and Correctness')
        plt.xlabel('Confidence')
        plt.ylabel('Count')
        plt.savefig(output_path / f"{prefix}confidence_distribution.png")
        
        # 間違えた問題のリスト
        wrong_answers = results_df[~results_df['is_correct']]
        wrong_answers.to_csv(output_path / f"{prefix}wrong_answers.csv", index=False)
        
        # 結果のCSVを保存
        results_df.to_csv(output_path / f"{prefix}grading_results.csv", index=False)
        
        print(f"結果を {output_dir} に保存しました")
    
    return results_df


def consolidate_results(all_results, all_accuracies):
    """複数ブロックの結果を統合する"""
    import pandas as pd
    # 全ての結果を結合
    combined_df = pd.concat(all_results, ignore_index=True)
    
    # 問題番号の4文字目（インデックス3）をブロックとして抽出
    combined_df['block'] = combined_df['question_number'].str[3]
    
    # 一般問題と必修問題を分類
    general_blocks = ['A', 'C', 'D', 'F']
    required_blocks = ['B', 'E']
    
    general_df = combined_df[combined_df['block'].isin(general_blocks)]
    required_df = combined_df[combined_df['block'].isin(required_blocks)]
    
    # 画像なしの問題だけを抽出
    no_image_df = combined_df[~combined_df['has_image']].copy()
    no_image_general_df = no_image_df[no_image_df['block'].isin(general_blocks)]
    no_image_required_df = no_image_df[no_image_df['block'].isin(required_blocks)]
    
    # デバッグ情報
    print(f"\n処理したブロック: {sorted(combined_df['block'].unique())}")
    print(f"一般問題（A,C,D,F）件数: {len(general_df)}")
    print(f"必修問題（B,E）件数: {len(required_df)}")
    print(f"画像なし問題件数: {len(no_image_df)}")
    
    # ヘルパー関数: 必修問題の得点（前半:1点、後半:3点）
    def required_point(question_number):
        try:
            num = int(question_number[4:])  # 例："119B17" → "17"
        except:
            num = 0
        return 1 if num <= 25 else 3

    # 各行の得点(score)と満点(possible)を計算する
    def calculate_points(row):
        block = row['block']
        if block in required_blocks:
            possible = required_point(row['question_number'])
            score = possible if row['is_correct'] else 0
        else:
            possible = 1
            score = 1 if row['is_correct'] else 0
        return pd.Series({'score': score, 'possible': possible})

    # 各行に対して得点と満点を計算
    combined_df[['score', 'possible']] = combined_df.apply(calculate_points, axis=1)
    
    # ブロック別得点率（全体）
    block_stats = combined_df.groupby('block').agg({'score': 'sum', 'possible': 'sum'}).reset_index()
    block_stats['score_rate'] = block_stats['score'] / block_stats['possible']
    block_score_rates = dict(zip(block_stats['block'], block_stats['score_rate']))

    # 画像なしの問題について、得点率計算
    no_image_df[['score', 'possible']] = no_image_df.apply(calculate_points, axis=1)
    block_stats_no_image = no_image_df.groupby('block').agg({'score': 'sum', 'possible': 'sum'}).reset_index()
    block_stats_no_image['score_rate'] = block_stats_no_image['score'] / block_stats_no_image['possible']
    no_image_block_score_rates = dict(zip(block_stats_no_image['block'], block_stats_no_image['score_rate']))

    # 画像なしのブロック別正答率を計算
    no_image_block_accuracies = no_image_df.groupby('block')['is_correct'].mean().to_dict()
    
    # 点数計算（一般問題:1点固定、必修問題は得点変動）
    general_score = general_df['is_correct'].sum()
    general_total = len(general_df)
    required_score = required_df.apply(
        lambda row: required_point(row['question_number']) if row['is_correct'] else 0, axis=1
    ).sum()
    required_total = required_df.apply(
        lambda row: required_point(row['question_number']), axis=1
    ).sum()
    
    total_score = general_score + required_score
    total_possible = general_total + required_total
    
    # 画像なしの問題点数計算
    no_image_general_score = no_image_general_df['is_correct'].sum()
    no_image_required_score = no_image_required_df.apply(
        lambda row: required_point(row['question_number']) if row['is_correct'] else 0, axis=1
    ).sum()
    no_image_general_total = len(no_image_general_df)
    no_image_required_total = no_image_required_df.apply(
        lambda row: required_point(row['question_number']), axis=1
    ).sum()
    
    no_image_total_score = no_image_general_score + no_image_required_score
    no_image_total_possible = no_image_general_total + no_image_required_total
    
    # 正答率の計算（全てカウントベース）
    general_accuracy = general_df['is_correct'].mean() if len(general_df) > 0 else 0
    required_accuracy = required_df['is_correct'].mean() if len(required_df) > 0 else 0
    total_accuracy = combined_df['is_correct'].mean() if len(combined_df) > 0 else 0

    no_image_general_accuracy = no_image_general_df['is_correct'].mean() if len(no_image_general_df) > 0 else 0
    no_image_required_accuracy = no_image_required_df['is_correct'].mean() if len(no_image_required_df) > 0 else 0
    no_image_total_accuracy = no_image_df['is_correct'].mean() if len(no_image_df) > 0 else 0

    
    stats = {
        'total': {
            'df': combined_df,
            'correct': combined_df['is_correct'].sum(),
            'total': len(combined_df),
            'accuracy': total_accuracy,
            'score': total_score,
            'possible_score': total_possible,
            'score_rate': total_score / total_possible if total_possible > 0 else 0,
            'block_score_rates': block_score_rates  # 全体のブロック別得点率
        },
        'general': {
            'df': general_df,
            'correct': general_df['is_correct'].sum(),
            'total': len(general_df),
            'accuracy': general_accuracy,
            'score': general_score,
            'possible_score': general_total,
            'score_rate': general_score / general_total if general_total > 0 else 0
        },
        'required': {
            'df': required_df,
            'correct': required_df['is_correct'].sum(),
            'total': len(required_df),
            'accuracy': required_accuracy,
            'score': required_score,
            'possible_score': required_total,
            'score_rate': required_score / required_total if required_total > 0 else 0
        },
        'block_accuracies': {block: acc for block, acc in all_accuracies.items()},
        'block_score_rates': block_score_rates,
        'no_image': {
            'df': no_image_df,
            'correct': no_image_df['is_correct'].sum(),
            'total': len(no_image_df),
            'accuracy': no_image_total_accuracy,
            'score': no_image_total_score,
            'possible_score': no_image_total_possible,
            'score_rate': no_image_total_score / no_image_total_possible if no_image_total_possible > 0 else 0,
            'general': {
                'df': no_image_general_df,
                'correct': no_image_general_df['is_correct'].sum(),
                'total': len(no_image_general_df),
                'accuracy': no_image_general_accuracy,
                'score': no_image_general_score,
                'possible_score': no_image_general_total,
                'score_rate': no_image_general_score / no_image_general_total if no_image_general_total > 0 else 0
            },
            'required': {
                'df': no_image_required_df,
                'correct': no_image_required_df['is_correct'].sum(),
                'total': len(no_image_required_df),
                'accuracy': no_image_required_accuracy,
                'score': no_image_required_score,
                'possible_score': no_image_required_total,
                'score_rate': no_image_required_score / no_image_required_total if no_image_required_total > 0 else 0
            },
            # 画像なしのブロック別結果として、正答率と得点率の両方を保存
            'block_accuracies': no_image_block_accuracies,
            'block_score_rates': no_image_block_score_rates
        }
    }
    
    return stats


def generate_consolidated_report(stats, output_dir=None, prefix=""):
    """統合された結果のレポートを生成する（summaryテキストファイル出力付き、得点率と正答率を順序変更）"""
    from pathlib import Path
    summary_lines = []

    # 全体の結果
    summary_lines.append("===== 全体の結果 =====")
    summary_lines.append(f"正解数: {stats['total']['correct']} / {stats['total']['total']}")
    summary_lines.append(f"正答率: {stats['total']['accuracy']:.2%}")
    summary_lines.append(f"合計点: {stats['total']['score']} / {stats['total']['possible_score']}")
    summary_lines.append(f"得点率: {stats['total']['score_rate']:.2%}")
    summary_lines.append("")

    # 一般問題の結果
    summary_lines.append("===== 一般問題 (A,C,D,F) =====")
    summary_lines.append(f"正解数: {stats['general']['correct']} / {stats['general']['total']}")
    summary_lines.append(f"正答率: {stats['general']['accuracy']:.2%}")
    summary_lines.append(f"合計点: {stats['general']['score']} / {stats['general']['possible_score']}")
    summary_lines.append(f"得点率: {stats['general']['score_rate']:.2%}")
    summary_lines.append("")

    # 必修問題の結果
    summary_lines.append("===== 必修問題 (B,E) =====")
    summary_lines.append(f"正解数: {stats['required']['correct']} / {stats['required']['total']}")
    summary_lines.append(f"正答率: {stats['required']['accuracy']:.2%}")
    summary_lines.append(f"合計点: {stats['required']['score']} / {stats['required']['possible_score']}")
    summary_lines.append(f"得点率: {stats['required']['score_rate']:.2%}")
    summary_lines.append("")

    # ブロック別結果（全体）
    summary_lines.append("===== ブロック別結果 =====")
    for block in sorted(stats['block_accuracies'].keys()):
        acc = stats['block_accuracies'][block]
        sr = stats['block_score_rates'][block]
        summary_lines.append(f"ブロック {block}: 正答率 {acc:.2%}, 得点率 {sr:.2%}")
    summary_lines.append("")

    # 画像なしの結果
    summary_lines.append("===== 画像なしの問題の結果 =====")
    summary_lines.append(f"正解数: {stats['no_image']['correct']} / {stats['no_image']['total']}")
    summary_lines.append(f"正答率: {stats['no_image']['accuracy']:.2%}")
    summary_lines.append(f"合計点: {stats['no_image']['score']} / {stats['no_image']['possible_score']}")
    summary_lines.append(f"得点率: {stats['no_image']['score_rate']:.2%}")
    summary_lines.append("")

    # 画像なしの一般問題
    summary_lines.append("===== 画像なしの一般問題 (A,C,D,F) =====")
    summary_lines.append(f"正解数: {stats['no_image']['general']['correct']} / {stats['no_image']['general']['total']}")
    summary_lines.append(f"正答率: {stats['no_image']['general']['accuracy']:.2%}")
    summary_lines.append(f"合計点: {stats['no_image']['general']['score']} / {stats['no_image']['general']['possible_score']}")
    summary_lines.append(f"得点率: {stats['no_image']['general']['score_rate']:.2%}")
    summary_lines.append("")

    # 画像なしの必修問題
    summary_lines.append("===== 画像なしの必修問題 (B,E) =====")
    summary_lines.append(f"正解数: {stats['no_image']['required']['correct']} / {stats['no_image']['required']['total']}")
    summary_lines.append(f"正答率: {stats['no_image']['required']['accuracy']:.2%}")
    summary_lines.append(f"合計点: {stats['no_image']['required']['score']} / {stats['no_image']['required']['possible_score']}")
    summary_lines.append(f"得点率: {stats['no_image']['required']['score_rate']:.2%}")
    summary_lines.append("")

    # 画像なしのブロック別結果
    summary_lines.append("===== 画像なしのブロック別結果 =====")
    for block in sorted(stats['no_image']['block_accuracies'].keys()):
        acc = stats['no_image']['block_accuracies'][block]
        sr = stats['no_image']['block_score_rates'][block]
        summary_lines.append(f"ブロック {block}: 正答率 {acc:.2%}, 得点率 {sr:.2%}")
    
    # 連結して1つの文字列にまとめる
    summary = "\n".join(summary_lines)

    # コンソール出力
    print(summary)

    # ファイルに保存
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)
        summary_file = output_path / f"{prefix}summary.txt"
        with open(summary_file, "w", encoding="utf-8") as f:
            f.write(summary)
        print(f"サマリを {summary_file} に保存しました")


def main():
    parser = argparse.ArgumentParser(description='モデルの解答を採点するスクリプト')
    parser.add_argument('--json_paths', '-j', nargs='+', help='モデル解答のJSONファイルへのパス（複数指定可能）')
    parser.add_argument('--json_path', help='モデル解答の単一JSONファイルへのパス（下位互換用）')
    parser.add_argument('--answers_path', '-a', help='正解データのパス（CSVまたはDataFrame）', default='results/correct_answers.csv')
    parser.add_argument('--output', '-o', help='結果の出力先ディレクトリ', default='./results')
    args = parser.parse_args()
    
    # 下位互換のために単一のjson_pathがある場合はjson_pathsに追加
    json_paths = args.json_paths or []
    if args.json_path and args.json_path not in json_paths:
        json_paths.append(args.json_path)
    
    # JSONファイルがない場合はエラー
    if not json_paths:
        parser.error("少なくとも1つのJSONファイルを指定してください（--json_paths または --json_path）")
    
    # 存在するJSONファイルのみを処理対象にする
    valid_json_paths = []
    for path in json_paths:
        if not Path(path).exists():
            print(f"警告: {path} が見つかりません。スキップします。")
        else:
            valid_json_paths.append(path)
    
    if not valid_json_paths:
        parser.error("処理可能なJSONファイルが見つかりません。")
    
    # 正解データを読み込む
    correct_df = load_correct_answers(args.answers_path)
    
    all_results = []
    all_accuracies = {}
    model_names = []
    base_numbers = []
    
    # 各JSONファイルを処理
    for json_path in valid_json_paths:
        print(f"\n{json_path} を処理中...")
        model_data = load_model_answers(json_path)
        
        # ファイル名から接頭辞を抽出
        filename = Path(json_path).stem  # 拡張子なしのファイル名
        # 例：119A_gemini-2.5-pro から "119A" と "gemini-2.5-pro" を抽出
        parts = filename.split('_', 1)
        if len(parts) >= 2:
            base_number_with_block = parts[0]  # 例：119A
            model_name = parts[1]  # 例：gemini-2.5-pro
        else:
            base_number_with_block = filename
            model_name = ""
        
        # ブロックIDを抽出 (例：119A から 'A')
        block_id = base_number_with_block[-1] if base_number_with_block else ""
        
        # ベース番号を抽出 (例：119A から '119')
        base_number = base_number_with_block[:-1] if base_number_with_block else ""
        
        # 出力ファイル名の接頭辞
        block_prefix = f"{base_number_with_block}_{model_name}_"
        
        # 情報を保存
        model_names.append(model_name)
        base_numbers.append(base_number)
        
        # 採点
        results_df, accuracy = grade_answers(model_data, correct_df)
        
        # ブロック別のレポート生成と結果の保存
        generate_report(results_df, accuracy, args.output, block_prefix)
        
        all_results.append(results_df)
        all_accuracies[block_id] = accuracy
    
    # 一つでもJSONファイルが処理された場合、統合レポートを生成
    if len(valid_json_paths) >= 1:
        print("\n全ブロックの結果を統合中...")
        consolidated_stats = consolidate_results(all_results, all_accuracies)
        
        # 共通のベース番号とモデル名を確認
        common_base_number = base_numbers[0] if all(bn == base_numbers[0] for bn in base_numbers) else ""
        common_model_name = model_names[0] if all(mn == model_names[0] for mn in model_names) else ""
        
        # 統合結果の接頭辞を生成
        all_prefix = f"{common_base_number}_all_{common_model_name}_"
        
        # 統合レポートを生成
        generate_consolidated_report(consolidated_stats, args.output, all_prefix)


if __name__ == "__main__":
    main()
