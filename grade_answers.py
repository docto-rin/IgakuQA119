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
        
        # 正解と照合
        correct_answer = correct_answers.get(question_number)
        is_correct = False
        
        if correct_answer is not None:
            if question_number == '119E28':
                is_correct = model_answer.lower() in ['a', 'c']
            else:
                is_correct = model_answer.lower() == correct_answer.lower()
            if is_correct:
                correct_count += 1
            total_questions += 1
        
        results.append({
            'question_number': question_number,
            'model': model_name,
            'model_answer': model_answer,
            'correct_answer': correct_answer,
            'is_correct': is_correct,
            'confidence': confidence
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
    # 全ての結果を結合
    combined_df = pd.concat(all_results, ignore_index=True)
    
    # ブロック情報を抽出（問題番号の最初の文字）
    combined_df['block'] = combined_df['question_number'].str[3]
    
    # 一般問題と必修問題を分類
    general_blocks = ['A', 'C', 'D', 'F']
    required_blocks = ['B', 'E']
    
    general_df = combined_df[combined_df['block'].isin(general_blocks)]
    required_df = combined_df[combined_df['block'].isin(required_blocks)]
    
    # データの検証とデバッグ情報
    print(f"\n処理したブロック: {sorted(combined_df['block'].unique())}")
    print(f"一般問題（A,C,D,F）データ件数: {len(general_df)}")
    print(f"必修問題（B,E）データ件数: {len(required_df)}")
    
    # 点数の計算（一般問題:1点、必修問題:2点）
    general_score = general_df['is_correct'].sum()
    required_score = required_df['is_correct'].sum() * 2
    
    general_total = len(general_df)
    required_total = len(required_df) * 2
    
    total_score = general_score + required_score
    total_possible = general_total + required_total
    
    # 正答率の計算
    general_accuracy = general_df['is_correct'].mean() if len(general_df) > 0 else 0
    required_accuracy = required_df['is_correct'].mean() if len(required_df) > 0 else 0
    total_accuracy = combined_df['is_correct'].mean() if len(combined_df) > 0 else 0
    
    # 結果の統計を返す
    stats = {
        'total': {
            'df': combined_df,
            'correct': combined_df['is_correct'].sum(),
            'total': len(combined_df),
            'accuracy': total_accuracy,
            'score': total_score,
            'possible_score': total_possible
        },
        'general': {
            'df': general_df,
            'correct': general_df['is_correct'].sum(),
            'total': len(general_df),
            'accuracy': general_accuracy,
            'score': general_score,
            'possible_score': general_total
        },
        'required': {
            'df': required_df,
            'correct': required_df['is_correct'].sum(),
            'total': len(required_df),
            'accuracy': required_accuracy,
            'score': required_score,
            'possible_score': required_total
        },
        'blocks': {block: acc for block, acc in all_accuracies.items()}
    }
    
    return stats


def generate_consolidated_report(stats, output_dir=None, prefix=""):
    """統合された結果のレポートを生成する"""
    # 全体の結果
    print("\n===== 全体の結果 =====")
    print(f"合計点: {stats['total']['score']} / {stats['total']['possible_score']}")
    print(f"全体の正答率: {stats['total']['accuracy']:.2%}")
    print(f"正解数: {stats['total']['correct']} / {stats['total']['total']}")
    
    # 一般問題の結果
    print("\n===== 一般問題 (A,C,D,F) =====")
    print(f"点数: {stats['general']['score']} / {stats['general']['possible_score']}")
    print(f"正答率: {stats['general']['accuracy']:.2%}")
    print(f"正解数: {stats['general']['correct']} / {stats['general']['total']}")
    
    # 必修問題の結果
    print("\n===== 必修問題 (B,E) =====")
    print(f"点数: {stats['required']['score']} / {stats['required']['possible_score']}")
    print(f"正答率: {stats['required']['accuracy']:.2%}")
    print(f"正解数: {stats['required']['correct']} / {stats['required']['total']}")
    
    # ブロック別の結果
    print("\n===== ブロック別正答率 =====")
    for block, accuracy in stats['blocks'].items():
        print(f"ブロック {block}: {accuracy:.2%}")
    
    # 結果をファイルに保存
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)
        
        # 全体の結果を保存
        stats['total']['df'].to_csv(output_path / f"{prefix}results.csv", index=False)
        
        # 間違えた問題のリスト
        wrong_answers = stats['total']['df'][~stats['total']['df']['is_correct']]
        wrong_answers.to_csv(output_path / f"{prefix}wrong_answers.csv", index=False)
        
        # 可視化
        plt.figure(figsize=(10, 6))
        sns.histplot(data=stats['total']['df'], x='confidence', hue='is_correct', bins=20, multiple='stack')
        plt.title('All Blocks - Confidence Distribution and Correctness')
        plt.xlabel('Confidence')
        plt.ylabel('Count')
        plt.savefig(output_path / f"{prefix}confidence_distribution.png")
        
        # ブロック別の正答率グラフ
        plt.figure(figsize=(10, 6))
        blocks = list(stats['blocks'].keys())
        accuracies = [stats['blocks'][block] for block in blocks]
        sns.barplot(x=blocks, y=accuracies)
        plt.title('Accuracy by Block')
        plt.xlabel('Block')
        plt.ylabel('Accuracy')
        plt.ylim(0, 1)
        for i, acc in enumerate(accuracies):
            plt.text(i, acc + 0.02, f"{acc:.2%}", ha='center')
        plt.savefig(output_path / f"{prefix}block_accuracies.png")
        
        print(f"統合結果を {output_dir} に保存しました")


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
    
    # 複数のJSONファイルが処理された場合、統合レポートを生成
    if len(valid_json_paths) > 1:
        print("\n全ブロックの結果を統合中...")
        consolidated_stats = consolidate_results(all_results, all_accuracies)
        
        # 共通のベース番号とモデル名を確認
        common_base_number = base_numbers[0] if all(bn == base_numbers[0] for bn in base_numbers) else ""
        common_model_name = model_names[0] if all(mn == model_names[0] for mn in model_names) else ""
        
        # 統合結果の接頭辞を生成
        all_prefix = f"{common_base_number}_all_{common_model_name}_"
        general_prefix = f"{common_base_number}_general_{common_model_name}_"
        required_prefix = f"{common_base_number}_required_{common_model_name}_"
        
        # 統合レポートを生成
        generate_consolidated_report(consolidated_stats, args.output, all_prefix)
        
        # 一般問題と必修問題の分割結果を保存
        if args.output:
            output_path = Path(args.output)
            output_path.mkdir(exist_ok=True, parents=True)
            
            # 一般問題の結果をファイルに保存（データがある場合のみ）
            if len(consolidated_stats['general']['df']) > 0:
                consolidated_stats['general']['df'].to_csv(output_path / f"{general_prefix}results.csv", index=False)
                wrong_general = consolidated_stats['general']['df'][~consolidated_stats['general']['df']['is_correct']]
                wrong_general.to_csv(output_path / f"{general_prefix}wrong_answers.csv", index=False)
                
                # 一般問題の可視化
                plt.figure(figsize=(10, 6))
                sns.histplot(data=consolidated_stats['general']['df'], x='confidence', hue='is_correct', bins=20, multiple='stack')
                plt.title('General Problems - Confidence Distribution and Correctness')
                plt.xlabel('Confidence')
                plt.ylabel('Count')
                plt.savefig(output_path / f"{general_prefix}confidence_distribution.png")
            else:
                print("警告: 一般問題のデータが見つかりません。(A,C,D,Fブロック)")
            
            # 必修問題の結果をファイルに保存（データがある場合のみ）
            if len(consolidated_stats['required']['df']) > 0:
                consolidated_stats['required']['df'].to_csv(output_path / f"{required_prefix}results.csv", index=False)
                wrong_required = consolidated_stats['required']['df'][~consolidated_stats['required']['df']['is_correct']]
                wrong_required.to_csv(output_path / f"{required_prefix}wrong_answers.csv", index=False)
                
                # 必修問題の可視化
                plt.figure(figsize=(10, 6))
                sns.histplot(data=consolidated_stats['required']['df'], x='confidence', hue='is_correct', bins=20, multiple='stack')
                plt.title('Required Problems - Confidence Distribution and Correctness')
                plt.xlabel('Confidence')
                plt.ylabel('Count')
                plt.savefig(output_path / f"{required_prefix}confidence_distribution.png")
            else:
                print("警告: 必修問題のデータが見つかりません。(B,Eブロック)")


if __name__ == "__main__":
    main()
