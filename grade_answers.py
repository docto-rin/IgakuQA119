import json
import pandas as pd
import argparse
from pathlib import Path

from src.file_utils import (
    load_model_answers,
    load_correct_answers,
    update_leaderboard_data,
    update_readme,
    parse_input_paths,
    save_skipped_list,
)
from src.report_utils import (
    generate_consolidated_report,
    generate_leaderboard_markdown,
    print_individual_report_summary,
    generate_consolidated_files
)
from src.grading_logic import grade_answers
from src.stats_utils import consolidate_results


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

    # 入力JSONパスの検証と情報抽出 (file_utils から呼び出す)
    try:
        valid_json_paths, file_identifier, consolidated_prefix, _, _ = parse_input_paths(args.json_paths)
    except ValueError as e:
         parser.error(str(e)) # エラーメッセージを表示して終了
    except Exception as e:
         parser.error(f"入力パスの解析中に予期せぬエラー: {e}")


    # 正解データを読み込み
    try:
        correct_df = load_correct_answers(args.answers_path)
    except (FileNotFoundError, ValueError) as e:
        parser.error(f"正解ファイルの読み込みエラー: {e}")
    except Exception as e:
         parser.error(f"正解ファイルの読み込み中に予期せぬエラー: {e}")


    all_results_dfs = []
    first_model_name_from_json = None
    all_skipped_questions = []

    # 各JSONファイルを処理して採点
    for i, json_path_str in enumerate(valid_json_paths):
        json_path = Path(json_path_str)
        print(f"\n--- [{i+1}/{len(valid_json_paths)}] {json_path.name} を処理中 ---")

        try:
            model_data = load_model_answers(json_path_str)
            # grade_answers を grading_logic から呼び出す
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

            # 最初のJSONからモデル名を取得 (Leaderboard用)
            if first_model_name_from_json is None and model_name_in_current_json and model_name_in_current_json != 'UnknownModel':
                 first_model_name_from_json = model_name_in_current_json

            # 有効な結果があれば個別サマリを表示し、リストに追加
            if has_results:
                # print_individual_report_summary を report_utils から呼び出す
                print_individual_report_summary(results_df, accuracy, json_path.name)
                all_results_dfs.append(results_df)

            # スキップされた問題があればリストに追加
            if has_skipped:
                all_skipped_questions.extend(skipped_q)

        except (FileNotFoundError, json.JSONDecodeError, ValueError) as e:
            print(f"エラー: {json_path.name} の処理中にエラーが発生しました: {e}。このファイルはスキップします。")
            continue
        except Exception as e:
             print(f"予期せぬエラー: {json_path.name} の処理中にエラー: {e}。このファイルはスキップします。")
             continue


    # Leaderboard/レポートタイトル用のエントリー名を決定
    effective_entry_name = "UnknownEntry"
    if args.entry_name:
        effective_entry_name = args.entry_name
        print(f"\n情報: コマンドライン引数からLeaderboard/レポートタイトル用エントリー名 '{effective_entry_name}' を使用します。")
    elif file_identifier != "UnknownExp": # parse_input_paths から取得した識別子を使用
        effective_entry_name = file_identifier
        print(f"\n情報: ファイル名から推測した識別子 '{effective_entry_name}' をLeaderboard/レポートタイトル用エントリー名として使用します。")
    elif first_model_name_from_json:
        # JSON内のモデル名をエントリー名として使用
        effective_entry_name = first_model_name_from_json
        print(f"\n情報: 最初のJSONファイルの 'model' フィールドからエントリー名 '{effective_entry_name}' をLeaderboard/レポートタイトル用に使用します。")
    else:
        print("\n警告: Leaderboard/レポートタイトルで使用するエントリー名を特定できませんでした。'UnknownEntry' として扱います。")


    # スキップされた問題があればファイルに出力 (file_utils から呼び出す)
    unique_skipped_questions = sorted(list(set(all_skipped_questions)))
    # all_results_dfs が空でもスキップリストは出力する可能性がある
    save_skipped_list(
        unique_skipped_questions,
        args.output,
        consolidated_prefix,
        file_identifier,
        args.skipped_output_file
    )

    # 有効な採点結果がなければ終了
    if not all_results_dfs:
        print("\n有効な採点結果がなかったため、統合レポートとLeaderboard更新をスキップします。")
        return

    # 全ブロックの結果を統合
    print("\n--- 全ブロックの結果を統合中 ---")
    try:
        # consolidate_results を stats_utils から呼び出す
        consolidated_stats = consolidate_results(all_results_dfs)
        # 統合後のDataFrameを取得 (ファイル生成用)
        consolidated_df = consolidated_stats.get('total', {}).get('df', pd.DataFrame())
    except Exception as e:
         print(f"エラー: 結果の統合処理中にエラーが発生しました: {e}。統合レポートとLeaderboard更新をスキップします。")
         return # エラーが発生したらここで終了

    # 統合サマリレポートを生成・保存 (report_utils から呼び出す)
    generate_consolidated_report(
        stats=consolidated_stats,
        entry_name_for_title=effective_entry_name,
        file_identifier=file_identifier,
        output_dir=args.output,
        prefix=consolidated_prefix
    )

    # 統合結果ファイル (プロット、CSV) を生成・保存 (report_utils から呼び出す)
    if not consolidated_df.empty:
        generate_consolidated_files(
            consolidated_df=consolidated_df,
            output_dir=args.output,
            prefix=consolidated_prefix,
            file_identifier=file_identifier
        )
    else:
        # 通常ここには到達しないはず
        print("情報: 統合DataFrameが空のため、統合ファイル (PNG, CSV) の生成をスキップします。")


    # Leaderboard を更新
    print("\n--- Leaderboard 更新 ---")
    if effective_entry_name == "UnknownEntry":
        print("警告: Leaderboard用エントリー名を特定できなかったため、Leaderboard は更新されません。")
    elif not consolidated_stats:
         # 通常ここには到達しないはず
         print("警告: 統合結果がないため、Leaderboard は更新されません。")
    else:
        try:
            # Leaderboardデータを更新/保存 (file_utils から呼び出す)
            leaderboard_data = update_leaderboard_data(args.leaderboard_data, effective_entry_name, consolidated_stats)
            # Markdownテーブルを生成 (report_utils から呼び出す)
            markdown_table = generate_leaderboard_markdown(leaderboard_data)
            # READMEファイルを更新 (file_utils から呼び出す)
            update_readme(args.readme, markdown_table)
        except Exception as e:
            print(f"エラー: Leaderboardの更新処理中に予期せぬエラーが発生しました: {e}")

    print("\n処理が完了しました。")


if __name__ == "__main__":
    main()