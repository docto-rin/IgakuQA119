import argparse
import subprocess
from pathlib import Path
from collections import defaultdict
import sys


def load_skipped_questions(filepath):
    """スキップリストファイルから問題番号のリストを読み込む"""
    path = Path(filepath)
    if not path.is_file():
        raise FileNotFoundError(f"スキップリストファイルが見つかりません: {filepath}")
    try:
        with open(path, 'r', encoding='utf-8') as f:
            questions = [line.strip() for line in f if line.strip()]
        if not questions:
             print(f"警告: スキップリストファイル '{filepath}' は空か、有効な問題番号が含まれていません。")
        return questions
    except Exception as e:
        raise IOError(f"スキップリストファイル '{filepath}' の読み込み中にエラーが発生しました: {e}")

def group_questions_by_block(questions):
    """問題番号のリストをブロックごとにグループ化する"""
    grouped = defaultdict(list)
    malformed = []
    for q_num in questions:
        if len(q_num) > 3 and q_num[3].isalpha():
            block = q_num[3].upper()
            grouped[block].append(q_num)
        else:
            malformed.append(q_num)
    if malformed:
        print(f"警告: 以下の問題番号は形式が不正かブロックを特定できませんでした: {malformed}")
    return dict(grouped)

def main():
    parser = argparse.ArgumentParser(
        description='スキップされた医師国家試験問題をブロックごとに再実行するスクリプト',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--skipped_list',
        required=True,
        help='スキップされた問題番号がリストされたテキストファイルへのパス (例: results/119_my-exp_skipped.txt)'
    )
    parser.add_argument(
        '--models',
        nargs='+',
        required=True,
        help='使用するLLMモデル (例: gpt-4o gemini-1.5-pro-latest)'
    )
    parser.add_argument(
        '--questions_dir',
        default='questions',
        help='問題JSONファイルが格納されているディレクトリ (例: questions/)'
    )
    parser.add_argument(
        '--exam_number',
        default='119',
        help='試験番号 (例: 119)'
    )
    parser.add_argument(
        '--rerun_exp',
        required=True,
        help='再実行時の **ベースとなる** 実験識別子 (--exp に付加される部分) (例: gemini-2.0-flash_retry)'
    )
    parser.add_argument(
        '--main_script',
        default='main.py',
        help='実行するメインスクリプトのパス'
    )
    parser.add_argument(
        '--runner',
        default='uv run',
        help='メインスクリプトを実行するためのコマンド (例: "uv run", "python")'
    )

    args = parser.parse_args()

    try:
        skipped_questions = load_skipped_questions(args.skipped_list)
        if not skipped_questions:
            print("再実行する問題がありません。終了します。")
            return

        grouped_questions = group_questions_by_block(skipped_questions)
        if not grouped_questions:
             print("有効なブロックを持つ問題が見つかりませんでした。終了します。")
             return

        questions_dir_path = Path(args.questions_dir)
        main_script_path = Path(args.main_script)

        if not main_script_path.is_file():
            print(f"エラー: メインスクリプト '{args.main_script}' が見つかりません。")
            sys.exit(1)

        print(f"--- 合計 {len(skipped_questions)} 件の問題を再実行します ---")

        failed_blocks = []
        for block, questions_in_block in grouped_questions.items():
            print(f"\n--- ブロック {block} ({len(questions_in_block)} 件) を処理中 ---")

            input_json_filename = f"{args.exam_number}{block}_json.json"
            input_json_path = questions_dir_path / input_json_filename

            if not input_json_path.is_file():
                print(f"警告: 入力JSONファイルが見つかりません: {input_json_path}。ブロック {block} をスキップします。")
                failed_blocks.append(block)
                continue

            # --- ★★★ 変更点 ★★★ ---
            # main.py に渡す --exp の値をブロックごとに生成する
            # 例: exam_number=119, block=A, rerun_exp="gemini-2.0-flash_retry" -> "119A_gemini-2.0-flash_retry"
            exp_for_this_block = f"{args.exam_number}{block}_{args.rerun_exp}"
            # --- ★★★ 変更点ここまで ★★★ ---

            command = args.runner.split()
            command.append(str(main_script_path))
            command.append(str(input_json_path))
            command.extend(['--models'] + args.models)
            command.extend(['--questions'] + questions_in_block)
            # 生成したブロック固有の --exp 値を渡す
            command.extend(['--exp', exp_for_this_block])

            print(f"実行コマンド: {' '.join(command)}")

            try:
                result = subprocess.run(command, capture_output=True, text=True, check=False, encoding='utf-8')

                print(f"標準出力:\n{result.stdout}")
                if result.stderr:
                    print(f"標準エラー:\n{result.stderr}", file=sys.stderr)

                if result.returncode == 0:
                    print(f"ブロック {block} の処理が正常に完了しました。")
                else:
                    print(f"エラー: ブロック {block} の処理中にエラーが発生しました (リターンコード: {result.returncode})。", file=sys.stderr)
                    failed_blocks.append(block)

            except FileNotFoundError:
                 print(f"エラー: コマンド '{args.runner}' が見つかりません。", file=sys.stderr)
                 sys.exit(1)
            except Exception as e:
                print(f"エラー: ブロック {block} のコマンド実行中に予期せぬエラーが発生しました: {e}", file=sys.stderr)
                failed_blocks.append(block)

        print("\n--- 再実行結果 ---")
        if not failed_blocks:
            print("すべてのブロックの再実行が正常に完了しました（main.py内でのエラーは除く）。")
        else:
            print(f"以下のブロックでエラーが発生しました: {', '.join(failed_blocks)}")
            print("詳細は上記のエラーログを確認してください。")

    except FileNotFoundError as e:
        print(f"エラー: {e}", file=sys.stderr)
        sys.exit(1)
    except IOError as e:
        print(f"エラー: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"予期せぬエラーが発生しました: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()