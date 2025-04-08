import argparse
import pandas as pd
import json
from pathlib import Path
import glob
import sys
import textwrap
from collections import Counter
import difflib # オプション
from typing import Optional, Dict, Any # 型ヒント追加
import os # LLMSolver用
from dotenv import load_dotenv # LLMSolver用

# .envファイルを読み込む (LLM APIキー用)
load_dotenv()

# srcディレクトリをパスに追加 (LLMSolverをインポートするため)
# スクリプトの場所に応じて調整が必要な場合があります
script_dir = Path(__file__).parent
project_root = script_dir.parent # このスクリプトが scripts/ にあると仮定
sys.path.append(str(project_root))

try:
    from src.llm_solver import LLMSolver # LLMSolverをインポート
except ImportError:
    print("エラー: src.llm_solver が見つかりません。PYTHONPATHを確認してください。", file=sys.stderr)
    sys.exit(1)


# --- ヘルパー関数 ---

def load_wrong_answers_csv(exp: str, results_dir: Path = Path("results")) -> set:
    """指定された実験の誤答CSVファイルを読み込み、誤答問題番号のセットを返す"""
    # (変更なし)
    csv_path = results_dir / f"119_{exp}_wrong_answers.csv"
    if not csv_path.is_file():
        print(f"警告: 誤答ファイルが見つかりません: {csv_path}。空のセットを返します。", file=sys.stderr)
        return set()
    try:
        df = pd.read_csv(csv_path)
        if 'question_number' not in df.columns:
            print(f"エラー: {csv_path} に 'question_number' カラムが見つかりません。", file=sys.stderr)
            return set()
        # question_number が NaN でない行のみを対象にする
        valid_q_nums = df.dropna(subset=['question_number'])['question_number'].astype(str)
        return set(valid_q_nums)
    except pd.errors.EmptyDataError:
        print(f"情報: 誤答ファイルは空です: {csv_path}。空のセットを返します。")
        return set()
    except Exception as e:
        print(f"エラー: 誤答ファイル {csv_path} の読み込み中にエラーが発生しました: {e}", file=sys.stderr)
        return set()

# ▼▼▼ `load_answers_json` 関数の修正 ▼▼▼
def load_answers_json(exp: str, answers_dir: Path = Path("answers")) -> dict:
    """指定された実験の全ブロックの回答JSONを読み込み、問題番号をキーとする辞書を返す (cotも含む)"""
    all_answers = {}
    json_files = glob.glob(str(answers_dir / f"119?_{exp}.json"))

    if not json_files:
         print(f"警告: {exp} に対応する回答JSONファイルが {answers_dir} に見つかりません。", file=sys.stderr)
         return {}

    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if "results" not in data or not isinstance(data["results"], list):
                     print(f"警告: {json_file} の形式が不正です ('results'リストが見つかりません)。スキップします。", file=sys.stderr)
                     continue

                for q_result in data["results"]:
                    q_num = q_result.get("question_number")
                    q_text = q_result.get("question_text", "")
                    choices = q_result.get("choices")
                    has_image = q_result.get("has_image")

                    if not q_num:
                         print(f"警告: {json_file} 内に問題番号のないデータがあります。スキップします。", file=sys.stderr)
                         continue

                    q_num_str = str(q_num) # 必ず文字列として扱う

                    answer_data = {}
                    # ★★★ cot を抽出するロジックを追加 ★★★
                    cot_value = None # デフォルトはNone
                    if q_result.get("answers") and isinstance(q_result["answers"], list) and len(q_result["answers"]) > 0:
                        ans_data = q_result["answers"][0]
                        if not isinstance(ans_data, dict): # ans_dataが辞書でない場合スキップ
                            print(f"警告: {q_num_str} の answers[0] が辞書形式ではありません。スキップします。 data: {ans_data}", file=sys.stderr)
                            continue

                        if "error" in ans_data:
                             answer_data = {
                                 "answer": f"ERROR: {ans_data['error']}",
                                 "explanation": "",
                                 "cot": None, # エラー時はcotもNone
                             }
                        else:
                            answer_data = {
                                "answer": ans_data.get("answer", "N/A"),
                                "explanation": ans_data.get("explanation", "N/A"),
                                "cot": ans_data.get("cot"), # cotを取得 (なければNone)
                            }
                    else:
                         answer_data = {
                             "answer": "No Answer Data",
                             "explanation": "",
                             "cot": None,
                         }

                    # 辞書に情報を追加
                    full_q_data = {
                        **answer_data, # answer, explanation, cot を展開
                        "question_text": q_text,
                        "choices": choices,
                        "has_image": has_image,
                    }

                    if q_num_str in all_answers:
                        print(f"情報: 問題番号 {q_num_str} が複数のファイルに存在します。{json_file} のデータで上書きします。")
                    all_answers[q_num_str] = full_q_data

        except FileNotFoundError:
            print(f"警告: 回答ファイルが見つかりません: {json_file}。スキップします。", file=sys.stderr)
        except json.JSONDecodeError:
            print(f"エラー: 回答ファイル {json_file} は不正なJSON形式です。", file=sys.stderr)
        except Exception as e:
            print(f"エラー: 回答ファイル {json_file} の処理中に予期せぬエラーが発生しました: {e}", file=sys.stderr)

    return all_answers
# ▲▲▲ `load_answers_json` 関数の修正 ▲▲▲

def load_correct_answers(filepath: Path = Path("results/correct_answers.csv")) -> dict:
    """正解データを読み込み、問題番号をキーとする辞書を返す"""
    # (変更なし)
    if not filepath.is_file():
        print(f"警告: 正解ファイルが見つかりません: {filepath}", file=sys.stderr)
        return {}
    try:
        df = pd.read_csv(filepath)
        if '問題番号' not in df.columns or '解答' not in df.columns:
            print(f"エラー: {filepath} に '問題番号' または '解答' カラムが見つかりません。", file=sys.stderr)
            return {}
        # 問題番号を文字列としてインデックスに設定
        df['問題番号'] = df['問題番号'].astype(str)
        return df.set_index('問題番号')['解答'].to_dict()
    except Exception as e:
        print(f"エラー: 正解ファイル {filepath} の読み込み中にエラーが発生しました: {e}", file=sys.stderr)
        return {}


# --- 分析・レポート生成関数 ---

def analyze_single_choice_errors(df: pd.DataFrame, exp1: str, exp2: str) -> Counter:
    """両モデルが誤答した単数選択問題の誤答パターンを集計する"""
    # (変更なし)
    error_patterns = Counter()
    both_wrong_single = df[
        (df['comparison_type'] == 'Both Wrong') &
        (df['correct_answer'].astype(str).str.match(r'^[a-z]$', na=False)) & # Correct answer is single letter
        (df[f'{exp1}_answer'].notna()) &
        (df[f'{exp2}_answer'].notna()) &
        (df[f'{exp1}_answer'].astype(str).str.match(r'^[a-z]+$', na=False)) & # Model 1 answer is letter(s)
        (df[f'{exp2}_answer'].astype(str).str.match(r'^[a-z]+$', na=False)) # Model 2 answer is letter(s)
    ].copy()

    if not both_wrong_single.empty:
        # Make sure answers are strings before creating pattern
        both_wrong_single[f'{exp1}_answer'] = both_wrong_single[f'{exp1}_answer'].astype(str)
        both_wrong_single[f'{exp2}_answer'] = both_wrong_single[f'{exp2}_answer'].astype(str)
        both_wrong_single['correct_answer'] = both_wrong_single['correct_answer'].astype(str) # Ensure correct answer is also string

        both_wrong_single['error_pattern'] = both_wrong_single.apply(
            lambda row: f"Correct:{row['correct_answer']} | {exp1}:{row[f'{exp1}_answer']} | {exp2}:{row[f'{exp2}_answer']}",
            axis=1
        )
        error_patterns = Counter(both_wrong_single['error_pattern'])

    return error_patterns


# ▼▼▼ `generate_markdown_report` 関数の修正 ▼▼▼
def generate_markdown_report(
    df_filtered: pd.DataFrame, # 特定の比較タイプでフィルタリングされたDF
    exp1: str,
    exp2: str,
    output_md_path: Path, # 出力ファイルパス
    comparison_type_title: str, # レポートタイトル用の比較タイプ名
    error_analysis: Optional[Counter] = None # Both Wrongの場合のみ渡す
):
    """特定の比較タイプのデータからMarkdownレポートを生成する (cot表示を追加)"""
    report_lines = []

    report_lines.append(f"# 誤答比較レポート ({comparison_type_title}): {exp1} vs {exp2}")
    report_lines.append(f"生成日時: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")

    # --- サマリー ---
    report_lines.append("## 1. サマリー")
    report_lines.append("")
    report_lines.append(f"- **対象比較タイプ:** {comparison_type_title}")
    report_lines.append(f"- **該当問題数:** {len(df_filtered)}")
    report_lines.append("")

    # --- 誤答パターン分析 (Both Wrongの場合のみ) ---
    if comparison_type_title == 'Both Wrong' and error_analysis:
        report_lines.append("### 1.1 両モデル共通誤答のパターン (単数選択問題)")
        if error_analysis:
            report_lines.append("```")
            # 出現回数が多い順に表示
            for pattern, count in sorted(error_analysis.items(), key=lambda item: item[1], reverse=True):
                report_lines.append(f"{pattern}: {count}件")
            report_lines.append("```")
        else:
            report_lines.append("該当する共通誤答パターンは見つかりませんでした。")
        report_lines.append("")

    # --- 問題別詳細 ---
    report_lines.append(f"## 2. 問題別 詳細 ({comparison_type_title})")
    report_lines.append("")

    if df_filtered.empty:
        report_lines.append("該当する問題はありませんでした。")
    else:
        # question_number でソート
        try:
             def qnum_to_sortkey(qnum_str):
                 import re
                 # qnum_strがNaNでないことを確認
                 if pd.isna(qnum_str):
                      return float('inf')
                 qnum = str(qnum_str) # 文字列に変換
                 match = re.match(r'(\d+)([A-I])(\d+)', qnum)
                 if match:
                     block_ord = ord(match.group(2)) - ord('A')
                     num_part1 = int(match.group(1))
                     num_part2 = int(match.group(3))
                     # 桁数を考慮したソートキー (例: 119*100000 + A(0)*1000 + 1 = 11900001)
                     return num_part1 * 100000 + block_ord * 1000 + num_part2
                 else:
                      # 数字のみなどの場合も考慮（エラーにならないように）
                      try:
                           return int(qnum) # 例: "119" のような場合
                      except ValueError:
                           return float('inf') # それ以外の形式は最後に

             # NaNが含まれている可能性があるため、dropna()するか、fillna()で対処
             df_filtered = df_filtered.dropna(subset=['question_number'])
             df_filtered['sort_key'] = df_filtered['question_number'].apply(qnum_to_sortkey)
             df_sorted = df_filtered.sort_values('sort_key').drop(columns=['sort_key'])
        except Exception as e:
             print(f"警告: 問題番号のソート中にエラーが発生しました。文字列としてソートします。エラー: {e}")
             # 文字列としてソートする前に NaN を除去
             df_sorted = df_filtered.dropna(subset=['question_number']).sort_values("question_number")


        wrapper = textwrap.TextWrapper(width=80, replace_whitespace=False, drop_whitespace=False)

        for _, row in df_sorted.iterrows():
            q_num = row['question_number']
            correct = row['correct_answer']
            q_text = row.get('question_text', '問題文不明')
            choices = row.get('choices')
            has_image = row.get('has_image')
            ans1 = row[f'{exp1}_answer']
            exp1_expl = row[f'{exp1}_explanation'] if pd.notna(row[f'{exp1}_explanation']) else ""
            # ★★★ cot1 を取得 ★★★
            exp1_cot = row.get(f'{exp1}_cot') if pd.notna(row.get(f'{exp1}_cot')) else None

            ans2 = row[f'{exp2}_answer']
            exp2_expl = row[f'{exp2}_explanation'] if pd.notna(row[f'{exp2}_explanation']) else ""
            # ★★★ cot2 を取得 ★★★
            exp2_cot = row.get(f'{exp2}_cot') if pd.notna(row.get(f'{exp2}_cot')) else None

            report_lines.append(f"### 問題: {q_num}")
            report_lines.append(f"- **正解:** `{correct}`")
            if has_image is not None:
                image_status = "あり" if has_image else "なし"
                report_lines.append(f"- **画像:** {image_status}")
            else:
                report_lines.append(f"- **画像:** 不明")

            report_lines.append(f"- **問題文:**")
            report_lines.append("```")
            report_lines.extend(wrapper.wrap(q_text))
            report_lines.append("```")

            if choices and isinstance(choices, list):
                report_lines.append("- **選択肢:**")
                for choice in choices:
                    if isinstance(choice, dict) and len(choice) == 1:
                        key = list(choice.keys())[0]
                        value = choice[key]
                        report_lines.append(f"  - `{key}`: {value}")
                    elif isinstance(choice, str):
                         report_lines.append(f"  - {choice}")
                    else:
                        report_lines.append(f"  - {str(choice)} (形式不明)")
            elif choices:
                report_lines.append("- **選択肢:** (形式不正または取得失敗)")
                report_lines.append(f"  ```\n{choices}\n```")

            report_lines.append(f"- **{exp1} の回答:** `{ans1}`")
            if exp1_expl:
                report_lines.append(f"  - **説明 ({exp1}):**")
                report_lines.append("  ```")
                report_lines.extend(["  " + line for line in wrapper.wrap(exp1_expl)])
                report_lines.append("  ```")
            # ★★★ cot1 の表示を追加 ★★★
            if exp1_cot:
                report_lines.append(f"  - **思考プロセス (CoT) ({exp1}):**")
                report_lines.append("    <details>")
                report_lines.append("      <summary>クリックして展開</summary>")
                report_lines.append("")
                report_lines.append("      ```")
                report_lines.extend(["      " + line for line in exp1_cot.splitlines()])
                report_lines.append("      ```")
                report_lines.append("    </details>")

            report_lines.append(f"- **{exp2} の回答:** `{ans2}`")
            if exp2_expl:
                report_lines.append(f"  - **説明 ({exp2}):**")
                report_lines.append("  ```")
                report_lines.extend(["  " + line for line in wrapper.wrap(exp2_expl)])
                report_lines.append("  ```")
            # ★★★ cot2 の表示を追加 ★★★
            if exp2_cot:
                report_lines.append(f"  - **思考プロセス (CoT) ({exp2}):**")
                report_lines.append("    <details>")
                report_lines.append("      <summary>クリックして展開</summary>")
                report_lines.append("")
                report_lines.append("      ```")
                report_lines.extend(["      " + line for line in exp2_cot.splitlines()])
                report_lines.append("      ```")
                report_lines.append("    </details>")

            report_lines.append("---")
            report_lines.append("")

    # ファイル書き込み
    try:
        output_md_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_md_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(report_lines))
        print(f"Markdownレポート ({comparison_type_title}) を {output_md_path} に保存しました。")
    except Exception as e:
         print(f"エラー: Markdownレポート ({comparison_type_title}) の書き込み中にエラーが発生しました: {e}", file=sys.stderr)
# ▲▲▲ `generate_markdown_report` 関数の修正 ▲▲▲


# --- ★★★ 新しい関数 (LLM分析用) ★★★ ---

def read_markdown_files(md_files: list[Path]) -> str:
    """複数のMarkdownファイルの内容を読み込んで結合する"""
    # (変更なし)
    combined_content = ""
    for md_file in md_files:
        if md_file.is_file():
            try:
                with open(md_file, 'r', encoding='utf-8') as f:
                    combined_content += f"--- コンテンツ開始: {md_file.name} ---\n\n"
                    combined_content += f.read()
                    combined_content += f"\n\n--- コンテンツ終了: {md_file.name} ---\n\n"
            except Exception as e:
                print(f"警告: Markdownファイル {md_file} の読み込みに失敗しました: {e}", file=sys.stderr)
        else:
            print(f"警告: Markdownファイルが見つかりません: {md_file}", file=sys.stderr)
    return combined_content

# ▼▼▼ `generate_llm_analysis_prompt` 関数の修正 ▼▼▼
def generate_llm_analysis_prompt(exp1: str, exp2: str, combined_report_content: str) -> str:
    """LLMに比較考察を依頼するプロンプトを生成する (cot考慮指示追加)"""
    prompt = f"""
あなたは医療分野とAIモデルの評価に詳しい専門家です。
以下の背景情報と誤答比較レポートの内容に基づいて、2つのLLMモデル（{exp1} と {exp2}）の医師国家試験問題における誤答傾向を詳細に比較・考察してください。

【背景情報】
- 分析対象は、日本の医師国家試験（第119回）の問題に対する2つのLLM（{exp1}, {exp2}）の回答結果です。
- これから提示するレポートには、両モデルが誤答した問題、{exp1}のみが誤答した問題、{exp2}のみが誤答した問題の詳細が含まれています。
- レポートには、問題番号、正解、画像有無、問題文、選択肢、各モデルの回答、回答理由（説明）、**そして利用可能な場合は思考プロセス（cot）**が含まれます。
- 「両モデル共通誤答のパターン (単数選択問題)」セクションには、特に両モデルが同じように間違えた選択肢のパターンがまとめられています。

【誤答比較レポートの内容】
{combined_report_content}

【考察依頼】
上記のレポート内容を踏まえ、以下の観点から{exp1}と{exp2}を比較・考察し、結果をMarkdown形式でまとめてください。

1.  **全体的な誤答傾向の比較:**
    *   両モデルに共通する弱点や誤答しやすい問題タイプはありますか？（例: 知識問題、臨床推論問題、画像問題、計算問題、選択肢の多い問題など）
    *   {exp1}と{exp2}で、誤答する問題の傾向に違いはありますか？それぞれのモデルが特に苦手とする分野や問題形式（画像有無、選択肢の特性なども考慮）を指摘してください。
    *   片方のモデルだけが間違える問題から、それぞれのモデルの強みや弱みを推測できますか？

2.  **具体的な誤答パターンの分析:**
    *   「両モデル共通誤答のパターン」から、特定の誤答（例: 正解がAなのに両方Cを選んでしまう）が頻発している場合、その原因として何が考えられますか？（例: 一般的な誤解、問題文や選択肢の曖昧さ、特定の知識領域の欠落など）
    *   各モデルの誤答において、回答理由（説明）や**思考プロセス（cot）**が提供されている場合、その内容から誤答の原因（知識不足、読解ミス、推論の誤り、**思考過程の問題点**など）を推測してください。**特に、cotがある場合は、その内容から思考のどの段階で誤りが生じたかを分析してください。**

3.  **総括:**
    *   今回の比較結果から、{exp1}と{exp2}の全体的な性能や特性について、どのようなことが言えますか？
    *   もし改善するとしたら、それぞれのモデルに対してどのような方向性（学習データの追加、プロンプトの改善、画像理解能力の強化、**思考プロセスの正確性向上**など）が考えられますか？

【出力形式】
- Markdown形式で、上記の考察依頼の項目（1, 2, 3）に対応するセクションを設けて記述してください。
- 専門的かつ客観的な視点で、具体的な問題番号や誤答パターン、選択肢の内容、画像有無、**思考プロセス（cot）**などに言及しながら考察を記述してください。
- 推測に基づく場合は、その旨を明記してください。
"""
    return prompt
# ▲▲▲ `generate_llm_analysis_prompt` 関数の修正 ▲▲▲

def call_llm_for_analysis(
    prompt: str,
    model_key: str,
    llm_solver: LLMSolver # LLMSolverのインスタンスを受け取る
) -> str:
    """指定されたLLMモデルで分析プロンプトを実行し、結果を返す"""
    # (変更なし)
    print(f"LLM ({model_key}) による分析を開始します...")
    try:
        config, client = llm_solver.get_config_and_client(model_key)

        max_output_tokens = 16384

        if config["client_type"] == "anthropic":
             messages=[{"role": "user", "content": prompt}]
             response = client.messages.create(
                 model=config["model_name"],
                 messages=messages,
                 max_tokens=max_output_tokens,
                 temperature=0.5
             )
             analysis_result = response.content[0].text
        else: # OpenAI互換API
             messages=[
                 {"role": "user", "content": prompt}
             ]
             response = client.chat.completions.create(
                 model=config["model_name"],
                 messages=messages,
                 max_tokens=max_output_tokens,
                 temperature=0.5
             )
             analysis_result = response.choices[0].message.content

        print(f"LLM ({model_key}) による分析が完了しました。")
        return analysis_result

    except Exception as e:
        print(f"エラー: LLM ({model_key}) での分析中にエラーが発生しました: {e}", file=sys.stderr)
        error_message = f"LLMによる分析中にエラーが発生しました: {e}"
        if "max_tokens" in str(e).lower():
            error_message += f"\n考えられる原因: max_tokens={max_output_tokens} がモデルの上限を超えている可能性があります。モデルのドキュメントを確認してください。"
        return error_message


# --- メイン処理 ---

# ▼▼▼ `compare_wrong_answers` 関数の修正 ▼▼▼
def compare_wrong_answers(
    exp1: str,
    exp2: str,
    output_dir: Path,
    results_dir: Path,
    answers_dir: Path,
    output_format: str,
    analyze_model_key: Optional[str] # LLM分析に使用するモデルキー
):
    """2つの実験の誤答を比較し、指定された形式で出力し、オプションでLLM分析を行う (cot対応)"""

    print(f"実験 {exp1} と {exp2} の誤答比較を開始します...")

    # 1. 誤答リストの読み込み
    print("誤答リストを読み込んでいます...")
    wrong1 = load_wrong_answers_csv(exp1, results_dir)
    wrong2 = load_wrong_answers_csv(exp2, results_dir)

    if not wrong1 and not wrong2:
        print("エラー: 両方の実験の誤答データが見つかりませんでした。処理を中断します。", file=sys.stderr)
        return

    # 2. 誤答パターンの分類
    both_wrong = wrong1.intersection(wrong2)
    exp1_wrong_only = wrong1.difference(wrong2)
    exp2_wrong_only = wrong2.difference(wrong1)

    print(f"  - 両方誤答: {len(both_wrong)} 件")
    print(f"  - {exp1} のみ誤答: {len(exp1_wrong_only)} 件")
    print(f"  - {exp2} のみ誤答: {len(exp2_wrong_only)} 件")

    # 3. 回答詳細データの読み込み
    print("回答詳細データを読み込んでいます...")
    answers1 = load_answers_json(exp1, answers_dir)
    answers2 = load_answers_json(exp2, answers_dir)

    if not answers1 and not answers2:
         print("警告: 両方の実験の回答詳細データが見つかりませんでした。回答情報は空になります。")

    # 4. 正解データの読み込み
    print("正解データを読み込んでいます...")
    correct_answers = load_correct_answers(results_dir / "correct_answers.csv")

    # 5. 比較結果のデータフレーム作成
    comparison_data = []
    all_compared_questions = both_wrong.union(exp1_wrong_only).union(exp2_wrong_only)

    # ★★★ デフォルト辞書に cot を追加 ★★★
    default_q_data = {"answer": "Data Not Found", "explanation": "", "cot": None, "question_text": "", "choices": None, "has_image": None}

    for q_num in sorted(list(all_compared_questions)):
        ans1_data = answers1.get(q_num, default_q_data)
        ans2_data = answers2.get(q_num, default_q_data)
        correct = correct_answers.get(q_num, "N/A")

        q_text = ans1_data["question_text"] or ans2_data["question_text"]
        choices = ans1_data["choices"] or ans2_data["choices"]
        has_image = ans1_data["has_image"] if ans1_data["has_image"] is not None else ans2_data["has_image"]

        if q_num in both_wrong:
            comp_type = "Both Wrong"
        elif q_num in exp1_wrong_only:
            comp_type = f"{exp1} Wrong Only"
            if ans2_data["answer"] == "Data Not Found":
                 ans2_data["answer"] = "Correct (or Data Not Found)"
        elif q_num in exp2_wrong_only:
             comp_type = f"{exp2} Wrong Only"
             if ans1_data["answer"] == "Data Not Found":
                 ans1_data["answer"] = "Correct (or Data Not Found)"
        else:
             comp_type = "Unknown"

        comparison_data.append({
            "question_number": q_num,
            "comparison_type": comp_type,
            "correct_answer": correct,
            "question_text": q_text,
            "choices": choices,
            "has_image": has_image,
            f"{exp1}_answer": ans1_data["answer"],
            f"{exp1}_explanation": ans1_data["explanation"],
            f"{exp1}_cot": ans1_data["cot"], # ★★★ cot1 を追加 ★★★
            f"{exp2}_answer": ans2_data["answer"],
            f"{exp2}_explanation": ans2_data["explanation"],
            f"{exp2}_cot": ans2_data["cot"], # ★★★ cot2 を追加 ★★★
        })

    if not comparison_data:
        print("比較対象となる誤答問題がありませんでした。")
        return

    df_comparison = pd.DataFrame(comparison_data)

    # ★★★ カラム順序に cot を追加 ★★★
    columns_order = [
        "question_number", "comparison_type", "correct_answer",
        "has_image", "question_text", "choices",
        f"{exp1}_answer", f"{exp1}_explanation", f"{exp1}_cot", # cot1 追加
        f"{exp2}_answer", f"{exp2}_explanation", f"{exp2}_cot"  # cot2 追加
    ]
    existing_columns = [col for col in columns_order if col in df_comparison.columns]
    other_columns = [col for col in df_comparison.columns if col not in existing_columns]
    df_comparison = df_comparison[existing_columns + other_columns]


    # 6. 出力形式に応じて処理
    output_basename = f"comparison_119_{exp1}_vs_{exp2}"
    md_files_generated = []

    # CSV出力
    if output_format in ['csv', 'all']:
        output_csv_path = output_dir / f"{output_basename}.csv"
        try:
            df_csv_out = df_comparison.copy()
            # choicesがリスト/辞書の場合、JSON文字列として保存
            if 'choices' in df_csv_out.columns:
                 df_csv_out['choices'] = df_csv_out['choices'].apply(lambda x: json.dumps(x, ensure_ascii=False) if x is not None else None)
            # cot も文字列として保存 (改行は保持されるはず)

            df_csv_out.to_csv(output_csv_path, index=False, encoding='utf-8-sig')
            print(f"比較結果CSVを {output_csv_path} に保存しました。")
        except Exception as e:
            print(f"エラー: 比較結果のCSVファイルへの書き込み中にエラーが発生しました: {e}", file=sys.stderr)

    # Markdownレポート出力 (分割)
    if output_format in ['md', 'all']:
        print("誤答パターンの簡易分析を実行中 (Both Wrong)...")
        error_analysis = analyze_single_choice_errors(df_comparison, exp1, exp2)

        print("Markdownレポートを生成中 (分割)...")
        comparison_types = ["Both Wrong", f"{exp1} Wrong Only", f"{exp2} Wrong Only"]

        for comp_type in comparison_types:
            df_filtered = df_comparison[df_comparison['comparison_type'] == comp_type]

            if not df_filtered.empty:
                safe_comp_type_name = comp_type.replace(' ', '_').replace('/','_').replace(':','_')
                output_md_path = output_dir / f"{output_basename}_{safe_comp_type_name}.md"
                error_analysis_arg = error_analysis if comp_type == "Both Wrong" else None
                generate_markdown_report(
                    df_filtered,
                    exp1,
                    exp2,
                    output_md_path,
                    comp_type,
                    error_analysis_arg
                )
                md_files_generated.append(output_md_path)
            else:
                 print(f"情報: 比較タイプ '{comp_type}' に該当する問題がないため、Markdownファイルは生成されません。")

    # 7. LLMによる分析 (オプション)
    if analyze_model_key:
        if not md_files_generated:
            print("警告: LLM分析の入力となるMarkdownレポートが生成されませんでした。LLM分析をスキップします。", file=sys.stderr)
            return

        print(f"\n--- LLM ({analyze_model_key}) による比較考察を実行します ---")
        print("生成されたMarkdownレポートを読み込んでいます...")
        combined_content = read_markdown_files(md_files_generated)

        if not combined_content:
             print("エラー: Markdownレポートの内容を読み込めませんでした。LLM分析をスキップします。", file=sys.stderr)
             return

        prompt = generate_llm_analysis_prompt(exp1, exp2, combined_content)
        # LLMSolverのインスタンス化
        llm_solver_instance = LLMSolver(system_prompt_type="default")
        analysis_result = call_llm_for_analysis(prompt, analyze_model_key, llm_solver_instance)

        analysis_output_path = output_dir / f"analysis_119_{exp1}_vs_{exp2}_by_{analyze_model_key}.md"
        try:
            with open(analysis_output_path, 'w', encoding='utf-8') as f:
                f.write(f"# LLM ({analyze_model_key}) による {exp1} vs {exp2} 誤答比較考察\n\n")
                f.write(f"分析実行日時: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                f.write(analysis_result)
            print(f"LLMによる分析結果を {analysis_output_path} に保存しました。")
        except Exception as e:
            print(f"エラー: LLM分析結果のMarkdownファイルへの書き込み中にエラーが発生しました: {e}", file=sys.stderr)

# ▲▲▲ `compare_wrong_answers` 関数の修正 ▲▲▲


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='2つのLLM実験の誤答パターンを比較し、CSVまたはMarkdown形式で出力します。\nMarkdownは比較タイプごとに分割され、オプションでLLMによる考察も生成します。',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # --- 引数定義 (変更なし) ---
    parser.add_argument(
        'exp1',
        help='比較対象の実験識別子1 (例: gemini-2.5-pro)'
    )
    parser.add_argument(
        'exp2',
        help='比較対象の実験識別子2 (例: CA-DSR1-DQ32B-JP)'
    )
    parser.add_argument(
        '--results_dir',
        default='results',
        help='誤答CSVファイルと正解CSVファイルが格納されているディレクトリ'
    )
    parser.add_argument(
        '--answers_dir',
        default='answers',
        help='回答詳細JSONファイルが格納されているディレクトリ'
    )
    parser.add_argument(
        '--output_dir',
        default='results/comparison',
        help='比較結果ファイルを出力するディレクトリ (デフォルト: ./results/comparison)'
    )
    parser.add_argument(
        '--output_format',
        choices=['csv', 'md', 'all'],
        default='all',
        help='比較レポートの出力形式 (csv: CSVのみ, md: Markdownのみ(分割), all: 両方)'
    )
    parser.add_argument(
        '--analyze_with_llm',
        metavar='MODEL_KEY',
        default=None,
        help='指定したLLMモデルで誤答比較レポートの考察を行い、結果をMarkdownで出力します (例: gemini-2.5-pro-exp-03-25, gpt-4o)。APIキーが必要です。'
    )

    args = parser.parse_args()

    results_path = Path(args.results_dir)
    answers_path = Path(args.answers_dir)
    output_path = Path(args.output_dir)

    output_path.mkdir(parents=True, exist_ok=True)

    compare_wrong_answers(
        args.exp1,
        args.exp2,
        output_path,
        results_path,
        answers_path,
        args.output_format,
        args.analyze_with_llm
    )