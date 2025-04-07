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


# --- ヘルパー関数 (変更なし) ---
# load_wrong_answers_csv, load_answers_json, load_correct_answers

def load_wrong_answers_csv(exp: str, results_dir: Path = Path("results")) -> set:
    """指定された実験の誤答CSVファイルを読み込み、誤答問題番号のセットを返す"""
    csv_path = results_dir / f"119_{exp}_wrong_answers.csv"
    if not csv_path.is_file():
        print(f"警告: 誤答ファイルが見つかりません: {csv_path}。空のセットを返します。", file=sys.stderr)
        return set()
    try:
        df = pd.read_csv(csv_path)
        if 'question_number' not in df.columns:
            print(f"エラー: {csv_path} に 'question_number' カラムが見つかりません。", file=sys.stderr)
            return set()
        return set(df['question_number'].astype(str))
    except pd.errors.EmptyDataError:
        print(f"情報: 誤答ファイルは空です: {csv_path}。空のセットを返します。")
        return set()
    except Exception as e:
        print(f"エラー: 誤答ファイル {csv_path} の読み込み中にエラーが発生しました: {e}", file=sys.stderr)
        return set()

def load_answers_json(exp: str, answers_dir: Path = Path("answers")) -> dict:
    """指定された実験の全ブロックの回答JSONを読み込み、問題番号をキーとする辞書を返す"""
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
                    if not q_num:
                         print(f"警告: {json_file} 内に問題番号のないデータがあります。スキップします。", file=sys.stderr)
                         continue

                    first_answer_data = {}
                    if q_result.get("answers") and isinstance(q_result["answers"], list) and len(q_result["answers"]) > 0:
                        ans_data = q_result["answers"][0]
                        if "error" in ans_data:
                             first_answer_data = {
                                 "answer": f"ERROR: {ans_data['error']}",
                                 "explanation": "",
                                 "question_text": q_text
                             }
                        else:
                            first_answer_data = {
                                "answer": ans_data.get("answer", "N/A"),
                                "explanation": ans_data.get("explanation", "N/A"),
                                "question_text": q_text
                            }
                    else:
                         first_answer_data = {
                             "answer": "No Answer Data",
                             "explanation": "",
                             "question_text": q_text
                         }

                    if q_num in all_answers:
                        print(f"情報: 問題番号 {q_num} が複数のファイルに存在します。{json_file} のデータで上書きします。")
                    all_answers[str(q_num)] = first_answer_data

        except FileNotFoundError:
            print(f"警告: 回答ファイルが見つかりません: {json_file}。スキップします。", file=sys.stderr)
        except json.JSONDecodeError:
            print(f"エラー: 回答ファイル {json_file} は不正なJSON形式です。", file=sys.stderr)
        except Exception as e:
            print(f"エラー: 回答ファイル {json_file} の処理中に予期せぬエラーが発生しました: {e}", file=sys.stderr)

    return all_answers

def load_correct_answers(filepath: Path = Path("results/correct_answers.csv")) -> dict:
    """正解データを読み込み、問題番号をキーとする辞書を返す"""
    if not filepath.is_file():
        print(f"警告: 正解ファイルが見つかりません: {filepath}", file=sys.stderr)
        return {}
    try:
        df = pd.read_csv(filepath)
        if '問題番号' not in df.columns or '解答' not in df.columns:
            print(f"エラー: {filepath} に '問題番号' または '解答' カラムが見つかりません。", file=sys.stderr)
            return {}
        return df.set_index(df['問題番号'].astype(str))['解答'].to_dict()
    except Exception as e:
        print(f"エラー: 正解ファイル {filepath} の読み込み中にエラーが発生しました: {e}", file=sys.stderr)
        return {}


# --- 分析・レポート生成関数 (generate_markdown_reportは変更なし) ---
# analyze_single_choice_errors, generate_markdown_report

def analyze_single_choice_errors(df: pd.DataFrame, exp1: str, exp2: str) -> Counter:
    """両モデルが誤答した単数選択問題の誤答パターンを集計する"""
    error_patterns = Counter()
    both_wrong_single = df[
        (df['comparison_type'] == 'Both Wrong') &
        (df['correct_answer'].str.match(r'^[a-z]$', na=False)) &
        (df[f'{exp1}_answer'].notna()) &
        (df[f'{exp2}_answer'].notna())
    ].copy()

    both_wrong_single = both_wrong_single[
        (both_wrong_single[f'{exp1}_answer'].str.match(r'^[a-z]+$', na=False)) &
        (both_wrong_single[f'{exp2}_answer'].str.match(r'^[a-z]+$', na=False))
    ]

    if not both_wrong_single.empty:
        both_wrong_single['error_pattern'] = both_wrong_single.apply(
            lambda row: f"Correct:{row['correct_answer']} | {exp1}:{row[f'{exp1}_answer']} | {exp2}:{row[f'{exp2}_answer']}",
            axis=1
        )
        error_patterns = Counter(both_wrong_single['error_pattern'])

    return error_patterns

def generate_markdown_report(
    df_filtered: pd.DataFrame, # 特定の比較タイプでフィルタリングされたDF
    exp1: str,
    exp2: str,
    output_md_path: Path, # 出力ファイルパス
    comparison_type_title: str, # レポートタイトル用の比較タイプ名
    error_analysis: Optional[Counter] = None # Both Wrongの場合のみ渡す
):
    """特定の比較タイプのデータからMarkdownレポートを生成する"""
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
            for pattern, count in error_analysis.most_common():
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
        df_sorted = df_filtered.sort_values("question_number")
        wrapper = textwrap.TextWrapper(width=80, replace_whitespace=False, drop_whitespace=False)

        for _, row in df_sorted.iterrows():
            q_num = row['question_number']
            correct = row['correct_answer']
            q_text = row.get('question_text', '問題文不明')
            ans1 = row[f'{exp1}_answer']
            exp1_expl = row[f'{exp1}_explanation'] if pd.notna(row[f'{exp1}_explanation']) else ""
            ans2 = row[f'{exp2}_answer']
            exp2_expl = row[f'{exp2}_explanation'] if pd.notna(row[f'{exp2}_explanation']) else ""

            report_lines.append(f"### 問題: {q_num}") # 見出しレベルを上げる
            report_lines.append(f"- **正解:** `{correct}`")
            report_lines.append(f"- **問題文:**")
            report_lines.append("```")
            report_lines.extend(wrapper.wrap(q_text))
            report_lines.append("```")
            report_lines.append(f"- **{exp1} の回答:** `{ans1}`")
            if exp1_expl:
                report_lines.append(f"  - **説明 ({exp1}):**")
                report_lines.append("  ```")
                report_lines.extend(["  " + line for line in wrapper.wrap(exp1_expl)])
                report_lines.append("  ```")
            report_lines.append(f"- **{exp2} の回答:** `{ans2}`")
            if exp2_expl:
                report_lines.append(f"  - **説明 ({exp2}):**")
                report_lines.append("  ```")
                report_lines.extend(["  " + line for line in wrapper.wrap(exp2_expl)])
                report_lines.append("  ```")

            # --- (オプション) 説明文の差分表示 ---
            # if comparison_type_title == 'Both Wrong' and exp1_expl and exp2_expl:
            #     report_lines.append(f"  - **説明文の差分:**")
            #     diff = difflib.ndiff(exp1_expl.splitlines(), exp2_expl.splitlines())
            #     report_lines.append("  ```diff")
            #     report_lines.extend([line for line in diff if line.strip()])
            #     report_lines.append("  ```")
            # ------------------------------------

            report_lines.append("---")
            report_lines.append("")

    # ファイル書き込み
    try:
        output_md_path.parent.mkdir(parents=True, exist_ok=True) # 念のため親ディレクトリ作成
        with open(output_md_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(report_lines))
        print(f"Markdownレポート ({comparison_type_title}) を {output_md_path} に保存しました。")
    except Exception as e:
         print(f"エラー: Markdownレポート ({comparison_type_title}) の書き込み中にエラーが発生しました: {e}", file=sys.stderr)


# --- ★★★ 新しい関数 (LLM分析用) ★★★ ---

def read_markdown_files(md_files: list[Path]) -> str:
    """複数のMarkdownファイルの内容を読み込んで結合する"""
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

def generate_llm_analysis_prompt(exp1: str, exp2: str, combined_report_content: str) -> str:
    """LLMに比較考察を依頼するプロンプトを生成する"""
    prompt = f"""
あなたは医療分野とAIモデルの評価に詳しい専門家です。
以下の背景情報と誤答比較レポートの内容に基づいて、2つのLLMモデル（{exp1} と {exp2}）の医師国家試験問題における誤答傾向を詳細に比較・考察してください。

【背景情報】
- 分析対象は、日本の医師国家試験（第119回）の問題に対する2つのLLM（{exp1}, {exp2}）の回答結果です。
- これから提示するレポートには、両モデルが誤答した問題、{exp1}のみが誤答した問題、{exp2}のみが誤答した問題の詳細が含まれています。
- レポートには、問題番号、正解、各モデルの回答、および可能であれば回答理由（説明）が含まれます。
- 「両モデル共通誤答のパターン (単数選択問題)」セクションには、特に両モデルが同じように間違えた選択肢のパターンがまとめられています。

【誤答比較レポートの内容】
{combined_report_content}

【考察依頼】
上記のレポート内容を踏まえ、以下の観点から{exp1}と{exp2}を比較・考察し、結果をMarkdown形式でまとめてください。

1.  **全体的な誤答傾向の比較:**
    *   両モデルに共通する弱点や誤答しやすい問題タイプはありますか？（例: 知識問題、臨床推論問題、画像問題、計算問題など）
    *   {exp1}と{exp2}で、誤答する問題の傾向に違いはありますか？それぞれのモデルが特に苦手とする分野や問題形式を指摘してください。
    *   片方のモデルだけが間違える問題から、それぞれのモデルの強みや弱みを推測できますか？

2.  **具体的な誤答パターンの分析:**
    *   「両モデル共通誤答のパターン」から、特定の誤答（例: 正解がAなのに両方Cを選んでしまう）が頻発している場合、その原因として何が考えられますか？（例: 一般的な誤解、問題文の曖昧さ、特定の知識領域の欠落など）
    *   各モデルの誤答において、回答理由（説明）が提供されている場合、その内容から誤答の原因（知識不足、読解ミス、推論の誤りなど）を推測してください。

3.  **総括:**
    *   今回の比較結果から、{exp1}と{exp2}の全体的な性能や特性について、どのようなことが言えますか？
    *   もし改善するとしたら、それぞれのモデルに対してどのような方向性（学習データの追加、プロンプトの改善など）が考えられますか？

【出力形式】
- Markdown形式で、上記の考察依頼の項目（1, 2, 3）に対応するセクションを設けて記述してください。
- 専門的かつ客観的な視点で、具体的な問題番号や誤答パターンに言及しながら考察を記述してください。
- 推測に基づく場合は、その旨を明記してください。
"""
    return prompt

def call_llm_for_analysis(
    prompt: str,
    model_key: str,
    llm_solver: LLMSolver # LLMSolverのインスタンスを受け取る
) -> str:
    """指定されたLLMモデルで分析プロンプトを実行し、結果を返す"""
    print(f"LLM ({model_key}) による分析を開始します...")
    try:
        config, client = llm_solver.get_config_and_client(model_key)

        # ★★★ max_tokensの値を増やす ★★★
        # モデルに応じて適切な最大値を設定（ここでは16384を試す）
        # モデルによってはこの値が大きすぎるとエラーになる可能性あり
        max_output_tokens = 16384

        if config["client_type"] == "anthropic":
             messages=[{"role": "user", "content": prompt}]
             response = client.messages.create(
                 model=config["model_name"],
                 messages=messages,
                 max_tokens=max_output_tokens, # ★ 変更
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
                 max_tokens=max_output_tokens, # ★ 変更
                 temperature=0.5
             )
             analysis_result = response.choices[0].message.content

        print(f"LLM ({model_key}) による分析が完了しました。")
        return analysis_result

    except Exception as e:
        print(f"エラー: LLM ({model_key}) での分析中にエラーが発生しました: {e}", file=sys.stderr)
        # エラーによっては、max_tokensが原因である可能性を示すメッセージを追加
        error_message = f"LLMによる分析中にエラーが発生しました: {e}"
        if "max_tokens" in str(e).lower():
            error_message += f"\n考えられる原因: max_tokens={max_output_tokens} がモデルの上限を超えている可能性があります。モデルのドキュメントを確認してください。"
        return error_message


# --- メイン処理 (変更箇所あり) ---

def compare_wrong_answers(
    exp1: str,
    exp2: str,
    output_dir: Path,
    results_dir: Path,
    answers_dir: Path,
    output_format: str,
    analyze_model_key: Optional[str] # LLM分析に使用するモデルキー
):
    """2つの実験の誤答を比較し、指定された形式で出力し、オプションでLLM分析を行う"""

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

    for q_num in sorted(list(all_compared_questions)):
        ans1 = answers1.get(q_num, {"answer": "Data Not Found", "explanation": "", "question_text": ""})
        ans2 = answers2.get(q_num, {"answer": "Data Not Found", "explanation": "", "question_text": ""})
        correct = correct_answers.get(q_num, "N/A")

        if q_num in both_wrong:
            comp_type = "Both Wrong"
        elif q_num in exp1_wrong_only:
            comp_type = f"{exp1} Wrong Only"
            ans2["answer"] = ans2["answer"] if ans2["answer"] != "Data Not Found" else "Correct (or Data Not Found)"
        elif q_num in exp2_wrong_only:
             comp_type = f"{exp2} Wrong Only"
             ans1["answer"] = ans1["answer"] if ans1["answer"] != "Data Not Found" else "Correct (or Data Not Found)"
        else:
             comp_type = "Unknown"

        comparison_data.append({
            "question_number": q_num,
            "comparison_type": comp_type,
            "correct_answer": correct,
            f"{exp1}_answer": ans1["answer"],
            f"{exp1}_explanation": ans1["explanation"],
            f"{exp2}_answer": ans2["answer"],
            f"{exp2}_explanation": ans2["explanation"],
            "question_text": ans1["question_text"] or ans2["question_text"]
        })

    if not comparison_data:
        print("比較対象となる誤答問題がありませんでした。")
        return

    df_comparison = pd.DataFrame(comparison_data)

    columns_order = [
        "question_number", "comparison_type", "correct_answer", "question_text",
        f"{exp1}_answer", f"{exp1}_explanation",
        f"{exp2}_answer", f"{exp2}_explanation"
    ]
    existing_columns = [col for col in columns_order if col in df_comparison.columns]
    df_comparison = df_comparison[existing_columns]

    # 6. 出力形式に応じて処理
    output_basename = f"comparison_119_{exp1}_vs_{exp2}"
    md_files_generated = [] # 生成されたMDファイルパスを記録

    # CSV出力
    if output_format in ['csv', 'all']:
        output_csv_path = output_dir / f"{output_basename}.csv"
        try:
            df_comparison.to_csv(output_csv_path, index=False, encoding='utf-8-sig')
            print(f"比較結果CSVを {output_csv_path} に保存しました。")
        except Exception as e:
            print(f"エラー: 比較結果のCSVファイルへの書き込み中にエラーが発生しました: {e}", file=sys.stderr)

    # Markdownレポート出力 (分割)
    if output_format in ['md', 'all']:
        print("誤答パターンの簡易分析を実行中 (Both Wrong)...")
        error_analysis = analyze_single_choice_errors(df_comparison, exp1, exp2) # Both Wrong分析はここで実行

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
                md_files_generated.append(output_md_path) # 生成したファイルパスを記録
            else:
                 print(f"情報: 比較タイプ '{comp_type}' に該当する問題がないため、Markdownファイルは生成されません。")

    # 7. LLMによる分析 (オプション)
    if analyze_model_key:
        if not md_files_generated:
            print("警告: LLM分析の入力となるMarkdownレポートが生成されませんでした。LLM分析をスキップします。", file=sys.stderr)
            return

        print(f"\n--- LLM ({analyze_model_key}) による比較考察を実行します ---")
        # 生成されたMarkdownファイルの内容を結合
        print("生成されたMarkdownレポートを読み込んでいます...")
        combined_content = read_markdown_files(md_files_generated)

        if not combined_content:
             print("エラー: Markdownレポートの内容を読み込めませんでした。LLM分析をスキップします。", file=sys.stderr)
             return

        # トークン数削減のため、簡易化（ここではコメントアウト。必要なら実装）
        # print("レポート内容を要約中（トークン数削減）...")
        # summarized_content = summarize_report_content(combined_content) # 要約関数を別途実装する必要あり
        # prompt = generate_llm_analysis_prompt(exp1, exp2, summarized_content)

        # 全文を使ってプロンプト生成
        prompt = generate_llm_analysis_prompt(exp1, exp2, combined_content)

        # LLMSolverのインスタンスを作成 (API呼び出しのため)
        # system_prompt_type はダミーでも良いが、念のため指定
        llm_solver = LLMSolver(system_prompt_type="v2")

        # LLM分析実行
        analysis_result = call_llm_for_analysis(prompt, analyze_model_key, llm_solver)

        # 分析結果を保存
        analysis_output_path = output_dir / f"analysis_119_{exp1}_vs_{exp2}_by_{analyze_model_key}.md"
        try:
            with open(analysis_output_path, 'w', encoding='utf-8') as f:
                f.write(f"# LLM ({analyze_model_key}) による {exp1} vs {exp2} 誤答比較考察\n\n")
                f.write(f"分析実行日時: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                f.write(analysis_result)
            print(f"LLMによる分析結果を {analysis_output_path} に保存しました。")
        except Exception as e:
            print(f"エラー: LLM分析結果のMarkdownファイルへの書き込み中にエラーが発生しました: {e}", file=sys.stderr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='2つのLLM実験の誤答パターンを比較し、CSVまたはMarkdown形式で出力します。\nMarkdownは比較タイプごとに分割され、オプションでLLMによる考察も生成します。',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # --- 引数定義 ---
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
        help='比較結果ファイルを出力するディレクトリ'
    )
    parser.add_argument(
        '--output_format',
        choices=['csv', 'md', 'all'],
        default='all',
        help='比較レポートの出力形式 (csv: CSVのみ, md: Markdownのみ(分割), all: 両方)'
    )
    # ★★★ LLM分析用の新しい引数 ★★★
    parser.add_argument(
        '--analyze_with_llm',
        metavar='MODEL_KEY',
        default=None, # デフォルトでは実行しない
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
        args.analyze_with_llm # 新しい引数を渡す
    )