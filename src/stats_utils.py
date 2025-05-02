import pandas as pd

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

    # 'correct_answer' が NaN でない行を採点対象とする
    graded_df = combined_df.dropna(subset=['correct_answer']).copy()
    if len(graded_df) != len(combined_df):
        print(f"情報: 統合された全 {len(combined_df)} 件のうち、正解データが存在しない {len(combined_df) - len(graded_df)} 件を集計から除外します。")

    if graded_df.empty:
        print("警告: 正解データが存在する有効な結果が統合後に見つかりませんでした。")
        empty_stats['total']['df'] = pd.DataFrame() # 空のDFを返すように修正
        return empty_stats

    try:
        graded_df['question_number'] = graded_df['question_number'].astype(str)
        graded_df['block'] = graded_df['question_number'].apply(
            lambda x: x[3].upper() if isinstance(x, str) and len(x) > 3 and x[3].isalpha() else 'Unknown'
        )
        graded_df['has_image'] = graded_df['has_image'].fillna(False).astype(bool)
        # is_correct は grade_answers で bool or None になっているので、ここでは bool に変換しない
        # graded_df['is_correct'] = graded_df['is_correct'].fillna(False).astype(bool)
        # is_correct が None の場合は False として扱う (集計のため)
        graded_df['is_correct'] = graded_df['is_correct'].fillna(False)

    except Exception as e:
        print(f"エラー: 統合データの前処理中にエラーが発生しました: {e}")
        empty_stats['total']['df'] = pd.DataFrame() # エラー時も空のDFを返す
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

    def required_point(question_number):
        # 必修問題 (B, Eブロック) の点数計算ロジック
        try:
            block = question_number[3].upper() if len(question_number) > 3 else ''
            if block in ['B', 'E']:
                # 問題番号の数字部分を取得 (例: 119B26 -> 26)
                num_part = ''.join(filter(str.isdigit, question_number[4:]))
                num = int(num_part) if num_part else 0
                # B26-B50, E26-E50 は3点、それ以外は1点
                return 3 if 26 <= num <= 50 else 1
            else: return 1 # 一般問題は1点
        except (IndexError, ValueError, TypeError):
            print(f"警告: 必修問題の点数計算で問題番号形式エラー: {question_number}。デフォルト1点とします。")
            return 1

    def calculate_points(row):
        # 各問題の獲得点数(score)と満点(possible)を計算
        block = row['block']
        is_correct = row['is_correct']
        question_number = row['question_number']
        possible = 0
        score = 0
        if block in required_blocks:
            possible = required_point(question_number)
            score = possible if is_correct else 0
        elif block in general_blocks:
            possible = 1 # 一般問題は1点満点
            score = 1 if is_correct else 0
        # 不明ブロックなどは score=0, possible=0 とする
        return pd.Series({'score': score, 'possible': possible})

    try:
        points_df = graded_df.apply(calculate_points, axis=1)
        graded_df[['score', 'possible']] = points_df
    except Exception as e:
        print(f"エラー: 点数計算中にエラーが発生しました: {e}。点数は0として扱われます。")
        graded_df['score'] = 0
        graded_df['possible'] = 0

    # 画像なし問題のデータフレームを作成 (点数計算後)
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

    def calculate_category_stats(df):
        # カテゴリごとの統計情報 (正解数, 問題数, 正答率, 獲得点, 満点, 得点率) を計算
        if df.empty:
            return {'correct': 0, 'total': 0, 'accuracy': 0.0, 'score': 0.0, 'possible_score': 0.0, 'score_rate': 0.0}
        # is_correct は bool 型のはず
        correct = int(df['is_correct'].sum())
        total = len(df)
        accuracy = correct / total if total > 0 else 0.0
        score = float(df['score'].sum())
        possible_score = float(df['possible'].sum())
        score_rate = score / possible_score if possible_score > 0 else 0.0
        return {'correct': correct, 'total': total, 'accuracy': accuracy,
                'score': score, 'possible_score': possible_score, 'score_rate': score_rate}

    def calculate_block_stats(df):
         # ブロックごとの正答率と得点率を計算
         if df.empty or 'block' not in df.columns: return {}, {}
         # 'Unknown' ブロックを除外して集計
         df_filtered = df[df['block'] != 'Unknown']
         if df_filtered.empty: return {}, {}
         block_agg = df_filtered.groupby('block').agg(
             correct=('is_correct', 'sum'),
             total=('is_correct', 'size'),
             score=('score', 'sum'),
             possible=('possible', 'sum')
         ).astype(float) # 集計結果を float に変換
         block_agg['accuracy'] = (block_agg['correct'] / block_agg['total']).fillna(0.0)
         block_agg['score_rate'] = (block_agg['score'] / block_agg['possible']).fillna(0.0)
         return block_agg['accuracy'].to_dict(), block_agg['score_rate'].to_dict()

    # 各カテゴリの統計情報を計算
    total_stats = calculate_category_stats(graded_df)
    general_stats = calculate_category_stats(graded_df[graded_df['is_general']])
    required_stats = calculate_category_stats(graded_df[graded_df['is_required']])
    no_image_total_stats = calculate_category_stats(no_image_df)
    no_image_general_stats = calculate_category_stats(no_image_df[no_image_df['is_general']])
    no_image_required_stats = calculate_category_stats(no_image_df[no_image_df['is_required']])
    # ブロック別統計を計算
    block_accuracies, block_score_rates = calculate_block_stats(graded_df)
    no_image_block_accuracies, no_image_block_score_rates = calculate_block_stats(no_image_df)

    # 最終的な統計情報を辞書にまとめる
    stats = {
        'total': {**total_stats, 'df': graded_df, 'block_accuracies': block_accuracies, 'block_score_rates': block_score_rates},
        'general': {**general_stats, 'df': graded_df[graded_df['is_general']]}, # df も含める
        'required': {**required_stats, 'df': graded_df[graded_df['is_required']]}, # df も含める
        'block_accuracies': block_accuracies, # トップレベルにも保持
        'block_score_rates': block_score_rates, # トップレベルにも保持
        'no_image': {
            **no_image_total_stats, 'df': no_image_df,
            'general': {**no_image_general_stats, 'df': no_image_df[no_image_df['is_general']]}, # df も含める
            'required': {**no_image_required_stats, 'df': no_image_df[no_image_df['is_required']]}, # df も含める
            'block_accuracies': no_image_block_accuracies,
            'block_score_rates': no_image_block_score_rates
        },
    }
    return stats