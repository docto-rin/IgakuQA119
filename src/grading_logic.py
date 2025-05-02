import pandas as pd

def grade_answers(model_data, correct_df):
    """
    モデルの解答を採点し、スキップされた問題番号も返す。
    戻り値: (results_df, accuracy, model_name_in_json, skipped_questions)
    """
    results = []
    correct_count = 0
    total_questions = 0
    model_name_in_json = None
    skipped_questions = []

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
             # confidence がなくても処理は続行できる (警告のみ)
             if 'answer' in missing_keys:
                 print(f"警告: answerデータに必須キー 'answer' がありません。スキップします: {question_number}")
                 if question_number != 'N/A':
                     skipped_questions.append(question_number)
                 continue
             else:
                 # confidence がない場合などの警告
                 print(f"警告: answerデータに必要なキーがありません ({missing_keys})。処理は続行しますが、一部情報が欠落する可能性があります: {question_number}")


        model_answer = answer_data.get('answer')
        model_name = answer_data.get('model', 'UnknownModel')
        # confidence がなくても None として取得
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
                # 複数選択肢の解答形式 (例: "a, b", "[a,b]") を統一的に処理
                model_answer_str = "".join(sorted(model_answer_str.replace('[', '').replace(']', '').replace(',', '').replace(' ', '')))
                correct_answer_str = "".join(sorted(correct_answer_str.replace('[', '').replace(']', '').replace(',', '').replace(' ', '')))


                if not model_answer_str:
                    is_correct = False
                # 119E28 は A or C で正解とする特別ルール
                elif question_number == '119E28':
                    is_correct = model_answer_str in ['a', 'c']
                else:
                    is_correct = model_answer_str == correct_answer_str

            except Exception as e:
                print(f"エラー: 問題 {question_number} の解答比較中にエラーが発生しました。 Model: '{model_answer}', Correct: '{correct_answer}'. Error: {e}")

            if is_correct:
                correct_count += 1
        else:
             # 正解データがない場合は採点対象外 (total_questions に加算しない)
             pass

        results.append({
            'question_number': question_number,
            'model': model_name,
            'model_answer': model_answer,
            'correct_answer': correct_answer,
            'is_correct': is_correct if correct_answer is not None else None, # 正解がない場合は None
            'confidence': confidence,
            'has_image': has_image
        })

    accuracy = correct_count / total_questions if total_questions > 0 else 0.0
    return pd.DataFrame(results), accuracy, model_name_in_json, sorted(list(set(skipped_questions)))