
# NMLE-RTA (National Medical License Examination - Real Time Analysis)

医師国家試験の問題をOCR処理し、複数のAIモデル（LLM）を使用して解答を生成するシステムです。  
本プロジェクトは、試験問題のOCRから問題整形、AI解答生成、結果の集約・確認までの一連の流れを実現します。

> **注意**  
> もともとは `answer_questions.py` を使用していましたが、出力の不安定さが課題となったため、  
> `main.py`、`llm_solver.py`、`process_llm_output.py` を追加し、解答生成の機能をこれらに置き換えました。

## 更新情報
- 2025-02-08-14:05: 119Aの解答結果を追加(gpt-4oのみ)
- 2025-02-08-15:10: deepseek, claudeの出力不安定のため別コード追記
- 2025-02-08-16:20: 119Bの解答結果を追加(gpt-4oのみ)
- 2025-02-08-18:50: 119A, 119Bの解答結果を追加(全てのモデル)
- 2025-02-08-21:15: 119Cの解答結果を追加(全てのモデル)
- 2025-02-09-13:40: 119Dの解答結果を追加(gpt-4oのみ)
- 2025-02-09-14:30: 119Dの解答結果を追加(全てのモデル)


## ディレクトリ構成

```
nmle-rta/
├── data/                    # 入力PDFファイルを配置
├── output/                  # OCR処理の出力ファイル
├── answer/                  # AIモデルによる解答結果
├── images/                  # 問題に関連する画像ファイル
├── fonts/                   # フォントファイル
│   └── BIZUDPMincho-Regular.ttf
├── scripts/                 # 補助スクリプト
├── .env                     # 環境変数設定ファイル
└── .env.example            # 環境変数のテンプレート
```

## 必要な環境

- Python 3.10 以上
- パッケージマネージャー `uv`  
- 各種APIキー（.env ファイルに設定してください）  
  - Azure Document Intelligence API Key/Endpoint
  - OpenAI API Key (gpt-4o, o1, o3-mini用)
  - Anthropic API Key (Claude用)
  - DeepSeek API Key
  - Gemini API Key

---

## セットアップ手順

1. **パッケージのインストール**  
   uv を用いてパッケージを同期します：
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   uv sync
   ```

2. **環境変数の設定**  
   `.env.example` をコピーして `.env` を作成し、必要な API キーを設定してください：
   ```env
   AZURE_ENDPOINT=your_azure_endpoint
   AZURE_API_KEY=your_azure_api_key
   OPENAI_API_KEY=your_openai_api_key
   DEEPSEEK_API_KEY=your_deepseek_api_key
   ANTHROPIC_API_KEY=your_anthropic_api_key
   GEMINI_API_KEY=your_gemini_api_key
   ```

3. **フォントの配置**  
   `fonts` ディレクトリに BIZUDPMincho-Regular.ttf を配置してください。

## 使用方法

### 1. PDFからテキスト抽出

```bash
python pdf2txt.py
```

入力PDFは`data/`ディレクトリに配置します（例：`data/118B.pdf`）。
処理結果は`output/`ディレクトリに保存されます：
- `{basename}_output.pdf`: OCRの結果を可視化したPDF
- `{basename}_ocr_text.txt`: 抽出されたテキスト
- `{basename}_ocr_results.json`: OCR詳細結果

出力例（`118B_ocr_text.txt`）:
```text
=== Page 1 ===
118B1以下の症例について問1～問3に答えよ。
54歳の男性。発熱と咳を主訴に来院した。
3日前から38℃台の発熱と咳が出現し、市販薬を内服したが改善しないため来院した。
身体所見：体温 38.5℃。呼吸数 24/分。SpO₂ 94%（室内気）。
胸部聴診で右下肺野に湿性ラ音を聴取する。
...
```

### 2. テキストのクリーニング

```bash
python cleaning_ocr_txt.py output/118B_ocr_text.txt output/118B_cleaned.txt
```
OCRテキストを整形し、問題形式に変換します。
この段階で人手による確認と修正を行うことを推奨します。

出力例（`118B_cleaned.txt`）:
```text
=== 118B1 ===
【問題】
54歳の男性。発熱と咳を主訴に来院した。
3日前から38℃台の発熱と咳が出現し、市販薬を内服したが改善しないため来院した。
身体所見：体温 38.5℃。呼吸数 24/分。SpO₂ 94%（室内気）。
胸部聴診で右下肺野に湿性ラ音を聴取する。

【選択肢】
a. 市中肺炎
b. 気管支喘息
c. 急性気管支炎
d. 肺結核
e. 肺癌
```

### 3. テキストからJSONへの変換

```bash
python txt2json.py output/118B_cleaned.txt output/118B_questions.json
```
整形されたテキストを構造化JSONデータに変換します。

出力例（`118B_questions.json`）:
```json
{
  "questions": [
    {
      "id": "118B1",
      "case_text": "54歳の男性。発熱と咳を主訴に来院した。\n3日前から38℃台の発熱と咳が出現し...",
      "sub_questions": [
        {
          "number": 1,
          "text": "最も考えられる診断は何か。",
          "options": [
            "市中肺炎",
            "気管支喘息",
            "急性気管支炎",
            "肺結核",
            "肺癌"
          ]
        }
      ],
      "has_image": false
    }
  ]
}
```

### 4. AIモデルによる問題解答

従来の `answer_questions.py` の代わりに、**新規追加** された `main.py` をエントリーポイントとして使用します。  
このスクリプトは内部で `llm_solver.py` と `process_llm_output.py` を利用し、複数の LLM（o1、gpt-4o、o3-mini、gemini、deepseek、claude など）で問題を解答します。

例：
```bash
uv run main.py --input question/119A_json.json --models claude o1 gpt-4o
```
- `--input` で JSON 形式の問題ファイルを指定します。  
- `--models` で使用するモデルを必要に応じて指定（複数指定可能、デフォルトは全モデル）。

実行後、解答結果は `answer/json/` 内にタイムスタンプ付きの JSON ファイルとして保存され、詳細な出力（生の出力や各モデル別のテキストファイル）は `answer/raw/` に保存されます。

### 5. 結果のCSV変換

解答結果の JSON ファイルを CSV に変換するには、以下のどちらかのスクリプトを使用してください。

- **json_to_csv_converter.py**  
  例：
  ```bash
  uv run python json_to_csv_converter.py answer/json/results_YYYYMMDD_HHMMSS_final.json
  ```

- **convert_json_to_csv.py**  
  （利用方法はスクリプト内ヘルプ参照）

## 画像問題の処理

画像を含む問題の場合、以下の手順で画像を準備します：

1. 画像ファイルを`images/`ディレクトリに配置
2. ファイル名の形式：
   - 単一画像：`{問題番号}.jpg`または`{問題番号}.png`
   - 複数画像：`{問題番号}-1.jpg`、`{問題番号}-2.jpg`など

例：
- `images/118B1.jpg`
- `images/118B2-1.png`、`images/118B2-2.png`


## 追加情報

- **出力ファイルについて**  
  - JSON 形式の結果は `answer/json/` に保存されます。  
  - 各モデルごとの生の出力は `answer/raw/` に保存され、結果のトラブルシュートにご利用いただけます。


## 注意事項

- OCR 結果は必ず人手で確認・修正してください。  
- API キーは `.env` ファイルで管理し、リポジトリにコミットしないよう注意してください。  
- 新規追加された `main.py`、`llm_solver.py`、`process_llm_output.py` の挙動に合わせ、出力内容やエラー出力を適宜確認してください。