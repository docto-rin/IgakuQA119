# NMLE-RTA (National Medical License Examination - Real Time Analysis)

医師国家試験の問題をOCR処理し、AIモデルを使用して解答を生成するシステム  
2025年2月8日、9日に行われる第119回医師国家試験の問題を解くことを前提にしています。

2025-01-17現在、以下のモデルを使用しています。
- o1-2024-12-17: 問題解答(Vision含む)
- gpt-4o-2024-08-06: 問題解答(Vision含む)
- claude-3-5-sonnet-20241022: 問題解答(Vision含む)
- gemini-2.0-flash-exp: 問題解答(Vision含む)
- DeepSeek-R1: 問題解答(Visionなし)


## ディレクトリ構造

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

## 必要なもの

- Python 3.10以上
- package manager: `uv`
- Azure Document Intelligence APIのアクセス情報
- OpenAI API Key
- Anthropic API Key
- DeepSeek API Key
- Gemini API Key

## セットアップ

1. uvのインストールおよび必要パッケージのインストール:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
```

2. 環境変数の設定:
`.env.example`を`.env`にコピーし、必要なAPI keyを設定します：

```env
AZURE_ENDPOINT=your_azure_endpoint
AZURE_API_KEY=your_azure_api_key
OPENAI_API_KEY=your_openai_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key
DEEPSEEK_API_KEY=your_deepseek_api_key
GEMINI_API_KEY=your_gemini_api_key
```

3. フォントのインストール:
`fonts`ディレクトリにBIZ UDPMinchoフォントを配置します。

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

### 4. 問題の解答

```bash
python solve_questions.py output/118B_questions.json
```
複数のAIモデルを使用して問題を解答します。
結果は`answer/`ディレクトリに保存されます。

#### コマンドラインオプション

- `--models`: 使用するモデルを指定（複数指定可）
  - 選択可能なモデル: `o1`, `gpt-4o`, `gemini`, `deepseek`, `claude`
  - デフォルト: すべてのモデルを使用
  - 例: `--models claude o1`（claudeとo1のみ使用）

- `--explanation`: 解答の説明を含める（デフォルト: 説明なし）
  - フラグを付けると各解答に説明が追加されます

使用例：
```bash
# すべてのモデルを使用（説明なし）
python solve_questions.py output/118B_questions.json

# 特定のモデルのみを使用（説明なし）
python solve_questions.py output/118B_questions.json --models claude o1

# 特定のモデルを使用し、説明を含める
python solve_questions.py output/118B_questions.json --models claude o1 --explanation
```

出力例（`answer/exam_results_20250117_120000_final.json`）:
```json
{
  "question_id": "118B1",
  "answers": {
    "claude": {
      "answer": "a",
      "confidence": 0.92,
      "explanation": "発熱、咳、SpO₂低下、右下肺野の湿性ラ音という所見から市中肺炎が最も考えられる"
    }
  }
}
```

## 画像問題の処理

画像を含む問題の場合、以下の手順で画像を準備します：

1. 画像ファイルを`images/`ディレクトリに配置
2. ファイル名の形式：
   - 単一画像：`{問題番号}.jpg`または`{問題番号}.png`
   - 複数画像：`{問題番号}-1.jpg`、`{問題番号}-2.jpg`など

例：
- `images/118B1.jpg`
- `images/118B2-1.png`、`images/118B2-2.png`

## 使用しているモデル

- Azure Document Intelligence: OCRテキスト抽出
- o1-2024-12-17: 問題解答(Vision含む)
- gpt-4o-2024-08-06: 問題解答(Vision含む)
- claude-3-5-sonnet-20241022: 問題解答(Vision含む)
- gemini-2.0-flash-exp: 問題解答(Vision含む)
- DeepSeek-V3: 問題解答(Visionなし)

`2025-01-17`現在、それぞれの最新モデルを使用しています。

以下は、各フォルダの役割と全体の処理フローをまとめた例です。既存のREADME.mdに以下のセクションを追加することで、システム全体の流れと各フォルダの目的が分かりやすくなります。


## フォルダ構造とワークフロー

本システムでは、以下のフォルダを用いて医師国家試験の問題処理から解答生成、解答確認までの一連の作業を管理しています。

- **pdf**  
  医師国家試験のPDFファイル（例：`119A.pdf`, `119B.pdf`）を配置するフォルダです。ここに配置されたPDFがOCR処理の入力となります。

- **output**  
  `pdf2txt.py` を実行して、`pdf` フォルダ内のPDFからAzure Document Intelligence によるOCR処理を行い、抽出されたテキストや詳細なOCR結果（例：`119A_ocr_text.txt`, `{basename}_ocr_results.json` など）を保存するフォルダです。

- **cleaned**  
  `output` 内のOCRテキスト（例：`119A_ocr_text.txt`）を `cleaning_ocr_txt.py` により自動整形した後、手作業でさらに修正・確認を行い、クリーンな状態にしたテキストファイル（例：`119A_cleaned.txt`）を保存するフォルダです。整形後のファイルは問題内容が正確に反映されるように確認してください。

- **question**  
  クリーンなテキストファイルを `txt2json.py` により解析し、構造化されたJSON形式の問題データ（例：`119A_json.json`）に変換して保存するフォルダです。JSONデータは各問題の問題文、選択肢、画像情報などを含みます。

- **answer**  
  `answer_questions.py` を実行して、`question` フォルダ内のJSON形式の問題データをもとに、各AIモデル（o1、gpt-4o、gemini、deepseek、claude など）で解答を生成します。生成された解答結果（例：`exam_results_YYYYMMDD_HHMMSS_intermediate.json` や最終結果ファイル）がこのフォルダに保存されます。

- **images**  
  問題文に関連する画像がある場合、対応する画像ファイルを配置するフォルダです。画像ファイルは、単一画像の場合は `{問題番号}.jpg/png`、複数画像の場合は `{問題番号}-1.jpg/png` などの命名規則で保存してください。

- **nmle-checker**  
  Next.js を用いた解答確認用のWebアプリケーションプロジェクトです。各ブロックごとに問題と解答（AIモデルによるものや自己採点機能など）をブラウザ上で確認することができます。

- **fonts**  
  PDF生成時に使用するフォントファイル（例：`BIZUDPMincho-Regular.ttf`）を格納しています。

- **scripts**  
  補助スクリプト（例：`structured_and_solve.py` など）が格納されており、場合によっては別の処理フローや追加の処理を実行するために利用します。

### 大まかな処理の流れ

1. **PDFファイルの配置**  
   - `pdf` フォルダに、医師国家試験の問題が含まれるPDFファイル（例：`119A.pdf`, `119B.pdf`）を保存します。

2. **OCRによるテキスト抽出**  
   - `pdf2txt.py` を実行し、`pdf` 内のPDFファイルからAzure Document Intelligenceを使用してOCR処理を行います。処理結果は `output` フォルダに、抽出テキスト（例：`119A_ocr_text.txt`）や詳細なOCRデータとして保存されます。

3. **テキストの整形・修正**  
   - `output` にあるOCRテキストファイルを `cleaning_ocr_txt.py` で自動整形し、その後、手作業で誤認識やレイアウトの崩れを修正して、より正確なテキスト（例：`119A_cleaned.txt`）を `cleaned` フォルダに保存します。

4. **JSON形式への変換**  
   - クリーンなテキストファイルを `txt2json.py` により解析し、構造化された問題データ（例：`119A_json.json`）を生成して `question` フォルダに保存します。

5. **AIによる解答生成**  
   - `answer_questions.py` を実行して、`question` フォルダ内のJSONファイルから各AIモデルを用いて解答を生成します。解答結果は `answer` フォルダに、モデルごとの結果ファイルとして保存されます。

6. **画像の利用（必要な場合）**  
   - 問題に関連する画像がある場合は、`images` フォルダに所定の命名規則に従って配置します。これにより、AIモデルは画像も考慮して解答を生成できます。

このように、各フォルダを役割ごとに分割することで、PDFの入力からOCR処理、テキストの整形、問題の構造化、AIによる解答生成、最終的な解答確認までの一連の作業が体系的に管理されています。


## 注意事項

- OCR結果は必ず人手で確認し、必要に応じて修正してください
- 画像問題は正確な画像の準備が重要です
- APIキーは必ず`.env`ファイルで管理し、リポジトリにコミットしないでください
