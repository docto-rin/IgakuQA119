# LLM (gemini-2.5-pro-exp-03-25) による CA-DSR1-DQ32B-JP vs CA-DSR1-DQ32B-JP-SFT 誤答比較考察

分析実行日時: 2025-04-12 16:44:11

## LLM医師国家試験誤答比較・考察レポート

### 1. 全体的な誤答傾向の比較

#### 1.1 両モデル共通の弱点・誤答しやすい問題タイプ

*   **該当問題数:** 60問
*   **共通誤答パターン:** レポートの「両モデル共通誤答のパターン」セクションを見ると、特定の誤答選択肢（例: 正解eに対し両モデルともcを選択: 5件）が複数回出現しています。これは、これらの問題が特定の医学的知識の曖昧さ、一般的な誤解、あるいは問題文や選択肢の解釈の難しさ（ひっかけ問題など）を含んでいる可能性を示唆します。
*   **問題タイプ:**
    *   **臨床推論:** 複数の情報（病歴、身体所見、検査結果）を統合し、診断や治療方針を決定する問題で共通の誤答が見られます（例: 119A22, 119A47, 119A55, 119C63, 119D16）。これは、複雑な情報の重み付けや鑑別診断のプロセスに共通の課題がある可能性を示唆します。
    *   **知識問題（特に細かい規定や最新ガイドライン）:** 法律（119B10 医療保険制度, 119C14 就業制限）、統計（119F17 医療費）、特定の疾患の診断基準や治療選択（119A1 自己免疫性膵炎, 119C10 尿所見, 119D4 食中毒予防）などで共通の誤答が見られます。これは、学習データに含まれる情報の網羅性や最新性に限界があるか、あるいは特定の知識領域の学習が不十分である可能性を示唆します。
    *   **画像問題:** 画像が関連する問題（119A15, 119A22, 119A28, 119A60, 119B9, 119C15, 119C22, 119D10, 119D16, 119D19, 119D54, 119D63, 119D68, 119E12, 119E23）も共通誤答に含まれています。画像解釈能力そのものの問題か、画像情報とテキスト情報を統合する能力の問題かは断定できませんが、画像が関わる問題は両モデルにとって依然として難しい課題である可能性があります。
    *   **計算問題:** 119A75（水分出納）、119C74（A-aDO2）、119C75（肥満度）、119F75（Na欠乏量）で共通の誤答が見られます。特に119A75と119C74では、両モデルとも計算プロセス自体に誤りが見られます（後述）。計算手順の正確な実行に課題があるようです。

#### 1.2 モデル間の誤答傾向の違い

*   **CA-DSR1-DQ32B-JP (ベースモデル) のみの誤答 (58問):**
    *   **弱点:**
        *   **臨床推論・治療選択:** ベースモデルは、SFTモデルが正答した臨床推論問題や治療選択問題で誤答するケースが多く見られます（例: 119A44 心不全治療, 119A52 腎梗塞疑い, 119B16 急性中耳炎, 119C51 神経性やせ症, 119D52 前立腺癌術前説明, 119F69 心筋梗塞初期対応）。これは、学習データだけでは獲得しきれない、より実践的な臨床判断能力やガイドライン知識の適用においてSFTモデルに劣る可能性を示唆します。
        *   **公衆衛生・法規:** 119F8（健診規定）、119F18（保健所業務）、119C9（診療録保存義務）など、制度や法律に関する知識問題で誤答が見られます。
        *   **画像問題:** SFTモデルが正答した画像関連問題（例: 119A21 腎生検, 119D2 心カテ波形, 119D18 皮膚生検）でも誤答しており、画像情報の解釈・統合能力に差がある可能性があります。
*   **CA-DSR1-DQ32B-JP-SFT (SFTモデル) のみの誤答 (46問):**
    *   **弱点:**
        *   **知識の偏り・過学習の可能性:** SFTモデルは、ベースモデルが正答した比較的単純な知識問題や定義問題で誤答するケースが見られます（例: 119A4 好発部位, 119B1 俗称, 119B14 在宅医療, 119B15 内視鏡体位, 119F3 新生児, 119F7 国際提言, 119F21 人口指数）。これは、SFTの過程で特定のデータに過剰適合したか、あるいは学習データに偏りがあった可能性を示唆します。
        *   **推論の硬直性？:** いくつかの臨床問題（例: 119A34 肺真菌症, 119C44 不正出血, 119D12 片頭痛）で、ベースモデルよりも単純な思考プロセスや、特定の情報に引きずられたような誤答が見られる場合があります。SFTによって柔軟性が失われた可能性も考えられます。
        *   **複数選択問題での選択ミス:** 119A12 (糖尿病診断基準、ceが正解なのにcのみ選択)、119D12 (片頭痛、cdが正解なのにdのみ選択)、119F35 (感染性廃棄物、bdeが正解なのにabeを選択)、119F42 (食事指導、cが正解なのにcdを選択) など、正解が複数ある問題で、正しい選択肢の一部しか選べていない、あるいは誤った組み合わせを選んでいるケースが見られます。これは指示の理解や複数選択肢の評価能力に課題がある可能性を示唆します。

#### 1.3 片方のみの誤答から推測される強み・弱み

*   **CA-DSR1-DQ32B-JP (ベースモデル):**
    *   **弱み:** 複雑な臨床推論、最新ガイドラインの適用、公衆衛生・法規関連の知識、画像情報の統合。
    *   **強み (相対的):** SFTモデルが間違えた基本的な知識問題や定義問題においては、より汎用的な知識を持っている可能性があります。
*   **CA-DSR1-DQ32B-JP-SFT (SFTモデル):**
    *   **強み:** ベースモデルと比較して、臨床推論能力や治療選択の精度が向上している傾向が見られます。SFTによって特定のタスク（この場合は医師国家試験問題解答）への適応が進んだと考えられます。
    *   **弱み:** 基本的な知識問題での誤答や、複数選択問題での選択ミスが見られ、知識の網羅性や指示理解、あるいは過学習による汎化性能の低下が示唆されます。推論プロセスがやや硬直化している可能性も考えられます。

### 2. 具体的な誤答パターンの分析

#### 2.1 両モデル共通誤答パターンの原因分析

*   **`Correct:e | CA-DSR1-DQ32B-JP:c | CA-DSR1-DQ32B-JP-SFT:c` (5件):**
    *   **119C15 (M蛋白分画):** 両モデルともM蛋白がγ分画(e)ではなくβ分画(c)に出現すると誤解している可能性があります。これは一般的な知識の誤りか、提示された模式図の解釈ミスかもしれません。CoTを見ると両モデルともγ分画を意識していますが、最終的にcを選んでいます。知識と最終判断の間に齟齬がある可能性があります。
    *   **119E36 (灯油誤飲):** 両モデルとも牛乳投与(c)を選択していますが、正解は経過観察(e)です。灯油のような揮発性物質の誤飲では、嘔吐誘発や胃洗浄は禁忌であり、牛乳投与も推奨されません。これは誤った応急処置の知識（牛乳による中和・希釈効果への過信）が原因と考えられます。CoTでも牛乳投与の有効性を前提としています。
    *   **119F44 (SLE患者管理):** 両モデルとも母親による管理継続(c)を選択していますが、正解は本人の病識確認(e)です。15歳という年齢と将来の内科移行を考慮すると、本人の主体性を促すアプローチが重要ですが、両モデルとも過去の怠薬歴を重視しすぎ、安全策としての現状維持を選択した可能性があります。CoTでもその傾向が見られます。
    *   **119F60 (甲状腺機能低下症の所見):** 両モデルとも非圧痕性浮腫(d)とアキレス腱反射遅延(e)の組み合わせを選んでいますが、正解はdeです。SFTモデルはdeを選択しており、ベースモデルの誤答(be)を修正できていません。甲状腺圧痛(b)は甲状腺炎を示唆しますが、この患者の主病態（化学放射線療法後の機能低下）とは合致しにくいです。両モデルとも甲状腺機能低下症の身体所見に関する知識が不完全、あるいは情報の重み付けに誤りがある可能性があります。
    *   **119F62 (甲状腺機能低下症の治療):** 両モデルとも甲状腺ホルモン補充(c)を選択していますが、正解は低カリウム血症に対する塩化カリウム投与(e)です。検査値(K 2.8mEq/L)を見落としたか、甲状腺機能低下症の治療を優先しすぎた可能性があります。CoTを見ると、両モデルとも甲状腺機能低下症の診断と治療に焦点が当たっており、電解質異常への言及が不足しています。
*   **計算問題での共通誤答:**
    *   **119A75 (水分出納):** ベースモデルは代謝水を摂取量に、SFTモデルは排出量に誤って計上しています。計算式の理解・適用に誤りがあります。
    *   **119C74 (A-aDO2):** ベースモデルは肺胞気酸素分圧(PAO2)の計算式 `PAO2 = (大気圧 - 飽和水蒸気圧) * FiO2 - (PaCO2 / 呼吸商)` を正しく適用できていません（`PBO2 - PaO2` という謎の計算）。SFTモデルはPAO2の計算自体を省略し、選択肢a（約150mmHg、これはPAO2の近似値）を回答しています。両モデルともA-aDO2の計算方法を正確に理解・実行できていません。
    *   **119C75 (肥満度):** ベースモデルは `(実測体重 - 標準体重) / 標準体重 * 100` を計算しようとしていますが、最終的な回答が `90.9` となっており計算ミスか解釈ミスがあります。SFTモデルは `(実測体重 / 標準体重) * 100` を計算しており、肥満度の定義（超過分を標準体重で割る）を誤解しています。
    *   **119F75 (Na欠乏量):** ベースモデルの回答 `-12.8` や `0.52` は意味不明です。SFTモデルは `(目標Na - 現在Na) * 細胞外液量` を計算していますが、細胞外液量の計算で体重の20% (2L) を使用しています。Na欠乏量の計算では、細胞外液だけでなく総体液量（体重 * 0.6程度、小児では割合が異なる）で補正係数（例: 0.6）を掛けて計算するのが一般的です。`不足Na量(mEq) = (目標Na - 実測Na) * 体重(kg) * 0.6`。SFTモデルは計算式の適用が不正確です。

#### 2.2 各モデルの誤答原因（説明・CoTからの推測）

*   **CA-DSR1-DQ32B-JP (ベースモデル):**
    *   **知識不足/不正確:** 法規(119F8, 119F18, 119C9)、統計(119F17)、特定の疾患知識（119A18 上咽頭癌, 119A54 副甲状腺機能亢進症）などで誤答が見られ、知識の網羅性や正確性に課題があるようです。
    *   **推論の誤り/単純化:** 臨床推論問題で、CoTを見ると、重要な情報を見落としたり、安易な結論に飛びついたりする傾向が見られます（例: 119A44 心不全治療でリハビリのみ選択、119A50 多発性硬化症疑いでRomberg徴候のみ重視）。複雑な情報を統合し、鑑別診断を進めるプロセスが弱い可能性があります。
    *   **計算エラー/式の間違い:** 計算問題(119A75, 119C74, 119C75, 119F75)で顕著な誤りが見られ、計算式の理解や実行能力に問題があります。
*   **CA-DSR1-DQ32B-JP-SFT (SFTモデル):**
    *   **知識の偏り/過学習:** 基本的な知識問題（119A4, 119B1, 119B14, 119B15, 119F3, 119F7, 119F21）での誤答は、SFTデータへの過学習や知識の抜け漏れを示唆します。CoTを見ると、時に自信過剰に誤った知識を前提として推論を進めている場合があります（例: 119B15 内視鏡体位で仰臥位が標準と断定）。
    *   **指示理解/複数選択の評価:** 複数選択問題（119A12, 119D12, 119F35, 119F42）で不完全な回答をしている点は、問題形式への対応能力や、複数の正解選択肢を適切に評価・選択する能力に課題があることを示します。
    *   **推論の硬直性（可能性）:** 一部の問題で、特定の情報に強く引きずられたり、代替案を十分に検討しなかったりするようなCoTが見られます（例: 119C44 子宮外妊娠疑いで超音波を最優先）。SFTにより特定のパターンへの反応が強化された一方で、柔軟な思考が抑制された可能性も否定できません。
    *   **計算エラー/式の間違い:** ベースモデル同様、計算問題（119A75, 119C74, 119C75, 119F75）で誤りが見られます。SFT後も計算能力の課題は残存しているようです。

### 3. 総括

#### 3.1 全体的な性能・特性

*   **SFTモデルの優位性:** SFTモデル (CA-DSR1-DQ32B-JP-SFT) は、ベースモデル (CA-DSR1-DQ32B-JP) よりも誤答数が少ない（共通誤答60問を除くと、ベース58問 vs SFT 46問）ことから、全体的な正答率は向上していると考えられます。特に、臨床推論や治療選択といった、より実践的な問題解決能力において改善が見られる傾向があります。これはSFTが特定のタスク（医師国家試験問題解答）への適応を促した結果と考えられます。
*   **SFTモデルの課題:** 一方で、SFTモデルはベースモデルが正答した基本的な知識問題で誤答したり、複数選択問題で不完全な回答をしたりするケースが見られます。これは、SFTによる過学習、知識の偏り、あるいは指示理解能力の課題を示唆しています。また、両モデルに共通して計算問題や特定の知識領域（法規、統計など）、複雑な臨床推論、画像関連問題に弱点が見られました。SFTを経てもこれらの根本的な課題が完全には解消されていない可能性があります。
*   **思考プロセス:** CoTが利用可能な場合、両モデルとも一定の論理的な思考プロセスを示そうとしていますが、知識の誤り、情報の見落とし、不適切な重み付け、計算ミスなどにより誤答に至るケースが多く見られます。SFTモデルのCoTは、時にベースモデルよりも構造化されているように見えることもありますが、それが必ずしも正答に結びつくわけではなく、誤った前提に基づいて詳細な推論を展開してしまうこともあります。

#### 3.2 今後の改善方向性

*   **CA-DSR1-DQ32B-JP (ベースモデル):**
    *   **臨床推論能力強化:** 臨床ケースを用いた学習データの拡充、複雑な症例に対する推論プロセス（鑑別診断、治療計画立案）を学習させるファインチューニング。
    *   **知識の網羅性・最新性向上:** 最新のガイドライン、法規、統計情報を含む学習データの定期的な更新と拡充。特に公衆衛生、法規、特定の専門分野（例: 循環器、内分泌）の知識強化。
    *   **計算能力向上:** 計算問題に特化した学習データセットでの訓練、計算ステップの正確性を検証する仕組みの導入。
    *   **画像理解能力:** （可能であれば）VLM（Vision-Language Model）としての能力向上、画像所見とテキスト情報を正確に統合する訓練。
*   **CA-DSR1-DQ32B-JP-SFT (SFTモデル):**
    *   **過学習の抑制と汎化性能向上:** SFTデータの多様化、正則化手法の導入、ベースモデルの知識を維持するような学習手法（例:知識蒸留の一部応用）の検討。
    *   **知識の補強:** ベースモデルと同様に、弱点分野（特にSFTで新たに見られた知識の抜け漏れ）に関する学習データの追加。
    *   **指示理解・複数選択問題への対応:** 様々な形式の問題（特に複数選択）を含むデータでのファインチューニング、プロンプトエンジニアリングによる指示理解の強化。
    *   **推論の柔軟性:** 多様な視点からの推論や代替案の検討を促すようなCoTプロンプトや学習手法の導入。
    *   **計算能力向上:** ベースモデルと同様。
*   **両モデル共通:**
    *   **思考プロセスの精度向上:** より正確で段階的な思考プロセス（CoT）を生成・評価する仕組みの導入。自己修正能力や批判的思考を促すプロンプトの活用。
    *   **誤答分析の活用:** 今回のような誤答分析結果をフィードバックし、モデルの弱点を特定し、的を絞った改善（追加学習やファインチューニング）を行うサイクルを構築する。