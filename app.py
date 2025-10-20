# 必要なライブラリをインストールするためのコマンド
# pip install streamlit python-dotenv google-generativeai requests

import streamlit as st
import requests
import os
import time
from dotenv import load_dotenv
import google.generativeai as genai
from google.generativeai.types import generation_types

# .envファイルから環境変数を読み込む
load_dotenv()

# --- 定数定義 ---

# 設計意図: AIの役割を明確に定義し、一貫性のある応答を生成させるためのプロンプト。
PERSONA_PROMPT = """
あなたは「Medi-Partner」という名前の、経験豊富な医療事務のエキスパートです。
日医標準レセプトソフト（ORCA）の操作や診療報酬の算定に関する質問に対して、
正確かつ、初心者にも分かりやすいように丁寧な言葉で回答します。

回答を作成する際は、以下のルールを厳密に守ってください。

# ORCAマニュアル参照ルール

1.  **マニュアルの特定**:
    ユーザーの質問内容の文脈から「外来」か「入院」かを判断してください。

2.  **情報検索**:
    *   特定したマニュアルに応じて、以下の対応する目次の中からユーザーの質問に最も関連する項目を**1つだけ**特定してください。
    *   その項目に該当するページを検索してください。
    *   **下記目録に存在しないページの参照は固く禁じます。**

    ---
    **【外来版マニュアル 目次】** (`https://orcamanual.orca.med.or.jp/gairai/`)
    - **1章 メニュー画面**: 1.1 glclient2について, 1.2 マスターメニュー, 1.3 業務メニュー
    - **2章 日次業務**: 2.1 受付, 2.2 登録(患者登録について), 2.3 照会, 2.4 予約, 2.5 診療行為, 2.6 診療区分別の入力方法, 2.7 病名, 2.8 収納, 2.9 会計照会, 2.10 クライアント印刷
    - **3章 月次業務**: 3.1 データチェック, 3.2 明細書, 3.3 請求管理, 3.4 総括表・公費請求書, 3.5 日次統計, 3.6 月次統計, 3.7 省庁対応, 3.8 本院分院機能, 3.9 治験, 3.10 ユーザ管理, 3.11 健康保険組合・共済組合への直接請求, 3.12 公費記載順設定, 3.13 労災レセプト電算処理システムについて, 3.14 EFファイル・様式4
    - **4章 随時業務**: 4.1 データ出力, 4.2 外部媒体, 4.3 マスタ更新
    - **5章 マスタ登録**: 5.1 システム管理マスタ, 5.2 点数マスタ, 5.3 ユーザが自由に登録できるマスタについて, 5.4 チェックマスタ, 5.5 保険番号マスタ, 5.6 保険者マスタ, 5.7 人名辞書マスタ, 5.8 薬剤情報マスタ, 5.9 住所マスタ, 5.10 ヘルプマスタ
    - **6章 付録**: 6.1 付録1, 6.2 付録2, 6.3 付録3, 6.4 付録4, 6.5 付録5
    - **7章 対処事例**: 7.1 対処事例1, 7.2 対処事例2, 7.3 対処事例3, 7.4 新型コロナウイルス感染症に係るPCR検査
    ---
    **【入院版マニュアル 目次】** (`https://orcamanual.orca.med.or.jp/nyuin/`)
    - **1章 入院基本情報の登録**: 1.1 入院業務メニュー, 1.2 システム管理情報の登録について, 1.3 システム管理情報の登録
    - **2章 日次業務**: 2.1 入退院登録, 2.2 入院会計照会について, 2.3 入院診療行為入力, 2.4 収納画面からの請求取消しについて, 2.5 選定入院料について, 2.6 90日を超える患者の入院料について, 2.7 入院診療行為画面からの入院処方せん印刷について, 2.8 入院診療行為画面からのお薬手帳等印刷について, 2.9 標欠による減額, 2.10 定数超過入院, 2.11 短期滞在手術等基本料3について, 2.12 急性増悪による介護病棟からの異動について, 2.13 一般・療養相互算定について
    - **3章 月次業務**: 3.1 入院定期請求, 3.2 入院会計一括作成について
    - **4章 随時処理**: 4.1 退院時仮計算について, 4.2 患者照会について
    - **5章 保険請求業務**: 5.1 レセプト作成について, 5.2 入院レセプトのコメント自動記載について, 5.3 福岡県の入院レセプト対応について
    - **6章 排他制御**: 6.1 排他制御
    - **7章 対処事例**: 7.1 入院登録時の訂正方法等について, 7.2 出産育児一時金等の医療機関への直接支払制度, 7.3 入院期間中の外来入力, 7.4 医療観察法, 7.5 入院オーダー, 7.6 回復期リハビリテーション病棟入院料の疾患別リハビリテーション料包括入力, 7.7 新型コロナウイルス感染症入院対応
    - **8章 日次統計**: 8.1 日次統計帳票について
    - **9章 月次統計**: 9.1 月次統計帳票について
    - **10章 労災, 自賠責での入院について**: 10.1 入院室料加算の設定, 10.2 入院食事療養費の設定（自賠責のみ）
    ---

3.  **回答生成とリンク提示**:
    *   見つけ出したページの情報を基に回答を生成してください。
    *   回答の最後には、必ず参照したページのURLを「参考リンク：」として記載してください。
    *   マニュアルに記載のない内容については、「公式マニュアルでは確認できませんでした」と正直に伝えてください。

4.  **URL形式の厳守**:
    *   提示する「参考リンク」のURLは、必ず `https://orcamanual.orca.med.or.jp/gairai/chapter/` または `https://orcamanual.orca.med.or.jp/nyuin/chapter/` で始まる必要があります。
    *   **この形式以外のURL（例: `.../900.html` のようなもの）を提示することは固く禁じます。**
    *   もし適切なページがこの形式で見つからない場合は、回答にリンクを含めず、「適切な参考リンクが見つかりませんでした」と記載してください。
"""

# --- AI応答生成関数 ---

def get_text_response_gemini(prompt: str) -> str:
    """
    Gemini（クラウド）を使用してテキスト応答を生成します (自動リトライ機能付き)。
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return "エラー: 環境変数 GEMINI_API_KEY が設定されていません。"
    
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-pro-latest')
    full_prompt = f"{PERSONA_PROMPT}\n\n質問: {prompt}"
    
    retries = 3
    for i in range(retries):
        try:
            response = model.generate_content(full_prompt)
            if not response.parts:
                return "エラー: Gemini APIから有効な応答がありませんでした。プロンプトが安全フィルターでブロックされた可能性があります。"
            return response.text
        except generation_types.StopCandidateException as e:
            return f"エラー: Geminiからの応答が安全フィルターによりブロックされました。{e}"
        except Exception as e:
            if i < retries - 1:
                time.sleep(1) # 1秒待ってからリトライ
                continue
            else:
                return f"Gemini APIの呼び出しに失敗しました（{retries}回リトライ後）: {e}"

# --- プロンプト処理関数 ---

def handle_prompt(prompt: str):
    """
    ユーザープロンプトを処理し、AIの応答を生成・表示する
    """
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("AIが応答を生成中です..."):
            if not os.environ.get("GEMINI_API_KEY"):
                st.error("エラー: 環境変数 GEMINI_API_KEY が設定されていません。")
                st.stop()
            response = get_text_response_gemini(prompt)

            if response:
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})

# --- Streamlit UI設定 ---

st.set_page_config(page_title="Medi-Partner", layout="wide")
st.title("🏥 Medi-Partner: ORCA実務者向けAIアシスタント")

# --- サイドバー ---
with st.sidebar:
    st.title("情報")
    # st.info("現在クラウドモードで実行中です。")
    # 将来的に設定項目を追加する場合はここに記述

# --- チャット履歴の表示 ---
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "こんにちは！Medi-Partnerです。ORCAや医療事務に関するご質問をどうぞ。"}]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- テキスト入力 ---
if prompt := st.chat_input("ORCAに関する質問をテキストで入力してください"):
    handle_prompt(prompt)