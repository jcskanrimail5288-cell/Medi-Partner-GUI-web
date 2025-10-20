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
回答は常に日本の医療制度と法律に基づき、具体的で実践的な内容を心がけてください。
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
    model = genai.GenerativeModel('gemini-pro')
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
