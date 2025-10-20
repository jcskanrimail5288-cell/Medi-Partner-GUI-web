# 必要なライブラリをインストールするためのコマンド
# pip install streamlit python-dotenv google-generativeai requests

import streamlit as st
import os
import re
from dotenv import load_dotenv
import google.generativeai as genai

# .envファイルから環境変数を読み込む
load_dotenv()

# --- 定数定義 ---

# AIへの指示（システムプロンプト）
SYSTEM_INSTRUCTION = """
あなたは「Medi-Partner」という、医療事務に詳しいAIアシスタントです。

ユーザーの質問と、関連する可能性のあるマニュアル項目が提供されます。
以下のルールを厳密に守って回答してください。

1. 提供された「関連マニュアル項目」リストの中から、ユーザーの質問に最も関連性の高いものを**1つだけ**選んでください。
2. 選んだ項目の**番号部分**（例: `2.2`）を特定します。
3. 特定した番号と、指定された「マニュアル種別」（`gairai` または `nyuin`）を使って、以下のルールでURLを生成してください。
    *   **URL形式**: `https://orcamanual.orca.med.or.jp/[マニュアル種別]/chapter/[変換後の章番号]/`
    *   **変換ルール**: 章番号に含まれる `.` を `-` に置き換えます。（例: `2.2` は `2-2` になります）
4. 生成したURLの情報を基に、ユーザーの質問に回答してください。
5. 回答の最後には、**必ず生成したURLを「参考リンク：」として記載してください。**
6. もしリスト内に適切な項目が見つからない場合は、「ご質問に関連するマニュアル項目が見つかりませんでした。」とだけ回答してください。
"""

# ORCAマニュアルの目次データ
GAIRAI_TOC = {
    "1.1": "glclient2について", "1.2": "マスターメニュー", "1.3": "業務メニュー",
    "2.1": "受付", "2.2": "登録(患者登録について)", "2.3": "照会", "2.4": "予約", "2.5": "診療行為", "2.6": "診療区分別の入力方法", "2.7": "病名", "2.8": "収納", "2.9": "会計照会", "2.10": "クライアント印刷",
    "3.1": "データチェック", "3.2": "明細書", "3.3": "請求管理", "3.4": "総括表・公費請求書", "3.5": "日次統計", "3.6": "月次統計", "3.7": "省庁対応", "3.8": "本院分院機能", "3.9": "治験", "3.10": "ユーザ管理", "3.11": "健康保険組合・共済組合への直接請求", "3.12": "公費記載順設定", "3.13": "労災レセプト電算処理システムについて", "3.14": "EFファイル・様式4",
    "4.1": "データ出力", "4.2": "外部媒体", "4.3": "マスタ更新",
    "5.1": "システム管理マスタ", "5.2": "点数マスタ", "5.3": "ユーザが自由に登録できるマスタについて", "5.4": "チェックマスタ", "5.5": "保険番号マスタ", "5.6": "保険者マスタ", "5.7": "人名辞書マスタ", "5.8": "薬剤情報マスタ", "5.9": "住所マスタ", "5.10": "ヘルプマスタ",
    "6.1": "付録1", "6.2": "付録2", "6.3": "付録3", "6.4": "付録4", "6.5": "付録5",
    "7.1": "対処事例1", "7.2": "対処事例2", "7.3": "対処事例3", "7.4": "新型コロナウイルス感染症に係るPCR検査"
}

NYUIN_TOC = {
    "1.1": "入院業務メニュー", "1.2": "システム管理情報の登録について", "1.3": "システム管理情報の登録",
    "2.1": "入退院登録", "2.2": "入院会計照会について", "2.3": "入院診療行為入力", "2.4": "収納画面からの請求取消しについて", "2.5": "選定入院料について", "2.6": "90日を超える患者の入院料について", "2.7": "入院診療行為画面からの入院処方せん印刷について", "2.8": "入院診療行為画面からのお薬手帳等印刷について", "2.9": "標欠による減額", "2.10": "定数超過入院", "2.11": "短期滞在手術等基本料3について", "2.12": "急性増悪による介護病棟からの異動について", "2.13": "一般・療養相互算定について",
    "3.1": "入院定期請求", "3.2": "入院会計一括作成について",
    "4.1": "退院時仮計算について", "4.2": "患者照会について",
    "5.1": "レセプト作成について", "5.2": "入院レセプトのコメント自動記載について", "5.3": "福岡県の入院レセプト対応について",
    "6.1": "排他制御",
    "7.1": "入院登録時の訂正方法等について", "7.2": "出産育児一時金等の医療機関への直接支払制度", "7.3": "入院期間中の外来入力", "7.4": "医療観察法", "7.5": "入院オーダー", "7.6": "回復期リハビリテーション病棟入院料の疾患別リハビリテーション料包括入力", "7.7": "新型コロナウイルス感染症入院対応",
    "8.1": "日次統計帳票について",
    "9.1": "月次統計帳票について",
    "10.1": "入院室料加算の設定", "10.2": "入院食事療養費の設定（自賠責のみ）"
}

# --- ヘルパー関数 ---

def find_relevant_sections(query: str, toc: dict) -> list[str]:
    """
    ユーザーの質問と目次から、関連性の高い項目をいくつか抽出する。
    """
    scores = []
    query_words = set(re.findall(r'[一-龠ぁ-んァ-ンA-Za-z0-9]+', query))
    
    for key, section in toc.items():
        section_words = set(re.findall(r'[一-龠ぁ-んァ-ンA-Za-z0-9]+', section))
        score = len(query_words.intersection(section_words))
        if score > 0:
            scores.append((score, f"{key} {section}"))
    
    scores.sort(key=lambda x: x[0], reverse=True)
    return [section for score, section in scores[:3]]

# --- AI応答生成関数 ---

def get_text_response_gemini(prompt: str) -> str:
    """
    ローカルでの目次検索と、Gemini API呼び出しを組み合わせた2段階処理で応答を生成します。
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return "エラー: 環境変数 GEMINI_API_KEY が設定されていません。"
    
    genai.configure(api_key=api_key)

    # ステップ1: ローカル検索
    if any(keyword in prompt for keyword in ["入院", "退院", "病棟", "入院料"]):
        toc = NYUIN_TOC
        manual_type = "nyuin"
    else:
        toc = GAIRAI_TOC
        manual_type = "gairai"
    
    relevant_sections = find_relevant_sections(prompt, toc)

    # ステップ2: AIへの指示
    context_message = f"マニュアル種別: {manual_type}\nユーザーの質問: 「{prompt}」\n"
    if relevant_sections:
        context_message += "\n関連マニュアル項目:\n- " + "\n- ".join(relevant_sections)
    else:
        return "関連するマニュアル項目が見つかりませんでした。質問を変えてお試しください。"

    try:
        model = genai.GenerativeModel(
            'gemini-1.5-flash-latest',
            system_instruction=SYSTEM_INSTRUCTION
        )
        response = model.generate_content(context_message)
        return response.text
    except Exception as e:
        return f"エラー: Gemini APIの呼び出し中に問題が発生しました。 {e}"

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
            response = get_text_response_gemini(prompt)
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

# --- Streamlit UI設定 ---

st.set_page_config(page_title="Medi-Partner", layout="wide")
st.title("🏥 Medi-Partner: ORCA実務者向けAIアシスタント")

# --- サイドバー ---
with st.sidebar:
    st.title("情報")

# --- チャット履歴の表示 ---
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "こんにちは！Medi-Partnerです。ORCAや医療事務に関するご質問をどうぞ。"}]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- テキスト入力 ---
if prompt := st.chat_input("ORCAに関する質問をテキストで入力してください"):
    handle_prompt(prompt)
