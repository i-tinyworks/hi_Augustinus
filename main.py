# ==========================================
# Hi Augustine — Streamlit RAG Chatbot
# Supabase + OpenAI Embedding + Cerebras LLM
# ==========================================

import os
import streamlit as st
from openai import OpenAI
from supabase import create_client
from dotenv import load_dotenv

load_dotenv()

# ==========================================
# 1) Supabase 연결 설정 (반드시 anon key)
# ==========================================
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_ANON_KEY")  # ✔ anon key only

if not SUPABASE_URL or not SUPABASE_KEY:
    st.error("❌ SUPABASE_URL 또는 SUPABASE_ANON_KEY가 로드되지 않았습니다.")
    st.stop()

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)


def check_supabase_connection():
    try:
        supabase.table("documents").select("id").limit(1).execute()
        return True, "정상 연결됨"
    except Exception as e:
        return False, str(e)


# ==========================================
# 2) Cerebras LLM (채팅 모델)
# ==========================================
client = OpenAI(
    base_url="https://api.cerebras.ai/v1",
    api_key=os.getenv("CEREBRAS_API_KEY")
)

if not os.getenv("CEREBRAS_API_KEY"):
    st.error("❌ CEREBRAS_API_KEY가 없습니다.")
    st.stop()


# ==========================================
# 3) OpenAI Embedding 모델
# ==========================================
embed_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

if not os.getenv("OPENAI_API_KEY"):
    st.error("❌ OPENAI_API_KEY가 없습니다.")
    st.stop()


def embed_text(text: str):
    try:
        res = embed_client.embeddings.create(
            model="text-embedding-3-large",  # 3072 vector
            input=text
        )
        return res.data[0].embedding
    except Exception as e:
        st.error(f"임베딩 오류: {str(e)}")
        return None


# ==========================================
# 4) Supabase 벡터 검색 (RPC 사용)
# ==========================================
def search_supabase(query_embedding, match_count=5):
    try:
        response = supabase.rpc(
            "match_documents",
            {
                "query_embedding": query_embedding,
                "match_threshold": 0.3,     # SQL 함수와 동일해야 함
                "match_count": match_count
            }
        ).execute()
        return response.data or []
    except Exception as e:
        st.error(f"Supabase RPC 오류: {str(e)}")
        return []


def build_context(question: str):
    emb = embed_text(question)
    if emb is None:
        return ""

    matches = search_supabase(emb, match_count=5)

    if not matches:
        return "본문에는 없습니다."

    return "\n\n".join([m["content"] for m in matches])


# ==========================================
# 5) 사이드바 — 모델 선택 / DB 연결상태 표시
# ==========================================
st.sidebar.title("⚙️ 설정")

ok, msg = check_supabase_connection()
if ok:
    st.sidebar.success("🟢 Supabase 연결됨")
else:
    st.sidebar.error(f"🔴 Supabase 연결 실패\n\n{msg}")

model_options = {
    "GPT-OSS 120B": "gpt-oss-120b",
    "QWen 32B": "qwen-3-32b",
    "LLaMA 3.1 8B": "llama3.1-8b",
}

selected_model_name = st.sidebar.selectbox(
    "🤖 LLM 선택",
    list(model_options.keys())
)

st.session_state["llm_model"] = model_options[selected_model_name]


# ==========================================
# 6) 시스템 메시지 — 어거스틴 역할 부여
# ==========================================
system_prompt = """
역할: 너는 히포의 어거스틴(Augustine of Hippo)의 역할을 수행한다.
네 말투는 따뜻하고 깊은 통찰을 가진 목사이자 철학자처럼 말한다.

답변 원칙:
1) 따뜻한 공감
2) 깊은 신학·철학적 통찰
3) 어거스틴 사상 반영
4) 성경적 부드러운 설명
5) 비기독교인도 포용
6) 핵심 요약
7) 마지막에 라틴어 한 문장 요약
8) context에 없는 내용: "본문에는 없습니다."
9) 대답 도중에 끝마치지 말고 반드시 마무리 하기.
"""


# ==========================================
# 7) LLM 응답 생성
# ==========================================
def ask_llm(question: str, context: str):
    rag_prompt = f"""
[Context: Augustine 문헌 발췌]
{context}

(주의: 위 context 내용만 참고하여 답하라.
context에 없는 내용은 반드시 "본문에는 없습니다."라고 답하라.)

질문: {question}
"""

    try:
        completion = client.chat.completions.create(
            model=st.session_state["llm_model"],
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": rag_prompt}
            ],
            temperature=0.4,
            max_completion_tokens=1000
        )
        return completion.choices[0].message.content
    except Exception as e:
        st.error(f"LLM 오류: {str(e)}")
        return "오류가 발생했습니다."


# ==========================================
# 8) UI 출력 — 대화 기록 표시
# ==========================================
st.title("Hi 어거스틴 😎✝️")

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "system", "content": system_prompt}]

for msg in st.session_state.messages:
    if msg["role"] != "system":
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])


# ==========================================
# 9) 사용자 입력 → RAG → LLM 대답
# ==========================================
if user_input := st.chat_input("신앙/신학 무엇이 궁금한가요?"):

    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    # 🔎 RAG Context 생성
    context = build_context(user_input)

    # 🤖 LLM 답변
    answer = ask_llm(user_input, context)

    with st.chat_message("assistant"):
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})


# ==========================================
# 10) 로컬 Streamlit 실행용
# ==========================================
if __name__ == "__main__":
    import subprocess, sys
    if not os.environ.get("STREAMLIT_RUNNING"):
        os.environ["STREAMLIT_RUNNING"] = "1"
        subprocess.run([sys.executable, "-m", "streamlit", "run", __file__])
