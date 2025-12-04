# 참고: https://docs.streamlit.io/develop/tutorials/chat-and-llm-apps/build-conversational-apps

from openai import OpenAI
import streamlit as st
import os
from supabase import create_client
from dotenv import load_dotenv

load_dotenv()

# ==============================
# 1) Supabase 연결 상태 체크 함수
# ==============================
def check_supabase_connection(supabase):
    try:
        res = supabase.table("documents").select("id").limit(1).execute()
        return True, "정상 연결됨"
    except Exception as e:
        return False, str(e)


# ==============================
# 2) Cerebras LLM 클라이언트
# ==============================
client = OpenAI(
    base_url="https://api.cerebras.ai/v1",
    api_key=os.getenv("CEREBRAS_API_KEY")
)

# ==============================
# 3) OpenAI Embedding 클라이언트
# ==============================
embed_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ==============================
# 4) Supabase 연결
# ==============================
supabase = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_SERVICE_KEY")
)

# ==============================
# 5) Sidebar: 모델 선택 + Supabase 상태
# ==============================
st.sidebar.title("⚙️ 설정")

supabase_ok, supabase_msg = check_supabase_connection(supabase)

if supabase_ok:
    st.sidebar.success("🟢 Supabase 연결됨")
else:
    st.sidebar.error(f"🔴 Supabase 연결 실패\n\n{supabase_msg}")

model_options = {
    "GPT-OSS 120B": "gpt-oss-120b",
    "QWen 32B": "qwen-3-32b",
    "LLaMA 3.1 8B": "llama3.1-8b",
}

selected_model_name = st.sidebar.selectbox(
    "🤖 사용할 언어 모델 선택",
    list(model_options.keys())
)

st.session_state["llm_model"] = model_options[selected_model_name]

# ==============================
# 6) UI 타이틀
# ==============================
st.title("Hi 어거스틴 😎✝️")

# ==============================
# 7) 시스템 메시지
# ==============================
prompt = """
역할: 너는 히포의 어거스틴(Augustine of Hippo)의 역할을 수행한다.
네 말투는 따뜻하고 지혜롭고 마음을 어루만지는 목사이자 철학자처럼 말한다.

대답의 원칙:
1) 따뜻한 공감
2) 깊은 철학·신학적 통찰
3) 은혜·사랑·성찰 중심의 어거스틴 사상 반영
4) 성경과 진리를 부드럽게 전달
5) 비기독교인도 포용
6) 복잡한 개념도 쉽게 설명
7) 핵심만 간결하게 요약
8) 마지막 문장에 라틴어 한 문장 요약 추가
9) context에 없는 내용은 "본문에는 없습니다"라고 답변
10) 대답은 도중에 끊기지 않고 마쳐져야 한다.
"""

# ==============================
# 8) RAG 검색 기능
# ==============================
def embed_text(text: str):
    res = embed_client.embeddings.create(
        model="text-embedding-3-large",
        input=text
    )
    return res.data[0].embedding

def search_supabase(query_embedding, match_count=5):
    response = supabase.rpc(
        "match_documents",
        {
            "query_embedding": query_embedding,
            "match_count": match_count
        }
    ).execute()
    return response.data or []

def build_context(question: str):
    emb = embed_text(question)
    matches = search_supabase(emb, match_count=5)
    return "\n\n".join([m["content"] for m in matches])

def ask_llm(question: str, context: str):
    rag_prompt = f"""
[Context: Augustine 문헌 자료]
{context}

너는 반드시 위 context 내용만 참고하여 답변해야 한다.
"""

    completion = client.chat.completions.create(
        model=st.session_state["llm_model"],
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": rag_prompt}
        ],
        temperature=0.4,
        max_completion_tokens=1000
    )

    return completion.choices[0].message.content

# ==============================
# 9) 이전 메시지 출력
# ==============================
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": prompt}
    ]

for msg in st.session_state.messages:
    if msg["role"] != "system":
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

# ==============================
# 10) 사용자 입력 처리
# ==============================
if user_input := st.chat_input("신앙/신학 무엇이 궁금한가요?"):

    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    context = build_context(user_input)
    answer = ask_llm(user_input, context)

    with st.chat_message("assistant"):
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})

# ==============================
# 11) Streamlit 로컬 실행
# ==============================
if __name__ == "__main__":
    import subprocess
    import sys
    if not os.environ.get("STREAMLIT_RUNNING"):
        os.environ["STREAMLIT_RUNNING"] = "1"
        subprocess.run([sys.executable, "-m", "streamlit", "run", __file__])
