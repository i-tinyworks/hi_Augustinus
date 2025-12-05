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
# 1) Supabase 연결 설정
# ==========================================
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_ANON_KEY")  # ✔ anon key only

if not SUPABASE_URL or not SUPABASE_KEY:
    st.error("❌ SUPABASE_URL 또는 SUPABASE_ANON_KEY가 설정되지 않았습니다.")
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
CEREBRAS_KEY = os.getenv("CEREBRAS_API_KEY")
if not CEREBRAS_KEY:
    st.error("❌ CEREBRAS_API_KEY가 없습니다.")
    st.stop()

client = OpenAI(
    base_url="https://api.cerebras.ai/v1",
    api_key=CEREBRAS_KEY
)

# ==========================================
# 3) OpenAI Embedding 모델
# ==========================================
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_KEY:
    st.error("❌ OPENAI_API_KEY가 없습니다.")
    st.stop()

embed_client = OpenAI(api_key=OPENAI_KEY)

def embed_text(text: str):
    try:
        res = embed_client.embeddings.create(
            model="text-embedding-3-large",  # vector size 3072
            input=text
        )
        return res.data[0].embedding
    except Exception as e:
        st.error(f"임베딩 오류 발생: {str(e)}")
        return None


# ==========================================
# 4) Supabase 벡터 검색 (RPC)
# ==========================================
def search_supabase(query_embedding, match_count=5):
    try:
        response = supabase.rpc(
            "match_documents",
            {
                "query_embedding": query_embedding,
                "match_threshold": 0.3,
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
# 5) Sidebar – 모델 선택 / DB 상태
# ==========================================
st.sidebar.title("⚙️ 설정")

ok, msg = check_supabase_connection()
if ok:
    st.sidebar.success("🟢 Supabase 연결됨")
else:
    st.sidebar.error(f"🔴 연결 실패: {msg}")

# ✔ 기본 모델을 Qwen-3-32B로 설정
model_options = {
    "Qwen 3-32B": "qwen-3-32b",
    "LLaMA 3.1 8B": "llama3.1-8b",
    "GPT-OSS 120B": "gpt-oss-120b"
}

selected_model_name = st.sidebar.selectbox(
    "🤖 LLM 선택",
    list(model_options.keys()),
    index=0,     # 기본값을 Qwen-3-32b로 설정
)

st.session_state["llm_model"] = model_options[selected_model_name]


# ==========================================
# 6) 시스템 프롬프트
# ==========================================
system_prompt = """
역할: 너는 어거스틴 이라는 이름의 목회자/신학자 이다.  
답변은 지혜롭고, 영적이며, 내적 성찰을 불러일으켜야 한다.

[말투 및 태도]
- 따뜻하지만 단호한 목회자 스타일
- 지적이며 성경의 진리에 해박한 깊이가 있는 신학자
- 인간의 내면을 읽고 마음을 어루만지는 상담가

[답변 원칙]
0) 폰트는 작지 않고 중간 사이즈로 하고 중요한 내용은 볼드체로 한다. 
1) 하나님의 은혜, 진리, 사랑을 중심으로 설명한다.
2) 질문자의 상태를 공감하며 부드럽게 인도한다.
3) 지나친 논쟁보다는 영적 깨달음을 주도록 설명한다.
4) 성경적 논리를 중심으로 표현하되, 신학적 통찰을 도입한다.
5) 인간 내면의 갈망과 하나님의 부르심을 연결하여 해석한다.
6) 복잡한 개념도 비유와 이미지로 쉽게 설명한다.
7) 대답의 마지막 문장은 반드시 라틴어 요약 문구(한글 번역을 포함한)로 마무리한다.
8) 제공된 RAG context 외의 내용은 마음대로 만들어내지 말고 
오직 “본문에는 없습니다.” 라는 문장을 출력한다. 
9) 답변은 자연스럽고 완결성 있게 끝마친다.
10) 이미 기술한 내용은 똑같이 반복하지 않는다.
11) 답변의 원칙을 잘 지킨다.
11) 그리고 금지로 정한 지침을 반드시 지킬 것

[금지]
※ 절대 <think> ....</think>, chain-of-thought, 내부 추론 과정, 모델의 사고 과정, 
   계획 단계, 분석 문장 등을 출력하지 말 것.
   사용자에게는 완성된 답변만 자연스럽게 제시한다.
"""

# ==========================================
# 7) LLM 응답 생성
# ==========================================
def ask_llm(question: str, context: str):
    rag_prompt = f"""
[Context: Augustine 문헌 발췌]
{context}

(주의: 위 context 내용만 참고하여 답하라.
context에 없는 내용은 반드시 "본문에는 없습니다."라고 답할 것.)

질문: {question}
"""
    try:
        completion = client.chat.completions.create(
            model=st.session_state["llm_model"],
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": rag_prompt}
            ],
            temperature=0.3,
            max_completion_tokens=1000
        )
        return completion.choices[0].message.content

    except Exception as e:
        st.error(f"LLM 오류: {str(e)}")
        return "오류가 발생했습니다."


# ==========================================
# 8) UI
# ==========================================
st.title("어거스틴에게 물어봐 😎✝️")

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "system", "content": system_prompt}]

for msg in st.session_state.messages:
    if msg["role"] != "system":
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])


# ==========================================
# 9) 사용자 입력 → RAG → LLM
# ==========================================
if user_input := st.chat_input("신앙/신학 질문을 입력하세요"):

    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    context = build_context(user_input)
    answer = ask_llm(user_input, context)

    with st.chat_message("assistant"):
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})


# ==========================================
# 10) 로컬 실행 모드
# ==========================================
if __name__ == "__main__":
    import subprocess, sys
    if not os.environ.get("STREAMLIT_RUNNING"):
        os.environ["STREAMLIT_RUNNING"] = "1"
        subprocess.run([sys.executable, "-m", "streamlit", "run", __file__])
