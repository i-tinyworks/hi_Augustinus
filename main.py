# 참고: https://docs.streamlit.io/develop/tutorials/chat-and-llm-apps/build-conversational-apps

from openai import OpenAI
import streamlit as st
import os

# Cerebras API를 사용하여 OpenAI API 클라이언트 초기화
client = OpenAI(
    base_url="https://api.cerebras.ai/v1",
    api_key=os.getenv("CEREBRAS_API_KEY")
)

# Cerebras 모델 사용
# https://inference-docs.cerebras.ai/models/overview
# "qwen-3-32b"
# "qwen-3-235b-a22b-instruct-2507",
# "qwen-3-coder-480b"
# "llama-4-scout-17b-16e-instruct"
# "qwen-3-235b-a22b-thinking-2507"
# "llama-3.3-70b"
# "llama3.1-8b"
# "gpt-oss-120b"
llm_model = "gpt-oss-120b"  
if "llm_model" not in st.session_state:
    st.session_state["llm_model"] = llm_model

st.title("나의 AI 친구 😎")

# prompt = """
# 역할:너는 공감을 잘해주는 나의 친구야.
# 네 이름은 제니, 대답은 한국어로 해줘.
# 답변마다, 현재 까지 대화 결과를 한문장의 영어 문장으로 요약해서 작성해줘.
# """

prompt = """
<persona>
당신은 시간여행이 가능한 역사학자입니다.
과거를 직접 방문했고, 미래도 다녀온 경험이 있습니다.
모든 현재의 질문을 시간축에서 입체적으로 바라봅니다.
</persona>

<temporal_perspective>
어떤 질문이든 3개 시점에서 답변:

📜 [PAST - 역사적 맥락]
- "500년 전이라면..." 또는 "1990년대만 해도..."
- 현재 상황이 어떻게 형성되었는지
- 과거 사람들은 비슷한 문제를 어떻게 해결했는지
- 잊혀진 지혜나 반복되는 패턴

⚡ [PRESENT - 현재 분석]  
- 지금 여기의 실용적 답변
- 하지만 "이것도 곧 역사가 된다"는 관점

🔮 [FUTURE - 미래 투사]
- "2050년 사람들이 지금을 돌아본다면..."
- 현재 선택이 만들 미래들
- 트렌드의 연장선상 예측
- 경고 또는 희망의 메시지
</temporal_perspective>

<narrative_style>
- 마치 타임머신에서 막 내린 것처럼 생생하게
- "흥미롭게도, 2087년에 내가 본 바로는..."
- 역사적 아이러니와 패턴 지적
- 시간의 흐름 속에서 상대성 강조
</narrative_style>

<wisdom>
"역사는 반복되지 않지만 운율을 맞춘다" - Mark Twain
모든 문제는 새로운 동시에 오래된 것
시간 여행자의 눈으로 보면 공황과 냉정함의 균형을 찾을 수 있음
</wisdom>
"""

# 시스템 프롬프트 설정
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "system", 
            "content": prompt
        }
    ]

for message in st.session_state.messages:
    if message["role"] == "system":
        continue
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("무엇이든 물어보세요."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # 스트리밍 응답 받기
        stream = client.chat.completions.create(
            model=st.session_state["llm_model"],
            messages=[
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
            ],
            temperature=0.7,
            max_completion_tokens=1000,
            stream=True
        )
        response = st.write_stream(stream)
    st.session_state.messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    import subprocess
    import sys
    
    # 환경 변수로 재실행 방지
    if not os.environ.get("STREAMLIT_RUNNING"):
        os.environ["STREAMLIT_RUNNING"] = "1"
        subprocess.run([sys.executable, "-m", "streamlit", "run", __file__])

# python -m streamlit run main.py
# streamlit run main.py
