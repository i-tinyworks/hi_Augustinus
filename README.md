# **어거스틴에게 물어봐 — Streamlit RAG Chatbot**

Supabase + OpenAI Embedding + Cerebras LLM 기반 신학 RAG 챗봇

---

## 📌 소개

**Hi Augustine**은 어거스틴(Augustinus)의 저작을 기반으로, 사용자의 신앙·신학 질문에 답변하는 **RAG 기반 챗봇**입니다.
질문 → 임베딩 → Supabase 벡터 검색 → Augustine 문맥 기반 LLM 응답의 전체 과정을 자동화합니다.

---

## 🚀 주요 기능

* Supabase Vector DB 기반 문맥 검색 (RPC: `match_documents`)
* OpenAI 임베딩 생성 (`text-embedding-3-large`)
* Cerebras LLM 선택(Qwen / LLaMA / GPT-OSS)
* Augustine 스타일의 시스템 프롬프트 적용
* Streamlit 챗 인터페이스 제공

---

## 📂 프로젝트 구조

```
Hi_Augustinus/
 ├── main.py
 ├── .env
 ├── requirements.txt
 └── README.md
```

---

## 🔧 설치 및 실행

### 1) 패키지 설치

```bash
pip install -r requirements.txt
```

### 2) 환경 변수 설정

프로젝트 루트에 `.env` 파일 생성:

```
SUPABASE_URL=your_supabase_url
SUPABASE_ANON_KEY=your_anon_key
OPENAI_API_KEY=your_openai_key
CEREBRAS_API_KEY=your_cerebras_key
```

### 3) 실행

```bash
streamlit run main.py
```

---

## 🧠 RAG 동작 흐름

1. 사용자 질문 입력
2. 질문을 OpenAI Embedding으로 변환
3. Supabase RPC(`match_documents`)로 유사 문헌 검색
4. 문맥(Context) 구성
5. 선택된 Cerebras LLM에 전달하여 Augustine 스타일로 응답 생성
6. Streamlit UI에서 출력

---

## 🗄 Supabase 설정 요약

### Documents 테이블

```sql
CREATE TABLE documents (
  id bigint generated always as identity primary key,
  content text,
  embedding vector(3072)
);
```

### 벡터 검색 RPC

```sql
create or replace function match_documents(
  query_embedding vector(3072),
  match_threshold float,
  match_count int
)
returns table (
  id bigint,
  content text,
  similarity float
)
language plpgsql
as $$
begin
  return query
  select d.id, d.content,
         1 - (d.embedding <=> query_embedding) as similarity
  from documents d
  order by similarity desc
  limit match_count;
end;
$$;
```

---

## 🎛 모델 선택

사이드바에서 다음 모델 중 선택 가능:

* **Qwen 3-32B**
* **LLaMA 3.1 8B**
* **GPT-OSS 120B**

---

## 📜 시스템 프롬프트 (요약)

* 역할: *어거스틴 스타일의 신학자·목회자*
* 원칙:

  * context 기반 답변
  * context에 없으면 “본문에는 없습니다.”
  * 부드럽고 지혜로운 목회자 스타일
  * 라틴어 문구로 결말

---

## 📝 개발자

Email: **[itinyworks@gmail.com](mailto:itinyworks@gmail.com)**

---


