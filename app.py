import json
import os

import streamlit as st
from dotenv import load_dotenv
from openai import (
    APIConnectionError,
    AuthenticationError,
    NotFoundError,
    OpenAI,
    OpenAIError,
    RateLimitError,
)

load_dotenv()

LANGUAGES = {
    "english": "English",
    "japanese": "日本語",
    "vietnamese": "Tiếng Việt",
}

SYSTEM_PROMPT = """You are a professional translator.
Detect the language of the user's text and translate it into English, Japanese, and Vietnamese.
Preserve the meaning and tone. Do not add explanations.
Keep the same line structure: each translation must have exactly as many lines as the input, with line breaks written as \\n.
Respond ONLY with a JSON object in this exact format:
{"english": "...", "japanese": "...", "vietnamese": "..."}"""


def translate(client: OpenAI, model: str, text: str) -> dict:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ],
        response_format={"type": "json_object"},
        temperature=0.2,
    )
    return json.loads(response.choices[0].message.content)


@st.cache_data(show_spinner=False)
def text_to_speech(_client: OpenAI, tts_model: str, text: str) -> bytes:
    response = _client.audio.speech.create(
        model=tts_model,
        voice="alloy",
        input=text,
        response_format="mp3",
    )
    return response.content


def error_message(e: OpenAIError) -> str:
    # OpenAI 원본 메시지에는 API 키 일부가 포함될 수 있어 그대로 보여주지 않는다
    if isinstance(e, AuthenticationError):
        return "API 키가 올바르지 않습니다. `.env`의 `OPENAI_API_KEY`를 확인해주세요."
    if isinstance(e, RateLimitError):
        return "요청 한도를 초과했거나 크레딧이 부족합니다. 잠시 후 다시 시도해주세요."
    if isinstance(e, APIConnectionError):
        return "OpenAI 서버에 연결하지 못했습니다. 인터넷 연결을 확인해주세요."
    if isinstance(e, NotFoundError):
        return "모델을 찾을 수 없습니다. `.env`의 모델 이름을 확인해주세요."
    return f"OpenAI API 오류가 발생했습니다. ({type(e).__name__})"


api_key = os.getenv("OPENAI_API_KEY")
model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
tts_model = os.getenv("OPENAI_TTS_MODEL", "gpt-4o-mini-tts")

st.set_page_config(page_title="AI 번역기", page_icon="🌐", layout="wide")
st.title("🌐 AI 번역기")
st.caption("입력한 텍스트를 영어 · 일본어 · 베트남어로 번역합니다.")

if not api_key:
    st.error("`.env` 파일에 `OPENAI_API_KEY`가 설정되어 있지 않습니다.")
    st.stop()

client = OpenAI(api_key=api_key)

text = st.text_area("번역할 텍스트", height=150, placeholder="번역할 텍스트를 입력하세요...")

if st.button("번역하기", type="primary"):
    # 빈 입력이나 오류 시 이전 번역 결과가 남아 보이지 않도록 먼저 비운다
    st.session_state.result = None
    st.session_state.speaking = None
    if not text.strip():
        st.warning("번역할 텍스트를 입력해주세요.")
    else:
        try:
            with st.spinner("번역 중..."):
                st.session_state.result = translate(client, model, text)
        except OpenAIError as e:
            st.error(error_message(e))
        except json.JSONDecodeError:
            st.error("번역 결과를 해석하지 못했습니다. 다시 시도해주세요.")

# 읽어주기 버튼을 누르면 스크립트가 다시 실행되므로 번역 결과는 session_state에 보관한다
result = st.session_state.get("result")
if result:
    for col, (key, label) in zip(st.columns(3), LANGUAGES.items()):
        translated = result.get(key, "")
        with col:
            st.subheader(label)
            st.code(translated, language=None, wrap_lines=True)
            if st.button("🔊 읽어주기", key=f"speak_{key}", disabled=not translated):
                st.session_state.speaking = key
            if st.session_state.get("speaking") == key:
                try:
                    with st.spinner("음성 생성 중..."):
                        audio = text_to_speech(client, tts_model, translated)
                    st.audio(audio, format="audio/mp3", autoplay=True)
                except OpenAIError as e:
                    st.error(f"음성 생성 실패: {error_message(e)}")
