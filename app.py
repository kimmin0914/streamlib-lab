# 📋 app.py · [실습 7-1] 화면(Streamlit · 프런트엔드)이 배포한 API(Vercel · 백엔드)를 부름 — 접근 코드를 헤더에 실어 보냄 (설명 한 번에 요청 1번)
# · Community Cloud 의 Secrets 에 API_URL · APP_ACCESS_CODE 두 줄을 넣음 (Gemini 키는 넣지 않음 — 키는 Vercel 에만)
import requests
import streamlit as st

try:
    API_URL = st.secrets["API_URL"]                  # Secrets 의 이름(API_URL)으로 꺼냄 — 대괄호 안에 주소를 쓰지 않음
except Exception:
    API_URL = "http://localhost:8000"                # Secrets 가 없으면(Colab 점검) 같은 런타임의 API 서버

try:
    ACCESS_CODE = st.secrets["APP_ACCESS_CODE"]      # 접근 코드도 Secrets 에서만 읽음 (코드 · GitHub 에 적지 않음)
except Exception:
    ACCESS_CODE = ""                                 # 접근 코드를 쓰지 않는 서버면 빈 값

st.title("📝 AI 용어 설명기 (API 버전)")
st.caption(f"API 주소 : {API_URL}")

style = st.selectbox("말투", ["초등학생도 알아듣게 쉽게", "전공 수업처럼 정확하게", "재미있는 비유를 들어"])
word = st.text_input("궁금한 용어", placeholder="예: API · 토큰 · 서버")

if st.button("설명 듣기", disabled=(word.strip() == "")):
    with st.spinner("API 에 묻는 중..."):
        try:
            response = requests.post(
                f"{API_URL}/explain",
                json={"word": word, "style": style},
                headers={"X-Access-Code": ACCESS_CODE},  # 서버의 APP_ACCESS_CODE 와 같아야 200 · 다르면 401
                timeout=90,
            )
        except requests.RequestException:
            st.error("❌ API 서버에 닿지 못함 — 화면의 'API 주소' 가 vercel.app 주소인지 확인하세요.")
        else:
            if response.status_code == 200:
                data = response.json()
                st.markdown(data["answer"])
                st.caption(f"상태 번호 200 · 토큰 {data['tokens']}")
            else:
                st.error(f"❌ 상태 번호 {response.status_code} : {response.json().get('detail')}")
