import streamlit as st
import json, random, pandas as pd, os, time
import google.generativeai as genai
from streamlit_mic_recorder import mic_recorder

# ===================== 1. CONFIG & MEDICAL UI =====================
st.set_page_config(layout="wide", page_title="ACLR Clinical Simulator", page_icon="🏥")

# Inject Custom Medical-Grade CSS (ทำให้แอปดูโปรขึ้นทันที)
st.markdown("""
    <style>
    .main { background-color: #f0f2f6; }
    .stButton>button { width: 100%; border-radius: 8px; height: 3.5em; background-color: #1976D2; color: white; font-weight: bold; }
    .stMetric { background-color: white; padding: 15px; border-radius: 10px; border-left: 5px solid #1976D2; }
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] { background-color: #e3f2fd; border-radius: 5px 5px 0 0; padding: 10px 20px; }
    .stTabs [aria-selected="true"] { background-color: #1976D2 !important; color: white !important; }
    </style>
    """, unsafe_allow_html=True)

# ===================== 2. API SETUP =====================
# ใส่ API Key ที่คุณได้มาตรงนี้
GEMINI_API_KEY = "AIzaSyDVy5Bh-RmscVwgzUIuYSK8CHa5ZAKnx_g"
genai.configure(api_key=GEMINI_API_KEY)

# ===================== 3. LOCAL DATABASE LOGIC (CSV) =====================
DB_FILE = "leaderboard.csv"

def save_score_local(name, role, score, block):
    """บันทึกคะแนนลงไฟล์ CSV ในเครื่อง"""
    new_data = {
        "Timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "Name": name,
        "Role": role.upper(),
        "Score": score,
        "Block": block
    }
    df = pd.DataFrame([new_data])
    if not os.path.isfile(DB_FILE):
        df.to_csv(DB_FILE, index=False)
    else:
        df.to_csv(DB_FILE, mode='a', index=False, header=False)

def get_ai_feedback(user_dx, user_reasoning, target_ans, role):
    """Winning Feature: Personalized AI Tutor using Gemini"""
    model = genai.GenerativeModel('gemini-1.5-flash')
    prompt = f"""
    You are a Senior Medical Clinical Instructor. 
    Evaluate this {role}'s clinical reasoning for this case.
    User's Diagnosis: {user_dx}
    User's Reasoning: {user_reasoning}
    Gold Standard for this role: {target_ans}
    
    Provide concise feedback (max 3 short bullets):
    1. Accuracy check.
    2. Critical points missed.
    3. Clinical pearl for improvement.
    Tone: Encouraging and professional.
    """
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"AI Feedback is resting. (Error: {e})"

# ===================== 4. DATA LOADING =====================
@st.cache_data
def load_cases():
    if os.path.exists("cases.json"):
        with open("cases.json", "r", encoding="utf-8") as f:
            return json.load(f)
    return [{"block":"General", "difficulty":"easy", "scenario":{"en":"Standard Case"}, "answer":"N/A"}]

cases = load_cases()

# ===================== 5. SESSION STATE =====================
if "case" not in st.session_state: st.session_state.case = cases[0]
if "submitted" not in st.session_state: st.session_state.submitted = False
if "ai_feedback" not in st.session_state: st.session_state.ai_feedback = ""

# ===================== 6. SIDEBAR =====================
with st.sidebar:
    st.title("🧠 ACLR PRO v7.5")
    st.markdown("---")
    menu = st.radio("Navigation", ["📖 User Guide", "🧪 Simulator", "🏆 Leaderboard"])
    st.divider()
    user_name = st.text_input("👤 Your Name", "Student_01")
    profession = st.selectbox("👩‍⚕️ Professional Role", ["Doctor", "Pharmacy", "Nursing", "AMS", "Vet", "Public Health"]).lower()
    
    if menu == "🧪 Simulator":
        if st.button("🔄 Generate New Case"):
            st.session_state.case = random.choice(cases)
            st.session_state.submitted = False
            st.session_state.ai_feedback = ""
            st.rerun()

# ===================== 7. MAIN PAGES =====================

if menu == "📖 User Guide":
    st.header("📖 Clinical Reasoning Guide")
    st.write("แพลตฟอร์มนี้จำลองสถานการณ์จริงเพื่อให้ทีมสหสาขาวิชาชีพฝึกการตัดสินใจร่วมกัน")
    st.info("💡 **Tip:** AI จะประเมินเหตุผลของคุณตามบทบาทวิชาชีพที่คุณเลือก")

elif menu == "🧪 Simulator":
    c = st.session_state.case
    st.title(f"🏥 Station: {c.get('block')} ({c.get('difficulty').upper()})")
    
    col_l, col_r = st.columns([2, 1])
    with col_l:
        tab1, tab2, tab3 = st.tabs(["📋 Scenario", "🧪 Diagnostics", "✍️ Your Analysis"])
        with tab1:
            st.info(c.get('scenario').get('en'))
            st.caption(f"TH: {c.get('scenario').get('th')}")
        with tab2:
            if c.get("labs"): st.table(pd.DataFrame(c["labs"]))
        with tab3:
            dx_in = st.text_input("🩺 Final Diagnosis")
            re_in = st.text_area("✍️ Rationale / Pathophysiology", height=150)
            
            p1, p2 = st.columns(2)
            u_step = p1.selectbox("Next Step", ["Emergency Procedure", "Start Medication", "Diagnostic Imaging", "Consult Specialist"])
            u_dispo = p2.selectbox("Disposition", ["Admit ICU/CCU", "Admit General Ward", "Discharge Home"])
            u_conf = st.slider("Confidence (%)", 0, 100, 50)

            if st.button("✅ Submit Decision"):
                if dx_in and re_in:
                    # Logic Score (Dummy for hackathon demo)
                    score = random.randint(7, 10) 
                    save_score_local(user_name, profession, score, c.get('block'))
                    
                    # Get AI Feedback from Gemini
                    target = c.get('interprofessional_answers', {}).get(profession, c.get('answer'))
                    with st.spinner("AI Clinical Instructor is evaluating..."):
                        st.session_state.ai_feedback = get_ai_feedback(dx_in, re_in, target, profession)
                    
                    st.session_state.submitted = True
                    st.rerun()

    if st.session_state.submitted:
        st.divider()
        res_l, res_r = st.columns(2)
        with res_l:
            st.subheader("🤖 AI Clinical Tutor Feedback")
            st.markdown(st.session_state.ai_feedback)
        with res_r:
            st.subheader("🎯 Gold Standard Answer")
            st.success(c.get('interprofessional_answers', {}).get(profession, c.get('answer')))
            with st.expander("View Team Perspectives"):
                for role, ans in c.get('interprofessional_answers', {}).items():
                    st.write(f"**{role.upper()}:** {ans}")

elif menu == "🏆 Leaderboard":
    st.header("🏆 Professional Ranking")
    if os.path.exists(DB_FILE):
        df = pd.read_csv(DB_FILE)
        st.dataframe(df.sort_values(by="Timestamp", ascending=False), use_container_width=True)
        
        # Simple Analytics for Hackathon
        st.subheader("📊 Performance Analytics")
        col1, col2 = st.columns(2)
        col1.metric("Total Cases Solved", len(df))
        col2.metric("Average Team Score", f"{df['Score'].mean():.1f}/10")
        st.bar_chart(df.groupby('Role')['Score'].mean())
    else:
        st.info("No data yet. Start a simulation!")

st.markdown("---")
st.caption("ACLR Professional v7.5 | Local DB Edition | Powered by Gemini 1.5 Flash")
