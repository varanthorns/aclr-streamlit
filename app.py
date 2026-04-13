import streamlit as st
import json, random, pandas as pd, os, time
import google.generativeai as genai

# ===================== ⚙️ GLOBAL CONFIG =====================
DB_FILE = "clinical_analytics_v10.csv"

# 🔐 API Setup
try:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
except:
    GEMINI_API_KEY = "DEMO_KEY"

genai.configure(api_key=GEMINI_API_KEY)

# ===================== 🔧 CORE FUNCTIONS =====================
def save_score_local(data):
    df_new = pd.DataFrame([data])
    if os.path.exists(DB_FILE):
        df_old = pd.read_csv(DB_FILE)
        df = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df = df_new
    df.to_csv(DB_FILE, index=False)

# ===================== 🧠 AI MENTOR LOGIC (UPGRADED) =====================
def get_ai_metacognitive_feedback(case, user_data):
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    # Prompt นี้ออกแบบตามทฤษฎี Reflection-on-action
    prompt = f"""
    Act as a Senior Clinical Professor. Analyze this student's clinical reasoning gap (FTF-CRA).
    
    [Case] Target: {case['answer']} | Role: {user_data['Role']}
    
    [Student Data]
    - First Thought (System 1): {user_data['Dx_FT']} (Confidence: {user_data['Conf_FT']}%)
    - Final Thought (System 2): {user_data['Dx_Final']} (Confidence: {user_data['Conf_Final']}%)
    - Reasoning Map: Positives({user_data['Pos_Findings']}), Noise({user_data['Noise']})
    
    [Tasks]
    1. Analyze 'Diagnostic Shift': Did they change their mind correctly based on Evidence?
    2. Identify 'Cognitive Biases': Any confirmation bias or premature closure?
    3. Calibration: If confidence is high but Dx is wrong, warn about 'Overconfidence'.
    4. Provide 1 Metacognitive Question to trigger reflection.
    """
    try:
        response = model.generate_content(prompt)
        return response.text
    except:
        return f"AI Mentor Offline. Gold Standard: {case['answer']}"

# ===================== 🎨 UI STYLING =====================
st.set_page_config(layout="wide", page_title="FTF-CRA Clinical Simulator")
st.markdown("""
    <style>
    .phase-box { padding: 20px; border-radius: 15px; margin-bottom: 20px; border-left: 10px solid #1976D2; background: #E3F2FD; }
    .stMetric { background: white; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    </style>
    """, unsafe_allow_html=True)

# ===================== 🔄 SESSION STATE =====================
if "phase" not in st.session_state: st.session_state.phase = "FT"
if "submitted" not in st.session_state: st.session_state.submitted = False

# ===================== 🧪 SIMULATOR PAGE =====================
# (จำลองการโหลดเคส)
case = {
    "block":"Cardiology", 
    "answer":"Acute STEMI",
    "scenario":{"en":"A 65-year-old male presents with 2 hours of crushing chest pain and diaphoresis."},
    "labs":[{"Test": "Troponin T", "Result": "480", "Unit": "ng/L", "Ref": "<14"}]
}

with st.sidebar:
    st.title("🩺 FTF-CRA Simulator")
    menu = st.radio("Menu", ["🧪 Clinical Simulator", "🏆 Analytics Hub"])
    user_name = st.text_input("Name", "User_01")
    profession = st.selectbox("Role", ["Doctor", "Nursing", "Pharmacy", "AMS"])
    if st.button("🔄 Reset Case"):
        st.session_state.phase = "FT"; st.session_state.submitted = False; st.rerun()

if menu == "🧪 Clinical Simulator":
    # PHASE 1: FIRST THOUGHT
    if st.session_state.phase == "FT":
        st.markdown("<div class='phase-box'><h3>⚡ Phase 1: First Thought (System 1)</h3></div>", unsafe_allow_html=True)
        st.info(case['scenario']['en'])
        dx_ft = st.text_input("Initial Diagnosis")
        conf_ft = st.slider("Confidence (%)", 0, 100, 50)
        
        if st.button("Unlock Labs & Analysis ➡️"):
            st.session_state.dx_ft = dx_ft
            st.session_state.conf_ft = conf_ft
            st.session_state.phase = "FINAL"
            st.rerun()

    # PHASE 2: FINAL THOUGHT
    elif st.session_state.phase == "FINAL" and not st.session_state.submitted:
        st.markdown("<div class='phase-box'><h3>🧐 Phase 2: Final Thought (System 2)</h3></div>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📋 Clinical Data")
            st.table(pd.DataFrame(case["labs"]))
        with col2:
            st.subheader("🧠 Reasoning Map")
            pos_f = st.text_area("Pertinent Positives (+)")
            noise_f = st.text_area("Clinical Noise (ข้อมูลลวง)")

        dx_final = st.text_input("Final Diagnosis", value=st.session_state.dx_ft)
        conf_final = st.slider("Final Confidence (%)", 0, 100, st.session_state.conf_ft)
        
        if st.button("🚀 Submit Final Decision"):
            payload = {
                "User": user_name, "Role": profession, "Score": 10 if dx_final.lower() in case['answer'].lower() else 5,
                "Dx_FT": st.session_state.dx_ft, "Conf_FT": st.session_state.conf_ft,
                "Dx_Final": dx_final, "Conf_Final": conf_final,
                "Pos_Findings": pos_f, "Noise": noise_f,
                "Timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            save_score_local(payload)
            st.session_state.ai_feedback = get_ai_metacognitive_feedback(case, payload)
            st.session_state.submitted = True
            st.rerun()

    # PHASE 3: RESULT
    if st.session_state.submitted:
        st.success("✅ Assessment Completed")
        st.subheader("👨‍🏫 AI Mentor: Metacognitive Debriefing")
        st.markdown(st.session_state.ai_feedback)
        
        # กราฟแสดง Calibration (ความต่างของความคิด)
        st.divider()
        st.subheader("📊 Decision Gap Analysis")
        gap_df = pd.DataFrame({
            "Phase": ["First Thought", "Final Thought"],
            "Confidence": [st.session_state.conf_ft, st.session_state.conf_final]
        })
        st.line_chart(gap_df.set_index("Phase"))
