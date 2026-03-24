import streamlit as st
import json, random, pandas as pd, time
from datetime import datetime
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from streamlit_mic_recorder import mic_recorder

# ===================== 1. CONFIG (ต้องอยู่บรรทัดแรกๆ) =====================
st.set_page_config(layout="wide", page_title="ACLR Ultimate Clinical Reasoning")

# ===================== 2. EVALUATION LOGIC =====================
def evaluate_base(dx, reasoning, case, profession):
    """ฟังก์ชันพื้นฐานสำหรับคำนวณคะแนน Dx, Evidence และ Logic"""
    target = case.get("interprofessional_answers", {}).get(profession, case.get("answer", ""))
    
    # 1. Accuracy Score (5 pts)
    try:
        vec = TfidfVectorizer().fit_transform([str(dx).lower(), str(target).lower()])
        sim = cosine_similarity(vec[0:1], vec[1:2])[0][0]
    except: sim = 0
    dx_score = 5 if sim > 0.75 else (3 if sim > 0.45 else 0)
    level = "correct" if sim > 0.75 else ("close" if sim > 0.45 else "wrong")

    # 2. Key Evidence Score (3 pts)
    found_keys = [k for k in case.get("key_points", []) if k.lower() in reasoning.lower()]
    r_score = min(3, len(found_keys))

    # 3. Clinical Logic Score (2 pts)
    logic_words = ["because", "therefore", "thus", "due to", "เนื่องจาก", "ดังนั้น", "ทำให้", "ส่งผล"]
    d_score = 2 if any(w in reasoning.lower() for w in logic_words) else 0

    return dx_score, r_score, d_score, target, found_keys, level

def evaluate_pro(dx, reasoning, case, profession, confidence, selected_ddx):
    """ฟังก์ชันประเมินผลระดับสูง รวมระบบ Safety และ Confidence"""
    dx_s, r_s, d_s, target, used, level = evaluate_base(dx, reasoning, case, profession)
    
    total_base = dx_s + r_s + d_s

    # 1. DDx Safety Check (ตรวจสอบโรคอันตรายที่ห้ามพลาด)
    must_exclude = case.get("must_exclude", ["Aortic Dissection", "PE"])
    safety_score = 1 if all(item in selected_ddx for item in must_exclude) else -1
    
    # 2. Confidence Calibration
    bonus = 0
    if level == "correct" and confidence > 80:
        bonus = 1 
    elif level == "wrong" and confidence > 90:
        bonus = -2 # Dangerous Overconfidence Penalty

    final_score = max(0, min(10, total_base + safety_score + bonus))
    return final_score, dx_s, r_s, d_s, safety_score, target, used, level

# ===================== 3. UTILS & DATA LOADING =====================
def safe_case(case):
    case.setdefault("block", "General")
    case.setdefault("difficulty", "medium")
    case.setdefault("must_exclude", ["Aortic Dissection", "PE"])
    case.setdefault("key_points", [])
    case.setdefault("labs", [])
    case.setdefault("reference", {"source":"Unknown","year":"2026"})
    case.setdefault("scenario", {"en": "No scenario provided."})
    case.setdefault("answer", "Unknown Diagnosis")
    case.setdefault("interprofessional_answers", {})
    return case

@st.cache_data
def load_cases():
    try:
        with open("cases.json","r",encoding="utf-8") as f:
            data = json.load(f)
            return [safe_case(c) for c in data]
    except FileNotFoundError:
        return [safe_case({
            "block":"Cardiovascular", 
            "difficulty":"hard", 
            "scenario":{"en":"A 62-year-old male presents with acute substernal chest pain. EKG shows ST-elevation in II, III, aVF."}, 
            "image_url": "https://p0.pikist.com/photos/403/619/ecg-heart-rate-frequency-medical-medicine-health-science-curve-pulse.jpg",
            "labs": [{"Test": "Troponin T", "Result": "450", "Unit": "ng/L", "Range": "< 14"}],
            "answer":"Inferior Wall MI", 
            "must_exclude": ["Aortic Dissection", "PE"],
            "key_points":["ST-elevation", "inferior", "chest pain"],
            "interprofessional_answers": {
                "medicine": "Primary PCI within 90 mins", 
                "pharmacy": "Aspirin 300mg + Heparin bolus",
                "nursing": "Monitor BP and Right-sided EKG"
            }
        })]

cases = load_cases()

# ===================== 4. SESSION STATE =====================
if "case" not in st.session_state:
    st.session_state.case = random.choice(cases)
if "submitted" not in st.session_state:
    st.session_state.submitted = False
if "voice_text" not in st.session_state:
    st.session_state.voice_text = ""

# ===================== 5. SIDEBAR (Navigation Defined Here) =====================
with st.sidebar:
    st.title("🧠 ACLR Professional")
    page = st.radio("Navigation", ["📖 User Guide", "🧪 Clinical Simulator", "🏆 Leaderboard"])
    
    st.divider()
    user_id = st.text_input("👤 User ID", value="Doctor_X")
    profession = st.selectbox("👩‍⚕️ Your Role", ["medicine", "nursing", "pharmacy", "dentistry", "ams", "public health"])
    
    if page == "🧪 Clinical Simulator":
        st.subheader("Station Control")
        if st.button("🔄 Next Random Case"):
            st.session_state.case = random.choice(cases)
            st.session_state.submitted = False
            st.session_state.voice_text = ""
            st.rerun()

# ===================== 6. MAIN PAGES LOGIC =====================

# --- PAGE 1: USER GUIDE ---
if page == "📖 User Guide":
    st.header("📖 Clinical Reasoning Manual")
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("💡 ACLR คืออะไร?")
        st.write("ระบบฝึกการตัดสินใจทางคลินิกขั้นสูงที่เน้นกระบวนการคิดมากกว่าการเลือกตอบ")
        st.markdown("""
        * **Active Reasoning:** วิเคราะห์พยาธิสภาพด้วยตัวเอง
        * **Safety Check:** คัดกรองโรคอันตราย (Must-Exclude)
        * **Calibration:** ฝึกความมั่นใจให้เหมาะสมกับความรู้
        """)
    with col_b:
        st.subheader("📊 Scoring Rubric")
        rubric_df = pd.DataFrame({
            "Category": ["Dx Accuracy", "Evidence", "Logic Flow", "Safety"],
            "Score": ["5", "3", "2", "+/- Bonus"]
        })
        st.table(rubric_df)

# --- PAGE 2: SIMULATOR ---
elif page == "🧪 Clinical Simulator":
    case = st.session_state.case
    st.title(f"🧠 Station: {case['block']} ({case['difficulty'].upper()})")
    
    col1, col2 = st.columns([2, 1])

    with col1:
        tab1, tab2, tab3 = st.tabs(["📋 Scenario", "🧪 Diagnostics", "✍️ Analysis & Decision"])
        
        with tab1:
            st.info(case["scenario"].get("en", ""))
            if case.get("image_url"):
                st.image(case["image_url"], caption="Clinical Presentation", use_container_width=True)
                
        with tab2:
            st.markdown("### 🔬 Laboratory & Investigation")
            if case["labs"]: st.table(pd.DataFrame(case["labs"]))
            else: st.write("No lab data available.")

        with tab3:
            st.warning(f"**Task:** {case.get('task', {}).get(profession, 'Diagnose and explain the pathophysiology.')}")
            
            # Voice Section
            audio = mic_recorder(start_prompt="🎙️ Record Summary", stop_prompt="Stop", key='recorder')
            if audio:
                st.session_state.voice_text = f"Input received (Simulated Dx: {case['answer']})"

            # Clinical Inputs
            selected_ddx = st.multiselect("🔍 Must-Exclude Differential Diagnosis", 
                                          ["MI", "PE", "Aortic Dissection", "Sepsis", "Pneumothorax", "GERD"])
            
            dx_input = st.text_input("🩺 Final Diagnosis", value=st.session_state.voice_text)
            reason_input = st.text_area("✍️ Pathophysiological Reasoning", placeholder="Describe the mechanism...")
            conf_input = st.slider("🎯 Confidence Level (%)", 0, 100, 50)
            
            if st.button("✅ Submit Decision"):
                if dx_input and reason_input:
                    st.session_state.submitted = True
                    # บันทึกคะแนน (Simulated CSV Save)
                    st.rerun()
                else:
                    st.error("Please fill in both Diagnosis and Reasoning.")

    # --- RESULTS SECTION ---
    if st.session_state.submitted:
        st.divider()
        score, dx_s, r_s, d_s, s_s, target, used, level = evaluate_pro(
            dx_input, reason_input, case, profession, conf_input, selected_ddx
        )
        
        c_left, c_right = st.columns([2, 1])
        with c_left:
            st.subheader(f"📊 Assessment Result: {score}/10")
            st.progress(score * 10)
            
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Dx Accuracy", f"{dx_s}/5")
            m2.metric("Key Evidence", f"{r_s}/3")
            m3.metric("Logic Flow", f"{d_s}/2")
            m4.metric("Safety Check", "PASS" if s_s > 0 else "FAIL", delta=s_s)
            
            st.write(f"**Correct Dx:** `{target}`")
            if s_s < 0:
                st.error(f"❌ Safety Warning: You missed critical Red Flags: {', '.join(case['must_exclude'])}")

        with c_right:
            st.subheader("👥 Team Feedback")
            for role, ans in case.get("interprofessional_answers", {}).items():
                with st.expander(f"From {role.upper()}"):
                    st.write(ans)

# --- PAGE 3: LEADERBOARD ---
elif page == "🏆 Leaderboard":
    st.header("🏆 Performance Dashboard")
    st.info("Leaderboard will be populated once you complete cases and save to responses.csv")
    # ตัวอย่างตาราง
    sample_data = pd.DataFrame({"User": ["Doctor_X"], "Block": ["Cardio"], "Score": [8.5], "Date": [datetime.now()]})
    st.dataframe(sample_data)

# ===================== FOOTER =====================
st.markdown("---")
st.caption("ACLR Ultimate v3.6 | 2026 Updated Edition | Built for Advanced Clinical Reasoning")

