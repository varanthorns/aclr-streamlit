import streamlit as st
import json, random, pandas as pd, time
from datetime import datetime
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from streamlit_mic_recorder import mic_recorder

# ===================== CONFIG =====================
st.set_page_config(layout="wide", page_title="ACLR Ultimate Clinical Reasoning")

# ===================== EVALUATION LOGIC =====================
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
    total = dx_s + r_s + d_score_val = d_s

    # 1. DDx Safety Check (ตรวจสอบโรคอันตรายที่ห้ามพลาด)
    must_exclude = case.get("must_exclude", ["Aortic Dissection", "PE"]) # Default safety checks
    safety_score = 1 if all(item in selected_ddx for item in must_exclude) else -1
    
    # 2. Confidence Calibration
    bonus = 0
    if level == "correct" and confidence > 80:
        bonus = 1 
    elif level == "wrong" and confidence > 90:
        bonus = -2 # Dangerous Overconfidence Penalty

    final_score = max(0, min(10, total + safety_score + bonus))
    return final_score, dx_s, r_s, d_s, safety_score, target, used, level

# ===================== UTILS & LOAD =====================
def safe_case(case):
    case.setdefault("block", "General")
    case.setdefault("difficulty", "medium")
    case.setdefault("must_exclude", ["Aortic Dissection", "PE"]) # เพิ่มระบบความปลอดภัยพื้นฐาน
    case.setdefault("key_points", [])
    case.setdefault("labs", [])
    case.setdefault("reference", {"source":"Unknown","year":"2026"})
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
            "interprofessional_answers": {"medicine": "Primary PCI", "pharmacy": "Aspirin + Heparin"}
        })]

cases = load_cases()

# ===================== SESSION STATE =====================
if "case" not in st.session_state:
    st.session_state.case = random.choice(cases)
if "submitted" not in st.session_state:
    st.session_state.submitted = False
if "voice_text" not in st.session_state:
    st.session_state.voice_text = ""

# ===================== SIDEBAR NAVIGATION =====================
with st.sidebar:
    st.title("🧠 ACLR Menu")
    page = st.radio("Go to Page", ["📖 User Guide & Scoring", "🧪 Clinical Simulator", "🏆 Leaderboard"])
    
    st.divider()
    user = st.text_input("👤 User ID", value="Doctor_X")
    profession = st.selectbox("👩‍⚕️ Your Role", ["medicine", "nursing", "pharmacy", "dentistry", "ams"])
    
    if page == "🧪 Clinical Simulator":
        if st.button("🔄 Next Case"):
            st.session_state.case = random.choice(cases)
            st.session_state.submitted = False
            st.session_state.voice_text = ""
            st.rerun()

# ===================== PAGE 1: USER GUIDE =====================
if page == "📖 User Guide & Scoring":
    st.header("📖 คู่มือการใช้งานและเกณฑ์การประเมิน")
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("💡 ระบบ Clinical Reasoning")
        st.write("เน้นการวิเคราะห์โรคอย่างเป็นระบบและการคัดกรองโรคอันตราย (DDx Safety Check)")
    with col_b:
        st.subheader("📊 Rubric (10 Points)")
        st.write("- Dx Accuracy: 5 pts\n- Evidence: 3 pts\n- Logic Flow: 2 pts\n- Safety/Confidence: +/- Points")

# ===================== PAGE 2: SIMULATOR =====================
elif page == "🧪 Clinical Simulator":
    case = st.session_state.case
    st.title(f"🏥 Station: {case['block']}")
    
    col1, col2 = st.columns([2, 1])

    with col1:
        tab1, tab2, tab3 = st.tabs(["📋 Scenario", "🧪 Diagnostics", "✍️ Analysis"])
        
        with tab1:
            st.info(case["scenario"].get("en", ""))
            if case.get("image_url"):
                st.image(case["image_url"], use_container_width=True)
                
        with tab2:
            if case["labs"]: st.table(pd.DataFrame(case["labs"]))
            else: st.write("No specific labs provided.")

        with tab3:
            st.warning(f"**Task:** {case.get('task', {}).get(profession, 'วินิจฉัยและระบุพยาธิสภาพ')}")
            
            # Voice Input
            audio = mic_recorder(start_prompt="🎙️ Voice Record", stop_prompt="Stop", key='recorder')
            if audio: st.session_state.voice_text = f"Recorded: {case['answer']}"
            
            # Clinical Inputs
            selected_ddx = st.multiselect("🔍 Differential Diagnosis (Select Important Ones)", 
                                          ["MI", "PE", "Aortic Dissection", "Pneumonia", "Sepsis", "GERD", "Pericarditis"])
            
            dx = st.text_input("🩺 Final Diagnosis", value=st.session_state.voice_text)
            reasoning = st.text_area("✍️ Pathophysiological Reasoning", placeholder="เหตุผลทางคลินิก...")
            conf = st.slider("🎯 Confidence Level (%)", 0, 100, 50)
            
            if st.button("✅ Submit Decision"):
                if dx and reasoning:
                    st.session_state.submitted = True
                    st.rerun()

    if st.session_state.submitted:
        st.divider()
        score, dx_s, r_s, d_s, s_s, target, used, level = evaluate_pro(dx, reasoning, case, profession, conf, selected_ddx)
        
        res_col, ipa_col = st.columns([2, 1])
        with res_col:
            st.subheader(f"📊 Total Score: {score}/10")
            st.progress(score * 10)
            
            # Metrics
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Dx Accuracy", f"{dx_s}/5")
            m2.metric("Evidence", f"{r_s}/3")
            m3.metric("Logic", f"{d_s}/2")
            m4.metric("Safety", "Pass" if s_s > 0 else "Fail", delta=s_s)
            
            st.write(f"**Correct Answer:** `{target}`")
            if s_s < 0: st.error(f"❌ Missed Red Flags: คุณไม่ได้เลือกโรคอันตรายที่ต้องแยกโรคออก ({', '.join(case['must_exclude'])})")

        with ipa_col:
            st.subheader("👥 Team Insights")
            for role, ans in case.get("interprofessional_answers", {}).items():
                with st.expander(f"From {role.upper()}"): st.write(ans)

# ===================== PAGE 3: LEADERBOARD =====================
elif page == "🏆 Leaderboard":
    st.header("🏆 Performance Dashboard")
    st.info("ระบบจะแสดงสถิติเมื่อมีการบันทึกข้อมูลใน responses.csv")

st.markdown("---")
st.caption("ACLR Professional v3.6 | Interprofessional Clinical Reasoning Simulator")

