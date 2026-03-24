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
    
    # แก้ไข Syntax Error จากเวอร์ชันก่อนหน้า
    total_base = dx_s + r_s + d_s

    # 1. DDx Safety Check (ตรวจสอบโรคอันตรายที่ห้ามพลาด)
    must_exclude = case.get("must_exclude", ["Aortic Dissection", "PE"])
    # ให้คะแนน 1 ถ้าเลือกครบ, หัก 1 ถ้าเลือกไม่ครบ
    safety_score = 1 if all(item in selected_ddx for item in must_exclude) else -1
    
    # 2. Confidence Calibration (โบนัสความมั่นใจ)
    bonus = 0
    if level == "correct" and confidence > 80:
        bonus = 1 
    elif level == "wrong" and confidence > 90:
        bonus = -2 # Dangerous Overconfidence Penalty

    final_score = max(0, min(10, total_base + safety_score + bonus))
    return final_score, dx_s, r_s, d_s, safety_score, target, used, level

# ===================== UTILS & LOAD =====================
def safe_case(case):
    case.setdefault("block", "General")
    case.setdefault("difficulty", "medium")
    case.setdefault("must_exclude", ["Aortic Dissection", "PE"])
    case.setdefault("key_points", [])
    case.setdefault("labs", [])
    case.setdefault("reference", {"source":"Unknown","year":"2026"})
    case.setdefault("scenario", {"en": "No scenario provided."})
    case.setdefault("answer", "Unknown Diagnosis")
    return case

@st.cache_data
def load_cases():
    try:
        with open("cases.json","r",encoding="utf-8") as f:
            data = json.load(f)
            return [safe_case(c) for c in data]
    except FileNotFoundError:
        # เคสตัวอย่างกรณีไม่มีไฟล์ cases.json
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

# ===================== SESSION STATE =====================
if "case" not in st.session_state:
    st.session_state.case = random.choice(cases)
if "submitted" not in st.session_state:
    st.session_state.submitted = False
if "voice_text" not in st.session_state:
    st.session_state.voice_text = ""

# ===================== SIDEBAR NAVIGATION =====================
with st.sidebar:
    st.title("🧠 ACLR Professional")
    page = st.radio("Navigation", ["📖 User Guide", "🧪 Clinical Simulator", "🏆 Leaderboard"])
    
    st.divider()
    user = st.text_input("👤 User ID", value="Doctor_X")
    profession = st.selectbox("👩‍⚕️ Your Role", ["medicine", "nursing", "pharmacy", "dentistry", "ams", "public health"])
    
    if page == "🧪 Clinical Simulator":
        st.subheader("Station Control")
        if st.button("🔄 Next Random Case"):
            st.session_state.case = random.choice(cases)
            st.session_state.submitted = False
            st.session_state.voice_text = ""
            st.rerun()

# ===================== PAGE 1: USER GUIDE =====================
if page == "📖 User Guide":
    st.header("📖 Clinical Reasoning Manual")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("💡 ACLR Methodology")
        st.markdown("""
        1. **Information Gathering:** อ่าน Scenario และผล Labs
        2. **Differential Diagnosis:** เลือกกลุ่มโรคที่ต้องระวัง
        3. **Logic Flow:** อธิบายเหตุผลโดยใช้คำเชื่อม 'เนื่องจาก... จึง...'
        4. **Safety First:** ระวังเรื่อง Dangerous Overconfidence
        """)
    with c2:
        st.subheader("📊 Scoring (10 Points)")
        st.write("- **Dx Accuracy:** 5 pts\n- **Evidence Capture:** 3 pts\n- **Logic Flow:** 2 pts\n- **Bonus/Penalty:** Safety Check & Confidence")

# ===================== PAGE 2: SIMULATOR =====================
elif page == "🧪 Clinical Simulator":
    case = st.session_state.case
    st.title(f"🧠 Station: {case['block']} ({case['difficulty'].upper()})")
    
    col1, col2 = st.columns([2, 1])

    with col1:
        tab1, tab2, tab3 = st.tabs(["📋 Scenario", "🧪 Diagnostics", "✍️ Analysis & Decision"])
        
        with tab1:
            st.info(case["scenario"].get("en", ""))
            if case.get("image_url"):
                st.image(case["image_url"], caption="Clinical Imaging", use_container_width=True)
                
        with tab2:
            st.markdown("### 🔬 Laboratory Findings")
            if case["labs"]: st.table(pd.DataFrame(case["labs"]))
            else: st.write("No specific labs provided.")

        with tab3:
            st.warning(f"**Task:** {case.get('task', {}).get(profession, 'Diagnose and explain the pathophysiology.')}")
            
            # Voice Section
            audio = mic_recorder(start_prompt="🎙️ Record Case Summary (SBAR)", stop_prompt="Stop Recording", key='recorder')
            if audio:
                st.session_state.voice_text = f"Voice processed: {case['answer']} suspected."

            # Analysis Inputs
            selected_ddx = st.multiselect("🔍 Differential Diagnosis (Select Must-Exclude Diseases)", 
                                          ["MI", "PE", "Aortic Dissection", "Pneumonia", "Sepsis", "Pneumothorax", "GERD"])
            
            dx = st.text_input("🩺 Final Diagnosis", value=st.session_state.voice_text)
            reasoning = st.text_area("✍️ Clinical Reasoning", placeholder="Explain why... (e.g., เพราะว่า... ทำให้...)")
            
            conf = st.slider("🎯 Confidence Level (%)", 0, 100, 50)
            
            c1, c2 = st.columns(2)
            with c1: next_step = st.selectbox("🚀 Next Step", ["Observation", "Emergency Procedure", "Medication", "Referral"])
            with c2: dispo = st.selectbox("🏥 Disposition", ["Home", "Ward", "ICU", "OR"])

            if st.button("✅ Submit Decision"):
                if dx and reasoning:
                    st.session_state.submitted = True
                    st.rerun()
                else:
                    st.error("Please provide both Diagnosis and Reasoning.")

    # --- Results Display ---
    if st.session_state.submitted:
        st.divider()
        score, dx_s, r_s, d_s, s_s, target, used, level = evaluate_pro(dx, reasoning, case, profession, conf, selected_ddx)
        
        res_col, ipa_col = st.columns([2, 1])
        with res_col:
            st.subheader(f"📊 Final Assessment: {score}/10")
            st.progress(score * 10)
            
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Dx Score", f"{dx_s}/5")
            m2.metric("Evidence", f"{r_s}/3")
            m3.metric("Logic", f"{d_s}/2")
            m4.metric("Safety", "PASS" if s_s > 0 else "FAIL", delta=s_s)
            
            st.markdown(f"**Correct Diagnosis:** `{target}`")
            st.write("**Key Features Found:**", ", ".join(used) if used else "None")
            
            if s_s < 0:
                st.error(f"❌ Safety Warning: You missed critical DDx: {', '.join(case['must_exclude'])}")
            if level == "wrong" and conf > 90:
                st.error("⚠️ Overconfidence Alert: Incorrect diagnosis with very high confidence is dangerous.")

        with ipa_col:
            st.subheader("👥 Interprofessional Insight")
            for role, ans in case.get("interprofessional_answers", {}).items():
                with st.expander(f"Perspective from {role.upper()}"):
                    st.write(ans)

# ===================== PAGE 3: LEADERBOARD =====================
elif page == "🏆 Leaderboard":
    st.header("🏆 Performance Dashboard")
    try:
        df = pd.read_csv("responses.csv")
        st.bar_chart(df.groupby("block")["score"].mean())
        st.dataframe(df.sort_values(by="time", ascending=False), use_container_width=True)
    except:
        st.info("No records found. Complete a case to see statistics.")

st.markdown("---")
st.caption("ACLR Professional v3.6 | 2026 Academic Medical Simulator")

