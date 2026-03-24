import streamlit as st
import json, random, pandas as pd, time
from datetime import datetime
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from streamlit_mic_recorder import mic_recorder

# ===================== 1. CONFIG =====================
st.set_page_config(layout="wide", page_title="ACLR Ultimate Clinical Reasoning", page_icon="🧠")

# ===================== 2. EVALUATION LOGIC =====================
def evaluate_pro(dx, reasoning, case, profession, confidence, selected_ddx):
    target = case.get("interprofessional_answers", {}).get(profession, case.get("answer", ""))
    
    # 1. Dx Accuracy (5 pts) - ใช้ AI ตรวจสอบความหมาย
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

    # 4. Safety Check
    must_exclude = case.get("must_exclude", [])
    safety_score = 1 if all(item in selected_ddx for item in must_exclude) else -1
    
    # 5. Confidence Calibration
    bonus = 0
    if level == "correct" and confidence > 80: bonus = 1 
    elif level == "wrong" and confidence > 90: bonus = -2 # Penalty สำหรับ Overconfidence

    final_score = max(0, min(10, dx_score + r_score + d_score + safety_score + bonus))
    return final_score, dx_score, r_score, d_score, safety_score, target, found_keys, level

# ===================== 3. DATA LOADING =====================
def safe_case(case):
    case.setdefault("block", "General")
    case.setdefault("difficulty", "easy")
    case.setdefault("must_exclude", [])
    case.setdefault("key_points", [])
    case.setdefault("labs", [])
    case.setdefault("scenario", {"en": "Sample Scenario: Chest pain."})
    case.setdefault("answer", "Acute MI")
    case.setdefault("interprofessional_answers", {"medicine": "PCI", "nursing": "Monitor", "pharmacy": "ASA"})
    return case

@st.cache_data
def load_cases():
    try:
        with open("cases.json", "r", encoding="utf-8") as f:
            data = json.load(f)
            return [safe_case(c) for c in data]
    except FileNotFoundError:
        return [safe_case({})] # Fallback case if file missing

cases = load_cases()

# ===================== 4. SESSION STATE =====================
if "case" not in st.session_state:
    st.session_state.case = cases[0]
if "submitted" not in st.session_state:
    st.session_state.submitted = False
if "voice_text" not in st.session_state:
    st.session_state.voice_text = ""

# ===================== 5. SIDEBAR NAVIGATION & FILTER =====================
with st.sidebar:
    st.title("🧠 ACLR Professional")
    page = st.radio("Navigation", ["📖 User Guide", "🧪 Clinical Simulator", "🏆 Leaderboard"])
    
    st.divider()
    if page == "🧪 Clinical Simulator":
        st.subheader("🎯 Station Selection")
        blocks = ["All"] + sorted(list(set(c['block'] for c in cases)))
        sel_block = st.selectbox("Select Block", blocks)
        sel_diff = st.select_slider("Select Difficulty", options=["easy", "medium", "hard"])
        
        if st.button("🔄 Generate New Case"):
            filtered = [c for c in cases if (sel_block == "All" or c['block'] == sel_block) and (c['difficulty'] == sel_diff)]
            if filtered:
                st.session_state.case = random.choice(filtered)
                st.session_state.submitted = False
                st.session_state.voice_text = ""
                st.rerun()
            else:
                st.warning("No cases found for this criteria.")

    st.divider()
    user_id = st.text_input("👤 User ID", value="Doctor")
    profession = st.selectbox("👩‍⚕️ Your Role", ["medicine", "nursing", "pharmacy", "ams"])

# ===================== 6. MAIN CONTENT =====================

# --- PAGE 1: USER GUIDE ---
if page == "📖 User Guide":
    st.header("📖 Clinical Reasoning Manual: ACLR Loop")
    st.write("ระบบจำลองการตัดสินใจทางคลินิกขั้นสูงที่เน้นกระบวนการคิดมากกว่าการเลือกตอบ")

    col_g1, col_g2 = st.columns(2)
    with col_g1:
        st.markdown("### 🛠 ขั้นตอนการใช้งาน")
        st.markdown("""
        1. **Information Gathering:** วิเคราะห์ Scenario และผล Lab อย่างละเอียด
        2. **Differential Diagnosis:** เลือกกลุ่มโรคที่ต้องเฝ้าระวัง (Red Flags)
        3. **Formulation:** พิมพ์วินิจฉัยและระบุเหตุผลทางพยาธิสภาพ
        4. **Calibration:** ประเมินความมั่นใจของตนเอง (Metacognition)
        5. **Team Debriefing:** เรียนรู้จากมุมมองสหสาขาวิชาชีพ
        """)
    
    with col_g2:
        st.markdown("### 📊 Scoring Rubric (10 Points)")
        rubric_data = {"หมวดหมู่": ["Diagnosis", "Evidence", "Logic", "Safety"],
                       "คะแนน": ["5", "3", "2", "+1/-1"],
                       "คำอธิบาย": ["ความถูกต้องแม่นยำ", "Keyword สำคัญ", "การใช้ตรรกะเหตุผล", "การเช็ก Red Flags"]}
        st.table(pd.DataFrame(rubric_data))

    st.divider()
    st.subheader("👥 ระบบ Team Board (Interprofessional Insight)")
    t1, t2, t3 = st.columns(3)
    t1.info("**Medicine:** การวินิจฉัยและการรักษาหลัก")
    t2.success("**Pharmacy:** การบริหารยาและข้อควรระวัง")
    t3.warning("**Nursing:** การพยาบาลและการเฝ้าระวัง")

# --- PAGE 2: SIMULATOR ---
elif page == "🧪 Clinical Simulator":
    case = st.session_state.case
    st.title(f"🏥 Station: {case['block']} | Level: {case['difficulty'].upper()}")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        tab1, tab2, tab3 = st.tabs(["📋 Scenario", "🧪 Diagnostics", "✍️ Analysis & Decision"])
        with tab1:
            st.info(case["scenario"].get("en", ""))
            if case.get("image_url"): st.image(case["image_url"], use_container_width=True)
        with tab2:
            if case["labs"]: st.table(pd.DataFrame(case["labs"]))
            else: st.write("No specific labs.")
        with tab3:
            st.warning(f"**Task:** วินิจฉัยและอธิบายเหตุผลในบทบาท {profession.upper()}")
            audio = mic_recorder(start_prompt="🎙️ Record Summary", stop_prompt="Stop", key='recorder')
            if audio: st.session_state.voice_text = f"Suspected {case['answer']}"
            
            ddx_list = ["MI", "PE", "Aortic Dissection", "Sepsis", "Pneumonia", "Pneumothorax"]
            selected_ddx = st.multiselect("🔍 Must-Exclude Differential Diagnosis", ddx_list)
            dx_input = st.text_input("🩺 Final Diagnosis", value=st.session_state.voice_text)
            re_input = st.text_area("✍️ Reasoning", placeholder="อธิบายกลไกพยาธิสภาพ...")
            conf_input = st.slider("🎯 Confidence Level (%)", 0, 100, 50)
            
            if st.button("✅ Submit Decision"):
                if dx_input and re_input:
                    st.session_state.submitted = True
                    st.rerun()
                else: st.error("กรุณากรอกข้อมูลให้ครบถ้วน")

    if st.session_state.submitted:
        st.divider()
        sc, dx_s, r_s, d_s, s_s, target, used, level = evaluate_pro(dx_input, re_input, case, profession, conf_input, selected_ddx)
        cl, cr = st.columns([2, 1])
        with cl:
            st.subheader(f"📊 Assessment Result: {sc}/10")
            st.progress(sc * 10)
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Dx Accuracy", f"{dx_s}/5")
            m2.metric("Evidence", f"{r_s}/3")
            m3.metric("Logic Flow", f"{d_s}/2")
            m4.metric("Safety", "PASS" if s_s > 0 else "FAIL", delta=s_s)
            st.write(f"**Correct Diagnosis:** `{target}`")
            if s_s < 0: st.error(f"❌ Safety Warning: Missed critical DDx!")
        with cr:
            st.subheader("👥 Interprofessional Feedback")
            for role, ans in case.get("interprofessional_answers", {}).items():
                with st.expander(f"From {role.upper()}"): st.write(ans)

# --- PAGE 3: LEADERBOARD ---
elif page == "🏆 Leaderboard":
    st.header("🏆 Performance Dashboard")
    st.info("Leaderboard is ready for integration with database.")

st.markdown("---")
st.caption("ACLR Ultimate v3.9 | 2026 Simulation Edition")
