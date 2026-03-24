import streamlit as st
import json, random, pandas as pd, time
from datetime import datetime
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from streamlit_mic_recorder import mic_recorder

# ===================== 1. CONFIG =====================
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

# ===================== 3. DATA LOADING =====================
def safe_case(case):
    case.setdefault("block", "General")
    case.setdefault("difficulty", "medium")
    case.setdefault("must_exclude", ["Aortic Dissection", "PE"])
    case.setdefault("key_points", [])
    case.setdefault("labs", [])
    case.setdefault("scenario", {"en": "A 62-year-old male presents with acute substernal chest pain. EKG shows ST-elevation in II, III, aVF."})
    case.setdefault("answer", "Inferior Wall MI")
    case.setdefault("interprofessional_answers", {
        "medicine": "Primary PCI within 90 mins", 
        "pharmacy": "Aspirin 300mg + Heparin bolus",
        "nursing": "Monitor BP and Right-sided EKG"
    })
    return case

@st.cache_data
def load_cases():
    try:
        with open("cases.json","r",encoding="utf-8") as f:
            data = json.load(f)
            return [safe_case(c) for c in data]
    except FileNotFoundError:
        return [safe_case({})]

cases = load_cases()

# ===================== 4. SESSION STATE =====================
if "case" not in st.session_state:
    st.session_state.case = random.choice(cases)
if "submitted" not in st.session_state:
    st.session_state.submitted = False
if "voice_text" not in st.session_state:
    st.session_state.voice_text = ""

# ===================== 5. SIDEBAR =====================
with st.sidebar:
    st.title("🧠 ACLR Professional")
    page = st.radio("Navigation", ["📖 User Guide", "🧪 Clinical Simulator", "🏆 Leaderboard"])
    st.divider()
    user_id = st.text_input("👤 User ID", value="Doctor_X")
    profession = st.selectbox("👩‍⚕️ Your Role", ["medicine", "nursing", "pharmacy", "dentistry", "ams", "public health"])
    
    if page == "🧪 Clinical Simulator":
        if st.button("🔄 Next Random Case"):
            st.session_state.case = random.choice(cases)
            st.session_state.submitted = False
            st.session_state.voice_text = ""
            st.rerun()

# ===================== 6. PAGE LOGIC =====================

# --- PAGE 1: USER GUIDE (DETAILED) ---
if page == "📖 User Guide":
    st.header("📖 Clinical Reasoning Manual: ACLR Loop")
    
    st.subheader("🎯 วัตถุประสงค์ของระบบ (Educational Philosophy)")
    st.write("""
    ระบบ **ACLR (Advanced Clinical Reasoning)** ไม่ใช่เครื่องมือทำข้อสอบทั่วไป แต่เป็น **Cognitive Simulator** ที่ออกแบบมาเพื่อฝึกทักษะการตัดสินใจในสภาวะจำลอง โดยเน้นการดึงความรู้จากสมอง (Active Recall) และความปลอดภัยของผู้ป่วย (Patient Safety)
    """)

    col_guide1, col_guide2 = st.columns(2)
    with col_guide1:
        st.markdown("### 🛠 ขั้นตอนการใช้งาน")
        st.markdown("""
        1. **Information Gathering:** วิเคราะห์ Scenario และผล Lab อย่างละเอียด
        2. **Differential Diagnosis:** เลือกกลุ่มโรคที่ต้องเฝ้าระวัง (Red Flags)
        3. **Formulation:** พิมพ์วินิจฉัยและระบุเหตุผลทางพยาธิสภาพ
        4. **Calibration:** ประเมินความมั่นใจของตนเอง (Metacognition)
        5. **Team Debriefing:** เรียนรู้จากมุมมองสหสาขาวิชาชีพ
        """)
        
    
    with col_guide2:
        st.markdown("### 📊 Scoring Rubric (10 Points)")
        rubric_data = {
            "หมวดหมู่": ["Diagnosis", "Evidence", "Logic Flow", "Safety"],
            "คะแนน": ["5", "3", "2", "+/- Bonus"],
            "คำอธิบาย": [
                "ความถูกต้องตามมาตรฐานการวินิจฉัย",
                "การระบุ Keyword สำคัญในเหตุผล",
                "การใช้คำเชื่อมเหตุและผล (เนื่องจาก... จึง...)",
                "หักคะแนนหากลืม Red Flags หรือมั่นใจผิดที่"
            ]
        }
        st.table(pd.DataFrame(rubric_data))

    st.divider()
    st.subheader("👥 ระบบ Team Board (Interprofessional Insight)")
    st.write("หัวใจของการรักษาคือการทำงานเป็นทีม หลังส่งคำตอบคุณจะพบกับ Insight จากวิชาชีพอื่น:")
    col_t1, col_t2, col_t3 = st.columns(3)
    col_t1.info("**Medicine:** เป้าหมายการรักษาหลัก (Definitive Rx)")
    col_t2.success("**Pharmacy:** การบริหารยาและข้อควรระวัง (Drug Safety)")
    col_t3.warning("**Nursing:** การพยาบาลและการเฝ้าระวัง (Monitoring)")
    

# --- PAGE 2: SIMULATOR ---
elif page == "🧪 Clinical Simulator":
    case = st.session_state.case
    st.title(f"🏥 Station: {case['block']} ({case['difficulty'].upper()})")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        tab1, tab2, tab3 = st.tabs(["📋 Scenario", "🧪 Diagnostics", "✍️ Analysis & Decision"])
        with tab1:
            st.info(case["scenario"].get("en", ""))
            if case.get("image_url"):
                st.image(case["image_url"], caption="Clinical Presentation", use_container_width=True)
        with tab2:
            st.markdown("### 🔬 Laboratory Findings")
            if case["labs"]: st.table(pd.DataFrame(case["labs"]))
            else: st.write("No lab data provided.")
        with tab3:
            st.warning(f"**Task:** วินิจฉัยและอธิบายเหตุผลในบทบาท {profession.upper()}")
            audio = mic_recorder(start_prompt="🎙️ Record Summary", stop_prompt="Stop", key='recorder')
            if audio: st.session_state.voice_text = f"Input received (Suspected {case['answer']})"
            
            selected_ddx = st.multiselect("🔍 Must-Exclude Differential Diagnosis (Red Flags)", 
                                          ["MI", "PE", "Aortic Dissection", "Sepsis", "Pneumothorax", "GERD"])
            dx_input = st.text_input("🩺 Final Diagnosis", value=st.session_state.voice_text)
            reason_input = st.text_area("✍️ Pathophysiological Reasoning", placeholder="เช่น เนื่องจาก... จึงทำให้เกิด...")
            conf_input = st.slider("🎯 Confidence Level (%)", 0, 100, 50)
            
            if st.button("✅ Submit Decision"):
                if dx_input and reason_input:
                    st.session_state.submitted = True
                    st.rerun()
                else: st.error("Please fill in Diagnosis and Reasoning.")

    if st.session_state.submitted:
        st.divider()
        score, dx_s, r_s, d_s, s_s, target, used, level = evaluate_pro(dx_input, reason_input, case, profession, conf_input, selected_ddx)
        c_l, c_r = st.columns([2, 1])
        with c_l:
            st.subheader(f"📊 Assessment: {score}/10")
            st.progress(score * 10)
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Dx Accuracy", f"{dx_s}/5")
            m2.metric("Evidence", f"{r_s}/3")
            m3.metric("Logic Flow", f"{d_s}/2")
            m4.metric("Safety", "PASS" if s_s > 0 else "FAIL", delta=s_s)
            st.write(f"**Correct Dx:** `{target}`")
            if s_s < 0: st.error(f"❌ Safety Warning: You missed critical Red Flags: {', '.join(case['must_exclude'])}")
            if level == "wrong" and conf_input > 90: st.error("⚠️ Overconfidence Alert: Incorrect diagnosis with very high confidence is dangerous.")
        with c_r:
            st.subheader("👥 Team Feedback")
            for role, ans in case.get("interprofessional_answers", {}).items():
                with st.expander(f"From {role.upper()}"): st.write(ans)

# --- PAGE 3: LEADERBOARD ---
elif page == "🏆 Leaderboard":
    st.header("🏆 Performance Dashboard")
    st.info("Leaderboard will populate here based on responses.csv")
    st.dataframe(pd.DataFrame({"User": [user_id], "Block": ["Cardiovascular"], "Score": ["-"], "Status": ["In Progress"]}))

st.markdown("---")
st.caption("ACLR Professional v3.6 | Master Edition 2026 | Simulation-Based Clinical Education")
