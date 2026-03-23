import streamlit as st
import json, random, pandas as pd, time
from datetime import datetime
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ตรวจสอบการติดตั้ง Library สำหรับบันทึกเสียง
try:
    from streamlit_mic_recorder import mic_recorder
    VOICE_AVAILABLE = True
except ImportError:
    VOICE_AVAILABLE = False

# ===================== CONFIG & LOAD =====================
st.set_page_config(layout="wide", page_title="ACLR Clinical Reasoning Ultimate")

def safe_case(case):
    case.setdefault("block", "General")
    case.setdefault("difficulty", "medium")
    case.setdefault("task", {})
    case.setdefault("interprofessional_answers", {})
    case.setdefault("reference", {"source":"Guidelines 2026","year":"2026"})
    case.setdefault("key_points", [])
    case.setdefault("labs", [])
    case.setdefault("image_url", None)
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
            "scenario":{"en":"A 62-year-old male with substernal chest pressure. EKG shows ST-elevation in II, III, aVF."}, 
            "image_url": "https://img.medscapestatic.com/pi/meds/qref/84/11984.jpg",
            "labs": [{"Test": "Troponin T", "Result": "450", "Unit": "ng/L", "Range": "< 14"}],
            "answer":"Inferior Wall MI", 
            "key_points":["ST-elevation", "inferior", "chest pain"],
            "interprofessional_answers": {"medicine": "Primary PCI", "pharmacy": "Aspirin + Heparin"}
        })]

cases = load_cases()

# ===================== EVALUATION LOGIC =====================
def evaluate(dx, reasoning, case, profession):
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

    total = dx_score + r_score + d_score
    return total, dx_score, r_score, d_score, target, found_keys, level

# ===================== SIDEBAR NAVIGATION =====================
with st.sidebar:
    st.title("🚀 ACLR Menu")
    page = st.radio("Go to Page", ["📖 User Guide & Scoring", "🧪 Clinical Simulator", "🏆 Leaderboard"])
    st.divider()
    user = st.text_input("👤 User ID", value="Doctor_X")
    profession = st.selectbox("👩‍⚕️ Your Role", ["medicine", "dentistry", "nursing","pharmacy","ams","public health", "veterinarian"])
    
    if page == "🧪 Clinical Simulator":
        st.header("⚙️ Station Settings")
        mode = st.radio("Mode", ["Practice", "OSCE (Timed)", "Battle"])
        block_choice = st.selectbox("📚 Block", ["All"] + list(set(c["block"] for c in cases)))
        if st.button("🔄 Next Case"):
            filtered = [c for c in cases if block_choice == "All" or c["block"] == block_choice]
            st.session_state.case = random.choice(filtered)
            st.session_state.submitted = False
            st.rerun()

# ===================== PAGE 1: USER GUIDE =====================
if page == "📖 User Guide & Scoring":
    st.header("📖 วิธีการใช้งานและเกณฑ์การประเมิน (Manual)")
    
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("💡 ACLR คืออะไร?")
        st.write("""
        **ACLR (Advanced Clinical Reasoning)** ออกแบบมาเพื่อก้าวข้ามการสอบแบบตัวเลือก (MCQ) 
        โดยเน้นการฝึก **Clinical Reasoning** หรือกระบวนการตัดสินใจทางคลินิกเสมือนจริง 
        แก้ปัญหา 'สอบผ่านแต่รักษาไม่ได้' หรือ 'วินิจฉัยถูกแต่ให้เหตุผลไม่ได้'
        """)
        st.markdown("""
        * **Production Over Recognition:** ต้องพิมพ์คำตอบเอง ไม่มีการตัดช้อยส์
        * **Interprofessional Collaboration:** เห็นมุมมองของวิชาชีพอื่นในทีมสุขภาพ
        * **Logic Extraction:** ระบบตรวจจับ 'ตรรกะ' ของการเชื่อมโยงอาการสู่โรค
        """)

    with col_b:
        st.subheader("🛠 วิธีใช้งาน")
        st.write("""
        1. **Study Scenario:** อ่านสถานการณ์ใน Tab 1 (Scenario)
        2. **Analyze Labs:** วิเคราะห์ผลแล็บและภาพ EKG/X-ray ใน Tab 2
        3. **Formulate Dx:** ระบุโรคที่เป็นไปได้มากที่สุด (Differential Diagnosis)
        4. **Write Reasoning:** อธิบายพยาธิสภาพว่าทำไมถึงคิดว่าเป็นโรคนั้น
        5. **Submit:** รับ Feedback ทันทีจาก AI Examiner
        """)

    st.divider()
    st.subheader("📊 เกณฑ์การให้คะแนน (Scoring Rubric - 10 Points)")
    rubric = {
        "หมวดหมู่": ["Diagnosis Accuracy", "Evidence Key Points", "Clinical Logic flow"],
        "คะแนน": ["5 คะแนน", "3 คะแนน", "2 คะแนน"],
        "เกณฑ์การประเมิน": [
            "ความถูกต้องของชื่อโรค (ใช้ AI วัดความใกล้เคียงของความหมาย)",
            "การระบุหลักฐานสำคัญจากอาการหรือผลแล็บที่โจทย์กำหนด (Key Features)",
            "การใช้คำเชื่อมแสดงเหตุและผล (เช่น 'เนื่องจาก...', 'ส่งผลให้...')"
        ]
    }
    st.table(pd.DataFrame(rubric))
    

# ===================== PAGE 2: SIMULATOR =====================
elif page == "🧪 Clinical Simulator":
    if "case" not in st.session_state:
        st.session_state.case = random.choice(cases)
    if "submitted" not in st.session_state:
        st.session_state.submitted = False

    case = st.session_state.case
    col1, col2 = st.columns([2, 1])

    with col1:
        tab1, tab2, tab3 = st.tabs(["📋 Scenario", "🧪 Diagnostics", "✍️ Answer Sheet"])
        
        with tab1:
            st.info(case["scenario"].get("en", ""))
            if case["image_url"]:
                st.image(case["image_url"], caption="Clinical Investigation", use_container_width=True)
                
        with tab2:
            st.markdown("### 🔬 Laboratory Findings")
            if case["labs"]: st.table(pd.DataFrame(case["labs"]))
            else: st.write("No specific labs provided.")

        with tab3:
            st.warning(f"**Task:** {case.get('task', {}).get(profession, 'Establish diagnosis and management.')}")
            
            # Voice Mode
            if VOICE_AVAILABLE:
                st.markdown("##### 🎙️ Voice SBAR Reporting")
                mic_recorder(start_prompt="Speak your report", stop_prompt="Stop", key='recorder')
            
            # Inputs
            ddx = st.multiselect("🔍 Differential Diagnosis", ["MI", "PE", "Sepsis", "Pneumonia", "Other"])
            dx = st.text_input("🩺 Final Diagnosis / Management Plan")
            reasoning = st.text_area("✍️ Clinical Reasoning (เชื่อมโยงหลักฐานเข้ากับพยาธิสภาพ)", height=150)
            
            row1_col1, row1_col2 = st.columns(2)
            with row1_col1: next_step = st.selectbox("🚀 Next Best Step", ["PCI", "Surgery", "Meds", "Refer"])
            with row1_col2: dispo = st.selectbox("🏥 Disposition", ["Home", "Ward", "ICU", "OR"])
            
            if st.button("✅ Submit Decision"):
                if dx and reasoning:
                    st.session_state.submitted = True
                    total, dx_s, r_s, d_s, target, used, level = evaluate(dx, reasoning, case, profession)
                    
                    # Save to History
                    res_df = pd.DataFrame([{"user": user, "block": case["block"], "score": total, "time": datetime.now()}])
                    try:
                        old = pd.read_csv("responses.csv")
                        res_df = pd.concat([old, res_df])
                    except: pass
                    res_df.to_csv("responses.csv", index=False)
                    st.rerun()

        # Display Result
        if st.session_state.submitted:
            st.divider()
            total, dx_s, r_s, d_s, target, used, level = evaluate(dx, reasoning, case, profession)
            st.markdown(f"### 📊 Result: {total}/10")
            st.progress(total * 10)
            
            c_a, c_b, c_c = st.columns(3)
            c_a.metric("Accuracy", f"{dx_s}/5")
            c_b.metric("Evidence", f"{r_s}/3")
            c_c.metric("Logic", f"{d_s}/2")
            
            st.markdown(f"**Gold Standard:** `{target}`")
            st.write("**Key Evidence Identified:**", used)

    with col2:
        st.subheader("👥 Interprofessional Insight")
        if st.session_state.submitted:
            ipa = case.get("interprofessional_answers", {})
            for role, ans in ipa.items():
                with st.expander(f"Insight from {role.upper()}", expanded=True):
                    st.write(ans)
        else:
            st.info("🔒 Decision required to unlock team insights.")

# ===================== PAGE 3: LEADERBOARD =====================
elif page == "🏆 Leaderboard":
    st.header("🏆 Performance Analytics")
    try:
        df = pd.read_csv("responses.csv")
        st.subheader("Average Score by Block")
        st.bar_chart(df.groupby("block")["score"].mean())
        st.subheader("Recent Activity")
        st.dataframe(df.sort_values(by="time", ascending=False), use_container_width=True)
    except:
        st.info("No records found. Start your first case!")

st.markdown("---")
st.caption("ACLR Engine v3.0 | 2026 Academic Edition")
