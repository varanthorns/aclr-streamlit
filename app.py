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
def evaluate_pro(dx, reasoning, case, profession, confidence):
    target = case.get("interprofessional_answers", {}).get(profession, case.get("answer", ""))
    
    # 1. Accuracy Score (5 pts) - Semantic Similarity
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

    # 4. Confidence Calibration (Bonus/Penalty)
    bonus = 0
    if level == "correct" and confidence > 80:
        bonus = 1 
    elif level == "wrong" and confidence > 90:
        total = max(0, total - 2) # Dangerous Overconfidence Penalty

    final_score = min(10, total + bonus)
    return final_score, dx_score, r_score, d_score, target, found_keys, level

# ===================== UTILS & LOAD =====================
def safe_case(case):
    case.setdefault("block", "General")
    case.setdefault("difficulty", "medium")
    case.setdefault("task", {})
    case.setdefault("interprofessional_answers", {})
    case.setdefault("reference", {"source":"Unknown","year":"2026"})
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
            "scenario":{"en":"A 62-year-old male presents with acute substernal chest pain. EKG shows ST-elevation in II, III, aVF."}, 
            "image_url": "https://p0.pikist.com/photos/403/619/ecg-heart-rate-frequency-medical-medicine-health-science-curve-pulse.jpg",
            "labs": [{"Test": "Troponin T", "Result": "450", "Unit": "ng/L", "Range": "< 14"}],
            "answer":"Inferior Wall MI", 
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
    st.title(" ACLR Menu")
    page = st.radio("Go to Page", ["📖 User Guide & Scoring", "🧪 Clinical Simulator", "🏆 Leaderboard"])
    
    st.divider()
    user = st.text_input("👤 User ID / Name", value="Doctor")
    if not user: st.stop()
    
    profession = st.selectbox("👩‍⚕️ Your Role", ["medicine", "dentistry", "nursing","pharmacy","ams","public health", "veterinarian"])
    
    if page == "🧪 Clinical Simulator":
        st.header("⚙️ Station Settings")
        mode = st.radio("Mode", ["Practice", "OSCE (Timed)", "Battle (Leaderboard)"])
        block_choice = st.selectbox("📚 Select Block", ["All"] + list(set(c["block"] for c in cases)))
        diff_choice = st.selectbox("🎯 Difficulty", ["easy","medium","hard"])

        if st.button("🔄 Next Case"):
            filtered = [c for c in cases if (block_choice == "All" or c["block"] == block_choice) and c["difficulty"] == diff_choice]
            st.session_state.case = random.choice(filtered) if filtered else random.choice(cases)
            st.session_state.submitted = False
            st.session_state.voice_text = ""
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
        """)
        st.markdown("""
        * **Production Over Recognition:** ต้องพิมพ์คำตอบเอง เพื่อฝึกการดึงข้อมูลจากสมอง
        * **Confidence Calibration:** ตรวจสอบความมั่นใจควบคู่ความถูกต้อง เพื่อความปลอดภัยของผู้ป่วย
        * **Logic Extraction:** ระบบตรวจจับ 'ตรรกะ' ของการเชื่อมโยงอาการสู่โรค
        """)

    with col_b:
        st.subheader("🛠 วิธีใช้งาน")
        st.write("""
        1. **Study Scenario:** อ่านสถานการณ์และวิเคราะห์ผลตรวจใน Tab 1 และ 2
        2. **Formulate Dx:** พิมพ์วินิจฉัยโรค และระบุเหตุผลพยาธิสภาพ
        3. **Confidence:** เลือกความมั่นใจของคุณ (มีผลต่อคะแนนความปลอดภัย)
        4. **Submit:** รับผลการประเมินและ Feedback จาก AI ทันที
        """)

    st.divider()
    st.subheader("📊 เกณฑ์การให้คะแนน (Scoring Rubric - 10 Points)")
    rubric_data = {
        "หมวดหมู่": ["Diagnosis Accuracy", "Evidence Key Points", "Clinical Logic", "Confidence Bonus/Penalty"],
        "คะแนน": ["5", "3", "2", "+1 / -2"],
        "เกณฑ์การประเมิน": [
            "ความถูกต้องของโรค (Semantic AI Check)",
            "การระบุหลักฐานสำคัญในช่องเหตุผล",
            "การใช้คำเชื่อมเหตุและผล (เช่น เนื่องจาก... ส่งผลให้...)",
            "วินิจฉัยถูก+มั่นใจสูง = โบนัส | วินิจฉัยผิด+มั่นใจสูง = หักคะแนน"
        ]
    }
    st.table(pd.DataFrame(rubric_data))
    st.info("💡 **Clinical Reasoning Tip:** พยายามเชื่อมโยงอาการเข้ากับพยาธิสภาพโดยใช้คำเชื่อม เพื่อคะแนนตรรกะที่สูงขึ้น")

# ===================== PAGE 2: SIMULATOR =====================
elif page == "🧪 Clinical Simulator":
    st.title("🧠 Clinical Simulator")
    case = st.session_state.case
    col1, col2 = st.columns([2, 1])

    with col1:
        tab1, tab2, tab3 = st.tabs(["📋 Scenario", "🧪 Diagnostics", "✍️ Analysis"])
        
        with tab1:
            st.info(case["scenario"].get("en", ""))
            if case["image_url"]:
                st.image(case["image_url"], caption="Clinical Imaging / EKG", use_container_width=True)
                
        with tab2:
            st.markdown("### 🔬 Laboratory Findings")
            if case["labs"]: st.table(pd.DataFrame(case["labs"]))
            else: st.write("No specific labs provided.")

        with tab3:
            st.warning(f"**Task:** {case.get('task', {}).get(profession, 'Diagnose and provide reasoning.')}")
            
            audio = mic_recorder(start_prompt="🎙️ Click to Speak (OSCE Mode)", stop_prompt="Stop Recording", key='recorder')
            if audio:
                st.session_state.voice_text = "Audio recorded (Simulated transcription: " + case["answer"] + " based on findings)"
            
            ddx = st.multiselect("🔍 Differential Diagnosis", ["MI", "PE", "Sepsis", "Pneumonia", "Aortic Dissection", "Other"])
            dx = st.text_input("🩺 Final Diagnosis", value=st.session_state.voice_text if st.session_state.voice_text else "")
            reasoning = st.text_area("✍️ Pathophysiological Reasoning", height=100, placeholder="อธิบายเหตุผลที่วินิจฉัยเช่นนี้...")
            
            conf = st.slider("🎯 Confidence Level (%)", 0, 100, 50)
            
            c1, c2 = st.columns(2)
            with c1: next_step = st.selectbox("🚀 Next Best Step", ["Observation", "Emergency Surgery", "Empirical Meds", "Referral"])
            with c2: dispo = st.selectbox("🏥 Disposition", ["Home", "Ward", "ICU", "OR"])

            if st.button("✅ Submit Decision"):
                if dx and reasoning:
                    st.session_state.submitted = True
                    total, dx_s, r_s, d_s, target, used, level = evaluate_pro(dx, reasoning, case, profession, conf)
                    
                    # บันทึกประวัติ
                    res_df = pd.DataFrame([{"user": user, "block": case["block"], "score": total, "time": datetime.now()}])
                    try:
                        old = pd.read_csv("responses.csv")
                        res_df = pd.concat([old, res_df])
                    except: pass
                    res_df.to_csv("responses.csv", index=False)
                    st.rerun()
                else:
                    st.error("กรุณากรอกข้อมูลวินิจฉัยและเหตุผลก่อนส่ง")

    if st.session_state.submitted:
        st.divider()
        total, dx_s, r_s, d_s, target, used, level = evaluate_pro(dx, reasoning, case, profession, conf)
        
        c_res, c_ipa = st.columns([2, 1])
        with c_res:
            st.markdown(f"### 📊 Score: {total}/10")
            st.progress(total * 10)
            
            r1, r2, r3 = st.columns(3)
            r1.metric("Dx Accuracy", f"{dx_s}/5")
            r2.metric("Evidence", f"{r_s}/3")
            r3.metric("Logic Flow", f"{d_s}/2")
            
            st.markdown(f"**Correct Answer:** `{target}`")
            st.write("**Key Points Detected:**", used)
            if level == "wrong" and conf > 90:
                st.error("⚠️ Warning: Dangerous Overconfidence Detected! (วินิจฉัยผิดแต่ความมั่นใจสูงเกินไป)")

        with c_ipa:
            st.markdown("### 👥 Interprofessional Insight")
            ipa = case.get("interprofessional_answers", {})
            for role, ans in ipa.items():
                with st.expander(f"Insight from {role.upper()}", expanded=True):
                    st.write(ans)

# ===================== PAGE 3: LEADERBOARD =====================
elif page == "🏆 Leaderboard":
    st.header("🏆 Performance Dashboard")
    try:
        df = pd.read_csv("responses.csv")
        st.subheader("Average Score by Block")
        st.bar_chart(df.groupby("block")["score"].mean())
        st.subheader("Recent History")
        st.dataframe(df.sort_values(by="time", ascending=False), use_container_width=True)
    except:
        st.info("ยังไม่มีข้อมูลการเล่น เริ่มจำลองสถานการณ์เพื่อสร้างสถิติ!")

st.markdown("---")
st.caption("ACLR Professional v3.5 | 2026 Academic Edition")


