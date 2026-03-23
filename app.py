import streamlit as st
import json, random, pandas as pd, time
from datetime import datetime
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ตรวจสอบ Library บันทึกเสียง
try:
    from streamlit_mic_recorder import mic_recorder
    VOICE_AVAILABLE = True
except ImportError:
    VOICE_AVAILABLE = False

# ===================== CONFIG & CSS =====================
st.set_page_config(layout="wide", page_title="ACLR Ultimate Clinical Reasoning")

# ตกแต่ง UI เล็กน้อย
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    .stTabs [data-baseweb="tab"] { height: 50px; white-space: pre-wrap; background-color: #ffffff; border-radius: 4px 4px 0 0; gap: 1px; padding-top: 10px; }
    </style>
    """, unsafe_allow_html=True)

# ===================== CORE FUNCTIONS =====================
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
    except:
        return [safe_case({
            "block":"Cardiovascular", "difficulty":"hard", 
            "scenario":{"en":"A 62-year-old male presents with acute substernal chest pain..."}, 
            "answer":"Inferior Wall MI", "key_points":["ST-elevation", "inferior"],
            "interprofessional_answers": {"medicine": "Primary PCI", "pharmacy": "Aspirin"}
        })]

def evaluate_detailed(dx, reasoning, case, profession):
    target = case.get("interprofessional_answers", {}).get(profession, case.get("answer", ""))
    
    # 1. Dx Accuracy (5 pts)
    try:
        vec = TfidfVectorizer().fit_transform([str(dx).lower(), str(target).lower()])
        sim = cosine_similarity(vec[0:1], vec[1:2])[0][0]
    except: sim = 0
    dx_score = 5 if sim > 0.75 else (3 if sim > 0.45 else 0)

    # 2. Key Evidence (3 pts)
    found_keys = [k for k in case.get("key_points", []) if k.lower() in reasoning.lower()]
    key_score = min(3, len(found_keys))

    # 3. Clinical Logic (2 pts)
    logic_words = ["because", "therefore", "thus", "due to", "เนื่องจาก", "ดังนั้น", "ทำให้"]
    logic_score = 2 if any(w in reasoning.lower() for w in logic_words) else 0

    total = dx_score + key_score + logic_score
    return total, dx_score, key_score, logic_score, target, found_keys

# ===================== SIDEBAR NAVIGATION =====================
with st.sidebar:
    st.title("🚀 ACLR Menu")
    page = st.radio("Go to", ["📖 User Guide & Rubric", "🧪 Clinical Simulation", "🏆 Leaderboard"])
    st.divider()
    user_id = st.text_input("👤 User ID", value="Doctor_Alpha")
    profession = st.selectbox("👩‍⚕️ Profession", ["medicine","nursing","pharmacy","ams","public_health"])

# ===================== PAGE 1: USER GUIDE =====================
if page == "📖 User Guide & Rubric":
    st.header("📖 วิธีการใช้งานและเกณฑ์การประเมิน (ACLR Guide)")
    
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("🎯 วัตถุประสงค์")
        st.write("""
        ACLR ไม่ใช่แค่การตอบคำถามถูกหรือผิด แต่เป็นการฝึก **Clinical Reasoning (การให้เหตุผลทางคลินิก)** ผ่านสถานการณ์จำลองที่ต้องทำงานร่วมกับสหสาขาวิชาชีพ โดยเน้น 3 ส่วนหลัก:
        1. **Data Synthesis**: การรวบรวมข้อมูลจากประวัติ ตรวจร่างกาย และ Lab
        2. **Logical Integration**: การเชื่อมโยงหลักพยาธิสรีรวิทยาเข้ากับการวินิจฉัย
        3. **Interprofessional Communication**: การมองภาพรวมของทีมรักษา
        """)
    
    with col_b:
        st.subheader("🛠 ขั้นตอนการใช้งาน")
        st.info("""
        1. **Read Scenario**: อ่านเคสและวิเคราะห์ข้อมูลใน Tab 1 และ 2
        2. **Interpret Labs/Images**: วิเคราะห์ผลเลือดหรือภาพถ่ายทางรังสีที่ปรากฏ
        3. **Draft DDx**: ระบุการวินิจฉัยแยกโรค (Differential Diagnosis)
        4. **Final Decision**: พิมพ์คำวินิจฉัยและ 'เหตุผล' (สำคัญที่สุด)
        5. **SBAR Voice**: (ถ้ามีไมค์) ฝึกพูดรายงานเคสสั้นๆ เพื่อจำลองสถานการณ์จริง
        """)

    st.divider()
    st.subheader("📊 เกณฑ์การให้คะแนน (Scoring Rubric - Total 10 Points)")
    
    rubric_data = {
        "หมวดหมู่": ["Diagnosis Accuracy", "Key Clinical Features", "Clinical Logic", "Management Step", "Disposition"],
        "คะแนนเต็ม": [5, 3, 2, "Bonus", "Bonus"],
        "เกณฑ์การพิจารณา": [
            "ความถูกต้องของชื่อโรคตามมาตรฐานสากล (Semantic Similarity > 75%)",
            "การระบุหลักฐานสำคัญจากโจทย์ (Key points) ในช่องเหตุผล",
            "การใช้คำเชื่อมแสดงตรรกะ (Causal linkage) เช่น 'เนื่องจาก...', 'ส่งผลให้...'",
            "การเลือกขั้นตอนถัดไปที่เหมาะสมที่สุด (Next Best Step)",
            "การเลือกสถานที่ส่งต่อผู้ป่วยที่เหมาะสม (Disposition)"
        ]
    }
    st.table(pd.DataFrame(rubric_data))

# ===================== PAGE 2: SIMULATION =====================
elif page == "🧪 Clinical Simulation":
    if "case" not in st.session_state:
        st.session_state.case = random.choice(load_cases())
    if "submitted" not in st.session_state:
        st.session_state.submitted = False

    case = st.session_state.case
    
    c1, c2 = st.columns([2, 1])
    with c1:
        tab1, tab2, tab3 = st.tabs(["📋 Scenario & Imaging", "🧪 Lab Results", "✍️ Answer Sheet"])
        
        with tab1:
            st.info(case["scenario"].get("en", ""))
            if case["image_url"]: st.image(case["image_url"], use_container_width=True)
                
        with tab2:
            if case["labs"]: st.table(pd.DataFrame(case["labs"]))
            else: st.write("No specific labs provided.")

        with tab3:
            st.warning(f"**Task for {profession.upper()}:** {case.get('task', {}).get(profession, 'Analyze and provide final decision.')}")
            
            # --- Input Section ---
            ddx = st.multiselect("🔍 Differential Diagnosis", ["MI", "PE", "Sepsis", "Pneumonia", "Other"])
            dx = st.text_input("🩺 Final Diagnosis (พิมพ์ชื่อโรคที่วินิจฉัย)")
            reasoning = st.text_area("✍️ Clinical Reasoning (อธิบายเหตุผลและพยาธิสภาพ)", height=150, 
                                     placeholder="ระบุเหตุผลที่เลือกโรคนี้ และหลักฐานที่สนับสนุนจาก Lab/Scenario...")
            
            col_in1, col_in2 = st.columns(2)
            with col_in1: n_step = st.selectbox("🚀 Next Best Step", ["PCI", "IV Fluids", "Antibiotics", "Refer"])
            with col_in2: dispo = st.selectbox("🏥 Disposition", ["Home", "Ward", "ICU", "OR"])

            if st.button("✅ Submit Decision"):
                if dx and reasoning:
                    st.session_state.submitted = True
                    score, d_s, k_s, l_s, target, found = evaluate_detailed(dx, reasoning, case, profession)
                    
                    st.divider()
                    st.header(f"🏆 Result: {score}/10")
                    r_col1, r_col2, r_col3 = st.columns(3)
                    r_col1.metric("Dx Accuracy", f"{d_s}/5")
                    r_col2.metric("Evidence Found", f"{k_s}/3")
                    r_col3.metric("Logic Flow", f"{l_s}/2")
                    
                    st.success(f"**Correct Diagnosis:** {target}")
                    st.write("**Key Evidence Identified:**", found)
                    
                    # บันทึกลงระบบ
                    res_df = pd.DataFrame([{"user": user_id, "block": case["block"], "score": score, "time": datetime.now()}])
                    res_df.to_csv("responses.csv", mode='a', header=not pd.io.common.file_exists("responses.csv"), index=False)
                else:
                    st.error("กรุณากรอกข้อมูลให้ครบถ้วนก่อนส่ง")

    with c2:
        st.subheader("👥 Interprofessional Insight")
        if st.session_state.submitted:
            ipa = case.get("interprofessional_answers", {})
            for role, ans in ipa.items():
                st.info(f"**{role.upper()}**: {ans}")
        else:
            st.caption("ระบบจะปลดล็อคคำตอบของทีมเมื่อคุณส่งคำตอบแล้ว")

# ===================== PAGE 3: LEADERBOARD =====================
elif page == "🏆 Leaderboard":
    st.header("🏆 Performance Dashboard")
    try:
        df = pd.read_csv("responses.csv")
        st.subheader("Top Performers by Block")
        st.bar_chart(df.groupby("block")["score"].mean())
        st.subheader("Recent History")
        st.dataframe(df.sort_values(by="time", ascending=False), use_container_width=True)
    except:
        st.info("No data available yet. Start a simulation!")

st.divider()
st.caption("ACLR Engine v3.2 | Built for Multi-disciplinary Excellence")
