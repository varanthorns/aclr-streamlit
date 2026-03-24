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
def evaluate_pro(dx, reasoning, plan, case, profession, confidence, selected_ddx):
    """ระบบประเมินผลประสิทธิภาพสูง ป้องกัน KeyError และประเมินครบทุกมิติ"""
    target_dx = case.get("interprofessional_answers", {}).get(profession, case.get("answer", ""))
    u_step = plan.get('step', 'Observation')
    u_dispo = plan.get('dispo', 'Admit General Ward')

    # 1. Dx Accuracy (5 pts)
    try:
        vec = TfidfVectorizer().fit_transform([str(dx).lower(), str(target_dx).lower()])
        sim = cosine_similarity(vec[0:1], vec[1:2])[0][0]
    except: sim = 0
    dx_score = 5 if sim > 0.75 else (3 if sim > 0.45 else 0)
    level = "correct" if sim > 0.75 else ("close" if sim > 0.45 else "wrong")

    # 2. Key Evidence & Logic (3 pts)
    found_keys = [k for k in case.get("key_points", []) if k.lower() in reasoning.lower()]
    logic_words = ["เพราะว่า", "เนื่องจาก", "ทำให้", "ส่งผลให้", "because", "therefore", "due to"]
    r_score = min(2, len(found_keys)) + (1 if any(w in reasoning.lower() for w in logic_words) else 0)

    # 3. Management Plan (2 pts)
    plan_score = 0
    if u_step == case.get("next_step_correct", "Observation"): plan_score += 1
    if u_dispo == case.get("dispo_correct", "Admit General Ward"): plan_score += 1

    # 4. Safety & Calibration
    must_exclude = case.get("must_exclude", [])
    safety_check = 1 if all(item in selected_ddx for item in must_exclude) else -1
    
    # Penalty Logic
    penalty = 0
    if level == "wrong" and u_dispo == "Discharge Home": penalty -= 2
    if level == "wrong" and confidence > 90: penalty -= 2 # Dangerous Overconfidence

    final_score = max(0, min(10, dx_score + r_score + plan_score + safety_check + penalty))
    return final_score, dx_score, r_score, plan_score, safety_check, target_dx, found_keys, level

# ===================== 3. DATA LOADING =====================
def safe_case(case):
    case.setdefault("block", "General")
    case.setdefault("difficulty", "medium")
    case.setdefault("must_exclude", [])
    case.setdefault("key_points", [])
    case.setdefault("labs", [])
    case.setdefault("scenario", {"en": "Case scenario not loaded."})
    case.setdefault("answer", "Unknown Diagnosis")
    case.setdefault("next_step_correct", "Observation")
    case.setdefault("dispo_correct", "Admit General Ward")
    case.setdefault("interprofessional_answers", {"medicine": "N/A", "pharmacy": "N/A", "nursing": "N/A"})
    return case

@st.cache_data
def load_cases():
    try:
        with open("cases.json", "r", encoding="utf-8") as f:
            data = json.load(f)
            return [safe_case(c) for c in data]
    except Exception:
        return [safe_case({"block":"Emergency", "difficulty":"easy", "answer":"Acute MI", "key_points":["chest pain", "ST elevation"]})]

cases = load_cases()

# ===================== 4. SESSION STATE =====================
if "case" not in st.session_state: st.session_state.case = cases[0]
if "submitted" not in st.session_state: st.session_state.submitted = False
if "voice_text" not in st.session_state: st.session_state.voice_text = ""
if "user_plan" not in st.session_state: st.session_state.user_plan = {"step": "Observation", "dispo": "Admit General Ward"}

# ===================== 5. SIDEBAR =====================
with st.sidebar:
    st.title("🧠 ACLR Professional")
    page = st.radio("เมนูหลัก", ["📖 User Guide", "🧪 Clinical Simulator", "🏆 Leaderboard"])
    st.divider()
    
    if page == "🧪 Clinical Simulator":
        st.subheader("🎯 ตั้งค่าสถานี")
        blocks = ["All"] + sorted(list(set(c['block'] for c in cases)))
        sel_block = st.selectbox("เลือก Block", blocks)
        sel_diff = st.select_slider("ระดับความยาก", options=["easy", "medium", "hard"])
        if st.button("🔄 สุ่มเคสใหม่"):
            filtered = [c for c in cases if (sel_block == "All" or c['block'] == sel_block) and (c['difficulty'] == sel_diff)]
            if filtered:
                st.session_state.case = random.choice(filtered)
                st.session_state.submitted = False
                st.session_state.voice_text = ""
                st.session_state.user_plan = {"step": "Observation", "dispo": "Admit General Ward"}
                st.rerun()
            else: st.warning("ไม่พบเคสในหมวดนี้")
    
    st.divider()
    user_id = st.text_input("👤 ID ผู้ใช้งาน", value="Doctor")
    profession = st.selectbox("👩‍⚕️ บทบาทของคุณ", ["medicine", "nursing", "pharmacy", "ams"])

# ===================== 6. PAGES =====================

# --- PAGE 1: USER GUIDE ---
if page == "📖 User Guide":
    st.header("📖 คู่มือการใช้งาน: ACLR Loop")
    st.subheader("🎯 วัตถุประสงค์ (Philosophy)")
    st.write("ระบบนี้คือ **Cognitive Simulator** เพื่อฝึกการตัดสินใจทางคลินิก โดยเน้นการดึงความรู้จริงออกมาใช้ (Active Recall) และการสร้าง Safety-First Mindset")
    
    col_u1, col_u2 = st.columns(2)
    with col_u1:
        st.markdown("### 🛠 ขั้นตอนการใช้งาน")
        st.markdown("""
        | ขั้นตอน | รายละเอียด |
        | :--- | :--- |
        | **1. Gathering** | วิเคราะห์ประวัติและผลแล็บอย่างละเอียด |
        | **2. DDx** | เลือกโรคอันตราย (Red Flags) ที่ต้องแยกออก |
        | **3. Decision** | วินิจฉัยและระบุเหตุผลพยาธิสภาพ |
        | **4. Action** | เลือก **Next Step** และ **Disposition** |
        | **5. Calibration** | ประเมินความมั่นใจของตนเอง |
        """)
        
    with col_u2:
        st.markdown("### 📊 เกณฑ์การประเมิน (10 คะแนน)")
        st.write("- **Dx Accuracy (5):** ความแม่นยำของการวินิจฉัย")
        st.write("- **Evidence & Logic (3):** การระบุจุดสำคัญและตรรกะ")
        st.write("- **Management (2):** ความเหมาะสมของแผนการจัดการ")
        st.error("⚠️ **Penalty:** หักคะแนนรุนแรงหากวินิจฉัยผิดแต่มั่นใจสูง (>90%) หรือปล่อยผู้ป่วยวิกฤตกลับบ้าน")

    st.divider()
    st.subheader("👥 ระบบสหสาขาวิชาชีพ (Team Board)")
    st.info("**Medicine:** เป้าหมายการรักษาหลัก | **Pharmacy:** ความปลอดภัยของยา | **Nursing:** การเฝ้าระวังผู้ป่วย")

# --- PAGE 2: SIMULATOR ---
elif page == "🧪 Clinical Simulator":
    case = st.session_state.case
    st.title(f"🏥 Station: {case['block']} | Level: {case['difficulty'].upper()}")
    
    col_main, col_side = st.columns([2, 1])
    with col_main:
        t1, t2, t3 = st.tabs(["📋 Scenario", "🧪 Diagnostics", "✍️ Analysis & Plan"])
        with t1:
            st.info(case["scenario"].get("en", ""))
            if case.get("image_url"): st.image(case["image_url"], use_container_width=True)
        with t2:
            if case["labs"]: st.table(pd.DataFrame(case["labs"]))
            else: st.write("ไม่มีข้อมูล Lab เพิ่มเติมสำหรับเคสนี้")
        with t3:
            st.warning(f"**Task:** วินิจฉัยและวางแผนในบทบาท {profession.upper()}")
            audio = mic_recorder(start_prompt="🎙️ สรุปด้วยเสียง (AI Preview)", stop_prompt="หยุดบันทึก", key='rec')
            if audio: st.session_state.voice_text = f"คาดว่าเป็น {case['answer']}"
            
            dx_in = st.text_input("🩺 การวินิจฉัย (Final Diagnosis)", value=st.session_state.voice_text)
            re_in = st.text_area("✍️ เหตุผลทางพยาธิสภาพ (Pathophysiological Reasoning)", placeholder="เช่น เนื่องจาก... จึงทำให้เกิด...")
            
            st.divider()
            st.subheader("🚀 Management Plan")
            p1, p2 = st.columns(2)
            with p1:
                next_step = st.selectbox("🎯 ขั้นตอนถัดไป", ["Observation", "Emergency Procedure", "Start Medication", "Diagnostic Imaging", "Consult Specialist", "Referral"])
            with p2:
                disposition = st.selectbox("🏥 การจัดการที่พำนัก", ["Discharge Home", "Admit General Ward", "Admit ICU/CCU", "Emergency Operation"])
            
            selected_ddx = st.multiselect("🔍 Must-Exclude (Red Flags)", ["MI", "PE", "Aortic Dissection", "Sepsis", "Stroke", "Pneumothorax"])
            conf_in = st.slider("🎯 ระดับความมั่นใจ (%)", 0, 100, 50)
            
            if st.button("✅ ยืนยันคำตอบ"):
                if dx_in and re_in:
                    st.session_state.user_plan = {"step": next_step, "dispo": disposition}
                    st.session_state.submitted = True
                    st.rerun()
                else: st.error("กรุณากรอกข้อมูลวินิจฉัยและเหตุผลให้ครบถ้วน")

    if st.session_state.submitted:
        st.divider()
        sc, dx_s, r_s, p_s, s_s, target, used, level = evaluate_pro(dx_in, re_in, st.session_state.user_plan, case, profession, conf_in, selected_ddx)
        
        res_l, res_r = st.columns([2, 1])
        with res_l:
            st.subheader(f"📊 ผลการประเมิน: {sc}/10 คะแนน")
            st.progress(sc * 10)
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Diagnosis", f"{dx_s}/5"); m2.metric("Reasoning", f"{r_s}/3")
            m3.metric("Plan", f"{p_s}/2"); m4.metric("Safety", "PASS" if s_s > 0 else "FAIL", delta=s_s)
            
            st.write(f"✅ **การวินิจฉัยที่ถูกต้อง:** `{target}`")
            st.write(f"➡️ **แผนของคุณ:** {st.session_state.user_plan['step']} ภายใน {st.session_state.user_plan['dispo']}")
            if level == "wrong" and st.session_state.user_plan['dispo'] == "Discharge Home":
                st.error("❌ **Critical Safety Error:** การปล่อยผู้ป่วยที่มีความเสี่ยงกลับบ้านเป็นความเสี่ยงระดับวิกฤต")
        with res_r:
            st.subheader("👥 ความเห็นทีมสหสาขา")
            for role, ans in case.get("interprofessional_answers", {}).items():
                with st.expander(f"มุมมองจาก {role.upper()}"): st.write(ans)

elif page == "🏆 Leaderboard":
    st.header("🏆 Performance Dashboard")
    st.info("ระบบกำลังเตรียมความพร้อมสำหรับการจัดอันดับผู้ใช้งาน")

st.markdown("---")
st.caption("ACLR Professional v4.2 Stable | Built for Clinical Excellence 2026")
