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
    """Robust evaluation system to prevent KeyErrors and assess all dimensions."""
    target_dx = case.get("interprofessional_answers", {}).get(profession, case.get("answer", ""))
    u_step = plan.get('step', 'N/A')
    u_dispo = plan.get('dispo', 'N/A')

    # 1. Dx Accuracy (5 pts)
    try:
        vec = TfidfVectorizer().fit_transform([str(dx).lower(), str(target_dx).lower()])
        sim = cosine_similarity(vec[0:1], vec[1:2])[0][0]
    except: sim = 0
    dx_score = 5 if sim > 0.75 else (3 if sim > 0.45 else 0)
    level = "correct" if sim > 0.75 else ("close" if sim > 0.45 else "wrong")

    # 2. Key Evidence & Logic (3 pts)
    found_keys = [k for k in case.get("key_points", []) if k.lower() in reasoning.lower()]
    logic_words = ["because", "therefore", "due to", "result in", "leads to", "since"]
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
    if level == "wrong" and confidence > 90: penalty -= 2 

    final_score = max(0, min(10, dx_score + r_score + plan_score + safety_check + penalty))
    return final_score, dx_score, r_score, plan_score, safety_check, target_dx, level

# ===================== 3. DATA LOADING =====================
@st.cache_data
def load_cases():
    try:
        with open("cases.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except:
        return [{"block":"Emergency", "difficulty":"easy", "scenario":{"en":"Sample: Chest pain case."}, "answer":"Acute MI", "key_points":["pain"], "next_step_correct":"Emergency Procedure", "dispo_correct":"Admit ICU/CCU"}]

cases = load_cases()

# ===================== 4. SESSION STATE =====================
if "case" not in st.session_state: st.session_state.case = cases[0]
if "submitted" not in st.session_state: st.session_state.submitted = False
if "user_plan" not in st.session_state: st.session_state.user_plan = {"step": "Waiting", "dispo": "Waiting"}
if "voice_text" not in st.session_state: st.session_state.voice_text = ""

# ===================== 5. SIDEBAR =====================
with st.sidebar:
    st.title("🧠 ACLR Professional")
    page = st.radio("Main Menu", ["📖 User Guide", "🧪 Clinical Simulator", "🏆 Leaderboard"])
    st.divider()
    
    if page == "🧪 Clinical Simulator":
        st.subheader("🎯 Station Control")
        all_blocks = ["All"] + sorted(list(set(c.get('block', 'General') for c in cases)))
        sel_block = st.selectbox("Select Block", all_blocks)
        sel_diff = st.select_slider("Select Difficulty", options=["easy", "medium", "hard"])
        
        if st.button("🔄 Generate New Case"):
            filtered = [c for c in cases if (sel_block == "All" or c.get('block') == sel_block) and (c.get('difficulty') == sel_diff)]
            if filtered:
                st.session_state.case = random.choice(filtered)
                st.session_state.submitted = False
                st.session_state.user_plan = {"step": "Waiting", "dispo": "Waiting"}
                st.session_state.voice_text = ""
                st.rerun()
            else: st.warning("No cases match this criteria.")
    
    st.divider()
    profession = st.selectbox("👩‍⚕️ Your Role", ["medicine", "nursing", "pharmacy", "ams"])

# ===================== 6. PAGES =====================

if page == "📖 User Guide":
    st.header("📖 Clinical Reasoning Manual: ACLR Loop")
    st.subheader("🎯 Educational Philosophy")
    st.write("This **Cognitive Simulator** is designed to train clinical decision-making through **Active Recall** and a **Safety-First Mindset**.")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 🛠 Workflow")
        st.markdown("""
        1. **Information Gathering:** Analyze the Scenario and Diagnostics.
        2. **Must-Exclude (Red Flags):** Identify life-threatening conditions.
        3. **Formulation:** Provide Diagnosis and Pathophysiological Reasoning.
        4. **Action Plan:** Select Next Step and Patient Disposition.
        5. **Calibration:** Rate your confidence level.
        """)
    with col2:
        st.markdown("### 📊 Scoring Rubric (10 Pts)")
        st.write("- **Diagnosis (5):** Accuracy of the primary condition.")
        st.write("- **Reasoning (3):** Keywords and clinical logic.")
        st.write("- **Management (2):** Appropriateness of the Action Plan.")
        st.error("⚠️ **Penalty:** Significant deduction for high-confidence errors or unsafe discharge of critical patients.")

elif page == "🧪 Clinical Simulator":
    case = st.session_state.case
    st.title(f"🏥 Station: {case.get('block', 'General')} | Level: {case.get('difficulty', 'medium').upper()}")
    
    col_main, col_side = st.columns([2, 1])
    with col_main:
        t1, t2, t3 = st.tabs(["📋 Scenario", "🧪 Diagnostics", "✍️ Analysis & Action"])
        with t1:
            st.info(case.get("scenario", {}).get("en", "No scenario info available."))
            if case.get("image_url"): st.image(case["image_url"])
        with t2:
            if case.get("labs"): st.table(pd.DataFrame(case["labs"]))
            else: st.write("No diagnostic data available.")
        with t3:
            st.warning(f"**Task:** Diagnose and Manage as **{profession.upper()}**")
            
            # Voice Recording Feature
            audio = mic_recorder(start_prompt="🎙️ Record Summary", stop_prompt="Stop Recording", key='rec')
            if audio: st.session_state.voice_text = f"Suspected {case['answer']}"
            
            dx_in = st.text_input("🩺 Final Diagnosis", value=st.session_state.voice_text)
            re_in = st.text_area("✍️ Pathophysiological Reasoning", placeholder="Explain the mechanism (e.g., 'Due to... results in...')")
            
            st.divider()
            st.subheader("🚀 Management Plan")
            p1, p2 = st.columns(2)
            with p1:
                next_step = st.selectbox("🎯 Next Immediate Step", ["Observation", "Emergency Procedure", "Start Medication", "Diagnostic Imaging", "Consult Specialist", "Referral"])
            with p2:
                disposition = st.selectbox("🏥 Patient Disposition", ["Discharge Home", "Admit General Ward", "Admit ICU/CCU", "Emergency Operation"])
            
            selected_ddx = st.multiselect("🔍 Must-Exclude (Red Flags)", ["MI", "PE", "Aortic Dissection", "Sepsis", "Stroke", "Pneumothorax"])
            conf_in = st.slider("🎯 Confidence Level (%)", 0, 100, 50)
            
            if st.button("✅ Submit Decision"):
                if dx_in and re_in:
                    st.session_state.user_plan = {"step": next_step, "dispo": disposition}
                    st.session_state.submitted = True
                    st.rerun()
                else: st.error("Please provide both Diagnosis and Reasoning.")

    if st.session_state.submitted:
        st.divider()
        sc, dx_s, r_s, p_s, s_s, target, level = evaluate_pro(dx_in, re_in, st.session_state.user_plan, case, profession, conf_in, selected_ddx)
        
        res_l, res_r = st.columns([2, 1])
        with res_l:
            st.subheader(f"📊 Assessment Score: {sc}/10")
            st.progress(sc * 10)
            
            curr_step = st.session_state.user_plan.get('step', 'N/A')
            curr_dispo = st.session_state.user_plan.get('dispo', 'N/A')
            
            st.write(f"✅ **Correct Diagnosis:** `{target}`")
            st.write(f"➡️ **Your Action Plan:** {curr_step} at {curr_dispo}")
            
            if level == "wrong" and curr_dispo == "Discharge Home":
                st.error("❌ **Critical Safety Warning:** Unsafe discharge plan for a high-risk patient.")

        with res_r:
            st.subheader("👥 Interprofessional Feedback")
            for role, ans in case.get("interprofessional_answers", {}).items():
                with st.expander(f"Perspective from {role.upper()}"): st.write(ans)

elif page == "🏆 Leaderboard":
    st.header("🏆 Performance Dashboard")
    st.info("Leaderboard data integration coming soon.")

st.markdown("---")
st.caption("ACLR Professional v4.3 English | Optimized for Clinical Training 2026")
