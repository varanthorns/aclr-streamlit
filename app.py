import streamlit as st
import json, random, pandas as pd, time, os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from streamlit_mic_recorder import mic_recorder

# ===================== 1. CONFIG =====================
st.set_page_config(layout="wide", page_title="ACLR Multi-Disciplinary Simulator", page_icon="🏥")

# ===================== 2. EVALUATION LOGIC =====================
def evaluate_pro(dx, reasoning, plan, case, profession, confidence, selected_ddx):
    """Adaptive evaluation based on Professional Role."""
    target_dx = case.get("interprofessional_answers", {}).get(profession, case.get("answer", "Standard Diagnosis"))
    
    u_step = plan.get('step', 'N/A')
    u_dispo = plan.get('dispo', 'N/A')

    # 1. Dx Accuracy (5 pts)
    try:
        vec = TfidfVectorizer().fit_transform([str(dx).lower(), str(target_dx).lower()])
        sim = cosine_similarity(vec[0:1], vec[1:2])[0][0]
    except: sim = 0
    dx_score = 5 if sim > 0.75 else (3 if sim > 0.40 else 0)
    level = "correct" if sim > 0.75 else ("close" if sim > 0.40 else "wrong")

    # 2. Key Evidence & Role-Based Logic (3 pts)
    role_keywords = {
        "pharmacy": ["dosage", "mechanism", "interaction", "contraindication", "side effect"],
        "nursing": ["monitor", "vital signs", "assessment", "comfort", "positioning"],
        "ams": ["sensitivity", "specificity", "biomarker", "assay", "culture"],
        "public health": ["outbreak", "prevention", "screening", "community", "risk factor"]
    }
    found_keys = [k for k in case.get("key_points", []) if k.lower() in reasoning.lower()]
    logic_words = ["because", "therefore", "due to", "result in", "indicates"]
    role_bonus = any(w in reasoning.lower() for w in role_keywords.get(profession, []))
    r_score = min(2, len(found_keys)) + (1 if any(w in reasoning.lower() for w in logic_words) or role_bonus else 0)

    # 3. Management Plan (2 pts)
    plan_score = 0
    if u_step == case.get("next_step_correct", "Observation"): plan_score += 1
    if u_dispo == case.get("dispo_correct", "Admit General Ward"): plan_score += 1

    # 4. Safety & Penalty
    must_exclude = case.get("must_exclude", [])
    safety_check = 1 if all(item in selected_ddx for item in must_exclude) else -1
    
    penalty = 0
    if level == "wrong" and u_dispo == "Discharge Home": penalty -= 2
    if level == "wrong" and confidence > 90: penalty -= 2 

    final_score = max(0, min(10, dx_score + r_score + plan_score + safety_check + penalty))
    return final_score, dx_score, r_score, plan_score, safety_check, target_dx, level

# ===================== 3. DATA LOADING (Robust Version) =====================
@st.cache_data
def load_cases():
    file_path = "cases.json"
    if os.path.exists(file_path):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list) and len(data) > 0:
                    return data
        except Exception as e:
            st.error(f"Error loading JSON: {e}")
    
    # Fallback Case if file is missing or corrupted
    return [{
        "block":"Emergency", 
        "difficulty":"easy", 
        "scenario":{"en":"Default Case: Chest pain in a 60-year-old male."}, 
        "answer":"Acute MI",
        "key_points":["chest pain"],
        "next_step_correct":"Emergency Procedure",
        "dispo_correct":"Admit ICU/CCU"
    }]

cases = load_cases()

# ===================== 4. SESSION STATE =====================
if "case" not in st.session_state: 
    st.session_state.case = cases[0]
if "submitted" not in st.session_state: 
    st.session_state.submitted = False
if "user_plan" not in st.session_state: 
    st.session_state.user_plan = {"step": "Waiting", "dispo": "Waiting"}
if "voice_text" not in st.session_state: 
    st.session_state.voice_text = ""

# ===================== 5. SIDEBAR =====================
with st.sidebar:
    st.title("🧠 ACLR Professional")
    page = st.radio("Main Menu", ["📖 Comprehensive User Guide", "🧪 Clinical Simulator", "🏆 Leaderboard"])
    st.divider()
    
    if page == "🧪 Clinical Simulator":
        st.subheader("🎯 Station Control")
        # Dynamic Blocks from JSON
        all_blocks = ["All"] + sorted(list(set(c.get('block', 'General') for c in cases)))
        sel_block = st.selectbox("Select Medical Block", all_blocks)
        sel_diff = st.select_slider("Select Difficulty", options=["easy", "medium", "hard"])
        
        if st.button("🔄 Generate New Case"):
            filtered = [c for c in cases if (sel_block == "All" or c.get('block') == sel_block) and (c.get('difficulty') == sel_diff)]
            if filtered:
                st.session_state.case = random.choice(filtered)
            else:
                st.session_state.case = random.choice(cases) # fallback random
            st.session_state.submitted = False
            st.session_state.user_plan = {"step": "Waiting", "dispo": "Waiting"}
            st.session_state.voice_text = ""
            st.rerun()
    
    st.divider()
    prof_list = ["Doctor", "Dentistry", "Nursing", "AMS", "Vet", "Pharmacy", "Public Health"]
    profession = st.selectbox("👩‍⚕️ Your Professional Role", prof_list).lower()

# ===================== 6. PAGES =====================

# --- 📖 COMPREHENSIVE USER GUIDE ---
if page == "📖 Comprehensive User Guide":
    st.header("📖 Clinical Reasoning Manual: ACLR Loop")
    st.markdown("---")
    st.subheader("🎯 Educational Philosophy")
    st.write("The **ACLR Loop** focuses on **Active Recall** and **Safety-First**. It challenges you to justify your decisions based on your specific professional background.")

    st.markdown("### 👥 Interprofessional Perspectives")
    guide_cols = st.columns(3)
    guide_cols[0].info("**Clinicians (Dr/Dent/Vet):** Focus on Definitive Diagnosis and Intervention.")
    guide_cols[1].success("**Pharmacy:** Focus on Pharmacotherapy and Drug Safety.")
    guide_cols[2].warning("**Nursing/AMS/PH:** Focus on Monitoring and Diagnostics.")

    st.divider()
    st.subheader("📊 Detailed Scoring Rubric (10 Pts)")
    rubric = {
        "Category": ["Diagnosis Accuracy", "Clinical Logic", "Management Plan", "Safety Check"],
        "Points": ["5 Pts", "3 Pts", "2 Pts", "+1 / -2"],
        "Criteria": ["Match with role-specific target.", "Evidence keywords and connectors.", "Next Step & Disposition.", "Must-Excludes & Safe Discharge."]
    }
    st.table(pd.DataFrame(rubric))

# --- 🧪 CLINICAL SIMULATOR ---
elif page == "🧪 Clinical Simulator":
    case = st.session_state.case
    st.title(f"🏥 Station: {case.get('block', 'General')} | {case.get('difficulty', 'medium').upper()}")
    
    col_main, col_side = st.columns([2, 1])
    with col_main:
        t1, t2, t3 = st.tabs(["📋 Scenario", "🧪 Diagnostics", "✍️ Professional Analysis"])
        with t1:
            st.info(case.get("scenario", {}).get("en", "No scenario data."))
            if case.get("image_url"): st.image(case["image_url"])
        with t2:
            if case.get("labs"): st.table(pd.DataFrame(case["labs"]))
            else: st.write("No diagnostic data.")
        with t3:
            st.warning(f"**Task:** Evaluate the case as a **{profession.capitalize()}**")
            
            audio = mic_recorder(start_prompt="🎙️ Voice-to-Dx Preview", stop_prompt="Stop Recording", key='rec')
            if audio: st.session_state.voice_text = f"Suspected {case.get('answer')}"
            
            dx_in = st.text_input("🩺 Professional Diagnosis / Finding", value=st.session_state.voice_text)
            re_in = st.text_area("✍️ Pathophysiological Reasoning", placeholder="Explain based on your professional expertise...")
            
            st.divider()
            st.subheader("🚀 Clinical Management Plan")
            p1, p2 = st.columns(2)
            with p1:
                next_step = st.selectbox("🎯 Immediate Next Step", ["Observation", "Emergency Procedure", "Start Medication", "Diagnostic Imaging", "Consult Specialist", "Referral", "Lab Investigation"])
            with p2:
                disposition = st.selectbox("🏥 Disposition", ["Discharge Home", "Admit General Ward", "Admit ICU/CCU", "Emergency Operation", "Transfer Facility"])
            
            selected_ddx = st.multiselect("🔍 Red Flags / Must-Exclude", ["MI", "PE", "Aortic Dissection", "Sepsis", "Stroke", "Pneumothorax"])
            conf_in = st.slider("🎯 Confidence Level (%)", 0, 100, 50)
            
            if st.button("✅ Submit Professional Decision"):
                if dx_in and re_in:
                    st.session_state.user_plan = {"step": next_step, "dispo": disposition}
                    st.session_state.submitted = True
                    st.rerun()
                else: st.error("Please provide both Diagnosis and Reasoning.")

    if st.session_state.submitted:
        st.divider()
        # Scoring logic
        sc, dx_s, r_s, p_s, s_s, target, level = evaluate_pro(dx_in, re_in, st.session_state.user_plan, case, profession, conf_in, selected_ddx)
        
        res_l, res_r = st.columns([2, 1])
        with res_l:
            st.subheader(f"📊 Assessment Score: {sc}/10")
            st.progress(sc * 10)
            st.write(f"✅ **Target Diagnosis for {profession.capitalize()}:** `{target}`")
            st.write(f"➡️ **Your Plan:** {st.session_state.user_plan.get('step')} at {st.session_state.user_plan.get('dispo')}")
            if level == "wrong" and st.session_state.user_plan.get('dispo') == "Discharge Home":
                st.error("❌ **Safety Violation:** Unsafe discharge plan for a high-risk case.")
        with res_r:
            st.subheader("👥 Interprofessional Debrief")
            for role, ans in case.get("interprofessional_answers", {}).items():
                with st.expander(f"Perspective from {role.upper()}"): st.write(ans)

# --- 🏆 LEADERBOARD ---
elif page == "🏆 Leaderboard":
    st.header("🏆 Performance Dashboard")
    st.info("Analytics for Multi-Disciplinary Teams coming soon.")

st.markdown("---")
st.caption("ACLR Professional v5.1 Multi-Disciplinary | Optimized for Clinical Decisions 2026")
