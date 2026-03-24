import streamlit as st
import json, random, pandas as pd, time, os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from streamlit_mic_recorder import mic_recorder

# ===================== 1. CONFIG =====================
st.set_page_config(layout="wide", page_title="ACLR Interprofessional Master", page_icon="🧠")

# ===================== 2. EVALUATION LOGIC =====================
def evaluate_pro(dx, reasoning, plan, case, profession, confidence, selected_ddx):
    """
    Advanced Logic: Checks for role-specific answers in JSON first.
    If 'pharmacy' is selected, it compares dx with case['interprofessional_answers']['pharmacy'].
    """
    # 0. Get Role-Specific Target
    role_answers = case.get("interprofessional_answers", {})
    target_dx = role_answers.get(profession, case.get("answer", "Standard Diagnosis"))
    
    u_step = plan.get('step', 'N/A')
    u_dispo = plan.get('dispo', 'N/A')

    # 1. Dx Accuracy (5 pts) - Semantic Comparison
    try:
        vec = TfidfVectorizer().fit_transform([str(dx).lower(), str(target_dx).lower()])
        sim = cosine_similarity(vec[0:1], vec[1:2])[0][0]
    except: sim = 0
    dx_score = 5 if sim > 0.80 else (3 if sim > 0.45 else 0)
    level = "correct" if sim > 0.80 else ("close" if sim > 0.45 else "wrong")

    # 2. Reasoning & Key Evidence (3 pts)
    found_keys = [k for k in case.get("key_points", []) if k.lower() in reasoning.lower()]
    logic_words = ["because", "due to", "leads to", "secondary to", "associated with"]
    r_score = min(2, len(found_keys)) + (1 if any(w in reasoning.lower() for w in logic_words) else 0)

    # 3. Management Plan (2 pts)
    plan_score = 0
    if u_step == case.get("next_step_correct", "Observation"): plan_score += 1
    if u_dispo == case.get("dispo_correct", "Admit General Ward"): plan_score += 1

    # 4. Safety Check & Penalty
    must_exclude = case.get("must_exclude", [])
    safety_check = 1 if all(item in selected_ddx for item in must_exclude) else -1
    
    penalty = 0
    if level == "wrong" and u_dispo == "Discharge Home": penalty -= 3 # Harsh penalty for unsafe discharge
    if level == "wrong" and confidence > 85: penalty -= 2 # Penalty for overconfidence in error

    final_score = max(0, min(10, dx_score + r_score + plan_score + safety_check + penalty))
    return final_score, dx_score, r_score, plan_score, safety_check, target_dx, level

# ===================== 3. DATA & STORAGE =====================
@st.cache_data
def load_cases():
    if os.path.exists("cases.json"):
        with open("cases.json", "r", encoding="utf-8") as f:
            return json.load(f)
    return [{"block":"General", "difficulty":"easy", "scenario":{"en":"Please upload cases.json"}, "answer":"N/A"}]

def save_score(name, profession, score):
    # Simulating a database with session state for the leaderboard
    if "db" not in st.session_state:
        st.session_state.db = pd.DataFrame(columns=["Timestamp", "Name", "Role", "Score"])
    
    new_data = pd.DataFrame({
        "Timestamp": [time.strftime("%H:%M:%S")],
        "Name": [name],
        "Role": [profession.upper()],
        "Score": [score]
    })
    st.session_state.db = pd.concat([st.session_state.db, new_data], ignore_index=True)

cases = load_cases()

# ===================== 4. SESSION STATE =====================
if "case" not in st.session_state: st.session_state.case = cases[0]
if "submitted" not in st.session_state: st.session_state.submitted = False
if "user_plan" not in st.session_state: st.session_state.user_plan = {}
if "voice_text" not in st.session_state: st.session_state.voice_text = ""
if "db" not in st.session_state: st.session_state.db = pd.DataFrame(columns=["Timestamp", "Name", "Role", "Score"])

# ===================== 5. SIDEBAR =====================
with st.sidebar:
    st.title("🧠 ACLR Master v6.0")
    st.caption("Interprofessional Decision Simulator")
    page = st.radio("Navigation", ["📖 Detailed Manual", "🧪 Clinical Simulator", "🏆 Leaderboard"])
    st.divider()
    
    if page == "🧪 Clinical Simulator":
        st.subheader("⚙️ Station Settings")
        all_blocks = ["All"] + list(set(c.get('block') for c in cases))
        sel_block = st.selectbox("Block", all_blocks)
        sel_diff = st.select_slider("Difficulty", ["easy", "medium", "hard"])
        
        if st.button("🔄 Next Random Case"):
            filtered = [c for c in cases if (sel_block == "All" or c['block'] == sel_block) and (c['difficulty'] == sel_diff)]
            st.session_state.case = random.choice(filtered) if filtered else random.choice(cases)
            st.session_state.submitted = False
            st.session_state.voice_text = ""
            st.rerun()

    profession = st.selectbox("👩‍⚕️ Your Role", ["doctor", "dentistry", "nursing", "ams", "vet", "pharmacy", "public health"])
    user_name = st.text_input("👤 Your Name", "Guest User")

# ===================== 6. PAGES =====================

# --- 📖 DETAILED MANUAL ---
if page == "📖 Detailed Manual":
    st.header("📖 User Guide & Scoring Methodology")
    st.write("Welcome to the **ACLR (Advanced Clinical Reasoning) Loop**. This tool simulates real-world clinical decision-making.")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🎯 Goal")
        st.markdown("""
        * **Role-Specific Diagnosis:** You are evaluated based on *your* profession's expected answer.
        * **Logic Justification:** Explain *why* you think so using pathophysiological reasoning.
        * **Safety Management:** Identify life-threatening 'Red Flags' (Must-Exclude).
        """)
    with col2:
        st.subheader("📊 Scoring Rubric (10 Max)")
        st.markdown("""
        1. **Dx Accuracy (5 pts):** How close is your Dx to the gold standard for your role?
        2. **Reasoning (3 pts):** Did you use the correct keywords and logical connectors?
        3. **Plan (2 pts):** Did you choose the right Next Step and Disposition?
        4. **Safety Check:** +1 for correct Red Flags, -3 for unsafe home discharge.
        """)
    
    st.divider()
    st.subheader("💡 Interprofessional Focus Table")
    focus_data = {
        "Role": ["Doctor", "Pharmacy", "Nursing", "AMS", "Dentistry", "Vet", "Public Health"],
        "Focus Area": ["Definitive Dx & Treatment", "Dosing & Interaction", "Monitoring & Safety", "Lab Interpretation", "Oral-Systemic Link", "Comparative Patho", "Community & Prevention"]
    }
    st.table(pd.DataFrame(focus_data))

# --- 🧪 CLINICAL SIMULATOR ---
elif page == "🧪 Clinical Simulator":
    c = st.session_state.case
    st.title(f"🏥 Station: {c.get('block')} ({c.get('difficulty').upper()})")
    
    m_col, s_col = st.columns([2, 1])
    with m_col:
        t1, t2, t3 = st.tabs(["📋 Scenario", "🧪 Diagnostics", "✍️ Analysis & Plan"])
        with t1:
            st.info(f"**Patient Presentation:**\n\n{c.get('scenario').get('en')}")
            st.caption(f"TH: {c.get('scenario').get('th')}")
        with t2:
            if c.get("labs"): st.table(pd.DataFrame(c["labs"]))
            else: st.warning("No lab data available for this case.")
        with t3:
            st.markdown(f"### Current Role: **{profession.upper()}**")
            
            # Voice Recording Simulation (Integration point for mic_recorder)
            audio = mic_recorder(start_prompt="🎙️ Voice-to-Text Dx", stop_prompt="🛑 Stop", key='recorder')
            if audio: st.session_state.voice_text = f"Suspected {c.get('answer')}" # Simplified logic for demo

            dx_in = st.text_input("🩺 Final Diagnosis / Assessment", value=st.session_state.voice_text)
            re_in = st.text_area("✍️ Pathophysiological Reasoning", placeholder="Explain the clinical logic...")
            
            st.divider()
            st.subheader("🚀 Management Plan")
            p1, p2 = st.columns(2)
            with p1:
                u_step = st.selectbox("Next Step", ["Observation", "Diagnostic Imaging", "Consult Specialist", "Emergency Procedure", "Start Medication"])
            with p2:
                u_dispo = st.selectbox("Disposition", ["Discharge Home", "Admit General Ward", "Admit ICU/CCU", "Emergency Operation"])
            
            u_ddx = st.multiselect("🔍 Must-Exclude (Red Flags)", ["MI", "PE", "Sepsis", "Stroke", "Aortic Dissection", "Pneumonia"])
            u_conf = st.slider("Confidence (%)", 0, 100, 50)
            
            if st.button("🚀 Submit My Decision"):
                if dx_in and re_in:
                    st.session_state.user_plan = {"step": u_step, "dispo": u_dispo}
                    st.session_state.submitted = True
                    # Calculate Score
                    sc, _, _, _, _, target, lvl = evaluate_pro(dx_in, re_in, st.session_state.user_plan, c, profession, u_conf, u_ddx)
                    save_score(user_name, profession, sc)
                    st.rerun()

    if st.session_state.submitted:
        st.divider()
        sc, dx_s, r_s, p_s, s_s, target, lvl = evaluate_pro(dx_in, re_in, st.session_state.user_plan, c, profession, u_conf, u_ddx)
        
        res_l, res_r = st.columns([2, 1])
        with res_l:
            st.header(f"📊 Your Score: {sc}/10")
            st.progress(sc * 10)
            st.success(f"✅ **Gold Standard for {profession.upper()}:** {target}")
            st.write(f"**Your Decision Path:** {u_step} ➔ {u_dispo}")
            if lvl == "wrong" and u_dispo == "Discharge Home":
                st.error("❌ **CRITICAL ERROR:** You discharged a high-risk patient. Potential Sentinel Event.")
        with res_r:
            st.subheader("👥 Team Perspectives")
            for role, ans in c.get("interprofessional_answers", {}).items():
                with st.expander(f"View {role.upper()} Logic"):
                    st.write(ans)

# --- 🏆 LEADERBOARD ---
elif page == "🏆 Leaderboard":
    st.header("🏆 Top Clinical Performers")
    if not st.session_state.db.empty:
        df = st.session_state.db.sort_values(by="Score", ascending=False)
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.bar_chart(df, x="Name", y="Score")
    else:
        st.info("No records yet. Be the first to complete a case!")

st.markdown("---")
st.caption(f"ACLR Professional v6.0 | Case ID: {st.session_state.case.get('case_id')} | © 2026 Interprofessional Education")
