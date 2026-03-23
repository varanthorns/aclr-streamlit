import streamlit as st
import json, random, pandas as pd, time
import os
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- 1. CONFIG (Must be first) ---
st.set_page_config(layout="wide", page_title="ACLR Pro - Clinical Reasoning Board", page_icon="🧠")

# --- 2. DATA PERSISTENCE & UTILS ---
SCORE_FILE = "user_stats.csv"

def save_score(user, block, score, dx_level):
    new_data = pd.DataFrame([[user, block, score, dx_level, datetime.now()]], 
                            columns=["User", "Block", "Score", "Level", "Timestamp"])
    if not os.path.isfile(SCORE_FILE):
        new_data.to_csv(SCORE_FILE, index=False)
    else:
        new_data.to_csv(SCORE_FILE, mode='a', header=False, index=False)

def get_user_stats(user):
    if os.path.isfile(SCORE_FILE):
        df = pd.read_csv(SCORE_FILE)
        return df[df["User"] == user]
    return pd.DataFrame()

def normalize(t): return str(t).lower().strip()

def semantic_score(a, b):
    try:
        vec = TfidfVectorizer().fit_transform([a, b])
        return cosine_similarity(vec[0:1], vec[1:2])[0][0]
    except: return 0

def extract_steps(reasoning):
    keys = ["because","therefore","thus","so","เนื่องจาก","ดังนั้น"]
    return [s.strip() for s in reasoning.split(".") if any(k in s.lower() for k in keys) and s.strip()]

# --- 3. ADVANCED SCORING LOGIC ---
def evaluate_clinical_reasoning(dx, reasoning, case):
    target = case.get("answer", "")
    sim = semantic_score(dx, target)
    
    # 1. Diagnosis Score (max 5)
    if normalize(dx) == normalize(target): dx_s, level = 5, "Correct"
    elif sim > 0.6: dx_s, level = 3, "Close"
    else: dx_s, level = 0, "Incorrect"
    
    # 2. Key Points Score (max 3)
    r_s = 0
    found_keys = []
    for k in case.get("key_points", []):
        if k.lower() in reasoning.lower():
            r_s += 1
            found_keys.append(k)
    r_s = min(3, r_s)
    
    # 3. Logic Steps Score (max 2)
    steps = extract_steps(reasoning)
    d_s = min(2, len(steps))
    
    total = dx_s + r_s + d_s
    return total, dx_s, r_s, d_s, level, found_keys, steps

# --- 4. LOAD CASES ---
@st.cache_data
def load_all_cases():
    try:
        with open("cases.json", "r", encoding="utf-8") as f:
            data = json.load(f)
            return data
    except:
        return [{
            "block": "Cardiology",
            "difficulty": "hard",
            "scenario": {"en": "65-year-old male with sudden onset of severe tearing chest pain radiating to the back."},
            "labs": [{"item": "BP", "value": "180/110 (Right Arm), 140/90 (Left Arm)", "status": "high"}],
            "answer": "Aortic Dissection",
            "key_points": ["tearing", "back", "BP discrepancy"],
            "interprofessional_answers": {"Nurse": "Patient is very agitated.", "Radiology": "Mediastinum looks widened."},
            "teaching_pearls": "Aortic dissection requires high index of suspicion with BP asymmetry."
        }]

# --- 5. SESSION STATE ---
if "submitted" not in st.session_state: st.session_state.submitted = False
if "case" not in st.session_state: st.session_state.case = random.choice(load_all_cases())
if "start_time" not in st.session_state: st.session_state.start_time = None

def reset_station():
    st.session_state.submitted = False
    st.session_state.case = random.choice(load_all_cases())
    st.session_state.start_time = None
    st.rerun()

# --- 6. SIDEBAR ---
with st.sidebar:
    st.header("👤 Medical Student Profile")
    user_id = st.text_input("Student ID", value="ST-2026")
    stats = get_user_stats(user_id)
    if not stats.empty:
        total = len(stats)
        st.metric("Cases Completed", total)
        st.metric("Avg Score", f"{stats['Score'].mean():.1f}/10")
        level = (total // 5) + 1
        st.write(f"**Level {level} Resident**")
        st.progress((total % 5) / 5)
    
    st.divider()
    mode = st.radio("Station Mode", ["Practice", "OSCE (Timed)"])
    if st.button("🔄 New Station"): reset_station()

# --- 7. MAIN CONTENT ---
st.title("🧠 ACLR: Advanced Clinical Learning & Reasoning")
case = st.session_state.case
tab_chart, tab_exam, tab_profile = st.tabs(["📋 Patient Chart", "✍️ Assessment", "📊 Scoring Profile"])

with tab_chart:
    c1, c2 = st.columns([3, 2])
    with c1:
        st.subheader("Scenario")
        st.info(case["scenario"]["en"])
        st.subheader("👥 Interprofessional Board")
        if st.session_state.submitted:
            ipa = case.get("interprofessional_answers", {})
            cols = st.columns(len(ipa))
            for i, (role, ans) in enumerate(ipa.items()):
                cols[i].success(f"**{role}**\n\n{ans}")
        else:
            st.warning("🔒 Team feedback is locked until submission.")
    with c2:
        st.subheader("🧪 Labs/Imaging")
        if case.get("labs"): st.table(pd.DataFrame(case["labs"]))

with tab_exam:
    if not st.session_state.submitted:
        if mode == "OSCE (Timed)":
            t_place = st.empty()
            if st.session_state.start_time is None:
                if st.button("▶️ Start OSCE Timer"):
                    st.session_state.start_time = time.time()
                    st.rerun()
            else:
                rem = 60 - int(time.time() - st.session_state.start_time)
                if rem > 0: t_place.metric("⏱ Time Left", f"{rem}s")
                else: t_place.error("⏰ TIME UP!")

        st.markdown("### 1. Differential Diagnosis")
        ddx_df = pd.DataFrame([{"Priority": "1", "Dx": "", "Evidence": ""}, {"Priority": "2", "Dx": "", "Evidence": ""}])
        st.data_editor(ddx_df, use_container_width=True, hide_index=True)

        st.markdown("### 2. Final Assessment")
        u_dx = st.text_input("🩺 Diagnosis")
        u_rs = st.text_area("✍️ Reasoning (Key findings & Pathophysiology)")
        
        if st.button("🚀 Submit Assessment"):
            if u_dx:
                score, dx_s, r_s, d_s, level, keys, steps = evaluate_clinical_reasoning(u_dx, u_rs, case)
                save_score(user_id, case["block"], score, level)
                st.session_state.submitted = True
                st.session_state.last_result = (score, dx_s, r_s, d_s, level, keys, steps, u_dx)
                st.rerun()
            else: st.error("Please enter a diagnosis.")
    else:
        score, dx_s, r_s, d_s, level, keys, steps, u_dx = st.session_state.last_result
        st.balloons()
        st.header(f"Results: {level}")
        st.subheader(f"Total Score: {score}/10")
        
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Diagnosis", f"{dx_s}/5")
        col_b.metric("Clinical Keys", f"{r_s}/3")
        col_c.metric("Logic Steps", f"{d_s}/2")

        with st.expander("Detailed Feedback"):
            st.write(f"**Correct Answer:** {case['answer']}")
            st.write(f"**Key Findings Found:** {', '.join(keys) if keys else 'None'}")
            st.warning(f"**Teaching Pearls:** {case['teaching_pearls']}")
        
        if st.button("🔄 Next Case"): reset_station()

with tab_profile:
    st.subheader(f"Performance Analysis for {user_id}")
    if not stats.empty:
        st.bar_chart(stats.groupby("Block")["Score"].mean())
        st.subheader("History")
        st.dataframe(stats.sort_values("Timestamp", ascending=False), use_container_width=True)
    else: st.info("No records found.")

# Timer Auto-refresh
if mode == "OSCE (Timed)" and st.session_state.start_time and not st.session_state.submitted:
    time.sleep(1)
    st.rerun()
