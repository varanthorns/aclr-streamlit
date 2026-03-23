import streamlit as st
import json, random, pandas as pd, time
import os
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- 1. CONFIG ---
st.set_page_config(layout="wide", page_title="ACLR Pro - Clinical Reasoning Board", page_icon="🧠")

# --- 2. UTILS & SCORING ---
SCORE_FILE = "user_stats.csv"

def save_score(user, block, score, dx_level):
    new_data = pd.DataFrame([[user, block, score, dx_level, datetime.now()]], 
                            columns=["User", "Block", "Score", "Level", "Timestamp"])
    new_data.to_csv(SCORE_FILE, mode='a', header=not os.path.exists(SCORE_FILE), index=False)

def get_user_stats(user):
    return pd.read_csv(SCORE_FILE) if os.path.exists(SCORE_FILE) else pd.DataFrame()

def semantic_score(a, b):
    try:
        vec = TfidfVectorizer().fit_transform([str(a).lower(), str(b).lower()])
        return cosine_similarity(vec[0:1], vec[1:2])[0][0]
    except: return 0

def evaluate_clinical_reasoning(dx, reasoning, case):
    target = case.get("answer", "")
    sim = semantic_score(dx, target)
    dx_s, level = (5, "Correct") if sim > 0.8 else (3, "Close") if sim > 0.4 else (0, "Incorrect")
    
    r_s = min(3, sum(1 for k in case.get("key_points", []) if k.lower() in reasoning.lower()))
    d_s = min(2, sum(1 for k in ["because","therefore","due to","ดังนั้น","เพราะ"] if k in reasoning.lower()))
    
    return dx_s + r_s + d_s, dx_s, r_s, d_s, level

# --- 3. LOAD DATA ---
@st.cache_data
def load_all_cases():
    try:
        with open("cases.json", "r", encoding="utf-8") as f: return json.load(f)
    except:
        return [{
            "block": "Cardiology", "difficulty": "hard",
            "scenario": {"en": "65-year-old male with tearing chest pain radiating to the back."},
            "labs": [{"item": "BP", "value": "180/110", "status": "high"}],
            "answer": "Aortic Dissection",
            "key_points": ["tearing", "back"],
            "interprofessional_answers": {"Nurse": "Pain is 10/10."},
            "teaching_pearls": "Think of Dissection in tearing pain."
        }]

# --- 4. SESSION STATE ---
if "submitted" not in st.session_state: st.session_state.submitted = False
if "case" not in st.session_state: st.session_state.case = random.choice(load_all_cases())
if "start_time" not in st.session_state: st.session_state.start_time = None

def reset_station():
    for key in ["submitted", "start_time", "last_result"]:
        if key in st.session_state: del st.session_state[key]
    st.session_state.case = random.choice(load_all_cases())
    st.rerun()

# --- 5. UI ---
with st.sidebar:
    st.header("👤 Profile")
    user_id = st.text_input("Student ID", value="ST-2026")
    mode = st.radio("Mode", ["Practice", "OSCE (Timed)"])
    if st.button("🔄 New Station"): reset_station()

st.title("🧠 ACLR: Clinical Reasoning Board")
case = st.session_state.case
tab_chart, tab_exam, tab_profile = st.tabs(["📋 Patient Chart", "✍️ Assessment", "📊 Scoring Profile"])

with tab_chart:
    c1, c2 = st.columns([3, 2])
    with c1:
        st.subheader("Scenario")
        st.info(case["scenario"]["en"])
        if st.session_state.submitted:
            st.subheader("👥 Interprofessional Board")
            ipa = case.get("interprofessional_answers", {})
            cols = st.columns(len(ipa))
            for i, (role, ans) in enumerate(ipa.items()):
                cols[i].success(f"**{role}**\n\n{ans}")
        else:
            st.warning("🔒 Team feedback locked until submission.")
    with c2:
        st.subheader("🧪 Labs/Imaging")
        if case.get("labs"): st.table(pd.DataFrame(case["labs"]))

with tab_exam:
    # --- TIMER DISPLAY ---
    if mode == "OSCE (Timed)" and not st.session_state.submitted:
        t_place = st.empty()
        if st.session_state.start_time:
            rem = 60 - int(time.time() - st.session_state.start_time)
            if rem > 0: t_place.metric("⏱ Time Left", f"{rem}s")
            else: t_place.error("⏰ TIME UP!")
        elif st.button("▶️ Start OSCE Timer"):
            st.session_state.start_time = time.time()
            st.rerun()

    # --- INPUT FORM (พิมพ์ตรงนี้) ---
    if not st.session_state.submitted:
        with st.form("assessment_form"):
            st.markdown("### 1. Differential Diagnosis")
            ddx_input = st.data_editor(pd.DataFrame([{"Priority": "1", "Dx": "", "Reason": ""}]), use_container_width=True, hide_index=True)
            
            st.markdown("### 2. Final Decision")
            u_dx = st.text_input("🩺 Diagnosis (พิมพ์คำตอบที่นี่)")
            u_rs = st.text_area("✍️ Reasoning (ให้เหตุผลที่นี่)")
            
            submit_btn = st.form_submit_button("🚀 Submit Assessment")
            
            if submit_btn:
                if u_dx:
                    score, dx_s, r_s, d_s, level = evaluate_clinical_reasoning(u_dx, u_rs, case)
                    save_score(user_id, case["block"], score, level)
                    st.session_state.submitted = True
                    st.session_state.last_result = (score, dx_s, r_s, d_s, level, u_dx)
                    st.rerun()
                else:
                    st.error("Please enter a diagnosis!")
    else:
        # ผลลัพธ์หลังส่ง
        score, dx_s, r_s, d_s, level, u_dx = st.session_state.last_result
        st.balloons()
        st.header(f"Results: {level} ({score}/10)")
        st.write(f"**Your Answer:** {u_dx} | **Correct:** {case['answer']}")
        st.warning(f"**Pearls:** {case['teaching_pearls']}")
        if st.button("🔄 Next Case"): reset_station()

with tab_profile:
    stats = get_user_stats(user_id)
    if not stats.empty:
        st.bar_chart(stats.groupby("Block")["Score"].mean())
        st.dataframe(stats.sort_values("Timestamp", ascending=False))

# Auto-refresh timer logic (Doesn't break st.form)
if mode == "OSCE (Timed)" and st.session_state.start_time and not st.session_state.submitted:
    time.sleep(1)
    st.rerun()
