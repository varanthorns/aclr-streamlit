import streamlit as st
import json, random, pandas as pd, time
from datetime import datetime
import os

# --- 1. CONFIG ---
st.set_page_config(layout="wide", page_title="ACLR Pro - Clinical Reasoning Board", page_icon="🧠")

# --- 2. DATA PERSISTENCE (Scoring System) ---
SCORE_FILE = "user_stats.csv"

def save_score(user, block, score):
    new_data = pd.DataFrame([[user, block, score, datetime.now()]], 
                            columns=["User", "Block", "Score", "Timestamp"])
    if not os.path.isfile(SCORE_FILE):
        new_data.to_csv(SCORE_FILE, index=False)
    else:
        new_data.to_csv(SCORE_FILE, mode='a', header=False, index=False)

def get_user_stats(user):
    if os.path.isfile(SCORE_FILE):
        df = pd.read_csv(SCORE_FILE)
        return df[df["User"] == user]
    return pd.DataFrame()

# --- 3. LOAD CASES ---
@st.cache_data
def load_all_cases():
    try:
        with open("cases.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except:
        # Emergency Mock Data
        return [{
            "id": "MOCK-01",
            "block": "Cardiology",
            "difficulty": "hard",
            "scenario": {"en": "65-year-old male with tearing chest pain radiating to the back. BP 180/100."},
            "labs": [{"item": "Troponin", "value": "Normal", "status": "normal"}, {"item": "CXR", "value": "Widened Mediastinum", "status": "high"}],
            "answer": "Aortic Dissection",
            "interprofessional_answers": {"Nurse": "BP is different in both arms.", "Radiology": "Suggest urgent CT."},
            "teaching_pearls": "Look for BP discrepancy and widened mediastinum."
        }]

# --- 4. SESSION STATE MANAGEMENT ---
if "submitted" not in st.session_state: st.session_state.submitted = False
if "case" not in st.session_state: st.session_state.case = random.choice(load_all_cases())
if "start_time" not in st.session_state: st.session_state.start_time = None
if "user_dx" not in st.session_state: st.session_state.user_dx = ""
if "user_rs" not in st.session_state: st.session_state.user_rs = ""

def reset_station():
    st.session_state.submitted = False
    st.session_state.case = random.choice(load_all_cases())
    st.session_state.start_time = None
    st.session_state.user_dx = ""
    st.session_state.user_rs = ""
    st.rerun()

# --- 5. SIDEBAR & USER PROFILE ---
with st.sidebar:
    st.header("👤 Medical Profile")
    user_id = st.text_input("Enter Student ID", value="ST-2026")
    
    # Dashboard Analytics
    stats = get_user_stats(user_id)
    if not stats.empty:
        total_cases = len(stats)
        avg_score = stats["Score"].mean()
        st.metric("Total Cases", total_cases)
        st.metric("Avg Score", f"{avg_score:.2f}/10")
        
        # XP Progress Bar
        level = (total_cases // 5) + 1
        st.write(f"**Level {level} Resident**")
        st.progress((total_cases % 5) / 5)
    
    st.divider()
    mode = st.radio("Simulation Mode", ["Practice", "OSCE (Timed)"])
    if st.button("🔄 New Case / Reset"): reset_station()

# --- 6. MAIN CONTENT ---
st.title("🧠 ACLR: Clinical Decision Training")

case = st.session_state.case
tab_chart, tab_exam, tab_profile = st.tabs(["📋 Patient Chart", "✍️ Your Assessment", "📊 Scoring History"])

# --- TAB 1: PATIENT CHART ---
with tab_chart:
    c1, c2 = st.columns([3, 2])
    with c1:
        st.subheader("Clinical Scenario")
        st.info(case["scenario"]["en"])
        
        st.subheader("👥 Interprofessional Board")
        if st.session_state.submitted:
            ipa = case.get("interprofessional_answers", {})
            cols = st.columns(len(ipa))
            for i, (role, ans) in enumerate(ipa.items()):
                cols[i].success(f"**{role}**\n\n{ans}")
        else:
            st.warning("🔒 Consultations are locked until you submit your diagnosis.")

    with c2:
        st.subheader("🧪 Lab/Imaging")
        if case.get("labs"):
            st.table(pd.DataFrame(case["labs"]))

# --- TAB 2: ASSESSMENT & TIMER ---
with tab_exam:
    if not st.session_state.submitted:
        # REAL-TIME TIMER
        if mode == "OSCE (Timed)":
            t_place = st.empty()
            if st.session_state.start_time is None:
                if st.button("▶️ Start OSCE Session"):
                    st.session_state.start_time = time.time()
                    st.rerun()
            else:
                elapsed = time.time() - st.session_state.start_time
                rem = 60 - int(elapsed)
                if rem > 0:
                    t_place.metric("⏱ Time Remaining", f"{rem}s")
                    # Note: In real Streamlit app, we use time.sleep(1) + rerun for true live countdown
                    # but for this script, it will update on interaction or we can use a loop.
                else:
                    t_place.error("⏰ TIME UP! Please submit immediately.")

        # INPUTS
        st.markdown("### 1. Differential Diagnosis Table")
        ddx_df = pd.DataFrame([{"Priority": "1", "Dx": "", "Reason": ""}, {"Priority": "2", "Dx": "", "Reason": ""}])
        st.data_editor(ddx_df, use_container_width=True, hide_index=True)

        st.markdown("### 2. Final Assessment")
        st.session_state.user_dx = st.text_input("🩺 Final Diagnosis", value=st.session_state.user_dx)
        st.session_state.user_rs = st.text_area("✍️ Clinical Reasoning", value=st.session_state.user_rs)
        
        if st.button("🚀 Submit to Examiner"):
            if st.session_state.user_dx:
                # Simple Logic Grade
                score = 10 if case["answer"].lower() in st.session_state.user_dx.lower() else 3
                save_score(user_id, case["block"], score)
                st.session_state.submitted = True
                st.rerun()
            else:
                st.error("Diagnosis is required!")
    
    else:
        # FEEDBACK SECTION
        st.balloons()
        st.header("🏁 Station Results")
        res1, res2 = st.columns(2)
        with res1:
            st.markdown(f"**Your Diagnosis:** `{st.session_state.user_dx}`")
            st.markdown(f"**Correct Answer:** `{case['answer']}`")
        with res2:
            st.markdown(f"**Key Pearls:**")
            st.warning(case["teaching_pearls"])
        
        if st.button("🔄 Next Case"): reset_station()

# --- TAB 3: SCORING HISTORY ---
with tab_profile:
    st.subheader(f"Performance Analysis: {user_id}")
    if not stats.empty:
        # Radar-like Bar Chart for Blocks
        block_scores = stats.groupby("Block")["Score"].mean().reset_index()
        st.bar_chart(block_scores.set_index("Block"))
        
        st.subheader("Recent Attempts")
        st.dataframe(stats.sort_values(by="Timestamp", ascending=False), use_container_width=True)
    else:
        st.info("No history found. Complete your first case to see results!")

# --- TIMER REFRESHER ---
# ส่วนนี้ช่วยให้ Timer อัปเดตทุกวินาทีโดยอัตโนมัติ
if mode == "OSCE (Timed)" and st.session_state.start_time is None == False and st.session_state.submitted == False:
    time.sleep(1)
    st.rerun()
