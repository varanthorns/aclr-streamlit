import streamlit as st

import json, random, pandas as pd, time

from datetime import datetime

import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.metrics.pairwise import cosine_similarity

from streamlit_mic_recorder import mic_recorder



# ===================== CONFIG =====================

st.set_page_config(layout="wide", page_title="ACLR Ultimate Clinical Reasoning")



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



def semantic_score(a, b):

    try:

        vec = TfidfVectorizer().fit_transform([str(a), str(b)])

        return cosine_similarity(vec[0:1], vec[1:2])[0][0]

    except: return 0



def extract_steps(reasoning):

    keys = ["because","therefore","thus","so","เนื่องจาก","ดังนั้น", "ทำให้", "ส่งผล"]

    return [s.strip() for s in reasoning.split(".") if any(k in s.lower() for k in keys) and s.strip()]



# ===================== SESSION STATE =====================

if "case" not in st.session_state:

    st.session_state.case = random.choice(cases)

if "submitted" not in st.session_state:

    st.session_state.submitted = False

if "voice_text" not in st.session_state:

    st.session_state.voice_text = ""



# ===================== HEADER =====================

st.title("🧠 ACLR – Clinical Reasoning Ultimate")

st.caption("Advanced Interprofessional OSCE Simulator 2026")



user = st.text_input("👤 User ID / Name", value="Doctor_X")

if not user: st.stop()



# ===================== SIDEBAR =====================

with st.sidebar:

    st.header("⚙️ Station Settings")

    profession = st.selectbox("👩‍⚕️ Your Role", ["medicine", "dentistry", "nursing","pharmacy","ams","public health", "veterinarian"])

    mode = st.radio("Mode", ["Practice", "OSCE (Timed)", "Battle (Leaderboard)"])

    

    st.divider()

    block_choice = st.selectbox("📚 Select Block", ["All"] + list(set(c["block"] for c in cases)))

    diff_choice = st.selectbox("🎯 Difficulty", ["easy","medium","hard"])



    if st.button("🔄 Next Case"):

        filtered = [c for c in cases if (block_choice == "All" or c["block"] == block_choice) and c["difficulty"] == diff_choice]

        st.session_state.case = random.choice(filtered) if filtered else random.choice(cases)

        st.session_state.submitted = False

        st.session_state.voice_text = ""

        st.rerun()



case = st.session_state.case



# ===================== MAIN LAYOUT =====================

col1, col2 = st.columns([2, 1])



with col1:

    tab1, tab2, tab3 = st.tabs(["📋 Scenario", "🧪 Diagnostics", "✍️ Analysis"])

    

    with tab1:

        st.info(case["scenario"].get("en", ""))

        if case["image_url"]:

            st.image(case["image_url"], caption="Clinical Imaging / EKG / Lab Slide", use_container_width=True)

            

            

    with tab2:

        st.markdown("### 🔬 Laboratory Findings")

        if case["labs"]: st.table(pd.DataFrame(case["labs"]))

        else: st.write("No specific labs provided.")



    with tab3:

        st.warning(f"**Task:** {case.get('task', {}).get(profession, 'Diagnose and provide reasoning.')}")

        

        # --- Voice Integration (SBAR Reporting) ---

        st.markdown("##### 🎙️ Voice SBAR Reporting (Optional)")

        audio = mic_recorder(start_prompt="Click to Speak (OSCE Mode)", stop_prompt="Stop Recording", key='recorder')

        if audio:

            st.session_state.voice_text = "Audio recorded (Simulated transcription: " + case["answer"] + " due to clinical findings)"

            st.success("Voice Captured!")



        # --- Input Fields ---

        ddx = st.multiselect("🔍 Differential Diagnosis", ["MI", "PE", "Sepsis", "Pneumonia", "Aortic Dissection", "Other"])

        dx = st.text_input("🩺 Final Diagnosis", value=st.session_state.voice_text if st.session_state.voice_text else "")

        reasoning = st.text_area("✍️ Pathophysiological Reasoning", height=100)

        

        c1, c2 = st.columns(2)

        with c1:

            next_step = st.selectbox("🚀 Next Best Step", ["Observation", "Emergency Surgery", "Empirical Meds", "Referral", "Discharge"])

        with c2:

            dispo = st.selectbox("🏥 Disposition", ["Home", "Ward", "ICU", "OR"])

            

        exclusion = st.text_area("🚫 Why exclude other DDx?", placeholder="Explain why it's not the other options...")



        if st.button("✅ Submit Decision"):

            if dx and reasoning:

                st.session_state.submitted = True

                total, dx_s, r_s, d_s, target, used, steps, level = evaluate(dx, reasoning, case, profession)

                

                # บันทึกลง responses.csv

                res_df = pd.DataFrame([{"user": user, "block": case["block"], "score": total, "time": datetime.now()}])

                try:

                    old = pd.read_csv("responses.csv")

                    res_df = pd.concat([old, res_df])

                except: pass

                res_df.to_csv("responses.csv", index=False)

                st.rerun()



    # --- Result Display ---

    if st.session_state.submitted:

        st.divider()

        total, dx_s, r_s, d_s, target, used, steps, level = evaluate(dx, reasoning, case, profession)

        st.markdown(f"### 📊 Result: {total}/10")

        st.progress(total * 10)

        

        r1, r2, r3 = st.columns(3)

        r1.metric("Dx Accuracy", f"{dx_s}/5")

        r2.metric("Reasoning", f"{r_s}/5")

        r3.metric("Logic Steps", f"{d_s}/3")

        

        st.markdown(f"**Correct Answer:** `{target}`")

        st.write("**AI Feedback:**", "Excellent" if level=="correct" else "Review the key features.")



# ===================== COL 2: TEAM & ANALYTICS =====================

with col2:

    if mode == "Battle (Leaderboard)":

        st.markdown("## 🏆 Global Leaderboard")

        try:

            df_all = pd.read_csv("responses.csv")

            leaderboard = df_all.groupby("block")["score"].mean().sort_values(ascending=False)

            st.dataframe(leaderboard, use_container_width=True)

            st.caption("Your Block Proficiency vs Global Average")

        except: st.write("Start playing to see rankings.")



    st.markdown("## 👥 Interprofessional Board")

    if st.session_state.submitted:

        ipa = case.get("interprofessional_answers", {})

        for role, ans in ipa.items():

            with st.expander(f"Insight from {role.upper()}", expanded=True):

                st.write(ans)

    else:

        st.info("🔒 Decision required to unlock team insights.")



    st.markdown("---")

    st.markdown("## 📖 Evidence Reference")

    st.caption(f"Source: {case['reference']['source']} ({case['reference']['year']})")

    

    # Performance Chart

    try:

        df = pd.read_csv("responses.csv")

        user_df = df[df["user"] == user]

        if not user_df.empty:

            st.line_chart(user_df.set_index("time")["score"])

    except: pass



st.markdown("---")

st.caption("ACLR Professional v3.0 | 2026 Updated Guidelines") 
