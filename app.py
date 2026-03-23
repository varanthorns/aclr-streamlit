import streamlit as st
import json, random, pandas as pd, time
from datetime import datetime
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ต้องอยู่บรรทัดแรกสุดของสคริปต์
st.set_page_config(layout="wide", page_title="ACLR Clinical Reasoning")

# ===================== UTILS & LOAD =====================
def safe_case(case):
    case.setdefault("task", {})
    case.setdefault("interprofessional_answers", {})
    case.setdefault("reference", {"source":"Unknown","year":"-"})
    case.setdefault("key_points", [])
    return case

@st.cache_data
def load_cases():
    # ตรวจสอบว่ามีไฟล์จริงไหม หรือสร้าง Mock data สำหรับทดสอบ
    try:
        with open("cases.json","r",encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return [{"block":"General", "difficulty":"easy", "scenario":{"en":"Patient with fever"}, "answer":"Flu", "key_points":["fever"]}]

cases = load_cases()

def normalize(t): return str(t).lower().strip()

def semantic_score(a, b):
    try:
        vec = TfidfVectorizer().fit_transform([a, b])
        return cosine_similarity(vec[0:1], vec[1:2])[0][0]
    except:
        return 0

def extract_steps(reasoning):
    keys = ["because","therefore","thus","so","เนื่องจาก","ดังนั้น"]
    return [s.strip() for s in reasoning.split(".") if any(k in s.lower() for k in keys) and s.strip()]

# ===================== SCORING =====================
def evaluate(dx, reasoning, case, profession):
    target = case.get("interprofessional_answers", {}).get(profession, case.get("answer", ""))
    sim = semantic_score(dx, target)

    if normalize(dx) == normalize(target):
        dx_score, level = 5, "correct"
    elif sim > 0.6:
        dx_score, level = 3, "close"
    else:
        dx_score, level = 0, "wrong"

    r_score = 0
    used = []
    for k in case.get("key_points", []):
        if k.lower() in reasoning.lower():
            r_score += 1
            used.append(k)

    steps = extract_steps(reasoning)
    d_score = min(3, len(steps))
    r_score = min(5, r_score + (1 if len(reasoning.split()) > 20 else 0))
    total = min(10, dx_score + r_score + d_score)

    return total, dx_score, r_score, d_score, target, used, steps, level

# ===================== SESSION STATE =====================
if "case" not in st.session_state:
    st.session_state.case = safe_case(random.choice(cases))

# ===================== HEADER =====================
st.title("🧠 ACLR – Clinical Reasoning Platform")
st.caption("UWorld + AMBOSS + OSCE + Interprofessional Simulation")

user = st.text_input("👤 User ID / Name")
if not user:
    st.info("Please enter your User ID to begin.")
    st.stop()

# ===================== SIDEBAR =====================
with st.sidebar:
    st.header("⚙️ Settings")
    profession = st.selectbox("👩‍⚕️ Profession", ["medicine","dentistry","nursing","vet","pharmacy","public_health","ams"])
    
    all_blocks = ["All"] + list(set(c["block"] for c in cases))
    block_choice = st.selectbox("📚 Block", all_blocks)
    diff_choice = st.selectbox("🎯 Difficulty", ["easy","medium","hard"])
    mode = st.radio("Mode", ["Practice", "OSCE", "Battle"])

    if st.button("🔄 New Case"):
        filtered = [c for c in cases if (block_choice == "All" or c["block"] == block_choice) and c["difficulty"] == diff_choice]
        if filtered:
            st.session_state.case = safe_case(random.choice(filtered))
            st.session_state.pop("start", None) # Reset timer
            st.rerun()

case = st.session_state.case

# ===================== MAIN LAYOUT =====================
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("## 📋 Clinical Scenario")
    st.info(case["scenario"].get("en", "No scenario available"))
    
    if case.get("additional"):
        st.caption(case["additional"].get("en", ""))

    st.markdown("## 🎯 Your Task")
    task_text = case.get("task", {}).get(profession, case.get("task", {}).get("medicine", "Provide your clinical decision"))
    st.warning(task_text)

    st.markdown("## 🧠 Think Step-by-Step")
    st.caption("1. Identify symptoms | 2. Link pathophysiology | 3. Consider differential | 4. Decide")

    if mode == "OSCE":
        if "start" not in st.session_state:
            if st.button("▶️ Start OSCE"):
                st.session_state.start = time.time()
                st.rerun()
        else:
            elapsed = int(time.time() - st.session_state.start)
            remaining = max(0, 60 - elapsed)
            st.metric("⏱ Time Left", f"{remaining}s")
            if remaining <= 0:
                st.error("⏰ Time up!")

    dx = st.text_input("停 Your Diagnosis / Answer")
    reasoning = st.text_area("✍️ Clinical Reasoning")
    confidence = st.slider("Confidence (%)", 0, 100, 50)

    if st.button("✅ Submit"):
        total, dx_s, r_s, d_s, target, used, steps, level = evaluate(dx, reasoning, case, profession)
        st.success(f"🏆 Score: {total}/10")
        
        # Feedback
        st.markdown("### 🔍 AI Examiner Feedback")
        if level == "correct": st.success("Excellent accuracy")
        elif level == "close": st.warning("Close but not precise")
        else: st.error("Incorrect diagnosis")
        
        st.write(f"**Correct Answer:** {target}")
        
        c_left, c_right = st.columns(2)
        with c_left:
            st.write("**Key Features Found:**", used)
            missing = [k for k in case.get("key_points", []) if k not in used]
            if missing: st.write("**Missing:**", missing)
        
        with c_right:
            st.write("**Logic Steps:**")
            for s in steps: st.write(f"✅ {s}")

        # Save History
        res_df = pd.DataFrame([{"user": user, "block": case["block"], "score": total, "time": datetime.now()}])
        try:
            old = pd.read_csv("responses.csv")
            res_df = pd.concat([old, res_df])
        except: pass
        res_df.to_csv("responses.csv", index=False)

with col2:
    st.markdown("## 👥 Team Decision Board")
    ipa = case.get("interprofessional_answers", {})
    for role, ans in ipa.items():
        if role == profession:
            st.success(f"🟢 {role}: {ans}")
        else:
            st.info(f"⚪ {role}: {ans}")

    st.markdown("## 📖 Reference")
    ref = case.get("reference", {})
    st.write(f"{ref.get('source','-')} ({ref.get('year','-')})")

    st.markdown("---")
    st.markdown("## 📊 Performance")
    try:
        df = pd.read_csv("responses.csv")
        user_df = df[df["user"] == user]
        if not user_df.empty:
            st.line_chart(user_df.set_index("time")["score"])
    except:
        st.info("No history yet.")
