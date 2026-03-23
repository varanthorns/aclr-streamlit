import streamlit as st
import json, random, pandas as pd, time
def safe_case(case):
    case.setdefault("task", {})
    case.setdefault("interprofessional_answers", {})
    case.setdefault("reference", {"source":"Unknown","year":"-"})
    return case
from datetime import datetime
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(layout="wide")

# ===================== LOAD =====================
@st.cache_data
def load_cases():
    with open("cases.json","r",encoding="utf-8") as f:
        return json.load(f)

cases = load_cases()

# ===================== UTILS =====================
def normalize(t): return str(t).lower().strip()

def semantic_score(a,b):
    try:
        vec = TfidfVectorizer().fit_transform([a,b])
        return cosine_similarity(vec[0:1], vec[1:2])[0][0]
    except:
        return 0

def extract_steps(reasoning):
    keys = ["because","therefore","thus","so","เนื่องจาก","ดังนั้น"]
    return [s for s in reasoning.split(".") if any(k in s.lower() for k in keys)]

# ===================== SCORING =====================
def evaluate(dx, reasoning, case, profession):

    target = case.get("interprofessional_answers",{}).get(profession, case["answer"])

    sim = semantic_score(dx, target)

    if normalize(dx) == normalize(target):
        dx_score = 5
        level = "correct"
    elif sim > 0.6:
        dx_score = 3
        level = "close"
    else:
        dx_score = 0
        level = "wrong"

    r_score = 0
    used = []

    for k in case.get("key_points",[]):
        if k.lower() in reasoning.lower():
            r_score += 1
            used.append(k)

    steps = extract_steps(reasoning)
    d_score = min(3,len(steps))

    r_score = min(5, r_score + (1 if len(reasoning.split())>20 else 0))

    total = min(10, dx_score + r_score + d_score)

    return total, dx_score, r_score, d_score, target, used, steps, level

# ===================== SESSION =====================
if "case" not in st.session_state:
    st.session_state.case = random.choice(cases)
    case = safe_case(case)
# ===================== HEADER =====================
st.title("🧠 ACLR – Clinical Reasoning Platform")
st.caption("UWorld + AMBOSS + OSCE + Interprofessional Simulation")

user = st.text_input("👤 User ID / Name")
if not user:
    st.stop()

# ===================== SIDEBAR =====================
with st.sidebar:
    st.header("⚙️ Settings")

    profession = st.selectbox(
        "👩‍⚕️ Profession",
        ["medicine","dentistry","nursing","vet","pharmacy","public_health","ams"]
    )

    block = st.selectbox("📚 Block",["All"]+list(set(c["block"] for c in cases)))
    difficulty = st.selectbox("🎯 Difficulty",["easy","medium","hard"])

    mode = st.radio("Mode",["Practice","OSCE","Battle"])

    if st.button("🔄 New Case"):
        st.session_state.case = random.choice(cases)

case = st.session_state.case

# ===================== FILTER =====================
filtered = [
    c for c in cases
    if (block=="All" or c["block"]==block)
    and c["difficulty"]==difficulty
]

if filtered:
    case = random.choice(filtered)
    st.session_state.case = case

# ===================== MAIN LAYOUT =====================
col1, col2 = st.columns([2,1])

# ==================================================
# LEFT: CASE + INPUT
# ==================================================
with col1:

    st.markdown("## 📋 Clinical Scenario")
    st.info(case["scenario"]["en"])

    if case.get("additional"):
        st.caption(case["additional"].get("en",""))
    # ===== TASK (FIXED) =====
    st.markdown("## 🎯 Your Task")
    
    task = case.get("task", {})
    
    task_text = task.get(
        profession,
        task.get("medicine", "Provide your clinical decision")
    )
    
    st.warning(task_text)
    
    # ===== TEAM BOARD (FIXED) =====
    st.markdown("## 👥 Team Decision Board")
    
    ipa = case.get("interprofessional_answers", {})
    
    for role, ans in ipa.items():
        if role == profession:
            st.success(f"🟢 {role}\n{ans}")
        else:
            st.info(f"{role}\n{ans}")
    
    # ===== REFERENCE (FIXED) =====
    st.markdown("## 📖 Reference")
    
    ref = case.get("reference", {})
    st.write(f"{ref.get('source','Unknown')} ({ref.get('year','-' )})")
        st.markdown("## 🎯 Your Task")
        task = case.get("task", {})

    task_text = task.get(
        profession,
        task.get("medicine", "Provide your clinical decision")
    )

st.warning(task_text)

    st.markdown("## 🧠 Think Step-by-Step")
    st.caption("""
    1. Identify key symptoms  
    2. Link pathophysiology  
    3. Consider differential  
    4. Decide
    """)

    # OSCE TIMER
    if mode == "OSCE":
        if "start" not in st.session_state:
            if st.button("▶️ Start OSCE"):
                st.session_state.start = time.time()

        else:
            remaining = 60 - int(time.time()-st.session_state.start)
            st.metric("⏱ Time Left", remaining)

            if remaining <= 0:
                st.error("⏰ Time up!")

    dx = st.text_input("🩺 Your Diagnosis / Answer")
    reasoning = st.text_area("✍️ Clinical Reasoning")

    confidence = st.slider("Confidence (%)",0,100,50)

    if st.button("✅ Submit"):

        total, dx_s, r_s, d_s, target, used, steps, level = evaluate(dx, reasoning, case, profession)

        st.success(f"🏆 Score: {total}/10")

        # =====================
        # FEEDBACK
        # =====================
        st.markdown("## 🧠 AI Examiner Feedback")

        if level=="correct":
            st.success("Excellent diagnostic accuracy")
        elif level=="close":
            st.warning("Close but not precise")
        else:
            st.error("Incorrect diagnosis")

        st.write("Correct Answer:", target)

        st.markdown("## 🔍 Key Features Used")
        st.write(used)

        missing = [k for k in case.get("key_points",[]) if k not in used]
        st.warning(f"Missing: {missing}")

        st.markdown("## 🌳 Your Clinical Thinking")
        if steps:
            for i,s in enumerate(steps):
                st.success(f"Step {i+1}: {s}")
        else:
            st.error("No structured reasoning detected")

        # Confidence check
        if confidence > 80 and total < 5:
            st.warning("⚠️ Overconfidence detected")

        # Save
        row = {
            "user": user,
            "block": case["block"],
            "profession": profession,
            "score": total,
            "time": datetime.now()
        }

        df = pd.DataFrame([row])

        try:
            old = pd.read_csv("responses.csv")
            df = pd.concat([old,df])
        except:
            pass

        df.to_csv("responses.csv",index=False)

# ==================================================
# RIGHT: TEAM + ANALYTICS
# ==================================================
with col2:

    st.markdown("## 👥 Team Decision Board")

    ipa = case.get("interprofessional_answers", {})

    for role, ans in ipa.items():
        if role == profession:
            st.success(f"🟢 {role}\n{ans}")
        else:
            st.info(f"{role}\n{ans}")

    st.markdown("## ❌ Common Mistakes")
    st.error("""
    - Missed key symptom  
    - Ignored lab data  
    - Jumped to conclusion  
    """)

    st.markdown("## 📖 Reference")
    ref = case.get("reference", {})
    st.write(f"{ref.get('source','Unknown')} ({ref.get('year','-')})")
    # =====================
    # ANALYTICS
    # =====================
    st.markdown("## 📊 Your Performance")

    try:
        df = pd.read_csv("responses.csv")

        user_df = df[df["user"]==user]

        if len(user_df)>0:
            st.line_chart(user_df["score"])

            weak_block = user_df.groupby("block")["score"].mean().idxmin()
            st.warning(f"Weakest Block: {weak_block}")

    except:
        st.info("No data yet")

# ==================================================
# BATTLE MODE
# ==================================================
if mode == "Battle":

    st.markdown("## ⚔️ Team Battle")

    players = st.text_input("Players (comma separated)")

    if players:
        players = [p.strip() for p in players.split(",")]

        scores = {}

        for p in players:
            dx = st.text_input(f"{p} Answer", key=f"{p}_dx")
            rs = st.text_area(f"{p} Reasoning", key=f"{p}_rs")

            if st.button(f"Submit {p}", key=f"{p}_btn"):
                total, *_ = evaluate(dx, rs, case, profession)
                scores[p] = total

        if scores:
            st.markdown("## 🏆 Leaderboard")
            leaderboard = sorted(scores.items(), key=lambda x: x[1], reverse=True)

            for i,(name,score) in enumerate(leaderboard):
                st.write(f"{i+1}. {name} — {score}")
