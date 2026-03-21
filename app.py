import streamlit as st
import json, random, pandas as pd
from datetime import datetime
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(layout="wide")

# -------------------- LOAD --------------------
@st.cache_data
def load_cases():
    with open("cases.json","r",encoding="utf-8") as f:
        return json.load(f)

cases = load_cases()

# -------------------- UTILS --------------------
def normalize(t): return t.lower().strip()

def semantic_score(a,b):
    try:
        vec = TfidfVectorizer().fit_transform([a,b])
        return cosine_similarity(vec[0:1], vec[1:2])[0][0]
    except:
        return 0

# -------------------- SCORING --------------------
def evaluate(dx, reasoning, case):

    sim = semantic_score(dx, case["answer"])

    dx_score = 5 if sim>0.7 else 3 if sim>0.4 else 0

    r_score = 0
    for k in case.get("key_points",[]):
        if k.lower() in reasoning.lower():
            r_score += 1

    r_score = min(5, r_score)

    total = dx_score + r_score

    return dx_score, r_score, total, sim

# -------------------- STATS --------------------
def compute_stats(df):

    results = {}

    scores = df["score"].values

    results["n"] = len(scores)
    results["mean"] = np.mean(scores)
    results["sd"] = np.std(scores)

    # split early vs late
    mid = len(scores)//2
    early = scores[:mid]
    late = scores[mid:]

    if len(early)>1 and len(late)>1:
        diff = np.mean(late) - np.mean(early)
        pooled_sd = np.sqrt((np.var(early)+np.var(late))/2)
        d = diff / pooled_sd if pooled_sd!=0 else 0

        results["early_mean"] = np.mean(early)
        results["late_mean"] = np.mean(late)
        results["effect_size"] = d

    return results

# -------------------- SESSION --------------------
if "difficulty" not in st.session_state:
    st.session_state.difficulty = "easy"

def adjust(score):
    if score>=8: return "hard"
    elif score>=5: return "medium"
    return "easy"

# -------------------- STYLE --------------------
st.markdown("""
<style>
.big-card {
    padding:20px;
    border-radius:15px;
    background-color:#f5f7fa;
    margin-bottom:10px;
}
</style>
""", unsafe_allow_html=True)

# -------------------- TABS --------------------
tab1, tab2, tab3 = st.tabs(["🧠 Practice","📊 Analytics","📜 History"])

# ==================================================
# 🧠 PRACTICE
# ==================================================
with tab1:

    st.title("🧠 ACLR Platform")

    col1, col2, col3 = st.columns(3)

    with col1:
        language = st.selectbox("Language",["English","Thai"])
    with col2:
        blocks = sorted(list(set([c["block"] for c in cases])))
        block = st.selectbox("Block",["All"]+blocks)
    with col3:
        mode = st.selectbox("Difficulty",["adaptive","easy","medium","hard"])

    lang = "en" if language=="English" else "th"

    filtered = cases
    if block!="All":
        filtered = [c for c in filtered if c["block"]==block]

    if mode!="adaptive":
        filtered = [c for c in filtered if c.get("difficulty","easy")==mode]
    else:
        filtered = [c for c in filtered if c.get("difficulty","easy")==st.session_state.difficulty]

    if not filtered:
        filtered = cases

    if "case" not in st.session_state:
        st.session_state.case = random.choice(filtered)

    if st.button("🔄 New Case"):
        st.session_state.case = random.choice(filtered)

    case = st.session_state.case

    st.markdown("### 📋 Case")
    st.markdown(f"<div class='big-card'>{case['scenario'][lang]}<br><br>{case['additional'][lang]}</div>", unsafe_allow_html=True)

    dx = st.text_input("Diagnosis")
    reasoning = st.text_area("Reasoning")

    if st.button("Submit"):

        dx_s, r_s, total, sim = evaluate(dx, reasoning, case)

        col1, col2, col3 = st.columns(3)
        col1.metric("Diagnosis", dx_s)
        col2.metric("Reasoning", r_s)
        col3.metric("Total", total)

        st.progress(total/10)

        if mode=="adaptive":
            st.session_state.difficulty = adjust(total)

        row = {
            "time": datetime.now(),
            "case_id": case["case_id"],
            "block": case["block"],
            "difficulty": case["difficulty"],
            "score": total
        }

        df = pd.DataFrame([row])

        try:
            old = pd.read_csv("responses.csv")
            df = pd.concat([old,df])
        except:
            pass

        df.to_csv("responses.csv",index=False)

# ==================================================
# 📊 ANALYTICS
# ==================================================
with tab2:

    st.title("📊 Analytics Dashboard")

    try:
        df = pd.read_csv("responses.csv")

        stats = compute_stats(df)

        col1, col2, col3 = st.columns(3)
        col1.metric("N", stats["n"])
        col2.metric("Mean Score", round(stats["mean"],2))
        col3.metric("SD", round(stats["sd"],2))

        if "effect_size" in stats:
            st.markdown("### 📈 Learning Gain")
            col1, col2, col3 = st.columns(3)
            col1.metric("Early Mean", round(stats["early_mean"],2))
            col2.metric("Late Mean", round(stats["late_mean"],2))
            col3.metric("Effect Size (d)", round(stats["effect_size"],2))

        st.line_chart(df["score"])
        st.bar_chart(df["block"].value_counts())

    except:
        st.info("No data yet")

# ==================================================
# 📜 HISTORY
# ==================================================
with tab3:

    st.title("📜 History")

    try:
        df = pd.read_csv("responses.csv")

        st.dataframe(df.sort_values("time",ascending=False))

        st.download_button(
            "Download CSV",
            df.to_csv(index=False),
            "results.csv"
        )

    except:
        st.info("No history yet")
