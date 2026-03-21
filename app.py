import streamlit as st
import json, random, pandas as pd
from datetime import datetime
import numpy as np
from scipy import stats
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util

st.set_page_config(layout="wide")

# ===================== LOAD =====================
@st.cache_data
def load_cases():
    with open("cases.json","r",encoding="utf-8") as f:
        return json.load(f)

cases = load_cases()

# ===================== LOAD BERT =====================
@st.cache_resource
def load_bert():
    return SentenceTransformer("all-MiniLM-L6-v2")

bert_model = load_bert()

# ===================== UTILS =====================
def normalize(t): return str(t).lower().strip()

def semantic_score(a,b):
    try:
        vec = TfidfVectorizer().fit_transform([a,b])
        return cosine_similarity(vec[0:1], vec[1:2])[0][0]
    except:
        return 0

def bert_similarity(a, b):
    try:
        emb1 = bert_model.encode(a, convert_to_tensor=True)
        emb2 = bert_model.encode(b, convert_to_tensor=True)
        return float(util.cos_sim(emb1, emb2))
    except:
        return 0

# ===================== DECISION TREE =====================
def extract_decision_steps(reasoning):
    steps = []
    keywords = ["because","therefore","thus","so","เนื่องจาก","ดังนั้น"]
    for s in reasoning.split("."):
        if any(k in s.lower() for k in keywords):
            steps.append(s.strip())
    return steps

def decision_tree_score(reasoning):
    steps = extract_decision_steps(reasoning)
    return min(3, len(steps)), steps

# ===================== SCORING =====================
def evaluate(dx, reasoning, case):

    sim = semantic_score(dx, case["answer"])
    
    if normalize(dx) == normalize(case["answer"]):
        dx_score = 5
    elif sim > 0.65:
        dx_score = 3
    else:
        dx_score = 0

    key_text = " ".join(case.get("key_points", []))
    bert_sim = bert_similarity(reasoning, key_text)

    if bert_sim > 0.75:
        r_score = 5
    elif bert_sim > 0.55:
        r_score = 3
    else:
        r_score = 1

    dt_score, steps = decision_tree_score(reasoning)

    total = dx_score + r_score + dt_score
    total = min(10, total)

    used = [k for k in case.get("key_points", []) if k.lower() in reasoning.lower()]
    missing = [k for k in case.get("key_points", []) if k not in used]

    return dx_score, r_score, dt_score, total, bert_sim, used, missing, steps

# ===================== STATS =====================
def compute_stats(df):

    scores = df["score"].values
    results = {
        "n": len(scores),
        "mean": np.mean(scores),
        "sd": np.std(scores)
    }

    if len(scores) > 4:
        mid = len(scores)//2
        early = scores[:mid]
        late = scores[mid:]

        t, p = stats.ttest_ind(late, early, equal_var=False)

        pooled_sd = np.sqrt((np.var(early)+np.var(late))/2)
        d = (np.mean(late)-np.mean(early))/pooled_sd if pooled_sd!=0 else 0

        results.update({
            "p_value": p,
            "effect_size": d
        })

    return results

# ===================== ADAPTIVE =====================
if "difficulty" not in st.session_state:
    st.session_state.difficulty = "easy"

if "recent_scores" not in st.session_state:
    st.session_state.recent_scores = []

def adjust(score):
    history = st.session_state.recent_scores
    history.append(score)

    if len(history) > 5:
        history.pop(0)

    avg = sum(history)/len(history)

    if avg >= 8:
        return "hard"
    elif avg >= 5:
        return "medium"
    return "easy"

# ===================== UI =====================
st.title("🧠 ACLR Platform (AI Clinical Reasoning Trainer)")

user_id = st.text_input("Enter Student ID / Name")

if not user_id:
    st.warning("Please enter user ID to start")
    st.stop()

tab1, tab2, tab3, tab4 = st.tabs([
    "🧠 Practice",
    "📊 Analytics",
    "📜 History",
    "📘 Guide"
])

# ===================== PRACTICE =====================
with tab1:

    col1, col2, col3 = st.columns(3)

    with col1:
        language = st.selectbox("Language",["English","Thai"])
    with col2:
        blocks = sorted(list(set([c["block"] for c in cases])))
        block = st.selectbox("Block",["All"]+blocks)
        mode_select = st.selectbox("Type",["All","keyword","scenario"])
    with col3:
        mode = st.selectbox("Difficulty",["adaptive","easy","medium","hard"])

    lang = "en" if language=="English" else "th"

    filtered = cases

    if block != "All":
        filtered = [c for c in filtered if c["block"] == block]

    if mode_select != "All":
        filtered = [c for c in filtered if c.get("mode","scenario")==mode_select]

    if mode!="adaptive":
        filtered = [c for c in filtered if c.get("difficulty","easy")==mode]
    else:
        filtered = [c for c in filtered if c.get("difficulty","easy")==st.session_state.difficulty]

    if not filtered:
        filtered = cases

    if "case" not in st.session_state:
        st.session_state.case = random.choice(filtered)

    if st.button("New Case"):
        st.session_state.case = random.choice(filtered)

    case = st.session_state.case

    st.subheader(f"{case['block']} | {case['difficulty']}")
    st.caption(f"Adaptive: {st.session_state.difficulty}")

    st.write(case["scenario"][lang])
    st.write(case.get("additional", {}).get(lang, ""))

    dx = st.text_input("Diagnosis")
    reasoning = st.text_area("Reasoning")

    if st.button("Submit"):

        dx_s, r_s, dt_s, total, bert_sim, used, missing, steps = evaluate(dx, reasoning, case)

        st.success(f"🔥 Score: {total}/10")

        st.write("Diagnosis:", dx_s)
        st.write("Reasoning (BERT):", r_s)
        st.write("Decision structure:", dt_s)
        st.write("Semantic similarity:", round(bert_sim,2))

        st.markdown("### 🌳 Reasoning Steps")
        for i, s in enumerate(steps):
            st.write(f"{i+1}. {s}")

        st.warning(f"Missing key points: {missing}")

        st.markdown("### 📚 Explanation")
        st.write(case.get("explanation",""))

        st.markdown("### 📖 Reference")
        st.write(f"{case['reference']['source']} ({case['reference']['year']})")

        if mode=="adaptive":
            st.session_state.difficulty = adjust(total)

        row = {
            "user": user_id,
            "score": total,
            "block": case["block"],
            "time": datetime.now()
        }

        df = pd.DataFrame([row])

        try:
            old = pd.read_csv("responses.csv")
            df = pd.concat([old, df])
        except:
            pass

        df.to_csv("responses.csv", index=False)

# ===================== ANALYTICS =====================
with tab2:

    try:
        df = pd.read_csv("responses.csv")

        st.write(compute_stats(df))

        st.subheader("Learning Curve")
        df["attempt"] = range(len(df))
        st.line_chart(df.set_index("attempt")["score"])

        st.subheader("Block Performance")
        st.bar_chart(df.groupby("block")["score"].mean())

    except:
        st.info("No data yet")

# ===================== HISTORY =====================
with tab3:

    try:
        df = pd.read_csv("responses.csv")
        st.dataframe(df[df["user"]==user_id])
    except:
        st.info("No history")

# ===================== GUIDE =====================
with tab4:

    st.markdown("""
## AI Clinical Reasoning Trainer

- Diagnosis (0–5)
- Reasoning via BERT (0–5)
- Decision steps (0–3)

Adaptive learning + analytics included
""")
