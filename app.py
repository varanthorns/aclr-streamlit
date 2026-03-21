import streamlit as st
import json, random, pandas as pd
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(layout="wide")

# -------------------- LOAD --------------------
@st.cache_data
def load_cases():
    with open("cases.json","r",encoding="utf-8") as f:
        return json.load(f)

cases = load_cases()

# -------------------- SYNONYMS --------------------
SYNONYMS = {
    "myocardial infarction": ["mi","heart attack","acute coronary syndrome"],
    "stroke": ["cva","cerebrovascular accident"],
    "diabetes mellitus": ["diabetes"],
    "pulmonary embolism": ["pe"],
}

def normalize(t): return t.lower().strip()

# -------------------- SEMANTIC --------------------
def semantic_score(a,b):
    try:
        vec = TfidfVectorizer().fit_transform([a,b])
        return cosine_similarity(vec[0:1], vec[1:2])[0][0]
    except:
        return 0

# -------------------- REASONING --------------------
def reasoning_analysis(text, case):
    text = normalize(text)
    score = 0
    explain = []

    for k in case.get("key_points",[]):
        if k.lower() in text:
            score += 1
            explain.append(f"✔ key: {k}")

    if "because" in text or "เนื่องจาก" in text:
        score += 1
        explain.append("✔ causal reasoning")

    if "therefore" in text or "ดังนั้น" in text:
        score += 1
        explain.append("✔ conclusion logic")

    return score, explain

# -------------------- SCORING --------------------
def evaluate(dx, reasoning, case):

    dx_n = normalize(dx)
    ans = normalize(case["answer"])

    sim = semantic_score(dx_n, ans)

    # diagnosis
    if sim > 0.7:
        dx_score = 5
    elif sim > 0.4:
        dx_score = 3
    elif any(s in dx_n for s in SYNONYMS.get(ans,[])):
        dx_score = 3
    else:
        dx_score = 0

    # differential bonus
    if dx_score == 0:
        if any(d.lower() in dx_n for d in str(case.get("distractor","")).split()):
            dx_score = 2

    # reasoning
    r_raw, r_exp = reasoning_analysis(reasoning, case)
    r_score = min(5, r_raw)

    # bias
    bias = []
    if len(reasoning.split()) < 5:
        bias.append("Premature closure")
    if "first" in reasoning:
        bias.append("Anchoring bias")
    if "definitely" in reasoning:
        bias.append("Overconfidence bias")

    total = dx_score + r_score

    feedback = f"""
Correct: {case['answer']}
Similarity: {round(sim,2)}

Reasoning:
{chr(10).join(r_exp)}

Improve:
- Add key features
- Use structured logic
"""

    return dx_score, r_score, total, bias, feedback

# -------------------- SAFE RANDOM --------------------
def safe_choice(primary, fallback):
    if primary: return random.choice(primary)
    if fallback: return random.choice(fallback)
    return None

# -------------------- ADAPTIVE --------------------
if "difficulty" not in st.session_state:
    st.session_state.difficulty = "easy"

def adjust(score):
    if score >= 8: return "hard"
    elif score >= 5: return "medium"
    else: return "easy"

# -------------------- UI --------------------
st.title("🧠 ACLR Full Offline System")

language = st.selectbox("Language",["English","Thai"])
lang = "en" if language=="English" else "th"

blocks = sorted(list(set([c["block"] for c in cases])))
block = st.selectbox("Block",["All"]+blocks)

mode = st.selectbox("Difficulty Mode",["adaptive","easy","medium","hard"])

# filter
filtered = cases

if block!="All":
    filtered = [c for c in filtered if c["block"]==block]

if mode!="adaptive":
    filtered = [c for c in filtered if c.get("difficulty","easy")==mode]
else:
    filtered = [c for c in filtered if c.get("difficulty","easy")==st.session_state.difficulty]

# fallback smart
if not filtered:
    st.warning("No exact match → fallback to closest difficulty")
    if st.session_state.difficulty == "hard":
        filtered = [c for c in cases if c.get("difficulty")=="medium"]
    elif st.session_state.difficulty == "medium":
        filtered = [c for c in cases if c.get("difficulty")=="easy"]
    else:
        filtered = cases

# spaced repetition (mistakes)
def get_weak_cases():
    try:
        df = pd.read_csv("responses.csv")
        weak_ids = df[df["score"]<5]["case_id"].values
        return [c for c in cases if c["case_id"] in weak_ids]
    except:
        return []

weak_cases = get_weak_cases()

if random.random() < 0.3 and weak_cases:
    filtered = weak_cases

# select case
if "case" not in st.session_state:
    st.session_state.case = safe_choice(filtered, cases)

if st.button("New Case"):
    st.session_state.case = safe_choice(filtered, cases)

case = st.session_state.case

if case is None:
    st.error("No cases available")
    st.stop()

# display
st.subheader("Case")
st.write(case.get("scenario",{}).get(lang,""))
st.write(case.get("additional",{}).get(lang,""))

dx = st.text_input("Diagnosis")
reasoning = st.text_area("Reasoning")

# -------------------- SUBMIT --------------------
if st.button("Submit"):

    dx_s, r_s, total, bias, feedback = evaluate(dx, reasoning, case)

    st.success(f"Score: {total}/10")
    st.write("Diagnosis:", dx_s)
    st.write("Reasoning:", r_s)
    st.write("Bias:", bias if bias else "None")
    st.write("Feedback")
    st.write(feedback)

    # adaptive
    if mode=="adaptive":
        st.session_state.difficulty = adjust(total)
        st.info(f"Next difficulty: {st.session_state.difficulty}")

    # save
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

# -------------------- DASHBOARD --------------------
st.divider()
st.header("📊 Dashboard")

try:
    df = pd.read_csv("responses.csv")

    st.line_chart(df["score"])
    st.write("Average:", round(df["score"].mean(),2))
    st.bar_chart(df["block"].value_counts())

    st.subheader("History")
    st.dataframe(df.tail(10))

except:
    st.info("No data yet")
