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

def bert_similarity(a,b):
    try:
        emb1 = bert_model.encode(a, convert_to_tensor=True)
        emb2 = bert_model.encode(b, convert_to_tensor=True)
        return float(util.cos_sim(emb1, emb2))
    except:
        return 0

# ===================== DECISION TREE =====================
def extract_steps(reasoning):
    keywords = ["because","therefore","thus","so","เนื่องจาก","ดังนั้น"]
    return [s.strip() for s in reasoning.split(".") if any(k in s.lower() for k in keywords)]

def decision_score(reasoning):
    steps = extract_steps(reasoning)
    return min(3,len(steps)), steps

# ===================== SCORING =====================
def evaluate(dx, reasoning, case, profession):

    if "interprofessional_answers" in case:
        target = case["interprofessional_answers"][profession]
    else:
        target = case["answer"]

    sim = semantic_score(dx, target)

    if normalize(dx) == normalize(target):
        dx_score = 5
    elif sim > 0.65:
        dx_score = 3
    else:
        dx_score = 0

    key_text = " ".join(case.get("key_points",[]))
    bert_sim = bert_similarity(reasoning, key_text)

    if bert_sim > 0.75:
        r_score = 5
    elif bert_sim > 0.55:
        r_score = 3
    else:
        r_score = 1

    dt_score, steps = decision_score(reasoning)

    total = min(10, dx_score + r_score + dt_score)

    used = [k for k in case.get("key_points",[]) if k.lower() in reasoning.lower()]
    missing = [k for k in case.get("key_points",[]) if k not in used]

    return dx_score, r_score, dt_score, total, bert_sim, used, missing, steps, target

# ===================== STATS =====================
def compute_stats(df):

    scores = df["score"].values

    result = {
        "n": len(scores),
        "mean": np.mean(scores),
        "sd": np.std(scores)
    }

    if len(scores)>4:
        mid = len(scores)//2
        early = scores[:mid]
        late = scores[mid:]

        t,p = stats.ttest_ind(late,early,equal_var=False)

        pooled = np.sqrt((np.var(early)+np.var(late))/2)
        d = (np.mean(late)-np.mean(early))/pooled if pooled!=0 else 0

        result.update({"p_value":p,"effect_size":d})

    return result

# ===================== ADAPTIVE =====================
if "difficulty" not in st.session_state:
    st.session_state.difficulty = "easy"

if "recent_scores" not in st.session_state:
    st.session_state.recent_scores = []

def adjust(score):
    hist = st.session_state.recent_scores
    hist.append(score)
    if len(hist)>5:
        hist.pop(0)
    avg = sum(hist)/len(hist)
    if avg>=8: return "hard"
    elif avg>=5: return "medium"
    return "easy"

# ===================== UI =====================
st.title("🧠 Healthcare Clinical Reasoning Platform")

user = st.text_input("Enter Student ID / Name", key="user_input")
if not user:
    st.stop()

tab1, tab2, tab3, tab4 = st.tabs(["🧠 Practice","📊 Analytics","📜 History","📘 Guide"])

# ===================== PRACTICE =====================
with tab1:

    col1,col2,col3,col4 = st.columns(4)

    with col1:
        language = st.selectbox("Language",["English","Thai"], key="lang")
    with col2:
        block = st.selectbox("Block",["All"]+list(set(c["block"] for c in cases)), key="block")
    with col3:
        difficulty = st.selectbox("Difficulty",["adaptive","easy","medium","hard"], key="difficulty")
    with col4:
        profession = st.selectbox(
            "Profession",
            ["medicine","nursing","pharmacy","lab","public_health","physio","dentistry"],
            key="profession"
        )

    mode_type = st.selectbox("Mode",["Learning","Exam"], key="mode")

    lang = "en" if language=="English" else "th"

    # ===== FILTER =====
    filtered = cases

    if block!="All":
        filtered = [c for c in filtered if c["block"]==block]

    if difficulty!="adaptive":
        filtered = [c for c in filtered if c.get("difficulty","easy")==difficulty]
    else:
        filtered = [c for c in filtered if c.get("difficulty","easy")==st.session_state.difficulty]

    if not filtered:
        st.warning("No cases available")
        st.stop()

    # ===== FIX CASE RANDOM =====
    current_filter = f"{block}_{difficulty}_{profession}"

    if "last_filter" not in st.session_state:
        st.session_state.last_filter = None

    if "case" not in st.session_state:
        st.session_state.case = random.choice(filtered)

    if st.session_state.last_filter != current_filter:
        st.session_state.case = random.choice(filtered)
        st.session_state.last_filter = current_filter

    if st.button("New Case", key="new_case_btn"):
        st.session_state.case = random.choice(filtered)

    case = st.session_state.case

    st.subheader(f"{case['block']} | {case['difficulty']}")
    st.caption(f"Adaptive: {st.session_state.difficulty}")

    # ===== CASE =====
    st.markdown("### 📋 Clinical Scenario")
    st.write(case["scenario"][lang])
    st.write(case.get("additional",{}).get(lang,""))

    # ===== TASK =====
    st.markdown("### 🎯 Task")
    if "task" in case:
        st.info(case["task"][profession])
    else:
        st.info("What is the diagnosis?")

    dx = st.text_input("Your Answer", key="dx")
    reasoning = st.text_area("Clinical Reasoning", key="reason")

    if st.button("Submit", key="submit_btn"):

        dx_s, r_s, dt_s, total, sim, used, missing, steps, target = evaluate(dx, reasoning, case, profession)

        st.success(f"🔥 Score: {total}/10")

        st.markdown("### 🎓 Competency")
        st.write({
            "Clinical Knowledge": dx_s,
            "Reasoning": r_s,
            "Decision Process": dt_s
        })

        st.write("Semantic similarity:", round(sim,2))

        st.markdown("### 🌳 Reasoning Steps")
        for i,s in enumerate(steps):
            st.write(f"{i+1}. {s}")

        st.warning(f"Missing key points: {missing}")

        if mode_type=="Learning":

            st.markdown("### ✅ Correct Answer")
            st.success(target)

            st.markdown("### 📚 Explanation")
            st.write(case.get("explanation",""))

            st.markdown("### 🎯 Learning Objective")
            st.write(case.get("learning_objective",""))

            st.markdown("### 👥 Interprofessional View")
            if "interprofessional_answers" in case:
                for role,ans in case["interprofessional_answers"].items():
                    st.write(f"{role}: {ans}")

            st.markdown("### 📖 Reference")
            st.write(f"{case['reference']['source']} ({case['reference']['year']})")

        if difficulty=="adaptive":
            st.session_state.difficulty = adjust(total)

        row = {
            "user":user,
            "score":total,
            "block":case["block"],
            "profession":profession,
            "time":datetime.now()
        }

        df = pd.DataFrame([row])

        try:
            old = pd.read_csv("responses.csv")
            df = pd.concat([old,df])
        except:
            pass

        df.to_csv("responses.csv",index=False)

# ===================== ANALYTICS =====================
with tab2:
    try:
        df = pd.read_csv("responses.csv")

        st.write(compute_stats(df))

        df["attempt"] = range(len(df))
        st.line_chart(df.set_index("attempt")["score"])

        st.subheader("By Block")
        st.bar_chart(df.groupby("block")["score"].mean())

        st.subheader("By Profession")
        st.bar_chart(df.groupby("profession")["score"].mean())

    except:
        st.info("No data")

# ===================== HISTORY =====================
with tab3:
    try:
        df = pd.read_csv("responses.csv")
        st.dataframe(df[df["user"]==user])
    except:
        st.info("No history")

# ===================== GUIDE =====================
with tab4:
    st.markdown("""
## 🧠 Healthcare Clinical Reasoning Platform

- Interprofessional (7 professions)
- AI reasoning (BERT)
- Decision thinking
- Adaptive difficulty
- Learning / Exam mode

Score:
- Knowledge (0–5)
- Reasoning (0–5)
- Decision (0–3)
""")
