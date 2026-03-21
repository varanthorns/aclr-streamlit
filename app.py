import streamlit as st
import json, random, pandas as pd
from datetime import datetime
import numpy as np
from scipy import stats
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

# ===================== SCORING =====================
def evaluate(dx, reasoning, case):

    sim = semantic_score(dx, case["answer"])
    dx_score = 5 if sim>0.7 else 3 if sim>0.4 else 0

    r_score = 0
    used = []
    for k in case.get("key_points",[]):
        if k.lower() in reasoning.lower():
            r_score += 1
            used.append(k)

    if "because" in reasoning.lower() or "ดังนั้น" in reasoning:
        r_score += 1

    r_score = min(5, r_score)
    total = dx_score + r_score

    return dx_score, r_score, total, sim, used

# ===================== STATS =====================
def compute_stats(df):

    results = {}

    scores = df["score"].values
    results["n"] = len(scores)
    results["mean"] = np.mean(scores)
    results["sd"] = np.std(scores)

    # early vs late
    mid = len(scores)//2
    early = scores[:mid]
    late = scores[mid:]

    if len(early)>1 and len(late)>1:
        t, p = stats.ttest_ind(late, early, equal_var=False)

        pooled_sd = np.sqrt((np.var(early)+np.var(late))/2)
        d = (np.mean(late)-np.mean(early))/pooled_sd if pooled_sd!=0 else 0

        results.update({
            "early_mean": np.mean(early),
            "late_mean": np.mean(late),
            "p_value": p,
            "t_stat": t,
            "effect_size": d
        })

    return results

# ===================== SESSION =====================
if "difficulty" not in st.session_state:
    st.session_state.difficulty = "easy"

def adjust(score):
    if score>=8: return "hard"
    elif score>=5: return "medium"
    return "easy"

# ===================== UI =====================
st.title("ACLR Platform")

# -------- USER LOGIN --------
user_id = st.text_input("Enter Student ID / Name")

if not user_id:
    st.warning("Please enter user ID to start")
    st.stop()

# -------- TABS --------
tab1, tab2, tab3, tab4 = st.tabs([
    "🧠 Practice",
    "📊 Analytics",
    "📜 History",
    "📘 Guide & Rubric"
])

# ==================================================
# 🧠 PRACTICE
# ==================================================
with tab1:

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

    if st.button("New Case"):
        st.session_state.case = random.choice(filtered)

    case = st.session_state.case

    st.subheader("Case")
    st.write(case["scenario"][lang])
    st.write(case["additional"][lang])

    dx = st.text_input("Diagnosis")
    reasoning = st.text_area("Reasoning")

    if st.button("Submit"):

        dx_s, r_s, total, sim, used = evaluate(dx, reasoning, case)

        st.success(f"Score: {total}/10")
        st.write("Diagnosis:", dx_s)
        st.write("Reasoning:", r_s)
        st.write("Similarity:", round(sim,2))
        st.write("Key features used:", used)

        if mode=="adaptive":
            st.session_state.difficulty = adjust(total)

        row = {
            "user": user_id,
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

    try:
        df = pd.read_csv("responses.csv")

        st.subheader("All Users Overview")
        stats_all = compute_stats(df)

        st.write(stats_all)

        st.line_chart(df["score"])

        st.subheader("Per User Analysis")
        user_df = df[df["user"]==user_id]

        if len(user_df)>2:
            stats_user = compute_stats(user_df)
            st.write(stats_user)

            st.line_chart(user_df["score"])

        # group comparison
        st.subheader("Group Comparison")
        group_mean = df.groupby("user")["score"].mean()
        st.bar_chart(group_mean)

    except:
        st.info("No data yet")

# ==================================================
# 📜 HISTORY
# ==================================================
with tab3:

    try:
        df = pd.read_csv("responses.csv")

        st.subheader("Your Attempts")
        st.dataframe(df[df["user"]==user_id].sort_values("time",ascending=False))

        st.subheader("All Data")
        st.dataframe(df.tail(20))

        st.download_button("Download CSV", df.to_csv(index=False), "results.csv")

    except:
        st.info("No history yet")

# ==================================================
# 📘 GUIDE
# ==================================================
with tab4:

    st.title("📘 User Guide & Scoring Rubric")

    st.markdown("""
## 🧠 วิธีใช้งาน
1. เลือก block และ difficulty
2. อ่านเคส
3. ใส่ diagnosis (คำเต็ม)
4. อธิบาย reasoning
5. กด Submit

---

## 📊 เกณฑ์การให้คะแนน

### Diagnosis (0–5)
- 5 = ถูกต้อง
- 3 = ใกล้เคียง
- 0 = ผิด

### Reasoning (0–5)
- ใช้ key features
- มีเหตุผลเชิง causal
- มี logical conclusion

---

## 📈 การแปลผล
- 8–10 = expert level
- 5–7 = intermediate
- <5 = needs improvement

---

## 📊 Statistical Analysis
ระบบคำนวณ:
- Mean / SD
- t-test (early vs late)
- p-value
- Effect size (Cohen’s d)

""")
