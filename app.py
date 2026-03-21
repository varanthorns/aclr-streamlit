
            r_score += 1
            used.append(k)

    logic_words = ["because","therefore","thus","so","เนื่องจาก","ดังนั้น"]
    if any(w in reasoning.lower() for w in logic_words):
        r_score += 1

    if len(reasoning.split()) > 20:
        r_score += 1

    r_score = min(5, r_score)

    total = dx_score + r_score

    # ===== Examiner Feedback =====
    feedback = case.get("examiner_feedback","")

    if dx_level == "wrong":
        feedback = "❌ Diagnosis incorrect. " + feedback
    elif dx_level == "close":
        feedback = "⚠️ Close, but not precise. " + feedback
    else:
        feedback = "✅ Good diagnostic accuracy. " + feedback

    return dx_score, r_score, total, sim, used, feedback

# ===================== STATS =====================
def compute_stats(df):

    results = {}

    scores = df["score"].values
    results["n"] = len(scores)
    results["mean"] = np.mean(scores)
    results["sd"] = np.std(scores)

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

# ===== NEW: BERT =====
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

# ===== NEW: BERT SEMANTIC =====
def bert_similarity(a, b):
    try:
        emb1 = bert_model.encode(a, convert_to_tensor=True)
        emb2 = bert_model.encode(b, convert_to_tensor=True)
        return float(util.cos_sim(emb1, emb2))
    except:
        return 0

# ===================== ORIGINAL SCORING =====================
def evaluate(dx, reasoning, case):

    sim = semantic_score(dx, case["answer"])
    
    if normalize(dx) == normalize(case["answer"]):
        dx_score = 5
        dx_level = "correct"
    elif sim > 0.65:
        dx_score = 3
        dx_level = "close"
    else:
        dx_score = 0
        dx_level = "wrong"

    r_score = 0
    used = []
    
    for k in case.get("key_points", []):
        if k.lower() in reasoning.lower():
            r_score += 1
            used.append(k)

    if "because" in reasoning.lower() or "ดังนั้น" in reasoning:
        r_score += 1

    if len(reasoning.split()) > 20:
        r_score += 1

    r_score = min(5, r_score)
    total = dx_score + r_score

    feedback = case.get("examiner_feedback","")

    return dx_score, r_score, total, sim, used, feedback

# ===================== NEW: BERT GRADING =====================
def bert_grade(reasoning, case):

    key_text = " ".join(case.get("key_points", []))
    sim = bert_similarity(reasoning, key_text)

    if sim > 0.75:
        score = 5
        level = "excellent"
    elif sim > 0.55:
        score = 3
        level = "moderate"
    else:
        score = 1
        level = "poor"

    return score, sim, level

# ===================== NEW: UWorld Reasoning =====================
def uworld_feedback(case):

    return f"""
### 🧠 Step 1: Key Clinical Clues
- {", ".join(case.get("key_points", []))}

### 🔍 Step 2: Interpretation
These findings suggest **{case["answer"]}**

### ⚖️ Step 3: Why not others?
Incorrect options are ruled out based on missing key features.

### 🎯 Step 4: Final Diagnosis
**{case["answer"]}**

### 📚 Concept Summary
{case.get("learning_objective","")}
"""

# ===================== STATS =====================
def compute_stats(df):

    results = {}

    scores = df["score"].values
    results["mean"] = np.mean(scores)
    results["sd"] = np.std(scores)

    return results

# ===================== UI =====================
st.title("🔥 AI Clinical Reasoning Platform (BERT + UWorld)")

user_id = st.text_input("Enter User ID")

if not user_id:
    st.stop()

block = st.selectbox("Block", ["All"] + list(set(c["block"] for c in cases)))
difficulty = st.selectbox("Difficulty", ["adaptive","easy","medium","hard"])

filtered = cases

if block != "All":
    filtered = [c for c in filtered if c["block"] == block]

case = random.choice(filtered)

st.subheader("📋 Clinical Case")
st.write(case["scenario"]["en"])

dx = st.text_input("Diagnosis")
reasoning = st.text_area("Clinical Reasoning")

if st.button("Submit"):

    dx_s, r_s, total, sim, used, fb = evaluate(dx, reasoning, case)

    # ===== NEW BERT =====
    bert_s, bert_sim, level = bert_grade(reasoning, case)

    final_score = dx_s + bert_s

    st.success(f"Total Score: {final_score}/10")

    st.write("Diagnosis Score:", dx_s)
    st.write("BERT Reasoning Score:", bert_s)
    st.write("BERT Similarity:", round(bert_sim,2))

    st.markdown("### 🧠 AI Examiner Feedback")
    st.info(fb)

    st.markdown("### 📊 NLP Evaluation")
    st.write(f"Reasoning quality: {level}")

    st.markdown("### 🧠 UWorld Explanation")
    st.markdown(uworld_feedback(case))

    st.markdown("### 📖 Reference")
    st.write(case["reference"]["source"], case["reference"]["year"])

    # SAVE
    row = {
        "user": user_id,
        "score": final_score,
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
try:
    df = pd.read_csv("responses.csv")
    st.subheader("📈 Performance")
    st.line_chart(df["score"])
except:
    pass

# ===================== UI =====================
st.title("🧠 ACLR Platform (AI Clinical Learning & Reasoning)")

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
        mode_select = st.selectbox("Question Type",["All","keyword","scenario"])
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

    st.subheader(f"{case['block']} | {case['difficulty']} | {case.get('level','')}")
    st.caption(f"Adaptive difficulty: {st.session_state.difficulty}")

    # ===== DISPLAY CASE =====
    if case.get("mode") == "keyword":
        st.markdown("### 🔑 Keywords")
        st.write(case["scenario"][lang])
    else:
        st.markdown("### 📋 Clinical Scenario")
        st.write(case["scenario"][lang])
        st.write(case.get("additional", {}).get(lang, ""))

    dx = st.text_input("Diagnosis")
    reasoning = st.text_area("Reasoning")

    if st.button("Submit"):

        dx_s, r_s, total, sim, used, feedback = evaluate(dx, reasoning, case)

        st.success(f"Score: {total}/10")
        st.write("Diagnosis score:", dx_s)
        st.write("Reasoning score:", r_s)
        st.write("Similarity:", round(sim,2))
        st.write("Key features used:", used)

        missing = [k for k in case.get("key_points",[]) if k not in used]
        st.warning(f"Missing key features: {missing}")

        st.markdown("### 🧠 AI Examiner Feedback")
        st.info(feedback)

        st.markdown("### 📚 Explanation")
        st.write(case.get("explanation",""))

        st.markdown("### 🎯 Learning Objective")
        st.write(case.get("learning_objective",""))

        st.markdown("### 📖 Reference")
        st.write(f"{case['reference']['source']} ({case['reference']['year']})")

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

        # Learning curve
        df["attempt"] = range(len(df))
        st.subheader("Learning Curve")
        st.line_chart(df.set_index("attempt")["score"])

        # Block performance
        st.subheader("Performance by Block")
        block_perf = df.groupby("block")["score"].mean()
        st.bar_chart(block_perf)

        st.subheader("Per User Analysis")
        user_df = df[df["user"]==user_id]

        if len(user_df)>2:
            stats_user = compute_stats(user_df)
            st.write(stats_user)
            st.line_chart(user_df["score"])

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
3. ใส่ diagnosis
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
- มี causal logic
- มี structured thinking

---

## 📈 การแปลผล
- 8–10 = expert
- 5–7 = intermediate
- <5 = improve

---

## 📊 Statistical Analysis
- Mean / SD
- t-test
- Effect size (Cohen’s d)

---

## 🤖 AI Features
- Adaptive difficulty
- Examiner-style feedback
- Learning curve tracking
""")
