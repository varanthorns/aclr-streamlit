import streamlit as st
import json, random, pandas as pd, os, time
import google.generativeai as genai

# ===================== ⚙️ 1. GLOBAL CONFIG & UI SETUP =====================
DB_FILE = "clinical_scores.csv"

# ต้องอยู่บรรทัดแรกๆ และมีแค่ที่เดียว!
st.set_page_config(layout="wide", page_title="FTF-CRA Clinical Platform", page_icon="🩺")

# 🔐 1.1 API Setup
try:
    if "GEMINI_API_KEY" in st.secrets:
        GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
    else:
        GEMINI_API_KEY = "DEMO_KEY"
except Exception:
    GEMINI_API_KEY = "DEMO_KEY"

genai.configure(api_key=GEMINI_API_KEY)

# 💾 1.2 Database & Adaptive Functions
def save_score_local(user, role, score, block, competency=None, time_taken=0):
    new_entry = {
        "User": user, "Role": role, "Score": score, "Block": block,
        "Time": time_taken, "Timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    if competency:
        new_entry.update(competency)
    df_new = pd.DataFrame([new_entry])
    if os.path.exists(DB_FILE):
        df_old = pd.read_csv(DB_FILE)
        df = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df = df_new
    df.to_csv(DB_FILE, index=False)

def get_adaptive_difficulty(user):
    if os.path.exists(DB_FILE):
        df = pd.read_csv(DB_FILE)
        user_df = df[df["User"] == user]
        if len(user_df) >= 3:
            avg = user_df["Score"].mean()
            if avg < 6: return "easy"
            elif avg < 8: return "medium"
            else: return "hard"
    return "easy"

# 🩺 1.3 AI Feedback Engine
def get_ai_feedback_v9_5(user_dx, user_re, user_map, target, role, time_taken, confidence, stress):
    model = genai.GenerativeModel('gemini-1.5-flash')
    prompt = f"""Act as a Senior Clinical Professor. Evaluate this {role}'s clinical reasoning.
    Diagnosis: {user_dx} | Rationale: {user_re} | Map: {user_map} | Target: {target}
    Time: {time_taken}s | Confidence: {confidence}% | Stress: {stress}/10
    Return evaluation in Markdown with Metrics, Insight, and Professor's Pearl."""
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"🚨 AI Mentor Error: {str(e)}"

# ===================== 🎨 2. STYLING & DATA LOADING =====================
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stButton>button { width: 100%; border-radius: 8px; height: 3.5em; background-color: #1976D2 !important; color: white !important; font-weight: bold; }
    .stMetric { background-color: white; padding: 20px; border-radius: 12px; border-left: 5px solid #1976D2; box-shadow: 0 4px 6px rgba(0,0,0,0.05); }
    .stress-timer { font-size: 28px; font-weight: bold; color: #d32f2f; text-align: center; border: 3px solid #d32f2f; padding: 10px; border-radius: 15px; background: white; }
    </style>
    """, unsafe_allow_html=True)

@st.cache_data
def load_cases():
    if os.path.exists("cases.json"):
        with open("cases.json", "r", encoding="utf-8") as f: return json.load(f)
    return [{"block":"Cardiology", "difficulty":"hard", "scenario":{"en":"65yo Male presents with chest pain..."}, "labs":[], "answer":"Acute STEMI"}]

all_cases = load_cases()

# 🔄 Session State Init
if "case" not in st.session_state: st.session_state.case = all_cases[0]
if "submitted" not in st.session_state: st.session_state.submitted = False
if "ai_feedback" not in st.session_state: st.session_state.ai_feedback = ""
if "start_time" not in st.session_state: st.session_state.start_time = time.time()
if "evolved" not in st.session_state: st.session_state.evolved = False

# ===================== 🧭 3. SIDEBAR =====================
with st.sidebar:
    st.title("🩺 FTF-CRA v9.9.5")
    menu = st.radio("Main Menu", ["📖 Manual & Standards", "🧪 Clinical Simulator", "🏆 Analytics Hub"])
    st.divider()
    user_name = st.text_input("👤 Name", "User_01")
    profession = st.selectbox("👩‍⚕️ Role", ["Doctor", "Pharmacy", "Nursing", "AMS", "Dentistry", "Vet"]).lower()
    
    adaptive_mode = st.checkbox("🧠 Adaptive Mode", value=False)
    f_diff = get_adaptive_difficulty(user_name) if adaptive_mode else st.select_slider("Difficulty", options=["easy", "medium", "hard"], value="medium")

# ===================== 🚥 4. PAGE ROUTING =====================

if menu == "📖 Manual & Standards":
    st.header("📖 Clinical Operations Guide")
    st.markdown("### **System Philosophy**")
    st.info("Adaptive Cognitive Load–Driven AI Loop")
    # ... รายละเอียด Manual ของนาย ...

elif menu == "🧪 Clinical Simulator":
    c = st.session_state.case
    elapsed = int(time.time() - st.session_state.start_time)
    
    col_h1, col_h2 = st.columns([3, 1])
    with col_h1: st.title(f"🏥 Case: {c.get('block')} ({f_diff.upper()})")
    with col_h2: st.markdown(f"<div class='stress-timer'>⏳ {600-elapsed}s</div>", unsafe_allow_html=True)

    t1, t2, t3 = st.tabs(["📋 Scenario", "🧠 Reasoning Map", "✍️ Entry"])
    
    with t1:
        st.write(c.get('scenario', {}).get('en'))
        if c.get("labs"): st.table(pd.DataFrame(c["labs"]))
        if st.button("⏩ Advance 24 Hours"): st.session_state.evolved = True
        if st.session_state.evolved: st.warning(c.get('evolution', 'Stable.'))

    with t2:
        map_pos = st.text_area("Pertinent Positives (+)", key="map_pos", height=150)
        map_neg = st.text_area("Pertinent Negatives (-)", key="map_neg", height=150)

    with t3:
        dx_in = st.text_input("Final Diagnosis")
        re_in = st.text_area("Rationale")
        u_conf = st.slider("Confidence", 0, 100, 80)
        stress_lv = st.slider("Stress", 0, 10, 5)

        if st.button("🚀 SUBMIT DECISION"):
            if dx_in and re_in:
                with st.spinner("AI Evaluating..."):
                    fb = get_ai_feedback_v9_5(dx_in, re_in, f"P:{map_pos} N:{map_neg}", c.get('answer'), profession, elapsed, u_conf, stress_lv)
                    st.session_state.ai_feedback = fb
                    st.session_state.submitted = True
                    # บันทึกคะแนนแบบง่าย
                    save_score_local(user_name, profession, 10 if dx_in.lower() in str(c.get('answer')).lower() else 5, c.get('block'), time_taken=elapsed)
                    st.rerun()

    if st.session_state.submitted:
        st.divider()
        st.markdown(st.session_state.ai_feedback)
        if st.button("🏁 Reset"):
            st.session_state.submitted = False
            st.session_state.start_time = time.time()
            st.rerun()

elif menu == "🏆 Analytics Hub":
    st.header("🏆 Performance Analytics")
    if os.path.exists(DB_FILE):
        df = pd.read_csv(DB_FILE)
        if not df.empty:
            c1, c2, c3 = st.columns(3)
            c1.metric("Total Cases", len(df))
            c2.metric("Avg Score", f"{df['Score'].mean():.1f}")
            c3.metric("Avg Time", f"{df['Time'].mean():.0f}s")
            st.line_chart(df.set_index("Timestamp")["Score"])
        else: st.warning("No data.")
    else: st.info("No database found.")

st.markdown("---")
st.caption("FTF-CRA Global v9.9.5 | © 2026")
