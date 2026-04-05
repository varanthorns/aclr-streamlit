import streamlit as st
import json, random, pandas as pd, os, time
import google.generativeai as genai

# ===================== 🔧 1. CONFIG & CORE SYSTEM =====================
DB_FILE = "aclr_scores.csv"  # เพิ่มประกาศชื่อไฟล์ฐานข้อมูล

# 🔐 FIX: ใช้ secrets แทน API key hardcode
try:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
except:
    GEMINI_API_KEY = "DEMO_KEY"

genai.configure(api_key=GEMINI_API_KEY)

# ✅ FIX: function บันทึกคะแนน
def save_score_local(user, role, score, block, competency=None, time_taken=0):
    new_entry = {
        "User": user,
        "Role": role,
        "Score": score,
        "Block": block,
        "Time": time_taken,
        "Timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
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

# ===================== 🧠 ADAPTIVE LEARNING =====================

def get_user_history(user):
    if os.path.exists(DB_FILE):
        df = pd.read_csv(DB_FILE)
        return df[df["User"] == user]
    return pd.DataFrame()

def get_adaptive_difficulty(user):
    df = get_user_history(user)
    if len(df) < 5: return "easy"
    avg_score = df["Score"].mean()
    if avg_score < 6: return "easy"
    elif avg_score < 8: return "medium"
    else: return "hard"

# ===================== 2. CONFIG & MEDICAL UI =====================
st.set_page_config(layout="wide", page_title="ACLR Clinical Analytics Platform", page_icon="🩺")

st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stButton>button { width: 100%; border-radius: 8px; height: 3.5em; background-color: #1976D2 !important; color: white !important; font-weight: bold; border: none; }
    .stButton>button:hover { background-color: #1565C0 !important; }
    .stMetric { background-color: white; padding: 20px; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); border-left: 5px solid #1976D2; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] { background-color: #e3f2fd; border-radius: 8px 8px 0 0; padding: 12px 24px; color: #1976D2; font-weight: 600; }
    .stTabs [aria-selected="true"] { background-color: #1976D2 !important; color: white !important; }
    div[data-testid="stExpander"] { border: 1px solid #e3f2fd; border-radius: 8px; background-color: white; }
    .stress-timer { font-size: 28px; font-weight: bold; color: #d32f2f; text-align: center; border: 3px solid #d32f2f; padding: 10px; border-radius: 15px; background: white; }
    </style>
    """, unsafe_allow_html=True)

# ===================== 3. API SETUP =====================
def get_ai_feedback_v9_5(user_dx, user_re, user_map, target, role, time_taken):
    model = genai.GenerativeModel('gemini-1.5-flash')
    prompt = f"""
    Act as a Senior Clinical Professor. Evaluate this {role}'s clinical reasoning process.
    [User Data]
    - Diagnosis: {user_dx}
    - Reasoning & SBAR: {user_re}
    - Clinical Reasoning Map: {user_map}
    - Gold Standard Reference: {target}
    - Time Taken: {time_taken} seconds.
    
    [Response Format]
    - Diagnosis Score (0-10), Reasoning Score (0-10), SBAR Score (0-10), Safety Score (0-10)
    - **Overall Score (0-10):**
    - **Strengths:**, **Critical Gaps:**, **Cognitive Bias:**, **Professional Pearl:**, **Well-being Tip:**
    English only.
    """
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"AI Mentor Offline (Error: {str(e)})"

# ===================== 4. DATA LOADING =====================
@st.cache_data
def load_cases():
    if os.path.exists("cases.json"):
        with open("cases.json", "r", encoding="utf-8") as f: return json.load(f)
    return [{"block":"Cardiology", "difficulty":"hard", "scenario":{"en":"65yo Male presents with 2 hours of crushing chest pain..."}, "labs":[{"Test": "Troponin T", "Result": "480", "Unit": "ng/L", "Ref": "<14"}], "answer":"Acute STEMI", "interprofessional_answers": {"doctor": "PCI"}, "evolution": "24h: SOB develops."}]

all_cases = load_cases()

# ===================== 5. SESSION STATE =====================
if "case" not in st.session_state: st.session_state.case = all_cases[0]
if "submitted" not in st.session_state: st.session_state.submitted = False
if "ai_feedback" not in st.session_state: st.session_state.ai_feedback = ""
if "start_time" not in st.session_state: st.session_state.start_time = time.time()
if "evolved" not in st.session_state: st.session_state.evolved = False

# ===================== 6. SIDEBAR =====================
with st.sidebar:
    st.title("ACLR Platform")
    menu = st.radio("Main Menu", ["📖 Manual & Standards", "🧪 Clinical Simulator", "🏆 Analytics Hub"])
    user_name = st.text_input("👤 Practitioner Name", "User_01")
    profession = st.selectbox("👩‍⚕️ Clinical Role", ["Doctor", "Pharmacy", "Nursing", "AMS", "Dentistry", "Vet", "Public Health"]).lower()
    
    st.subheader("🎯 Session Filters")
    blocks = sorted(list(set([c.get('block', 'General') for c in all_cases])))
    f_block = st.selectbox("Select Block", ["All Blocks"] + blocks)
    f_diff = st.select_slider("Select Difficulty", options=["easy", "medium", "hard"], value="medium")
    
    if st.checkbox("🧠 Adaptive Learning Mode"):
        f_diff = get_adaptive_difficulty(user_name)
        st.success(f"AI Difficulty → {f_diff.upper()}")

    if menu == "🧪 Clinical Simulator" and st.button("🔄 Generate Filtered Case"):
        pool = [c for c in all_cases if (f_block == "All Blocks" or c.get('block') == f_block) and c.get('difficulty') == f_diff]
        st.session_state.case = random.choice(pool) if pool else random.choice(all_cases)
        st.session_state.submitted = False
        st.session_state.start_time = time.time()
        st.session_state.evolved = False
        st.rerun()

# ===================== 7. PAGES =====================
if menu == "📖 Manual & Standards":
    st.header("📖 Clinical Operations & User Guide")
    with st.expander("🌐 1. System Philosophy", expanded=True):
        st.info("Bridge the gap between medical theory and practice via Cognitive Load management.")
    
    col_w = st.columns(3)
    col_w[0].success("Step 1: Calibration")
    col_w[1].success("Step 2: Synthesis")
    col_w[2].success("Step 3: Execution")

elif menu == "🧪 Clinical Simulator":
    c = st.session_state.case
    elapsed = int(time.time() - st.session_state.start_time)
    remaining = max(0, 600 - elapsed)
    
    col_h1, col_h2 = st.columns([3, 1])
    col_h1.title(f"🏥 Simulation: {c.get('block')} | {c.get('difficulty').upper()}")
    col_h2.markdown(f"<div class='stress-timer'>⏳ {remaining}s</div>", unsafe_allow_html=True)

    col_main, col_info = st.columns([2, 1])
    with col_main:
        t1, t2, t3 = st.tabs(["📋 Case Details", "🧠 Reasoning Map", "✍️ Entry"])
        
        with t1:
            st.info(c.get('scenario', {}).get('en'))
            if c.get("labs"): st.table(pd.DataFrame(c["labs"]))
            if st.button("⏩ Advance 24 Hours"): st.session_state.evolved = True
            if st.session_state.evolved: st.warning(c.get('evolution'))
            st.json({"Patient ID": "ACLR-001", "Vitals": "90/60, HR 120"})

        with t2:
            pos_f = st.text_area("Pertinent Positives (+)", key="map_pos_main")
            neg_f = st.text_area("Pertinent Negatives (-)", key="map_neg_main")

        with t3:
            dx_in = st.text_input("🩺 Final Diagnosis", key="entry_dx")
            role_info = ""
            if profession == "doctor":
                ddx = st.multiselect("🔍 DDx", ["Sepsis", "MI", "Stroke", "IE"], key="dr_ddx")
                plan = st.text_input("💊 Plan", key="dr_plan")
                role_info = f"DDx: {ddx}, Plan: {plan}"
            elif profession == "pharmacy":
                dosing = st.text_input("⚖️ Dosing", key="ph_dose")
                role_info = f"Dosing: {dosing}"
            elif profession == "nursing":
                vitals = st.multiselect("📉 Watch", ["BP", "SpO2"], key="ns_vitals")
                role_info = f"Watch: {vitals}"

            re_in = st.text_area("✍️ Rationale", key="entry_re")
            st.divider()
            st.markdown("#### 🗣️ SBAR Handover")
            h_s = st.text_input("S", key="s_s")
            h_b = st.text_input("B", key="s_b")
            h_a = st.text_area("A", key="s_a")
            h_r = st.text_area("R", key="s_r")
            
            if st.button("🚀 SUBMIT DECISION"):
                comp = {"Diagnosis": random.randint(6,10), "Reasoning": random.randint(6,10), "SBAR": random.randint(6,10), "Safety": random.randint(6,10)}
                save_score_local(user_name, profession, int(sum(comp.values())/4), c.get('block'), competency=comp, time_taken=elapsed)
                st.success("Submitted! Check Analytics Hub.")

elif menu == "🏆 Analytics Hub":
    st.header("🏆 Performance Analytics")
    if os.path.exists(DB_FILE):
        df = pd.read_csv(DB_FILE)
        st.dataframe(df.sort_values(by="Timestamp", ascending=False))
        c1, c2, c3 = st.columns(3)
        c1.metric("Simulations", len(df))
        c2.metric("Mean Score", f"{df['Score'].mean():.1f}")
        
        st.subheader("📈 Learning Curve")
        if "Timestamp" in df.columns:
            df["Timestamp"] = pd.to_datetime(df["Timestamp"])
            st.line_chart(df.set_index("Timestamp")["Score"])
            
        st.subheader("🧠 Competency Breakdown")
        comp_cols = ["Diagnosis", "Reasoning", "SBAR", "Safety"]
        existing = [col for col in comp_cols if col in df.columns]
        if existing: st.bar_chart(df[existing].mean())
    else:
        st.info("No data yet.")

st.markdown("---")
st.caption("ACLR Global v9.9.5 | © 2026")
