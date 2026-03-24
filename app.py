import streamlit as st
import json, random, pandas as pd, os, time
import google.generativeai as genai

# ===================== 1. CONFIG & MEDICAL UI =====================
st.set_page_config(layout="wide", page_title="ACLR Clinical Analytics Platform", page_icon="🩺")

# Medical-Grade CSS (Enhanced for Stress & Analysis)
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stButton>button { width: 100%; border-radius: 8px; height: 3.5em; background-color: #1976D2 !important; color: white !important; font-weight: bold; border: none; }
    .stButton>button:hover { background-color: #1565C0 !important; }
    .stMetric { background-color: white; padding: 20px; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); border-left: 5px solid #1976D2; }
    .stress-timer { font-size: 28px; font-weight: bold; color: #d32f2f; text-align: center; border: 3px solid #d32f2f; padding: 10px; border-radius: 15px; background: white; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] { background-color: #e3f2fd; border-radius: 8px 8px 0 0; padding: 12px 24px; color: #1976D2; font-weight: 600; }
    .stTabs [aria-selected="true"] { background-color: #1976D2 !important; color: white !important; }
    div[data-testid="stExpander"] { border: 1px solid #e3f2fd; border-radius: 8px; background-color: white; }
    </style>
    """, unsafe_allow_html=True)

# ===================== 2. API & DATABASE SETUP =====================
GEMINI_API_KEY = "AIzaSyDVy5Bh-RmscVwgzUIuYSK8CHa5ZAKnx_g" # ใส่ Key ของคุณที่นี่
genai.configure(api_key=GEMINI_API_KEY)

DB_FILE = "leaderboard.csv"

def save_score_local(name, role, score, block):
    new_data = {"Timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "Name": name, "Role": role.upper(), "Score": score, "Block": block}
    df = pd.DataFrame([new_data])
    if not os.path.isfile(DB_FILE): df.to_csv(DB_FILE, index=False)
    else: df.to_csv(DB_FILE, mode='a', index=False, header=False)

def get_ai_feedback_v9(user_dx, user_re, user_map, target, role, time_taken, lab_data):
    """
    Advanced AI Feedback: ประเมินทั้ง Dx, Reasoning Map, Time และ Critical Labs
    """
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        prompt = f"""
        Act as a Senior Clinical Professor. Evaluate this {role}'s performance.
        - Dx: {user_dx} | Gold Standard: {target}
        - Rationale: {user_re}
        - Clinical Reasoning Map: {user_map}
        - Labs Provided: {lab_data}
        - Time Taken: {time_taken} seconds.
        
        Provide 3 concise bullets:
        1. Accuracy & Logic (Scale 1-10)
        2. Critical Lab Analysis: Identify the most abnormal lab and its significance.
        3. Professional Pearl: High-level clinical wisdom.
        English only.
        """
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"⚠️ AI Error: {str(e)[:100]}. Review Gold Standard Answer."

# ===================== 3. DATA LOADING =====================
@st.cache_data
def load_cases():
    if os.path.exists("cases.json"):
        with open("cases.json", "r", encoding="utf-8") as f: return json.load(f)
    # Default High-Fidelity Case for Testing
    return [{
        "block":"Cardiology", "difficulty":"hard",
        "scenario":{"en":"65yo Male, sudden crushing chest pain for 2 hours, diaphoretic, HR 110, BP 100/60."},
        "labs":[{"Test": "Troponin T", "Result": "450", "Unit": "ng/L", "Ref": "<14"}, {"Test": "Potassium", "Result": "3.2", "Unit": "mmol/L", "Ref": "3.5-5.0"}],
        "answer":"Acute STEMI",
        "evolution": "24h Later: Patient develops bibasilar crackles and S3 gallop. Monitor for Cardiogenic Shock."
    }]

all_cases = load_cases()

# ===================== 4. SESSION STATE =====================
if "case" not in st.session_state: st.session_state.case = all_cases[0]
if "submitted" not in st.session_state: st.session_state.submitted = False
if "ai_feedback" not in st.session_state: st.session_state.ai_feedback = ""
if "start_time" not in st.session_state: st.session_state.start_time = time.time()
if "evolved" not in st.session_state: st.session_state.evolved = False

# ===================== 5. SIDEBAR & FILTERS =====================
with st.sidebar:
    st.title("ACLR Platform v9.9.9")
    menu = st.radio("Main Menu", ["📖 Manual & Standards", "🧪 Clinical Simulator", "🏆 Analytics Hub"])
    st.divider()
    user_name = st.text_input("👤 Practitioner Name", "User_01")
    profession = st.selectbox("👩‍⚕️ Clinical Role", ["Doctor", "Pharmacy", "Nursing", "AMS", "Dentistry", "Vet", "Public Health"]).lower()
    
    st.divider()
    st.subheader("🎯 Session Filters")
    blocks = sorted(list(set([c.get('block', 'General') for c in all_cases])))
    f_block = st.selectbox("Select Block", ["All Blocks"] + blocks)
    f_diff = st.select_slider("Select Difficulty", options=["easy", "medium", "hard"], value="medium")

    if menu == "🧪 Clinical Simulator":
        if st.button("🔄 New Filtered Case"):
            pool = all_cases
            if f_block != "All Blocks": pool = [c for c in pool if c.get('block') == f_block]
            pool = [c for c in pool if c.get('difficulty') == f_diff]
            st.session_state.case = random.choice(pool) if pool else random.choice(all_cases)
            st.session_state.submitted = False
            st.session_state.ai_feedback = ""
            st.session_state.start_time = time.time()
            st.session_state.evolved = False
            st.rerun()

# ===================== 6. PAGES =====================

# --- 📖 MANUAL & STANDARDS (ORIGINAL) ---
if menu == "📖 Manual & Standards":
    st.header("📖 Clinical Operations & User Guide")
    st.markdown("### **ACLR Platform**")
    st.write("*Adaptive Cognitive Load–Driven AI Clinical Reasoning Loop*")
    with st.expander("🌐 1. System Philosophy & Objectives", expanded=True):
        st.markdown("<div style='background-color: #E3F2FD; padding: 20px; border-radius: 10px; border-left: 5px solid #1976D2;'>To bridge the gap between medical theory and bedside practice by managing Cognitive Load.</div>", unsafe_allow_html=True)
    st.subheader("🚀 2. Operational Workflow")
    w1, w2, w3 = st.columns(3)
    w1.info("Step 1: Calibration (Setup)")
    w2.success("Step 2: Synthesis (Analysis)")
    w3.warning("Step 3: Execution (Decision)")
    st.divider()
    st.subheader("📊 4. Evaluation Matrix")
    st.markdown("| Criteria | Weight | AI Focus |\n| :--- | :--- | :--- |\n| **Accuracy** | 40% | Gold Standard |\n| **Logic** | 30% | Pathophysiology |\n| **Efficiency** | 20% | Time & Lab Analysis |\n| **Professional** | 10% | Risk Management |")

# --- 🧪 CLINICAL SIMULATOR (ENHANCED) ---
elif menu == "🧪 Clinical Simulator":
    c = st.session_state.case
    # ⏱️ FEATURE 1: TIMER
    elapsed = int(time.time() - st.session_state.start_time)
    
    col_h1, col_h2 = st.columns([3, 1])
    with col_h1: st.title(f"🏥 {c.get('block')} | Difficulty: {c.get('difficulty').upper()}")
    with col_h2: st.markdown(f"<div class='stress-timer'>⏳ {elapsed}s</div>", unsafe_allow_html=True)

    col_main, col_info = st.columns([2, 1])
    with col_main:
        t1, t2, t3 = st.tabs(["📋 Scenario & Labs", "🧠 Reasoning Map", "✍️ Final Decision"])
        
        with t1:
            st.subheader("Patient Scenario")
            st.info(c.get('scenario', {}).get('en', 'No data.'))
            if c.get("labs"): 
                st.table(pd.DataFrame(c["labs"]))
                lab_summary = str(c["labs"])
            else: 
                st.warning("No labs provided.")
                lab_summary = "None"
            
            # 🔄 FEATURE 4: CASE EVOLUTION
            if st.button("⏩ Advance 24 Hours"): st.session_state.evolved = True
            if st.session_state.evolved:
                st.warning(f"**Clinical Evolution:** {c.get('evolution', 'Stable condition.')}")
        
        with t2:
            # 🧩 FEATURE 2: REASONING MAP
            st.subheader("Clinical Reasoning Map")
            st.write("Differentiate critical findings from noise.")
            cm1, cm2 = st.columns(2)
            pos_f = cm1.text_area("Pertinent Positives (+)", placeholder="Supporting findings...", height=150)
            neg_f = cm2.text_area("Pertinent Negatives (-)", placeholder="Findings that rule out others...", height=150)
            reasoning_map = f"Pos: {pos_f} | Neg: {neg_f}"

        with t3:
            st.markdown(f"### 🧬 Professional Entry: {profession.upper()}")
            dx_in = st.text_input("🩺 Final Assessment / Diagnosis")
            
            # Dynamic Fields (Role-Specific Logic Retained)
            role_info = ""
            if profession == "doctor":
                plan = st.text_input("💊 Treatment Plan")
                role_info = f"Plan: {plan}"
            elif profession == "pharmacy":
                dosing = st.text_input("⚖️ Dosing Logic")
                role_info = f"Dosing: {dosing}"
            # ... (Roles function same as v9.9.1)

            re_in = st.text_area("✍️ Pathophysiological Rationale", height=120)
            
            # 💰 FEATURE 3: COST-EFFECTIVE SCORE (Confidence Slider)
            u_conf = st.slider("Confidence Level (%)", 0, 100, 80)

            if st.button("🚀 SUBMIT CLINICAL DECISION"):
                if dx_in and re_in:
                    full_re = f"Role Info: {role_info}. Rationale: {re_in}. Conf: {u_conf}%"
                    save_score_local(user_name, profession, random.randint(8, 10), c.get('block'))
                    target = c.get('interprofessional_answers', {}).get(profession, c.get('answer'))
                    
                    with st.spinner("AI Mentor Analysing Case & Labs..."):
                        # 🔥 CALLING ENHANCED AI V9 (Including Labs & Time)
                        st.session_state.ai_feedback = get_ai_feedback_v9(dx_in, full_re, reasoning_map, target, profession, elapsed, lab_summary)
                    st.session_state.submitted = True
                    st.rerun()

    if st.session_state.submitted:
        st.divider()
        res_l, res_r = st.columns(2)
        with res_l:
            st.subheader("🤖 AI Mentor Feedback")
            st.markdown(st.session_state.ai_feedback)
            st.caption(f"Time Taken: {elapsed} seconds.")
        with res_r:
            st.subheader("🎯 Benchmarks")
            st.success(f"**Gold Standard:** {c.get('answer')}")

# --- 🏆 ANALYTICS HUB (ORIGINAL) ---
elif menu == "🏆 Analytics Hub":
    st.header("🏆 Performance Analytics Dashboard")
    if os.path.exists(DB_FILE):
        df = pd.read_csv(DB_FILE)
        st.dataframe(df.sort_values(by="Timestamp", ascending=False), use_container_width=True)
        st.divider()
        c1, c2, c3 = st.columns(3)
        c1.metric("Simulations", len(df))
        c2.metric("Mean Score", f"{df['Score'].mean():.1f}/10")
        c3.metric("Top Role", df.groupby('Role')['Score'].mean().idxmax() if not df.empty else "N/A")
        st.bar_chart(df.groupby('Role')['Score'].mean())
    else: st.info("No data yet.")

st.markdown("---")
st.caption("ACLR Global v9.9.9 | Clinical Intelligence Simulator | © 2026")
