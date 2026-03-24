import streamlit as st
import json, random, pandas as pd, os, time
import google.generativeai as genai

# ===================== 1. CONFIG & MEDICAL UI =====================
st.set_page_config(layout="wide", page_title="ACLR Clinical Analytics Platform", page_icon="🩺")

# Medical-Grade CSS
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
    </style>
    """, unsafe_allow_html=True)

# ===================== 2. API & DATABASE SETUP =====================
GEMINI_API_KEY = "AIzaSyDVy5Bh-RmscVwgzUIuYSK8CHa5ZAKnx_g"
genai.configure(api_key=GEMINI_API_KEY)

DB_FILE = "leaderboard.csv"

def save_score_local(name, role, score, block):
    new_data = {"Timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "Name": name, "Role": role.upper(), "Score": score, "Block": block}
    df = pd.DataFrame([new_data])
    if not os.path.isfile(DB_FILE): df.to_csv(DB_FILE, index=False)
    else: df.to_csv(DB_FILE, mode='a', index=False, header=False)

def get_ai_feedback(user_dx, user_re, target, role):
    model = genai.GenerativeModel('gemini-1.5-flash')
    prompt = f"Act as a Senior Clinical Professor. Evaluate this {role}'s reasoning. Dx: {user_dx} | Rationale: {user_re} | Gold Standard: {target}. Provide 3 concise bullets: Accuracy, Logic Gaps, Professional Pearl. English only."
    try: return model.generate_content(prompt).text
    except: return "AI Mentor is offline. Review Gold Standard Answer."

# ===================== 3. DATA LOADING =====================
@st.cache_data
def load_cases():
    if os.path.exists("cases.json"):
        with open("cases.json", "r", encoding="utf-8") as f: return json.load(f)
    return [{"block":"General", "difficulty":"medium", "scenario":{"en":"Sample Case Loaded"}, "answer":"N/A"}]

all_cases = load_cases()

# ===================== 4. SESSION STATE =====================
if "case" not in st.session_state: st.session_state.case = all_cases[0]
if "submitted" not in st.session_state: st.session_state.submitted = False
if "ai_feedback" not in st.session_state: st.session_state.ai_feedback = ""

# ===================== 5. SIDEBAR & FILTERS =====================
with st.sidebar:
    st.title("ACLR Platform v9.9")
    menu = st.radio("Main Menu", ["📖 Manual & Standards", "🧪 Clinical Simulator", "🏆 Analytics Hub"])
    st.divider()
    user_name = st.text_input("👤 Practitioner Name", "User_01")
    profession = st.selectbox("👩‍⚕️ Clinical Role", ["Doctor", "Pharmacy", "Nursing", "AMS", "Dentistry", "Vet", "Public Health"]).lower()
    
    st.divider()
    st.subheader("🎯 Session Filters")
    blocks = sorted(list(set([c['block'] for c in all_cases])))
    f_block = st.selectbox("Select Block", ["All Blocks"] + blocks)
    f_diff = st.select_slider("Select Difficulty", options=["easy", "medium", "hard"], value="medium")

    if menu == "🧪 Clinical Simulator":
        if st.button("🔄 Generate Filtered Case"):
            pool = all_cases
            if f_block != "All Blocks": pool = [c for c in pool if c['block'] == f_block]
            pool = [c for c in pool if c['difficulty'] == f_diff]
            st.session_state.case = random.choice(pool) if pool else random.choice(all_cases)
            st.session_state.submitted = False
            st.session_state.ai_feedback = ""
            st.rerun()

# ===================== 6. PAGES =====================

# --- 📖 MANUAL & STANDARDS (NEW UPGRADED) ---
if menu == "📖 Manual & Standards":
    st.header("📖 Clinical Operations & User Guide")
    st.markdown("### **ACLR Platform**")
    st.write("*Adaptive Cognitive Load–Driven AI Clinical Reasoning Loop*")
    
    # --- SECTION 1: SYSTEM PHILOSOPHY ---
    with st.expander("🌐 1. System Philosophy & Objectives", expanded=True):
        st.markdown("""
        <div style="background-color: #E3F2FD; padding: 20px; border-radius: 10px; border-left: 5px solid #1976D2;">
            <h4 style="color: #1976D2;">Core Objective</h4>
            <p>To bridge the gap between medical theory and bedside practice by simulating high-fidelity clinical scenarios. 
            The system manages <b>Cognitive Load</b> by filtering content into blocks and difficulty levels, allowing learners to focus on specific clinical reasoning pathways.</p>
        </div>
        """, unsafe_allow_html=True)
        st.write("")

    # --- SECTION 2: WORKFLOW STEPS ---
    st.divider()
    st.subheader("🚀 2. Operational Workflow (How to Play)")
    
    w1, w2, w3 = st.columns(3)
    with w1:
        st.markdown("""
        <div style="background-color: #FFF3E0; padding: 15px; border-radius: 10px; min-height: 250px;">
            <h4 style="color: #E65100;">Step 1: Calibration</h4>
            <p><b>Configuration:</b></p>
            <ul>
                <li>Enter your <b>Practitioner Name</b>.</li>
                <li>Select <b>Clinical Role</b> to activate specialized input fields.</li>
                <li>Apply <b>Block Filters</b> (e.g., Cardio) and <b>Difficulty</b>.</li>
                <li>Click <b>'Generate Filtered Case'</b>.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
    with w2:
        st.markdown("""
        <div style="background-color: #E8F5E9; padding: 15px; border-radius: 10px; min-height: 250px;">
            <h4 style="color: #2E7D32;">Step 2: Analysis</h4>
            <p><b>Data Synthesis:</b></p>
            <ul>
                <li>Read the <b>Patient Scenario</b> carefully.</li>
                <li>Interpret <b>Diagnostic Data</b> (Labs/Imaging) provided in the table.</li>
                <li>Identify <b>Red Flags</b> and key clinical indicators.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with w3:
        st.markdown("""
        <div style="background-color: #F3E5F5; padding: 15px; border-radius: 10px; min-height: 250px;">
            <h4 style="color: #7B1FA2;">Step 3: Execution</h4>
            <p><b>Clinical Entry:</b></p>
            <ul>
                <li>Formulate your <b>Diagnosis</b> and <b>Rationale</b>.</li>
                <li>Complete <b>Role-Specific</b> tasks (e.g., Dosing for Pharmacy).</li>
                <li>Submit for <b>AI Mentor Debriefing</b>.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    # --- SECTION 3: DYNAMIC LOGIC ---
    st.divider()
    st.subheader("🧬 3. Interprofessional Dynamic Logic")
    st.write("Each role faces different 'Cognitive Loads'. The UI adapts to reflect your real-world responsibilities:")
    
    r1, r2 = st.columns(2)
    with r1:
        st.markdown("""
        - <b style="color:#1976D2;">Doctor/Dentist:</b> Focus on <i>Differential Diagnosis (DDx)</i> and definitive treatment.
        - <b style="color:#D32F2F;">Pharmacy:</b> Focus on <i>Pharmacotherapy</i>, safety, and precise dosing.
        - <b style="color:#388E3C;">Nursing:</b> Focus on <i>Vitals Monitoring</i> and immediate patient stabilization.
        """, unsafe_allow_html=True)
    with r2:
        st.markdown("""
        - <b style="color:#FBC02D;">AMS:</b> Focus on <i>Lab Integrity</i> and recommending advanced diagnostics.
        - <b style="color:#7B1FA2;">Public Health/Vet:</b> Focus on <i>Population Safety</i> and epidemiological links.
        """, unsafe_allow_html=True)

    # --- SECTION 4: SCORING ---
    st.divider()
    st.subheader("📊 4. Scoring & AI Evaluation Matrix")
    
    st.markdown("""
    | Criteria | Weight | Description |
    | :--- | :--- | :--- |
    | **Clinical Accuracy** | 40% | Is the diagnosis aligned with the Gold Standard? |
    | **Logical Rationale** | 30% | Does the pathophysiological explanation make sense? |
    | **Patient Safety** | 20% | Is the disposition (ICU/Ward) and next step appropriate? |
    | **Risk Management** | 10% | Were confidence levels and red flags acknowledged? |
    """)
    
    st.info("""
    💡 **AI Mentor Feedback:** Powered by Google Gemini 1.5 Flash. 
    It doesn't just score; it provides **'Professional Pearls'**—nuggets of wisdom that help you think like a Senior Consultant.
    """)

    st.divider()
    st.caption("Educational Reference Standards: Harrison's Principles of Internal Medicine, UpToDate, and WHO Clinical Protocols.")

# --- 🏆 ANALYTICS HUB ---
elif menu == "🏆 Analytics Hub":
    st.header("🏆 Performance Analytics Dashboard")
    if os.path.exists(DB_FILE):
        df = pd.read_csv(DB_FILE)
        st.dataframe(df.sort_values(by="Timestamp", ascending=False), use_container_width=True)
        st.divider()
        c1, c2, c3 = st.columns(3)
        c1.metric("Simulations", len(df))
        c2.metric("Mean Score", f"{df['Score'].mean():.1f}/10")
        c3.metric("Top Role", df.groupby('Role')['Score'].mean().idxmax())
        st.bar_chart(df.groupby('Role')['Score'].mean())
    else: st.info("No data yet.")

st.markdown("---")
st.caption("ACLR Global v9.9 | Adaptive Cognitive Load–Driven AI Clinical Reasoning Loop | © 2026")
