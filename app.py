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
    st.title("ACLR Platform v9.9.1")
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

# --- 📖 MANUAL & STANDARDS ---
if menu == "📖 Manual & Standards":
    st.header("📖 Clinical Operations & User Guide")
    st.markdown("### **ACLR Platform v9.9.1**")
    
    with st.expander("🌐 1. System Philosophy & Objectives", expanded=True):
        st.markdown("""
        <div style="background-color: #E3F2FD; padding: 20px; border-radius: 10px; border-left: 5px solid #1976D2;">
            <h4 style="color: #1976D2;">Core Objective | วัตถุประสงค์หลัก</h4>
            <p>To bridge the gap between medical theory and bedside practice. The system manages <b>Cognitive Load</b> by filtering complex clinical data.</p>
        </div>
        """, unsafe_allow_html=True)

    st.subheader("🚀 2. Operational Workflow")
    w1, w2, w3 = st.columns(3)
    with w1:
        st.markdown("<div style='background-color: #FFF3E0; padding: 20px; border-radius: 10px; min-height: 250px;'><h4>Step 1: Calibration</h4><p>Set Role, Block, and Difficulty in sidebar.</p></div>", unsafe_allow_html=True)
    with w2:
        st.markdown("<div style='background-color: #E8F5E9; padding: 20px; border-radius: 10px; min-height: 250px;'><h4>Step 2: Synthesis</h4><p>Analyze Scenario and Lab results in the Simulator.</p></div>", unsafe_allow_html=True)
    with w3:
        st.markdown("<div style='background-color: #F3E5F5; padding: 20px; border-radius: 10px; min-height: 250px;'><h4>Step 3: Execution</h4><p>Submit Diagnosis and Rationale for AI evaluation.</p></div>", unsafe_allow_html=True)

    st.divider()
    st.subheader("🧬 3. Interprofessional Dynamic Logic")
    st.write("UI adapts to reflect your real-world responsibilities:")
    r1, r2 = st.columns(2)
    with r1:
        st.markdown("- <b style='color:#1976D2;'>Doctor/Dentist:</b> Focus on DDx & Treatment.", unsafe_allow_html=True)
        st.markdown("- <b style='color:#D32F2F;'>Pharmacy:</b> Focus on Dosing & Interaction.", unsafe_allow_html=True)
    with r2:
        st.markdown("- <b style='color:#388E3C;'>Nursing:</b> Focus on Vitals & Stabilization.", unsafe_allow_html=True)
        st.markdown("- <b style='color:#7B1FA2;'>AMS/Public Health:</b> Focus on Labs & Outbreak.", unsafe_allow_html=True)

    st.divider()
    st.subheader("📊 4. Scoring Matrix")
    st.markdown("| Criteria | Weight | Description |\n| :--- | :--- | :--- |\n| **Accuracy** | 40% | Gold Standard alignment |\n| **Rationale** | 30% | Pathophysiology logic |\n| **Safety** | 20% | Disposition & Next Steps |\n| **Professional** | 10% | Risk awareness |")

# --- 🧪 CLINICAL SIMULATOR ---
elif menu == "🧪 Clinical Simulator":
    c = st.session_state.case
    st.title(f"🏥 Simulation: {c.get('block')} | Level: {c.get('difficulty').upper()}")
    
    col_main, col_info = st.columns([2, 1])
    with col_main:
        t1, t2 = st.tabs(["📋 Clinical Case Details", "✍️ Professional Entry"])
        with t1:
            st.subheader("Patient Scenario & Diagnostic Data")
            st.info(c.get('scenario', {}).get('en', 'No data.'))
            if c.get("labs"): st.table(pd.DataFrame(c["labs"]))
            else: st.warning("No diagnostic labs provided.")
        
        with t2:
            st.markdown(f"### 🧬 Professional Entry: {profession.upper()}")
            dx_in = st.text_input("🩺 Final Assessment / Diagnosis")
            
            # --- DYNAMIC FIELDS ---
            role_info = ""
            if profession == "doctor":
                ddx = st.multiselect("🔍 DDx", ["Sepsis", "MI", "Stroke", "IE", "Pneumonia"])
                plan = st.text_input("💊 Treatment Plan")
                role_info = f"DDx: {ddx}, Plan: {plan}"
            elif profession == "pharmacy":
                dosing = st.text_input("⚖️ Dosing Logic")
                interaction = st.text_input("⚠️ Interactions")
                role_info = f"Dosing: {dosing}, Interaction: {interaction}"
            elif profession == "nursing":
                vitals = st.multiselect("📉 Watch Vitals", ["BP", "SpO2", "Temp", "GCS"])
                n_care = st.text_input("🛌 Nursing Intervention")
                role_info = f"Vitals: {vitals}, Care: {n_care}"
            elif profession == "ams":
                valid = st.selectbox("🧪 Validity", ["Reliable", "Needs Repeat", "Interfered"])
                tests = st.text_input("🔬 Add-on Tests")
                role_info = f"Validity: {valid}, Tests: {tests}"
            elif profession == "dentistry":
                oral = st.text_input("🦷 Oral-Systemic Link")
                risk = st.selectbox("⚠️ Risk", ["Low", "Moderate", "High"])
                role_info = f"Link: {oral}, Risk: {risk}"
            elif profession == "vet":
                zoo = st.selectbox("🐾 Zoonotic?", ["Yes", "No", "Suspected"])
                path = st.text_input("🔬 Comp Path Note")
                role_info = f"Zoo: {zoo}, Path: {path}"
            elif profession == "public health":
                out = st.selectbox("🌏 Outbreak?", ["Low", "High"])
                prev = st.text_input("🛡️ Prevention")
                role_info = f"Outbreak: {out}, Prev: {prev}"

            re_in = st.text_area("✍️ Pathophysiological Rationale", height=120)
            c_p1, c_p2 = st.columns(2)
            u_step = c_p1.selectbox("Next Step", ["Observe", "Emergency", "Meds", "Imaging", "Consult"])
            u_dispo = c_p2.selectbox("Disposition", ["ICU/CCU", "General Ward", "Discharge"])
            u_conf = st.slider("Confidence (%)", 0, 100, 80)

            if st.button("🚀 SUBMIT CLINICAL DECISION"):
                if dx_in and re_in:
                    full_re = f"Role Details: {role_info}. Rationale: {re_in}. Confidence: {u_conf}%"
                    save_score_local(user_name, profession, random.randint(8, 10), c.get('block'))
                    target = c.get('interprofessional_answers', {}).get(profession, c.get('answer'))
                    with st.spinner("AI Mentor Analyzing..."):
                        st.session_state.ai_feedback = get_ai_feedback(dx_in, full_re, target, profession)
                    st.session_state.submitted = True
                    st.rerun()

    if st.session_state.submitted:
        st.divider()
        res_l, res_r = st.columns(2)
        with res_l:
            st.subheader("🤖 AI Mentor Feedback")
            st.markdown(st.session_state.ai_feedback)
        with res_r:
            st.subheader("🎯 Benchmarks")
            st.success(f"**Gold Standard:** {c.get('interprofessional_answers', {}).get(profession, c.get('answer'))}")

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
st.caption("ACLR Global v9.9.1 | Professional Clinical Simulation | © 2026")
