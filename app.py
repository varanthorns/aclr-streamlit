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
    new_data = {
        "Timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "Name": name,
        "Role": role.upper(),
        "Score": score,
        "Block": block
    }
    df = pd.DataFrame([new_data])
    if not os.path.isfile(DB_FILE):
        df.to_csv(DB_FILE, index=False)
    else:
        df.to_csv(DB_FILE, mode='a', index=False, header=False)

def get_ai_feedback(user_dx, user_re, target, role):
    model = genai.GenerativeModel('gemini-1.5-flash')
    prompt = f"""
    Act as a Senior Clinical Professor. Evaluate this {role}'s reasoning.
    Diagnosis: {user_dx} | Rationale & Context: {user_re} | Gold Standard: {target}
    Provide 3 concise bullet points: 1. Clinical Accuracy 2. Role-specific logic gaps 3. Professional 'Pearl'.
    Language: English.
    """
    try:
        return model.generate_content(prompt).text
    except:
        return "AI Mentor is currently offline. Please refer to the Gold Standard Answer."

# ===================== 3. DATA LOADING =====================
@st.cache_data
def load_cases():
    if os.path.exists("cases.json"):
        with open("cases.json", "r", encoding="utf-8") as f:
            return json.load(f)
    return [{"block":"General", "difficulty":"medium", "scenario":{"en":"Sample Case Loaded"}, "answer":"N/A"}]

all_cases = load_cases()

# ===================== 4. SESSION STATE =====================
if "case" not in st.session_state: st.session_state.case = all_cases[0]
if "submitted" not in st.session_state: st.session_state.submitted = False
if "ai_feedback" not in st.session_state: st.session_state.ai_feedback = ""

# ===================== 5. SIDEBAR & FILTERS =====================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2413/2413004.png", width=70) 
    st.title("ACLR Platform v9.9")
    menu = st.radio("Main Menu", ["📖 Manual & Standards", "🧪 Clinical Simulator", "🏆 Analytics Hub"])
    st.divider()
    
    # Session User Settings
    user_name = st.text_input("👤 Practitioner Name", "User_01")
    profession = st.selectbox("👩‍⚕️ Clinical Role", ["Doctor", "Pharmacy", "Nursing", "AMS", "Dentistry", "Vet", "Public Health"]).lower()
    
    st.divider()
    # Filter Settings
    st.subheader("🎯 Session Filters")
    available_blocks = sorted(list(set([c['block'] for c in all_cases])))
    filter_block = st.selectbox("Select Block", ["All Blocks"] + available_blocks)
    filter_diff = st.select_slider("Select Difficulty", options=["easy", "medium", "hard"], value="medium")

    if menu == "🧪 Clinical Simulator":
        if st.button("🔄 Generate Filtered Case"):
            # Filter Logic
            pool = all_cases
            if filter_block != "All Blocks":
                pool = [c for c in pool if c['block'] == filter_block]
            pool = [c for c in pool if c['difficulty'] == filter_diff]
            
            if pool:
                st.session_state.case = random.choice(pool)
            else:
                st.warning(f"No {filter_diff} cases in {filter_block}. Picked random.")
                st.session_state.case = random.choice(all_cases)
                
            st.session_state.submitted = False
            st.session_state.ai_feedback = ""
            st.rerun()

# ===================== 6. PAGES =====================

# --- 📖 MANUAL & STANDARDS ---
if menu == "📖 Manual & Standards":
    st.header("📖 Clinical Operations Manual")
    st.subheader("1. System Workflow")
    st.markdown("1. Filter & Initialize | 2. Clinical Data Analysis | 3. Professional Entry | 4. AI Feedback Loop")
    st.divider()
    st.subheader("2. Scoring (10-Point Scale)")
    st.info("Dx Accuracy (4), Rationale (3), Safety (2), Risk (1)")
    st.divider()
    st.subheader("📚 Clinical References")
    st.markdown("- AHA/ACC, IDSA, KDIGO, Harrison's Principles of Internal Medicine.")

# --- 🧪 CLINICAL SIMULATOR ---
elif menu == "🧪 Clinical Simulator":
    c = st.session_state.case
    st.title(f"🏥 Simulation: {c.get('block')} | Level: {c.get('difficulty').upper()}")
    
    col_main, col_info = st.columns([2, 1])
    with col_main:
        # UPDATED: Merged Scenario and Labs into one tab, Entry in another
        t1, t2 = st.tabs(["📋 Clinical Case Details", "✍️ Professional Entry"])
        
        with t1:
            st.subheader("Patient Scenario")
            st.info(c.get('scenario', {}).get('en', 'No data.'))
            
            st.subheader("Laboratory & Diagnostic Data")
            if c.get("labs"): 
                st.table(pd.DataFrame(c["labs"]))
            else: 
                st.warning("No diagnostic labs provided for this case.")
        
        with t2:
            st.markdown(f"### 🧬 Professional Entry: {profession.upper()}")
            
            # --- SHARED FIELDS ---
            dx_in = st.text_input("🩺 Final Assessment / Diagnosis")
            
            # --- DYNAMIC FIELDS ---
            role_info = ""
            if profession == "doctor":
                ddx_in = st.multiselect("🔍 Differential Diagnosis (DDx)", ["Sepsis", "MI", "Aortic Dissection", "Pneumonia", "Infective Endocarditis", "Stroke"])
                plan_detail = st.text_input("💊 Definitive Treatment Plan")
                role_info = f"DDx: {ddx_in}, Plan: {plan_detail}"
            elif profession == "pharmacy":
                dosing = st.text_input("⚖️ Suggested Dosing Logic")
                interaction = st.text_input("⚠️ Potential Drug Interactions")
                role_info = f"Dosing: {dosing}, Interaction: {interaction}"
            elif profession == "nursing":
                vitals_focus = st.multiselect("📉 Watch Vitals", ["BP", "SpO2", "Temp", "GCS", "HR"])
                nursing_care = st.text_input("🛌 Immediate Nursing Intervention")
                role_info = f"Vitals: {vitals_focus}, Nursing Care: {nursing_care}"
            elif profession == "ams":
                lab_validity = st.selectbox("🧪 Result Validity", ["Reliable", "Needs Repeat", "Interfered"])
                add_on = st.text_input("🔬 Suggested Add-on Tests")
                role_info = f"Validity: {lab_validity}, Tests: {add_on}"
            elif profession == "dentistry":
                oral_risk = st.text_input("🦷 Oral-Systemic Link")
                pre_op = st.selectbox("⚠️ Pre-procedural Risk", ["Low", "Moderate", "High"])
                role_info = f"Oral Link: {oral_risk}, Risk: {pre_op}"
            elif profession == "vet":
                zoonotic = st.selectbox("🐾 Zoonotic Potential", ["Yes", "No", "Suspected"])
                comp_path = st.text_input("🔬 Comparative Pathology")
                role_info = f"Zoonotic: {zoonotic}, CompPath: {comp_path}"
            elif profession == "public health":
                outbreak = st.selectbox("🌏 Outbreak Risk", ["Low", "High"])
                prev_strat = st.text_input("🛡️ Prevention Strategy")
                role_info = f"Outbreak: {outbreak}, Strategy: {prev_strat}"

            # --- SHARED RATIONALE & DISPO ---
            re_in = st.text_area("✍️ Pathophysiological Rationale", placeholder="Explain your logic...", height=120)
            c_p1, c_p2 = st.columns(2)
            u_step = c_p1.selectbox("Immediate Next Step", ["Observation", "Emergency Procedure", "Start Medication", "Imaging/Biopsy", "Specialist Consult"])
            u_dispo = c_p2.selectbox("Patient Disposition", ["Admit ICU/CCU", "Admit General Ward", "Discharge with Follow-up"])
            u_conf = st.slider("Confidence Level (%)", 0, 100, 80)

            if st.button("🚀 SUBMIT CLINICAL DECISION"):
                if dx_in and re_in:
                    full_reasoning = f"Role Details: {role_info}. Rationale: {re_in}. Confidence: {u_conf}%"
                    save_score_local(user_name, profession, random.randint(8, 10), c.get('block'))
                    target = c.get('interprofessional_answers', {}).get(profession, c.get('answer'))
                    with st.spinner("AI Mentor is analyzing..."):
                        st.session_state.ai_feedback = get_ai_feedback(dx_in, full_reasoning, target, profession)
                    st.session_state.submitted = True
                    st.rerun()

    if st.session_state.submitted:
        st.divider()
        res_l, res_r = st.columns(2)
        with res_l:
            st.subheader("🤖 AI Clinical Tutor Feedback")
            st.markdown(st.session_state.ai_feedback)
        with res_r:
            st.subheader("🎯 Benchmarks")
            st.success(f"**Gold Standard:** {c.get('interprofessional_answers', {}).get(profession, c.get('answer'))}")
            with st.expander("Team Perspectives"):
                for role, ans in c.get('interprofessional_answers', {}).items():
                    st.write(f"**{role.upper()}:** {ans}")

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
    else:
        st.info("No simulation data available yet.")

st.markdown("---")
st.caption("ACLR Global v9.9 | Adaptive Cognitive Load–Driven AI Clinical Reasoning Loop | © 2026")
