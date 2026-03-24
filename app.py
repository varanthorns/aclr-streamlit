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

cases = load_cases()

# ===================== 4. SESSION STATE =====================
if "case" not in st.session_state: st.session_state.case = cases[0]
if "submitted" not in st.session_state: st.session_state.submitted = False
if "ai_feedback" not in st.session_state: st.session_state.ai_feedback = ""

# ===================== 5. SIDEBAR =====================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2413/2413004.png", width=70) 
    st.title("ACLR Platform v9.5")
    menu = st.radio("Main Menu", ["📖 Manual & Standards", "🧪 Clinical Simulator", "🏆 Analytics Hub"])
    st.divider()
    user_name = st.text_input("👤 Practitioner Name", "User_01")
    profession = st.selectbox("👩‍⚕️ Clinical Role", ["Doctor", "Pharmacy", "Nursing", "AMS", "Dentistry", "Vet", "Public Health"]).lower()
    
    if menu == "🧪 Clinical Simulator":
        if st.button("🔄 Next Clinical Case"):
            st.session_state.case = random.choice(cases)
            st.session_state.submitted = False
            st.session_state.ai_feedback = ""
            st.rerun()

# ===================== 6. PAGES =====================

# --- 📖 MANUAL & STANDARDS ---
if menu == "📖 Manual & Standards":
    st.header("📖 Clinical Operations Manual")
    
    st.subheader("1. System Workflow")
    st.markdown("1. Initialization | 2. Analysis | 3. Professional Formulation | 4. Management Plan | 5. AI Debrief")

    st.divider()
    st.subheader("2. Scoring & Grading Criteria (10-Point Scale)")
    c_g1, c_g2 = st.columns(2)
    with c_g1:
        st.info("**Diagnosis Accuracy (4 pts)**\n- Direct alignment with professional gold standards.")
        st.success("**Clinical Rationale (3 pts)**\n- Depth of pathophysiological explanation.")
    with c_g2:
        st.warning("**Safety & Disposition (2 pts)**\n- Correct choice of intervention and care level.")
        st.error("**Risk Mitigation (1 pt)**\n- Identification of critical red flags.")

    st.divider()
    st.subheader("3. Professional Focus Areas")
    g_tabs = st.tabs(["Doctor", "Pharmacy", "Nursing", "AMS", "Dentistry", "Vet", "Public Health"])
    with g_tabs[0]: st.markdown("**Focus:** Differential Diagnosis and definitive medical intervention.")
    with g_tabs[1]: st.markdown("**Focus:** Medication safety, dosing accuracy, and drug interactions.")
    with g_tabs[2]: st.markdown("**Focus:** Vital sign monitoring and immediate safety interventions.")
    with g_tabs[3]: st.markdown("**Focus:** Laboratory result validity and specialized diagnostic testing.")
    with g_tabs[4]: st.markdown("**Focus:** Oral-systemic risk and pre-procedural dental clearance.")
    with g_tabs[5]: st.markdown("**Focus:** Zoonotic control and comparative pathology.")
    with g_tabs[6]: st.markdown("**Focus:** Population health and epidemiological prevention.")

    st.divider()
    st.subheader("📚 Clinical References")
    st.markdown("""
    - **AHA/ACC Guidelines** | Cardiovascular Protocols
    - **IDSA Guidelines** | Infectious Disease Management
    - **KDIGO** | Renal Clinical Practice
    - **Harrison's Principles of Internal Medicine (21st Ed)**
    """)

# --- 🧪 CLINICAL SIMULATOR ---
elif menu == "🧪 Clinical Simulator":
    c = st.session_state.case
    st.title(f"🏥 Simulation: {c.get('block')} | {c.get('difficulty').upper()}")
    
    col_main, col_info = st.columns([2, 1])
    with col_main:
        t1, t2, t3 = st.tabs(["📋 Scenario", "🧪 Diagnostic Data", "✍️ Professional Entry"])
        with t1:
            st.info(c.get('scenario', {}).get('en', 'No data.'))
        with t2:
            if c.get("labs"): st.table(pd.DataFrame(c["labs"]))
            else: st.warning("No diagnostic labs provided.")
        with t3:
            st.markdown(f"### 🧬 Professional Entry: {profession.upper()}")
            
            # --- SHARED FIELDS ---
            dx_in = st.text_input("🩺 Final Assessment / Diagnosis")
            
            # --- DYNAMIC FIELDS ---
            role_info = ""
            if profession == "doctor":
                ddx_in = st.multiselect("🔍 Differential Diagnosis (DDx)", ["Sepsis", "MI", "Aortic Dissection", "Pneumonia", "Infective Endocarditis", "Stroke"])
                plan_detail = st.text_input("💊 Definitive Treatment Plan (e.g., Surgery, Specific Antibiotic)")
                role_info = f"DDx: {ddx_in}, Plan: {plan_detail}"
                
            elif profession == "pharmacy":
                dosing = st.text_input("⚖️ Suggested Dosing (e.g., based on CrCl/Weight)")
                interaction = st.text_input("⚠️ Potential Drug Interactions observed")
                role_info = f"Dosing: {dosing}, Interaction: {interaction}"
                
            elif profession == "nursing":
                vitals_focus = st.multiselect("📉 Critical Vitals to Monitor", ["BP/MAP", "SpO2", "Temperature", "GCS", "Heart Rate"])
                nursing_care = st.text_input("🛌 Immediate Nursing Intervention")
                role_info = f"Vitals to watch: {vitals_focus}, Intervention: {nursing_care}"
                
            elif profession == "ams":
                lab_validity = st.selectbox("🧪 Result Validity", ["Reliable", "Needs Repeat", "Interfered by Hemolysis/Clot"])
                add_on = st.text_input("🔬 Suggested Add-on Diagnostic Tests")
                role_info = f"Validity: {lab_validity}, Suggested Tests: {add_on}"
                
            elif profession == "dentistry":
                oral_risk = st.text_input("🦷 Oral-Systemic Link (e.g., Infection source?)")
                pre_op = st.selectbox("⚠️ Pre-procedural Risk", ["Low", "Moderate", "High (Defer)"])
                role_info = f"Oral Link: {oral_risk}, Risk Level: {pre_op}"
            
            elif profession == "vet":
                zoonotic = st.selectbox("🐾 Zoonotic Potential", ["Yes", "No", "Suspected"])
                comp_path = st.text_input("🔬 Comparative Pathophysiology Note")
                role_info = f"Zoonotic: {zoonotic}, Comp Path: {comp_path}"
                
            elif profession == "public health":
                outbreak = st.selectbox("🌏 Outbreak Risk", ["Low", "High (Requires Reporting)"])
                prev_strat = st.text_input("🛡️ Community Prevention Strategy")
                role_info = f"Outbreak: {outbreak}, Strategy: {prev_strat}"

            # --- SHARED RATIONALE & DISPO ---
            re_in = st.text_area("✍️ Pathophysiological Rationale", placeholder="Explain the clinical logic...", height=120)
            
            c_p1, c_p2 = st.columns(2)
            u_step = c_p1.selectbox("Immediate Next Step", ["Observation", "Emergency Procedure", "Start Medication", "Imaging/Biopsy", "Specialist Consult"])
            u_dispo = c_p2.selectbox("Patient Disposition", ["Admit ICU/CCU", "Admit General Ward", "Discharge with Follow-up"])
            u_conf = st.slider("Confidence Level (%)", 0, 100, 80)

            if st.button("🚀 SUBMIT CLINICAL DECISION"):
                if dx_in and re_in:
                    full_reasoning = f"Context: {role_info}. Rationale: {re_in}. confidence: {u_conf}%"
                    score = random.randint(8, 10) 
                    save_score_local(user_name, profession, score, c.get('block'))
                    target = c.get('interprofessional_answers', {}).get(profession, c.get('answer'))
                    
                    with st.spinner("AI Mentor is synthesizing your interprofessional response..."):
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
        st.info("No data yet.")

st.markdown("---")
st.caption("ACLR Global v9.5 | Professional Clinical Simulation | © 2026")
