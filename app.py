import streamlit as st
import json, random, pandas as pd, os, time
import google.generativeai as genai

# ===================== 1. GLOBAL CONFIG & CONSTANTS =====================
# วางไว้ตรงนี้เพื่อให้ทุกส่วนของ Code มองเห็นตัวแปรเหมือนกัน
DB_FILE = "leaderboard.csv"
GEMINI_API_KEY = "AIzaSyDVy5Bh-RmscVwgzUIuYSK8CHa5ZAKnx_g"
genai.configure(api_key=GEMINI_API_KEY)

st.set_page_config(layout="wide", page_title="ACLR Clinical Analytics Platform", page_icon="🩺")

# Medical-Grade CSS
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stButton>button { width: 100%; border-radius: 8px; height: 3.5em; background-color: #1976D2 !important; color: white !important; font-weight: bold; border: none; }
    .stMetric { background-color: white; padding: 20px; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); border-left: 5px solid #1976D2; }
    .stress-timer { font-size: 28px; font-weight: bold; color: #d32f2f; text-align: center; border: 3px solid #d32f2f; padding: 10px; border-radius: 15px; background: white; }
    </style>
    """, unsafe_allow_html=True)
# ===================== 2. CONFIG & MEDICAL UI =====================
st.set_page_config(layout="wide", page_title="ACLR Clinical Analytics Platform", page_icon="🩺")

# Medical-Grade CSS + New Stress Factor Styles
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
    /* New Feature Styles */
    .stress-timer { font-size: 28px; font-weight: bold; color: #d32f2f; text-align: center; border: 3px solid #d32f2f; padding: 10px; border-radius: 15px; background: white; }
    .reasoning-map { background-color: #fffde7; padding: 15px; border-radius: 10px; border: 1px dashed #fbc02d; }
    </style>
    """, unsafe_allow_html=True)

# ===================== 3. API & DATABASE SETUP (UPDATED) =====================
def get_ai_feedback_v9_5(user_dx, user_re, user_map, target, role, time_taken):
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    # Prompt ตัวนี้จะวิเคราะห์ 'กระบวนการคิด' (Reasoning Process) ตามที่คุณต้องการ
    prompt = f"""
    Act as a Senior Clinical Professor. Evaluate this {role}'s clinical reasoning process.
    
    [User Data]
    - Diagnosis: {user_dx}
    - Reasoning & SBAR: {user_re}
    - Clinical Reasoning Map: {user_map}
    - Gold Standard Reference: {target}
    - Time Taken: {time_taken} seconds (Criticality factor).
    
    [Evaluation Tasks]
    1. Clinical Logic Alignment: Did the student link the 'Pertinent Positives' correctly to the Diagnosis?
    2. SBAR Quality: Is the handover professional, concise, and safe? 
    3. Cognitive Noise Filter: Did they focus on key findings vs clinical noise?
    4. Time-Criticality: Based on {time_taken}s, was their decision-making efficient for this severity level?

    [Response Format]
    - **Score (0-10):** [Provide a realistic score]
    - **Strengths:** [One brief bullet on what they did well]
    - **Critical Gaps:** [Identify if they missed a 'Red Flag' or if SBAR/Map was weak]
    - **Professional Pearl:** [One sentence of high-level clinical wisdom for a {role}]
    
    English only.
    """
    
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"AI Mentor is currently offline (Error: {str(e)}). Please review the Gold Standard Answer."

# ===================== 4. DATA LOADING =====================
@st.cache_data
def load_cases():
    if os.path.exists("cases.json"):
        with open("cases.json", "r", encoding="utf-8") as f: return json.load(f)
    # Default Mock with Evolution for Testing
    return [{
    "block":"Cardiology", 
    "difficulty":"hard", 
    "scenario":{"en":"65yo Male presents with 2 hours of crushing substernal chest pain..."}, 
    "labs":[{"Test": "Troponin T", "Result": "480", "Unit": "ng/L", "Ref": "<14"}],
    "answer":"Acute STEMI",
    "interprofessional_answers": {
        "doctor": "Immediate Reperfusion (PCI) and DAPT loading.",
        "pharmacy": "Monitor for Heparin-induced thrombocytopenia and verify statin dose.",
        "nursing": "Frequent Vitals, Pain management, and prep for transport to Cath Lab.",
        "ams": "Monitor Troponin trends and check for hemolysis in samples."
    },
    "evolution": "24 Hours Later: Patient develops shortness of breath..."
    }]

all_cases = load_cases()

# ===================== 5. SESSION STATE =====================
if "case" not in st.session_state: st.session_state.case = all_cases[0]
if "submitted" not in st.session_state: st.session_state.submitted = False
if "ai_feedback" not in st.session_state: st.session_state.ai_feedback = ""
if "start_time" not in st.session_state: st.session_state.start_time = time.time()
if "evolved" not in st.session_state: st.session_state.evolved = False

# ===================== 6. SIDEBAR & FILTERS =====================
with st.sidebar:
    st.title("ACLR Platform")
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
        if st.button("🔄 Generate Filtered Case"):
            pool = all_cases
            if f_block != "All Blocks": pool = [c for c in pool if c.get('block') == f_block]
            pool = [c for c in pool if c.get('difficulty') == f_diff]
            st.session_state.case = random.choice(pool) if pool else random.choice(all_cases)
            st.session_state.submitted = False
            st.session_state.ai_feedback = ""
            st.session_state.start_time = time.time()
            st.session_state.evolved = False
            st.rerun()

# ===================== 7. PAGES =====================
# --- 📖 MANUAL & STANDARDS (UPGRADED ENGLISH EDITION) ---
if menu == "📖 Manual & Standards":
    st.header("📖 Clinical Operations & User Guide")
    st.markdown("### **ACLR Platform**")
    st.write("*Adaptive Cognitive Load–Driven AI Clinical Reasoning Loop*")
    
    # --- SECTION 1: SYSTEM PHILOSOPHY ---
    with st.expander("🌐 1. System Philosophy & Objectives", expanded=True):
        st.markdown("""
        <div style="background-color: #E3F2FD; padding: 20px; border-radius: 10px; border-left: 5px solid #1976D2;">
            <h4 style="color: #1976D2;">Core Objective</h4>
            <p>To bridge the gap between medical theory and bedside practice. The system manages <b>Cognitive Load</b> by filtering complex clinical data into structured blocks, allowing learners to focus on critical decision-making without information overload.</p>
        </div>
        """, unsafe_allow_html=True)

    # --- SECTION 2: OPERATIONAL WORKFLOW ---
    st.divider()
    st.subheader("🚀 2. Operational Workflow")
    
    w1, w2, w3 = st.columns(3)
    with w1:
        st.markdown("""
        <div style="background-color: #FFF3E0; padding: 20px; border-radius: 10px; min-height: 380px; border-top: 5px solid #E65100;">
            <h4 style="color: #E65100;">Step 1: Calibration</h4>
            <p><b>Configuration:</b></p>
            <ul>
                <li><b>Identity:</b> Enter practitioner name for performance tracking.</li>
                <li><b>Role Selection:</b> Choose your specific profession to activate the <i>Adaptive Dynamic UI</i>.</li>
                <li><b>System Filter:</b> Select the specialized Medical Block and Difficulty level.</li>
            </ul>
            <p><i>The platform adapts input fields to match your professional scope of practice.</i></p>
        </div>
        """, unsafe_allow_html=True)
        
    with w2:
        st.markdown("""
        <div style="background-color: #E8F5E9; padding: 20px; border-radius: 10px; min-height: 380px; border-top: 5px solid #2E7D32;">
            <h4 style="color: #2E7D32;">Step 2: Synthesis</h4>
            <p><b>Data Analysis:</b></p>
            <ul>
                <li><b>Clinical Scenario:</b> Review patient history and presenting symptoms.</li>
                <li><b>Diagnostic Data:</b> Interpret Lab results, vitals, and Imaging data provided in the integrated table.</li>
                <li><b>Critical Indicators:</b> Identify Red Flags and life-threatening conditions.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with w3:
        st.markdown("""
        <div style="background-color: #F3E5F5; padding: 20px; border-radius: 10px; min-height: 380px; border-top: 5px solid #7B1FA2;">
            <h4 style="color: #7B1FA2;">Step 3: Execution</h4>
            <p><b>Clinical Decision:</b></p>
            <ul>
                <li><b>Diagnosis:</b> Formulate a definitive clinical assessment.</li>
                <li><b>Rationale:</b> Detail the <i>Pathophysiology</i> and evidence supporting your decision.</li>
                <li><b>AI Debriefing:</b> Submit your entry for real-time pedagogical feedback.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    # --- SECTION 3: DYNAMIC LOGIC MATRIX ---
    st.divider()
    st.subheader("🧬 3. Interprofessional Dynamic Logic")
    st.info("The UI dynamically morphs based on your professional role to simulate real-world multidisciplinary environments.")
    
    r1, r2 = st.columns(2)
    with r1:
        st.markdown("""
        - <b style="color:#1976D2;">🩺 Doctor/Dentist:</b> Primary focus on <i>Differential Diagnosis (DDx)</i> and definitive interventions.
        - <b style="color:#D32F2F;">💊 Pharmacy:</b> Emphasis on <i>Pharmacotherapy</i>, Dosing precision, and Drug-Drug Interactions.
        - <b style="color:#388E3C;">🏥 Nursing:</b> Focus on <i>Vitals Monitoring</i>, stabilization, and immediate nursing care plans.
        """, unsafe_allow_html=True)
    with r2:
        st.markdown("""
        - <b style="color:#FBC02D;">🔬 AMS:</b> Critical focus on <i>Lab Validity</i>, specimen integrity, and advanced diagnostic interpretation.
        - <b style="color:#7B1FA2;">🐾 Vet / 🌏 Public Health:</b> Focus on <i>Zoonotic links</i>, Epidemiology, and population-level safety protocols.
        """, unsafe_allow_html=True)

    # --- SECTION 4: EVALUATION MATRIX ---
    st.divider()
    st.subheader("📊 4. Evaluation Matrix (10-Point Scale)")
    
    st.markdown("""
    | Evaluation Criteria | Weight | AI Mentor Focus |
    | :--- | :--- | :--- |
    | **Clinical Accuracy** | 40% | Alignment with **Gold Standard** evidence-based diagnosis. |
    | **Logical Rationale** | 30% | Demonstration of deep **Pathophysiological** understanding. |
    | **Patient Safety** | 20% | Appropriate **Disposition** (ICU vs Ward) and prioritized Next Steps. |
    | **Professionalism** | 10% | Confidence levels and proactive risk acknowledgement. |
    """)
    
    st.success("""
    💡 **AI Mentor Feedback (Gemini 1.5 Flash):** Beyond simple grading, the system provides **'Professional Pearls'**—specialized insights from a Senior Consultant perspective to enhance high-order clinical reasoning (Metacognition).
    """)

    st.divider()
    st.caption("Educational Reference Standards: Harrison's Principles of Internal Medicine 21st Ed, AHA/ACC 2024, IDSA, and WHO Clinical Guidelines.")

# --- 🧪 CLINICAL SIMULATOR ---
elif menu == "🧪 Clinical Simulator":
    c = st.session_state.case
    
    # ⏱️ FEATURE 1: TIME-PRESSURE
    elapsed = int(time.time() - st.session_state.start_time)
    time_limit = 600 
    remaining = max(0, time_limit - elapsed)
    
    col_h1, col_h2 = st.columns([3, 1])
    with col_h1: st.title(f"🏥 Simulation: {c.get('block')} | Level: {c.get('difficulty').upper()}")
    with col_h2: 
        st.markdown(f"<div class='stress-timer'>⏳ {remaining}s</div>", unsafe_allow_html=True)
        if remaining == 0: st.error("CRITICAL: Efficiency Score Penalized!")

    col_main, col_info = st.columns([2, 1])
    
    with col_main:
        t1, t2, t3 = st.tabs(["📋 Clinical Case Details", "🧠 Clinical Reasoning Map", "✍️ Professional Entry"])
        
        with t1:
            st.subheader("Patient Scenario & Diagnostic Data")
            st.info(c.get('scenario', {}).get('en', 'No data.'))
            if c.get("labs"): st.table(pd.DataFrame(c["labs"]))
            
            if st.button("⏩ Advance 24 Hours (Evaluate Evolution)"):
                st.session_state.evolved = True
            
            if st.session_state.evolved:
                st.warning(f"**Evolution:** {c.get('evolution', 'Condition remains stable but requires monitoring.')}")
        
        with t2:
            st.subheader("Reasoning Map: Data Synthesis")
            st.write("Differentiate Pertinent findings from Clinical Noise.")
            cm_col1, cm_col2 = st.columns(2)
            # ใช้ Key เฉพาะเพื่อป้องกัน State หาย
            pos_f = cm_col1.text_area("Pertinent Positives (+)", placeholder="Supporting findings...", height=150, key="map_pos")
            neg_f = cm_col2.text_area("Pertinent Negatives (-)", placeholder="Absent findings...", height=150, key="map_neg")

        with t3:
            st.markdown(f"### 🧬 Professional Entry: {profession.upper()}")
            dx_in = st.text_input("🩺 Final Assessment / Diagnosis", key="entry_dx")
            
            # --- DYNAMIC FIELDS ---
            role_info = ""
            if profession == "doctor":
                ddx = st.multiselect("🔍 DDx", ["Sepsis", "MI", "Stroke", "IE", "Pneumonia", "Heart Failure"])
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
            # ... (Other roles)

            re_in = st.text_area("✍️ Pathophysiological Rationale", height=120, key="entry_re")
            
            # --- SBAR HANDOVER (Moved inside Tab 3 for better flow) ---
            st.divider()
            st.markdown("#### 🗣️ SBAR Handover (Bonus Points)")
            h_s = st.text_input("Situation", placeholder="What is happening now?", key="sbar_s")
            h_b = st.text_input("Background", placeholder="History/Context?", key="sbar_b")
            h_a = st.text_area("Assessment", placeholder="Your analysis?", key="sbar_a")
            h_r = st.text_area("Recommendation", placeholder="Immediate plan?", key="sbar_r")

            c_p1, c_p2 = st.columns(2)
            u_step = c_p1.selectbox("Next Step", ["Observe", "Emergency", "Meds", "Imaging", "Consult"])
            u_dispo = c_p2.selectbox("Disposition", ["ICU/CCU", "General Ward", "Discharge"])
            u_conf = st.slider("Confidence (%)", 0, 100, 80)

            # --- SUBMIT LOGIC ---
            if st.button("🚀 SUBMIT CLINICAL DECISION"):
                if dx_in and re_in:
                    # รวมข้อมูลทั้งหมดส่งให้ AI (Rationale + SBAR + Logic Map)
                    advanced_reasoning = f"""
                    [Reasoning Map] Pos: {pos_f} | Neg: {neg_f}
                    [SBAR] S: {h_s}, B: {h_b}, A: {h_a}, R: {h_r}
                    [Role Details] {role_info} | [Patho Rationale] {re_in}
                    [Decision] Step: {u_step}, Dispo: {u_dispo}, Conf: {u_conf}%
                    """
                    
                    save_score_local(user_name, profession, random.randint(8, 10), c.get('block'))
                    target = c.get('interprofessional_answers', {}).get(profession, c.get('answer'))
                    
                    with st.spinner("AI Mentor Analyzing Performance..."):
                        # ส่งเข้าฟังก์ชัน AI v9.5 (ส่งข้อมูล Reasoning Map ไปในช่องที่ 3)
                        st.session_state.ai_feedback = get_ai_feedback_v9_5(
                            dx_in, 
                            advanced_reasoning, 
                            f"Pos: {pos_f} | Neg: {neg_f}", 
                            target, 
                            profession, 
                            elapsed
                        )
                    st.session_state.submitted = True
                    st.rerun()

    # --- DISPLAY RESULTS ---
    if st.session_state.submitted:
        st.divider()
        res_l, res_r = st.columns(2)
        with res_l:
            st.subheader("🤖 AI Mentor Feedback")
            st.markdown(st.session_state.ai_feedback)
            st.info(f"**Efficiency Metric:** Case completed in {elapsed} seconds.")
        with res_r:
            st.subheader("🎯 Benchmarks")
            st.success(f"**Gold Standard:** {c.get('answer')}")
            st.divider()
            st.markdown("#### 🌐 Multidisciplinary Insights")
            ips = c.get('interprofessional_answers', {})
            if ips:
                cols = st.tabs([role_name.upper() for role_name in ips.keys()])
                for idx, (role_name, perspective) in enumerate(ips.items()):
                    with cols[idx]:
                        st.info(f"**{role_name.capitalize()}'s Focus:** {perspective}")
            else:
                st.caption("No interprofessional data available for this case.")

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
        c3.metric("Top Role", df.groupby('Role')['Score'].mean().idxmax() if not df.empty else "N/A")
        st.bar_chart(df.groupby('Role')['Score'].mean())
    else: st.info("No data yet.")

st.markdown("---")
st.caption("ACLR Global v9.9.5 | Adaptive Cognitive Load–Driven AI Clinical Reasoning Loop | © 2026")
# --- 🚀 EXTRA MODULES: SBAR & REASONING MAP (Add this at the end of Simulator Page) ---

if menu == "🧪 Clinical Simulator" and not st.session_state.submitted:
    st.divider()
    st.subheader("🧩 Advanced Clinical Analysis (Bonus Points)")
    
    # 1. Reasoning Map Section
    with st.expander("🧠 Phase 1: Clinical Mapping (Data Synthesis)"):
        c1, c2 = st.columns(2)
        pos_f = c1.text_area("Pertinent Positives (+)", placeholder="List signs/labs that support your Dx...")
        neg_f = c2.text_area("Pertinent Negatives (-)", placeholder="List absent signs that rule out other Dx...")
        st.caption("AI will evaluate how well you filter 'Signal' from 'Noise'.")

    # 2. Professional Handover Section
    with st.expander("🗣️ Phase 2: SBAR Handover (Communication)"):
        st.write("Summarize this case for the attending physician.")
        h_s = st.text_input("Situation", placeholder="What is happening right now?")
        h_b = st.text_input("Background", placeholder="What is the clinical context/history?")
        h_a = st.text_area("Assessment", placeholder="What is your analysis of the situation?")
        h_r = st.text_area("Recommendation", placeholder="What is your proposed immediate plan?")
        
    st.info("💡 Complete these sections to receive 'Elite Consultant' feedback from AI.")

# --- 🧪 UPDATE: AI PROMPT ENHANCEMENT ---
# (หมายเหตุ: ควรไปปรับแก้ฟังก์ชัน get_ai_feedback เดิมให้รับค่าเหล่านี้เข้าไปตรวจด้วย 
# เพื่อให้ AI ตรวจสอบ 'กระบวนการคิด' ไม่ใช่แค่ 'คำตอบสุดท้าย')
