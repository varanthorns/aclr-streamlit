import streamlit as st
import json, random, pandas as pd, os, time
import google.generativeai as genai
# ===================== ⚙️ GLOBAL CONFIG =====================
DB_FILE = "clinical_scores.csv"  #

# ===================== 🔧 1. FIX + NEW CORE SYSTEM =====================

# 🔐 FIX: ใช้ secrets แทน API key hardcode (ปลอดภัย)
try:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]# --- 🏆 ANALYTICS HUB ---
elif menu == "🏆 Analytics Hub":
    st.header("🏆 Performance Analytics Dashboard")
    
    if os.path.exists(DB_FILE):
        df = pd.read_csv(DB_FILE)
        
        if not df.empty:
            # 1. แสดงตารางข้อมูลล่าสุด
            st.dataframe(df.sort_values(by="Timestamp", ascending=False), use_container_width=True)
            st.divider()
            
            # 2. คำนวณ Metrics หลัก
            c1, c2, c3 = st.columns(3)
            c1.metric("Total Simulations", len(df))
            c2.metric("Mean Diagnosis Score", f"{df['Score'].mean():.1f}/10")
            c3.metric("Avg Time Taken", f"{df['Time'].mean():.0f}s")
            
            # 3. กราฟ Learning Curve (Score over time)
            st.subheader("📈 Learning Curve (Overall Score)")
            df["Timestamp"] = pd.to_datetime(df["Timestamp"])
            st.line_chart(df.set_index("Timestamp")["Score"])
            
            # 4. Competency Breakdown (แก้จุดที่ Error)
            st.subheader("🧠 Competency Radar")
            comp_cols = ["Diagnosis", "Reasoning", "SBAR", "Safety"]
            
            # ตรวจสอบว่าในไฟล์มี Column เหล่านี้จริงๆ ไหม
            existing_cols = [col for col in comp_cols if col in df.columns]
            
            if existing_cols:
                # คำนวณค่าเฉลี่ย
                avg_series = df[existing_cols].mean()
                
                # แสดงเป็น Bar Chart แบบง่ายก่อน
                st.bar_chart(avg_series)
                
                # (Optional) ถ้าลง plotly ไว้ จะสวยมากครับ
                try:
                    import plotly.graph_objects as go
                    fig = go.Figure(data=go.Scatterpolar(
                        r=avg_series.values.tolist() + [avg_series.values[0]],
                        theta=existing_cols + [existing_cols[0]],
                        fill='toself',
                        line_color='#1976D2'
                    ))
                    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 10])))
                    st.plotly_chart(fig, use_container_width=True)
                except ImportError:
                    st.info("Tip: Install 'plotly' to view the Radar Chart.")
                    
        else:
            st.warning("Database is empty. Please complete a case in the Simulator first.")
    else: 
        st.info("No simulation data found. Please start by using the Clinical Simulator.")
except:
    GEMINI_API_KEY = "DEMO_KEY"

genai.configure(api_key=GEMINI_API_KEY)

# ✅ FIX: function ที่หาย
def save_score_local(user, role, score, block, competency=None, time_taken=0):
    new_entry = {
        "User": user,
        "Role": role,
        "Score": score,
        "Block": block,
        "Time": time_taken,
        "Timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    # เพิ่ม competency tracking
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
    if len(df) < 5:
        return "easy"
    
    avg_score = df["Score"].mean()
    
    if avg_score < 6:
        return "easy"
    elif avg_score < 8:
        return "medium"
    else:
        return "hard"
# ===================== 2. CONFIG & MEDICAL UI =====================
st.set_page_config(layout="wide", page_title="FTF-CRA Clinical Analytics Platform", page_icon="🩺")

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
    - Diagnosis Score (0-10)
    - Reasoning Score (0-10)
    - SBAR Score (0-10)
    - Safety Score (0-10)
    - **Overall Score (0-10):**
    - **Strengths:**
    - **Critical Gaps:**
    - **Cognitive Bias (if any):**
    - **Professional Pearl:**
    - **Well-being Tip:**
    
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
    st.title("FTF-CRA Platform")
    menu = st.radio("Main Menu", ["📖 Manual & Standards", "🧪 Clinical Simulator", "🏆 Analytics Hub"])
    st.divider()
    user_name = st.text_input("👤 Practitioner Name", "User_01")
    profession = st.selectbox("👩‍⚕️ Clinical Role", ["Doctor", "Pharmacy", "Nursing", "AMS", "Dentistry", "Vet", "Public Health"]).lower()
    
    st.divider()
    st.subheader("🎯 Session Filters")
    blocks = sorted(list(set([c.get('block', 'General') for c in all_cases])))
    f_block = st.selectbox("Select Block", ["All Blocks"] + blocks)
    f_diff = st.select_slider("Select Difficulty", options=["easy", "medium", "hard"], value="medium")
    # 🧠 Adaptive Mode
    adaptive_mode = st.checkbox("🧠 Adaptive Learning Mode", value=False)
    if adaptive_mode:
        f_diff = get_adaptive_difficulty(user_name)
        st.success(f"AI adjusted difficulty → {f_diff.upper()}")

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
    st.markdown("### **FTF-CRA Platform**")
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
    
    # ⏱️ FEATURE 1: TIME-PRESSURE LOGIC
    elapsed = int(time.time() - st.session_state.start_time)
    time_limit = 600  # 10 minutes
    remaining = max(0, time_limit - elapsed)
    
    # Header Section
    col_h1, col_h2 = st.columns([3, 1])
    with col_h1: 
        st.title(f"🏥 Simulation: {c.get('block')} | Level: {c.get('difficulty').upper()}")
    with col_h2: 
        st.markdown(f"<div class='stress-timer'>⏳ {remaining}s</div>", unsafe_allow_html=True)
        if remaining == 0: 
            st.error("CRITICAL: Efficiency Score Penalized!")

    col_main, col_info = st.columns([2, 1])
    
    with col_main:
        t1, t2, t3 = st.tabs(["📋 Clinical Case Details", "🧠 Clinical Reasoning Map", "✍️ Professional Entry"])
        
        with t1:
            st.subheader("Patient Scenario & Diagnostic Data")
            st.info(c.get('scenario', {}).get('en', 'No data.'))
            
            if c.get("labs"): 
                st.markdown("**🔬 Laboratory Findings:**")
                st.table(pd.DataFrame(c["labs"]))
            
            if st.button("⏩ Advance 24 Hours (Evaluate Evolution)"):
                st.session_state.evolved = True
            
            if st.session_state.evolved:
                st.warning(f"**Evolution:** {c.get('evolution', 'Condition remains stable but requires monitoring.')}")
            
            # 📂 EHR Integration Snapshot
            ehr_data = {
                "Patient ID": "FTF-CRA-001",
                "Vitals": {"BP": "90/60", "HR": 120, "SpO2": "92%"},
                "Status": "ER Admission",
                "Note": "High-risk cardiac event"
            }
            st.divider()
            st.subheader("📂 EHR Snapshot")
            st.json(ehr_data)
        
        with t2:
            st.subheader("Reasoning Map: Data Synthesis")
            st.write("Differentiate Pertinent findings from Clinical Noise.")
            cm_col1, cm_col2 = st.columns(2)
            # ใช้ st.session_state มาคุมค่า เพื่อให้ค่าไม่หายเวลาสลับ Tab
            pos_f = cm_col1.text_area("Pertinent Positives (+)", placeholder="Supporting findings...", height=200, key="map_pos")
            neg_f = cm_col2.text_area("Pertinent Negatives (-)", placeholder="Absent findings...", height=200, key="map_neg")

        with t3:
            st.markdown(f"### 🧬 Professional Entry: {profession.upper()}")
            dx_in = st.text_input("🩺 Final Assessment / Diagnosis", key="entry_dx", placeholder="Enter definitive diagnosis...")
            
            # --- DYNAMIC FIELDS BASED ON ROLE ---
            role_info = ""
            if profession == "doctor":
                ddx = st.multiselect("🔍 Differential Diagnosis (DDx)", ["Sepsis", "MI", "Stroke", "IE", "Pneumonia", "Heart Failure"], key="doctor_ddx")
                plan = st.text_input("💊 Immediate Treatment Plan", key="doctor_plan")
                role_info = f"DDx: {ddx}, Plan: {plan}"
            elif profession == "pharmacy":
                dosing = st.text_input("⚖️ Dosing Logic & Calculation", key="ph_dose")
                interaction = st.text_input("⚠️ Critical Drug Interactions", key="ph_inter")
                role_info = f"Dosing: {dosing}, Interaction: {interaction}"
            elif profession == "nursing":
                vitals_mon = st.multiselect("📉 Priority Vitals to Monitor", ["BP", "SpO2", "Temp", "GCS", "Urine Output"], key="nu_vit")
                n_care = st.text_input("🛌 Immediate Nursing Intervention", key="nu_care")
                role_info = f"Vitals: {vitals_mon}, Care: {n_care}"
            
            re_in = st.text_area("✍️ Pathophysiological Rationale (System 2)", height=150, key="entry_re", placeholder="Explain the underlying mechanism...")
            
            # --- SBAR HANDOVER ---
            st.divider()
            st.markdown("#### 🗣️ SBAR Handover")
            s_col1, s_col2 = st.columns(2)
            h_s = s_col1.text_input("Situation", key="sbar_s")
            h_b = s_col2.text_input("Background", key="sbar_b")
            h_a = s_col1.text_area("Assessment", key="sbar_a", height=100)
            h_r = s_col2.text_area("Recommendation", key="sbar_r", height=100)

            st.divider()
            c_p1, c_p2 = st.columns(2)
            u_step = c_p1.selectbox("Next Step", ["Observe", "Emergency Intervention", "Medication Loading", "Urgent Imaging", "Consult Specialist"])
            u_dispo = c_p2.selectbox("Disposition", ["ICU/CCU", "General Ward", "Discharge / Home"])
            u_conf = st.slider("Diagnostic Confidence (%)", 0, 100, 80)
            
            st.markdown("### 🧘 Reflection & Stress")
            stress_level = st.slider("😓 Current Perceived Stress Level", 0, 10, 5)
            if stress_level > 8:
                st.warning("⚠️ High stress detected. Take a deep breath before final submission.")

            # --- SUBMIT LOGIC ---
            if st.button("🚀 SUBMIT CLINICAL DECISION"):
                if dx_in and re_in:
                    with st.spinner("⚕️ AI Mentor is evaluating your reasoning..."):
                        # 1. Synthesis Reasoning Map Data
                        user_map = f"Positives: {st.session_state.get('map_pos', '')}, Negatives: {st.session_state.get('map_neg', '')}"
                        
                        # 2. Call AI Feedback
                        ai_response = get_ai_feedback_v9_5(
                            user_dx=dx_in, 
                            user_re=f"Rationale: {re_in} | SBAR: {h_s}, {h_b}, {h_a}, {h_r} | Role Info: {role_info}",
                            user_map=user_map,
                            target=c.get('answer'),
                            role=profession,
                            time_taken=elapsed
                        )
                        
                        # 3. Store in State
                        st.session_state.ai_feedback = ai_response
                        
                        # 4. Scoring & Competency Logging
                        target_ans_str = str(c.get('answer')).lower()
                        score = 10 if dx_in.lower() in target_ans_str else 5
                        
                        competency = {
                            "Diagnosis": score,
                            "Reasoning": 8 if len(re_in) > 50 else 4,
                            "SBAR": 10 if all([h_s, h_b, h_a, h_r]) else 5,
                            "Safety": 10 if (u_dispo == "ICU/CCU" and score == 10) else 6
                        }
                        
                        save_score_local(user_name, profession, score, c.get('block'), competency, elapsed)
                        st.session_state.submitted = True
                        st.rerun()
                else:
                    st.error("Please complete the Diagnosis and Rationale before submitting.")

    # --- RESULTS AREA (DISPLAYED AFTER SUBMISSION) ---
    if st.session_state.submitted:
        st.divider()
        
        # 🚨 NEW: COGNITIVE BIAS ALERT BOX
        def extract_bias_alert(ai_text):
            if "Cognitive Bias" in ai_text:
                parts = ai_text.split("Cognitive Bias")
                if len(parts) > 1:
                    return parts[1].split("\n")[0].replace(":", "").replace("**", "").strip()
            return None

        bias_msg = extract_bias_alert(st.session_state.ai_feedback)
        if bias_msg and "none" not in bias_msg.lower():
            st.error(f"🚨 **Cognitive Bias Detected:** {bias_msg}")
        else:
            st.success("✅ **Cognitive Calibration:** No significant bias identified in your reasoning.")

        col_fb1, col_fb2 = st.columns([2, 1])
        with col_fb1:
            st.subheader("👨‍🏫 AI Mentor Clinical Debriefing")
            st.markdown(st.session_state.ai_feedback)
        
        with col_fb2:
            st.subheader("🔑 Gold Standard")
            st.success(f"**Target Diagnosis:** {c.get('answer')}")
            st.info(f"**{profession.capitalize()} Perspective:**\n\n{c.get('interprofessional_answers', {}).get(profession, 'Standard clinical protocol applies.')}")
            
        if st.button("🏁 Start New Simulation Session"):
            st.session_state.submitted = False
            st.session_state.ai_feedback = ""
            st.session_state.start_time = time.time()
            st.session_state.evolved = False
            st.rerun()

# --- 🏆 ANALYTICS HUB ---
elif menu == "🏆 Analytics Hub":
    st.header("🏆 Performance Analytics Dashboard")
    st.markdown("Detailed breakdown of your clinical reasoning journey and competency growth.")
    
    # 1. ตรวจสอบไฟล์ Database
    if os.path.exists(DB_FILE):
        df = pd.read_csv(DB_FILE)
        
        if not df.empty:
            # --- 📈 ส่วนที่ 1: KEY PERFORMANCE METRICS ---
            c1, c2, c3, c4 = st.columns(4)
            
            # คำนวณค่าต่างๆ
            total_sims = len(df)
            avg_score = df['Score'].mean()
            avg_time = df['Time'].mean()
            
            # แสดง Metrics
            c1.metric("Total Cases", total_sims)
            c2.metric("Avg Score", f"{avg_score:.1f}/10")
            c3.metric("Avg Speed", f"{avg_time:.0f}s")
            
            # คำนวณ Success Rate (สมมติว่า Score >= 8 คือ Success)
            success_rate = (len(df[df['Score'] >= 8]) / total_sims) * 100
            c4.metric("Success Rate", f"{success_rate:.0f}%")
            
            st.divider()

            # --- 📊 ส่วนที่ 2: COMPETENCY RADAR & LEARNING CURVE ---
            col_left, col_right = st.columns([1, 1])
            
            with col_left:
                st.subheader("🧠 Competency Radar")
                # นิยาม Columns ที่เราต้องการวัดผล
                comp_cols = ["Diagnosis", "Reasoning", "SBAR", "Safety"]
                # กรองเอาเฉพาะที่มีอยู่ในไฟล์ CSV จริงๆ
                existing_cols = [col for col in comp_cols if col in df.columns]
                
                if existing_cols:
                    avg_values = df[existing_cols].mean().tolist()
                    
                    try:
                        import plotly.graph_objects as go
                        # สร้าง Radar Chart ด้วย Plotly
                        fig = go.Figure(data=go.Scatterpolar(
                            r=avg_values + [avg_values[0]], # ปิดวงกลม
                            theta=existing_cols + [existing_cols[0]],
                            fill='toself',
                            name='Average Competency',
                            line_color='#1976D2',
                            fillcolor='rgba(25, 118, 210, 0.3)'
                        ))
                        
                        fig.update_layout(
                            polar=dict(
                                radialaxis=dict(visible=True, range=[0, 10])
                            ),
                            showlegend=False,
                            margin=dict(l=40, r=40, t=20, b=20)
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.bar_chart(df[existing_cols].mean())
                        st.info("Install 'plotly' for enhanced Radar visualization.")
                else:
                    st.warning("No competency data found in logs.")

            with col_right:
                st.subheader("📈 Learning Progress")
                if "Timestamp" in df.columns:
                    # แปลงเวลาและพลอตกราฟเส้น
                    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
                    # เรียงลำดับตามเวลาก่อนพลอต
                    df_sorted = df.sort_values("Timestamp")
                    st.line_chart(df_sorted.set_index("Timestamp")["Score"])
                    st.caption("Overall Score trends across your simulation sessions.")

            st.divider()

            # --- 📋 ส่วนที่ 3: RAW DATA LOGS ---
            with st.expander("📂 View Full Simulation Logs"):
                # แสดงตารางข้อมูล เรียงจากล่าสุดขึ้นก่อน
                st.dataframe(
                    df.sort_values(by="Timestamp", ascending=False), 
                    use_container_width=True,
                    hide_index=True
                )
                
                # ปุ่มล้างข้อมูล (สำหรับ Debug/Reset)
                if st.button("🗑️ Clear Local History"):
                    if os.path.exists(DB_FILE):
                        os.remove(DB_FILE)
                        st.success("History cleared. Please refresh.")
                        st.rerun()
        else:
            st.warning("Your clinical record is currently empty. Complete a case in the Simulator to see your analytics.")
            
    else: 
        # กรณีรันครั้งแรกแล้วยังไม่มีไฟล์ .csv
        st.info("🚀 Welcome to FTF-CRA! No simulation data found. Start your first clinical case to activate the dashboard.")
    if existing_cols:
        st.subheader("🕸️ Competency Radar")
        avg_values = df[existing_cols].mean().tolist()
        
        fig = go.Figure(data=go.Scatterpolar(
          r=avg_values + [avg_values[0]], # ปิดวงกลม
          theta=existing_cols + [existing_cols[0]],
          fill='toself',
          line_color='#1976D2'
        ))
        fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 10])))
        st.plotly_chart(fig, use_container_width=True)

# --- ⚠️ สำคัญ: ลบโค้ดที่ซ้ำซ้อนหรือหลุดอยู่ล่างสุดของไฟล์ทิ้ง ---
# ตรวจสอบว่าไม่มีบรรทัด 'existing_cols = ...' หลุดอยู่นอกแนว (Indentation) ของ elif นะครับ
    
st.markdown("---")
st.caption("FTF-CRA Global v9.9.5 | Adaptive Cognitive Load–Driven AI Clinical Reasoning Loop | © 2026")
# --- 🧪 UPDATE: AI PROMPT ENHANCEMENT ---
# (หมายเหตุ: ควรไปปรับแก้ฟังก์ชัน get_ai_feedback เดิมให้รับค่าเหล่านี้เข้าไปตรวจด้วย 
# เพื่อให้ AI ตรวจสอบ 'กระบวนการคิด' ไม่ใช่แค่ 'คำตอบสุดท้าย')
