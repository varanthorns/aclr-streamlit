import streamlit as st
import json, random, pandas as pd, os, time
import google.generativeai as genai
import plotly.graph_objects as go
# ===================== ⚙️ GLOBAL CONFIG =====================
DB_FILE = "clinical_scores.csv"  #

# ===================== 🔧 1. FIX + NEW CORE SYSTEM =====================

# 🔐 FIX: ใช้ secrets แทน API key hardcode (ปลอดภัย)
try:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
except:
    GEMINI_API_KEY = "AIzaSyDyceNZMEAV61jtsPlfvXwKhwlVXcfaEDYY"

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

# ===================== 3. UPGRADED AI MENTOR PROMPT (FTF-CRA V10) =====================
def get_ai_feedback_v9_5(user_dx, user_re, user_map, target, role, time_taken, confidence, stress, first_thought="Not recorded"):
    """
    Advanced Clinical Reasoning Evaluator: 
    Analyzes the shift from Intuition (First Thought) to Analysis (Final Thought).
    """
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    prompt = f"""
    Act as a Senior Medical Educator and Cognitive Psychologist specialized in Clinical Reasoning.
    Your task is to evaluate a {role}'s clinical decision-making process.

    [PHASE 1: THE DATA]
    - First Impression (Intuition): {first_thought}
    - Final Diagnosis (Decision): {user_dx}
    - Supporting Evidence (Reasoning Map): {user_map}
    - Rationale & Communication (SBAR): {user_re}
    - Gold Standard Reference: {target}

    [PHASE 2: PERFORMANCE METRICS]
    - Time Taken: {time_taken}s (Pressure: {"High" if time_taken < 120 else "Moderate"})
    - Reported Confidence: {confidence}%
    - Subjective Stress: {stress}/10

    [PHASE 3: EVALUATION TASKS]
    1. INTUITION VS ANALYSIS: Compare 'First Impression' to 'Final Diagnosis'. Did the student evolve their thinking based on labs, or did they stick to a wrong initial guess?
    2. DATA SYNTHESIS: Did the 'Reasoning Map' include critical Pertinent Negatives? (Identify if they only looked for data that confirmed their bias).
    3. COGNITIVE BIAS DETECTION: Identify specific biases:
       - Anchoring: Stuck on first thought despite conflicting labs.
       - Premature Closure: Decided too quickly without full analysis.
       - Overconfidence: High confidence (>80%) with incorrect diagnosis.
    4. SBAR & SAFETY: Evaluate the handover quality and the safety of the proposed next steps.

    [RESPONSE FORMAT - MANDATORY]
    - Professional Feedback: (Direct, empathetic, but intellectually challenging).
    - Cognitive Bias Alert: (Explicitly name the bias found or 'None detected').
    - Professional Pearl: (One expert-level takeaway for a {role}).
    
    MUST END WITH THIS JSON BLOCK FOR ANALYTICS:
    {{
      "scores": {{
        "Diagnosis": 0-10, 
        "Reasoning": 0-10, 
        "SBAR": 0-10, 
        "Safety": 0-10
      }},
      "bias_detected": "string",
      "pearl": "string"
    }}
    """
    
    try:
        # กำหนดค่าการตอบกลับให้เสถียรขึ้น
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.2, # ลดความเพ้อเจ้อ ให้วิเคราะห์ตาม Fact
            )
        )
        return response.text
    except Exception as e:
        return f"AI Mentor error: {str(e)}"

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
    
    # ⏱️ FEATURE 1: TIME-PRESSURE
    elapsed = int(time.time() - st.session_state.start_time)
    time_limit = 600 
    remaining = max(0, time_limit - elapsed)
    
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
            st.subheader("📋 Clinical Case Details")
            
            # 1. แสดงแค่ Scenario (เนื้อเรื่อง)
            st.info(c.get('scenario', {}).get('en', 'No data.'))
            
            # 2. ให้กรอก First Impression ทันทีที่อ่านเนื้อเรื่องจบ (ก่อนดู Lab)
            st.markdown("""
                <div style="background-color: #e3f2fd; padding: 10px; border-radius: 8px; border-left: 5px solid #1976D2;">
                    <p style="margin-bottom:0; font-weight:bold; color: #1976D2;">💡 First Thought (Intuition Phase)</p>
                </div>
            """, unsafe_allow_html=True)
            
            st.text_input(
                "What is your initial 'Gut' feeling?", 
                key="first_thought", 
                placeholder="พิมพ์ความสงสัยแรกของคุณที่นี่..."
            )
            
            st.divider()

            # 3. ค่อยแสดงผล Labs และข้อมูลเชิงลึก
            st.subheader("🔬 Diagnostic Data & Labs")
            if c.get("labs"): 
                st.table(pd.DataFrame(c["labs"]))
            
            # 🏥 Mock EHR Integration
            ehr_data = {
                "Patient ID": "FTF-CRA-001",
                "Vitals": {"BP": "90/60", "HR": 120, "SpO2": "92%"},
                "Status": "ER Admission",
                "Note": "High-risk cardiac event"
            }
            with st.expander("📂 View EHR Snapshot"):
                st.json(ehr_data)
            
            # ⏩ Evolution Button
            if st.button("⏩ Advance 24 Hours (Evaluate Evolution)"):
                st.session_state.evolved = True
            
            if st.session_state.evolved:
                st.warning(f"**Evolution:** {c.get('evolution', 'Condition remains stable.')}")
        
        with t2:
            st.subheader("Reasoning Map: Data Synthesis")
            st.write("Differentiate Pertinent findings from Clinical Noise.")
            cm_col1, cm_col2 = st.columns(2)
            pos_f = cm_col1.text_area("Pertinent Positives (+)", placeholder="Supporting findings...", height=150, key="map_pos")
            neg_f = cm_col2.text_area("Pertinent Negatives (-)", placeholder="Absent findings...", height=150, key="map_neg")

        with t3:
            st.markdown(f"### 🧬 Professional Entry: {profession.upper()}")
            
            # --- Analysis Phase Header ---
            st.markdown("""
                <div style="background-color: #f1f8e9; padding: 10px; border-radius: 8px; border-left: 5px solid #4CAF50; margin-bottom: 20px;">
                    <p style="margin-bottom:0; font-weight:bold; color: #2E7D32;">🧠 Final Thought (Analysis Phase)</p>
                    <small style="color: #558b2f;">วินิจฉัยสรุปหลังจากวิเคราะห์ข้อมูลทั้งหมด (Final Assessment)</small>
                </div>
            """, unsafe_allow_html=True)

            # 1. Final Diagnosis
            dx_in = st.text_input("🩺 Final Assessment / Diagnosis", key="entry_dx", placeholder="สรุปการวินิจฉัยสุดท้ายของคุณที่นี่...")
            
            st.divider()

            # 2. Dynamic Professional Fields
            st.markdown("#### 🛠️ Specialized Clinical Actions")
            role_info = ""
            if profession == "doctor":
                ddx = st.multiselect("🔍 Differential Diagnosis (DDx)", ["Sepsis", "MI", "Stroke", "IE", "Pneumonia", "Heart Failure", "PE"], key="doc_ddx")
                plan = st.text_input("💊 Immediate Intervention Plan", key="doc_plan")
                role_info = f"DDx: {ddx}, Plan: {plan}"
            elif profession == "pharmacy":
                dosing = st.text_input("⚖️ Pharmacokinetic/Dosing Logic", key="pharma_dosing")
                interaction = st.text_input("⚠️ Drug-Drug Interactions (DDI)", key="pharma_interact")
                role_info = f"Dosing: {dosing}, Interaction: {interaction}"
            elif profession == "nursing":
                vitals_focus = st.multiselect("📉 Critical Vitals to Monitor", ["BP", "SpO2", "Temp", "GCS", "MAP", "Urine Output"], key="nurse_vitals")
                n_care = st.text_input("🛌 Immediate Nursing Intervention", key="nurse_care")
                role_info = f"Vitals Focus: {vitals_focus}, Care Plan: {n_care}"
            elif profession == "ams":
                validity = st.selectbox("🔬 Specimen Validity/Quality", ["Optimal", "Hemolyzed", "Clotted", "Inadequate Volume"], key="ams_valid")
                lab_interp = st.text_area("🧪 Advanced Lab Interpretation", key="ams_lab")
                role_info = f"Validity: {validity}, Lab Interp: {lab_interp}"
            # ... (เพิ่มอาชีพอื่นๆ ได้ตามต้องการในรูปแบบเดียวกัน)

            # 3. Pathophysiology & SBAR
            st.divider()
            re_in = st.text_area("✍️ Pathophysiological Rationale", height=120, key="entry_re", placeholder="อธิบายเหตุผลทางพยาธิสรีรวิทยา...")
            
            with st.expander("🗣️ SBAR Handover (Bonus Points)"):
                h_s = st.text_input("Situation", key="sbar_s_new")
                h_b = st.text_input("Background", key="sbar_b_new")
                h_a = st.text_area("Assessment", key="sbar_a_new")
                h_r = st.text_area("Recommendation", key="sbar_r_new")

            # 4. Metrics & Reflection
            st.markdown("#### 📊 Clinical Decision Metrics")
            c_p1, c_p2 = st.columns(2)
            u_step = c_p1.selectbox("Next Step", ["Observe", "Emergency", "Meds", "Imaging", "Consult"], key="u_step")
            u_dispo = c_p2.selectbox("Disposition", ["ICU/CCU", "General Ward", "Discharge"], key="u_dispo")
            u_conf = st.slider("Confidence (%)", 0, 100, 80, key="u_conf")
            
            st.markdown("### 🧘 Reflection")
            stress_level = st.slider("😓 Stress Level (0-10)", 0, 10, 5, key="u_stress")
            if stress_level > 8:
                st.warning("⚠️ High stress detected. Consider taking a short break.")

            # --- 🚀 SINGLE SUBMIT LOGIC ---
            st.divider()
            if st.button("🚀 SUBMIT CLINICAL DECISION", key="final_submit_btn"):
                if dx_in and re_in:
                    with st.spinner("⚕️ AI Mentor is analyzing your reasoning process..."):
                        f_thought = st.session_state.get('first_thought', 'Not recorded')
                        user_map = f"Positives: {st.session_state.get('map_pos', '')}, Negatives: {st.session_state.get('map_neg', '')}"
                        
                        ai_response = get_ai_feedback_v9_5(
                            user_dx=dx_in, 
                            user_re=f"Rationale: {re_in} | Role Data: {role_info} | SBAR: {h_s}, {h_b}, {h_a}, {h_r}",
                            user_map=user_map,
                            target=c.get('answer'),
                            role=profession,
                            time_taken=elapsed,
                            confidence=u_conf,
                            stress=stress_level,
                            first_thought=f_thought
                        )
                        
                        st.session_state.ai_feedback = ai_response
                        target_ans_str = str(c.get('answer')).lower()
                        score = 10 if dx_in.lower() in target_ans_str else 5
                        
                        competency = {
                            "Diagnosis": score, "Reasoning": 8, 
                            "SBAR": 10 if all([h_s, h_b, h_a, h_r]) else 5,
                            "Safety": 10 if u_dispo == "ICU/CCU" else 7
                        }
                        
                        save_score_local(user_name, profession, score, c.get('block'), competency, elapsed)
                        st.session_state.submitted = True
                        st.rerun()
                else:
                    st.error("Please provide both Diagnosis and Rationale before submitting.")
        
           # --- SUBMIT LOGIC ---
            if st.button("🚀 SUBMIT CLINICAL DECISION"):
                if dx_in and re_in:
                    with st.spinner("⚕️ AI Mentor is analyzing your reasoning process..."):
                        # 1. ดึงข้อมูล First Thought และ Map (เช็คย่อหน้าให้ตรงกัน)
                        f_thought = st.session_state.get('first_thought', 'Not recorded')
                        user_map = f"Positives: {st.session_state.get('map_pos', '')}, Negatives: {st.session_state.get('map_neg', '')}"
                        
                        # 2. เรียกใช้ AI Mentor
                        ai_response = get_ai_feedback_v9_5(
                            user_dx=dx_in, 
                            user_re=f"Rationale: {re_in} | SBAR: {h_s}, {h_b}, {h_a}, {h_r}",
                            user_map=user_map,
                            target=c.get('answer'),
                            role=profession,
                            time_taken=elapsed,
                            confidence=u_conf,
                            stress=stress_level,
                            first_thought=f_thought
                        )
                        
                        # 3. เก็บผลลัพธ์และคำนวณคะแนน
                        st.session_state.ai_feedback = ai_response
                        target_ans_str = str(c.get('answer')).lower()
                        score = 10 if dx_in.lower() in target_ans_str else 5
                        
                        competency = {
                            "Diagnosis": score,
                            "Reasoning": 8, 
                            "SBAR": 10 if all([h_s, h_b, h_a, h_r]) else 5,
                            "Safety": 10 if u_dispo == "ICU/CCU" else 7
                        }
                        
                        save_score_local(user_name, profession, score, c.get('block'), competency, elapsed)
                        st.session_state.submitted = True
                        st.rerun()
    # --- ส่วนแสดงผลหลังจาก Submit แล้ว (ต่อจาก col_main) ---
    if st.session_state.submitted:
        st.divider()
        st.subheader("👨‍🏫 AI Mentor Clinical Debriefing")
        st.markdown(st.session_state.ai_feedback)
        
        with st.expander("🔑 View Gold Standard Answer"):
            st.success(f"**Target Diagnosis:** {c.get('answer')}")
            
            # ดึงคำตอบตาม Profession ที่เลือกมาแสดง
            target_role_answer = c.get('interprofessional_answers', {}).get(profession, "Consult Senior Staff for professional specific guidance.")
            
            st.write(f"**Professional Perspective ({profession.upper()}):**")
            st.info(target_role_answer)
        
        if st.button("🏁 Finish & Start New Case"):
            st.session_state.submitted = False
            st.session_state.ai_feedback = ""
            st.session_state.start_time = time.time()
            st.rerun()

# --- 🏆 ANALYTICS HUB ---
elif menu == "🏆 Analytics Hub":
    st.header("🏆 Performance Analytics Dashboard")
    
    # 1. เช็คว่ามีไฟล์ Database หรือไม่
    if os.path.exists(DB_FILE):
        df = pd.read_csv(DB_FILE)
        
        # 2. เช็คว่าในไฟล์มีข้อมูลหรือไม่
        if not df.empty:
            # จัดรูปแบบวันที่เพื่อให้เรียงลำดับและทำกราฟได้ถูกต้อง
            if "Timestamp" in df.columns:
                df["Timestamp"] = pd.to_datetime(df["Timestamp"])
            
            # --- ส่วนแสดงตารางข้อมูลล่าสุด ---
            with st.expander("📂 View Raw Simulation Data", expanded=False):
                st.dataframe(df.sort_values(by="Timestamp", ascending=False), use_container_width=True)
            
            st.divider()

            # --- ส่วน Metrics สรุปผล ---
            c1, c2, c3 = st.columns(3)
            c1.metric("Total Simulations", len(df))
            c2.metric("Average Score", f"{df['Score'].mean():.1f}/10")
            if "Time" in df.columns:
                c3.metric("Avg. Speed (sec)", f"{df['Time'].mean():.0f}s")
            
            st.divider()

            # --- ส่วนกราฟพัฒนาการ (Learning Curve & Competency) ---
            col_graph1, col_graph2 = st.columns(2)

            with col_graph1:
                st.subheader("📈 Learning Curve")
                if "Timestamp" in df.columns:
                    # แสดงกราฟเส้นความต่อเนื่องของคะแนน
                    chart_data = df.set_index("Timestamp")["Score"]
                    st.line_chart(chart_data)
                else:
                    st.info("Timestamp data missing for trend chart.")

            with col_graph2:
                st.subheader("🧠 Multi-Dimensional Competency")
                # เตรียมข้อมูลสำหรับ Spider Chart
                comp_cols = ["Diagnosis", "Reasoning", "SBAR", "Safety"]
                # กรองเฉพาะ Column ที่มีอยู่จริงในไฟล์ CSV
                existing_cols = [c for c in comp_cols if c in df.columns]

                if len(existing_cols) >= 3:
                    # คำนวณค่าเฉลี่ยของแต่ละทักษะ
                    avg_scores = df[existing_cols].mean().tolist()
                    
                    # สร้าง Spider Chart ด้วย Plotly
                    fig = go.Figure()
                    fig.add_trace(go.Scatterpolar(
                        r=avg_scores + [avg_scores[0]], # เพิ่มจุดแรกปิดท้ายลูป
                        theta=existing_cols + [existing_cols[0]],
                        fill='toself',
                        line_color='#1976D2',
                        fillcolor='rgba(25, 118, 210, 0.3)'
                    ))

                    fig.update_layout(
                        polar=dict(
                            radialaxis=dict(visible=True, range=[0, 10], tickfont=dict(size=10))
                        ),
                        showlegend=False,
                        margin=dict(l=40, r=40, t=40, b=40)
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Need at least 3 competency metrics to generate Spider Chart.")

            # --- กราฟแท่งแสดงค่าเฉลี่ยรายด้าน (Bar Chart) ---
            if existing_cols:
                with st.expander("📊 Detailed Competency Breakdown (Bar Chart)"):
                    st.bar_chart(df[existing_cols].mean())

        else:
            st.warning("Database is empty. Please complete a case in the Simulator.")
            
    else: 
        st.info("No simulation data found. Please start by using the Clinical Simulator.")

# --- ส่วนท้ายไฟล์ (วางนอกเงื่อนไข IF/ELIF ทุกตัว) ---
st.markdown("---")
st.caption("FTF-CRA Global v9.9.5 | Adaptive Cognitive Load–Driven AI Clinical Reasoning Loop | © 2026")
