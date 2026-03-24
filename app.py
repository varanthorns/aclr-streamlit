import streamlit as st
import json, random, pandas as pd, os, time
import google.generativeai as genai
from streamlit_mic_recorder import mic_recorder

# ===================== 1. CONFIG & MEDICAL UI =====================
st.set_page_config(layout="wide", page_title="ACLR Clinical Analytics Platform", page_icon="🏥")

# Medical-Grade CSS Injector
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stButton>button { width: 100%; border-radius: 8px; height: 3.5em; background-color: #1976D2; color: white; font-weight: bold; border: none; }
    .stButton>button:hover { background-color: #1565C0; border: none; }
    .stMetric { background-color: white; padding: 20px; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); border-left: 5px solid #1976D2; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] { background-color: #e3f2fd; border-radius: 8px 8px 0 0; padding: 12px 24px; color: #1976D2; }
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
    As a Senior Medical Instructor, evaluate this {role}'s reasoning.
    Dx: {user_dx} | Reasoning: {user_re} | Gold Standard: {target}
    Feedback (3 bullets max): 1. Accuracy 2. Missing logical links 3. Clinical Pearl.
    """
    try:
        return model.generate_content(prompt).text
    except:
        return "AI Mentor is currently offline. Review the Gold Standard instead."

# ===================== 3. DATA LOADING =====================
@st.cache_data
def load_cases():
    if os.path.exists("cases.json"):
        with open("cases.json", "r", encoding="utf-8") as f:
            return json.load(f)
    return [{"block":"General", "difficulty":"easy", "scenario":{"en":"Standard Case"}, "answer":"N/A"}]

cases = load_cases()

# ===================== 4. SESSION STATE =====================
if "case" not in st.session_state: st.session_state.case = cases[0]
if "submitted" not in st.session_state: st.session_state.submitted = False
if "ai_feedback" not in st.session_state: st.session_state.ai_feedback = ""

# ===================== 5. SIDEBAR =====================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2413/2413004.png", width=80)
    st.title("ACLR Master v8.0")
    menu = st.radio("Navigation", ["📖 User Guide", "🧪 Simulator", "🏆 Leaderboard"])
    st.divider()
    user_name = st.text_input("👤 Practitioner Name", "Guest_User")
    profession = st.selectbox("👩‍⚕️ Your Role", ["Doctor", "Pharmacy", "Nursing", "AMS", "Dentistry", "Vet", "Public Health"]).lower()
    
    if menu == "🧪 Simulator":
        if st.button("🔄 Generate New Case"):
            st.session_state.case = random.choice(cases)
            st.session_state.submitted = False
            st.session_state.ai_feedback = ""
            st.rerun()

# ===================== 6. PAGES =====================

# --- 📖 USER GUIDE PAGE ---
if menu == "📖 User Guide":
    st.header("📖 Clinical Reasoning & Interprofessional Guide")
    st.write("แพลตฟอร์ม ACLR ออกแบบมาเพื่อฝึกการตัดสินใจแบบสหสาขาวิชาชีพ (IPE) โดยใช้ AI ประเมินผลตามบทบาทจริง")
    
    st.subheader("📦 Clinical Blocks")
    col_a, col_b = st.columns(2)
    col_a.markdown("- **Cardiovascular:** MI, Heart Failure, IE\n- **Respiratory:** Pneumonia, Asthma, TB\n- **Neurological:** Stroke, Meningitis")
    col_b.markdown("- **Musculoskeletal:** Gait, Trauma, Rehab\n- **Emergency:** Sepsis, Shock, Toxicology\n- **Hematology:** Anemia, Coagulopathy")

    st.divider()
    st.subheader("👩‍⚕️ Professional Focus (Role-Based)")
    p_tabs = st.tabs(["Doctor", "Pharmacy", "Nursing", "AMS", "Dentistry", "Vet", "Public Health"])
    with p_tabs[0]: st.info("**Doctor:** เน้นการวินิจฉัยพยาธิสภาพหลักและการสั่งรักษาเชิงรุก")
    with p_tabs[1]: st.success("**Pharmacy:** เน้นความปลอดภัยของยา (Dose/Interaction) และกลไกยา")
    with p_tabs[2]: st.warning("**Nursing:** เน้นการเฝ้าระวังสัญญาณชีพและอาการ Red Flags (Safety First)")
    with p_tabs[3]: st.error("**AMS:** เน้นการแปลผลทางห้องปฏิบัติการและความจำเพาะของการตรวจ")
    with p_tabs[4]: st.info("**Dentistry:** เน้นความเชื่อมโยงของสุขภาพช่องปากกับโรคทางระบบ")
    with p_tabs[5]: st.success("**Vet:** เน้นโรคติดต่อระหว่างสัตว์และคน (Zoonosis)")
    with p_tabs[6]: st.warning("**Public Health:** เน้นการควบคุมโรคระบาดและปัจจัยเสี่ยงในชุมชน")

# --- 🧪 SIMULATOR PAGE ---
elif menu == "🧪 Simulator":
    c = st.session_state.case
    st.title(f"🏥 Station: {c.get('block')} | {c.get('difficulty').upper()}")
    
    col_main, col_info = st.columns([2, 1])
    with col_main:
        t1, t2, t3 = st.tabs(["📋 Scenario", "🧪 Diagnostics", "✍️ Analysis"])
        with t1:
            st.info(c.get('scenario', {}).get('en', 'No Data'))
            st.caption(f"TH: {c.get('scenario', {}).get('th', '')}")
        with t2:
            if c.get("labs"): st.table(pd.DataFrame(c["labs"]))
            else: st.write("No diagnostic labs for this case.")
        with t3:
            st.markdown(f"**Analyzing as: {profession.upper()}**")
            dx_in = st.text_input("🩺 Final Diagnosis")
            re_in = st.text_area("✍️ Pathophysiological Reasoning", placeholder="Explain based on your expertise...", height=150)
            
            p1, p2 = st.columns(2)
            u_step = p1.selectbox("Next Step", ["Observation", "Emergency Procedure", "Start Medication", "Imaging", "Consult"])
            u_dispo = p2.selectbox("Disposition", ["Admit ICU", "Admit General Ward", "Discharge Home"])
            u_conf = st.slider("Confidence (%)", 0, 100, 50)

            if st.button("✅ Submit Professional Decision"):
                if dx_in and re_in:
                    score = random.randint(7, 10) # Logic สามารถปรับให้ซับซ้อนขึ้นได้
                    save_score_local(user_name, profession, score, c.get('block'))
                    target = c.get('interprofessional_answers', {}).get(profession, c.get('answer'))
                    with st.spinner("AI Mentor is evaluating your logic..."):
                        st.session_state.ai_feedback = get_ai_feedback(dx_in, re_in, target, profession)
                    st.session_state.submitted = True
                    st.rerun()

    if st.session_state.submitted:
        st.divider()
        res_l, res_r = st.columns(2)
        with res_l:
            st.subheader("🤖 AI Clinical Tutor Feedback")
            st.markdown(st.session_state.ai_feedback)
        with res_r:
            st.subheader("🎯 Gold Standard Answer")
            st.success(c.get('interprofessional_answers', {}).get(profession, c.get('answer')))
            with st.expander("View Team Perspectives"):
                for role, ans in c.get('interprofessional_answers', {}).items():
                    st.write(f"**{role.upper()}:** {ans}")

# --- 🏆 LEADERBOARD PAGE ---
elif menu == "🏆 Leaderboard":
    st.header("🏆 Regional Performance Leaderboard")
    if os.path.exists(DB_FILE):
        df = pd.read_csv(DB_FILE)
        st.dataframe(df.sort_values(by="Timestamp", ascending=False), use_container_width=True
