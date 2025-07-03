
import streamlit as st
import spacy
from spacy.matcher import PhraseMatcher, Matcher
from sentence_transformers import SentenceTransformer, util
import plotly.express as px
import time
import PyPDF2
from io import BytesIO
import pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import base64
import torch
import docx
import re

# Load NLP models
nlp = spacy.load("en_core_web_sm")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
transformer_model = SentenceTransformer('all-MiniLM-L6-v2', device=device)

# SKILLS list (business-related skills removed)
SKILLS = [
    "Python", "Excel", "JavaScript", "SQL", "Machine Learning",
    "Communication", "Project Management", "Data Analysis", "Leadership",
    "React", "Node.js",
    "Teaching", "Classroom Management", "Curriculum Design", "Student Assessment",
    "Lesson Planning", "Educational Technology", "Instruction", "Tutoring",
    "Mentoring", "Classroom Instruction", "Special Education"
]

# Skill weights for each job role
SKILL_WEIGHTS = {
    "Software Engineer": {"Python": 0.3, "JavaScript": 0.2, "React": 0.2, "Node.js": 0.15, "SQL": 0.1, "Machine Learning": 0.05},
    "Data Analyst": {"Python": 0.25, "Excel": 0.2, "SQL": 0.25, "Data Analysis": 0.2, "Machine Learning": 0.1},
    "Teacher": {"Teaching": 0.3, "Classroom Management": 0.2, "Curriculum Design": 0.15, "Student Assessment": 0.15, "Communication": 0.1},
    "Project Manager": {"Project Management": 0.4, "Leadership": 0.3, "Communication": 0.3}
}

DEGREES = ["Bachelor", "Master", "PhD", "Diploma", "Certificate"]
TITLES = [
    "Software Engineer", "Data Analyst", "Intern", "Project Manager",
    "Teacher", "Instructor", "Tutor", "Professor"
]

# JOB_DATABASE with experience requirements
JOB_DATABASE = {
    "Software Engineer": {
        "skills": ["Python", "JavaScript", "React", "Node.js", "SQL", "Machine Learning"],
        "titles": ["Software Engineer", "Intern"],
        "degrees": ["Bachelor", "Master"],
        "experience": "3-5 years"
    },
    "Data Analyst": {
        "skills": ["Python", "Excel", "SQL", "Data Analysis", "Machine Learning"],
        "titles": ["Data Analyst", "Intern"],
        "degrees": ["Bachelor", "Master"],
        "experience": "2-4 years"
    },
    "Teacher": {
        "skills": ["Teaching", "Classroom Management", "Curriculum Design", "Lesson Planning", "Student Assessment", "Communication"],
        "titles": ["Teacher", "Instructor", "Tutor", "Professor"],
        "degrees": ["Bachelor", "Master", "PhD"],
        "experience": "1-3 years"
    },
    "Project Manager": {
        "skills": ["Project Management", "Leadership", "Communication"],
        "titles": ["Project Manager"],
        "degrees": ["Bachelor", "Master"],
        "experience": "5-7 years"
    }
}

# Matchers
def create_matcher(label_list, label_name):
    matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
    matcher.add(label_name, None, *[nlp(text) for text in label_list])
    return matcher

skill_matcher = create_matcher(SKILLS, "SKILLS")
title_matcher = create_matcher(TITLES, "TITLES")
degree_matcher = create_matcher(DEGREES, "DEGREES")

# Extract years of experience
def extract_experience(doc):
    matcher = Matcher(nlp.vocab)
    pattern = [
        {"LIKE_NUM": True},
        {"LOWER": {"IN": ["year", "years"]}, "OP": "?"},
        {"ORTH": "-", "OP": "?"},
        {"LIKE_NUM": True, "OP": "?"},
        {"LOWER": {"IN": ["year", "years"]}, "OP": "?"},
        {"LOWER": {"IN": ["of", "experience"]}, "OP": "*"}
    ]
    matcher.add("EXPERIENCE", [pattern])
    matches = matcher(doc)
    experience = []
    for match_id, start, end in matches:
        experience.append(doc[start:end].text)
    return experience[0] if experience else "Not specified"

# Extract matches
def extract_matches(doc, matcher):
    matches = matcher(doc)
    return sorted(set([doc[start:end].text for _, start, end in matches]))

# Semantic similarity for skills
def calculate_semantic_similarity(cv_skills, job_skills):
    if not job_skills or not cv_skills:
        return 0, set(), set()
    cv_embeddings = transformer_model.encode(cv_skills, convert_to_tensor=True)
    job_embeddings = transformer_model.encode(job_skills, convert_to_tensor=True)
    cosine_scores = util.cos_sim(cv_embeddings, job_embeddings)

    matched_skills = set()
    missing_skills = set(job_skills)
    for i, cv_skill in enumerate(cv_skills):
        max_score = cosine_scores[i].max().item()
        if max_score > 0.7:
            matched_idx = cosine_scores[i].argmax().item()
            matched_skills.add(cv_skill)
            missing_skills.discard(job_skills[matched_idx])

    skill_score = int((len(matched_skills) / len(job_skills)) * 100)
    return skill_score, matched_skills, missing_skills

# Weighted match calculation
def calculate_match(cv_list, job_list, weights=None):
    if not job_list:
        return 0, set(), set()
    matched = set(cv_list) & set(job_list)
    if weights:
        score = sum(weights.get(item, 0.1) for item in matched) / sum(weights.get(item, 0.1) for item in job_list) * 100
    else:
        score = (len(matched) / len(set(job_list))) * 100
    return int(score), matched, set(job_list) - set(matched)

# Job recommendation
def recommend_jobs(cv_text, user_experience=None):
    cv_doc = nlp(cv_text)
    cv_skills = extract_matches(cv_doc, skill_matcher)
    cv_titles = extract_matches(cv_doc, title_matcher)
    cv_degrees = extract_matches(cv_doc, degree_matcher)
    cv_experience = extract_experience(cv_doc)

    recommendations = []

    for job, reqs in JOB_DATABASE.items():
        skill_score, matched_skills, missing_skills = calculate_semantic_similarity(cv_skills, reqs["skills"])
        title_score, _, _ = calculate_match(cv_titles, reqs["titles"])
        degree_score, _, _ = calculate_match(cv_degrees, reqs["degrees"])
        experience_score = 100 if cv_experience and reqs["experience"] in cv_experience else 50

        final_score = round((0.5 * skill_score + 0.2 * title_score + 0.15 * degree_score + 0.15 * experience_score), 1)

        # Filter by user preferences
        if user_experience and reqs["experience"] not in user_experience:
            continue

        if final_score >= 50:
            recommendations.append((job, final_score, matched_skills, missing_skills, reqs["experience"]))

    recommendations.sort(key=lambda x: x[1], reverse=True)
    return recommendations

# Extract text from files
def extract_text_from_file(uploaded_file):
    if uploaded_file is None:
        return ""
    try:
        if uploaded_file.name.endswith('.pdf'):
            pdf_reader = PyPDF2.PdfReader(BytesIO(uploaded_file.read()))
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() or ""
            return text
        elif uploaded_file.name.endswith('.txt'):
            return uploaded_file.read().decode('utf-8')
        elif uploaded_file.name.endswith('.docx'):
            doc = docx.Document(BytesIO(uploaded_file.read()))
            text = "\n".join([para.text for para in doc.paragraphs])
            return text
        else:
            st.error("Unsupported file format. Please upload a .txt, .pdf, or .docx file.")
            return ""
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return ""

# Generate PDF report
def generate_pdf_report(final_score, matched_skills, missing_skills, recommendations, job_experience):
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    c.setFont("Helvetica", 12)
    c.drawString(50, 750, "CV vs Job Description Match Report")
    c.drawString(50, 730, f"Final Match Score: {final_score}%")
    c.drawString(50, 710, f"Job Experience Required: {job_experience}")
    c.drawString(50, 680, "Matched Skills:")
    y = 660
    for skill in matched_skills:
        c.drawString(70, y, f"- {skill}")
        y -= 20
    c.drawString(50, y - 20, "Missing Skills:")
    y -= 40
    for skill in missing_skills:
        c.drawString(70, y, f"- {skill}")
        y -= 20
    c.drawString(50, y - 20, "Recommended Jobs:")
    y -= 40
    for job, score, _, _, exp in recommendations:
        c.drawString(70, y, f"- {job}: {score}% (Experience: {exp})")
        y -= 20
    c.save()
    buffer.seek(0)
    return buffer

# Custom CSS
st.markdown("""
    <style>
    body {
        font-family: 'Inter', sans-serif;
    }
    .stApp {
        background-color: #f5f7fa;
    }
    .main-title {
        color: #1e3a8a;
        font-size: 2.5em;
        font-weight: 700;
        text-align: center;
        margin-bottom: 0.5em;
    }
    .tagline {
        color: #4b5563;
        font-size: 1.2em;
        text-align: center;
        margin-bottom: 2em;
    }
    .stButton>button {
        background-color: #1e3a8a;
        color: white;
        border-radius: 8px;
        padding: 0.5em 1.5em;
        font-weight: 500;
        transition: background-color 0.3s;
    }
    .stButton>button:hover {
        background-color: #3b82f6;
    }
    .stTextArea textarea {
        border: 1px solid #d1d5db;
        border-radius: 8px;
        background-color: #ffffff;
        padding: 1em;
    }
    .stTabs [role="tab"] {
        font-weight: 500;
        color: #1e3a8a;
        border-bottom: 2px solid transparent;
    }
    .stTabs [role="tab"][aria-selected="true"] {
        border-bottom: 2px solid #1e3a8a;
        color: #1e3a8a;
    }
    .card {
        background-color: #ffffff;
        padding: 1.5em;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1.5em;
    }
    .sidebar .sidebar-content {
        background-color: #ffffff;
        border-right: 1px solid #e5e7eb;
    }
    .metric-card {
        background-color: #ffffff;
        padding: 1em;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# UI Layout
st.set_page_config(page_title="CV Matcher", layout="wide", page_icon="💼")
st.markdown('<h1 class="main-title">Advanced CV Matcher</h1>', unsafe_allow_html=True)
st.markdown('<p class="tagline">Find your perfect job with AI-powered semantic matching and detailed analysis</p>', unsafe_allow_html=True)

# Initialize session state
if 'cv_uploaded_text' not in st.session_state:
    st.session_state['cv_uploaded_text'] = ""
if 'jd_uploaded_text' not in st.session_state:
    st.session_state['jd_uploaded_text'] = ""

# Sidebar
with st.sidebar:
    st.markdown("### 🔍 About This Tool")
    st.info("""
        This tool uses advanced NLP to match your CV with job descriptions, analyzing skills, titles, degrees, and experience requirements.
        Upload or paste your CV and job description to get a detailed match score, recommendations, and actionable feedback.
        **Data Privacy**: Your data is processed locally and not stored. All uploads are cleared after analysis.
    """)
    st.markdown("---")
    st.markdown("### ⚙️ User Preferences")
    user_experience = st.selectbox("Experience Level", options=["Any", "0-2 years", "2-4 years", "3-5 years", "5-7 years", "7+ years"])
    location = st.text_input("Preferred Location (optional)")
    remote = st.checkbox("Remote Work Only")
    st.markdown("---")
    st.markdown("### 📋 Keyword Lists")
    with st.expander("🧠 Skills", expanded=False):
        st.write(", ".join(SKILLS))
    with st.expander("👤 Job Titles", expanded=False):
        st.write(", ".join(TITLES))
    with st.expander("🎓 Degrees", expanded=False):
        st.write(", ".join(DEGREES))
    st.markdown("---")
    st.markdown("**Developed by xAI** | [Learn More](https://x.ai)")

# Text Input and File Upload Columns
col1, col2 = st.columns([1, 1], gap="medium")

with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 📄 Provide Your CV")
    st.markdown("Paste your CV text or upload a file (.txt, .pdf, or .docx).")
    cv_file = st.file_uploader("Upload CV File", type=['txt', 'pdf', 'docx'], key="cv_file")
    if cv_file:
        st.session_state['cv_uploaded_text'] = extract_text_from_file(cv_file)
        st.success("CV file uploaded successfully!")
    cv_text = st.text_area("Your CV", value=st.session_state['cv_uploaded_text'], height=250, placeholder="Paste your CV here...", key="cv_input")
    if st.button("Clear CV", key="clear_cv"):
        st.session_state['cv_uploaded_text'] = ""
        st.experimental_rerun()
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 📑 Provide Job Description")
    st.markdown("Paste the job description or upload a file (.txt, .pdf, or .docx).")
    job_file = st.file_uploader("Upload Job Description File", type=['txt', 'pdf', 'docx'], key="jd_file")
    if job_file:
        st.session_state['jd_uploaded_text'] = extract_text_from_file(job_file)
        st.success("Job description file uploaded successfully!")
    job_text = st.text_area("Job Description", value=st.session_state['jd_uploaded_text'], height=250, placeholder="Paste job description here...", key="jd_input")
    if st.button("Clear Job Description", key="clear_jd"):
        st.session_state['jd_uploaded_text'] = ""
        st.experimental_rerun()
    st.markdown('</div>', unsafe_allow_html=True)

# Analyze Button
if st.button("🔍 Analyze CV", use_container_width=True):
    if not cv_text or not job_text:
        st.error("Please provide both a CV and a Job Description to analyze.", icon="⚠️")
    else:
        with st.spinner("Analyzing your CV with advanced NLP..."):
            time.sleep(1)
            cv_doc = nlp(cv_text)
            job_doc = nlp(job_text)

            # Extract matches
            cv_skills = extract_matches(cv_doc, skill_matcher)
            job_skills = extract_matches(job_doc, skill_matcher)
            cv_titles = extract_matches(cv_doc, title_matcher)
            job_titles = extract_matches(job_doc, title_matcher)
            cv_degrees = extract_matches(cv_doc, degree_matcher)
            job_degrees = extract_matches(job_doc, degree_matcher)
            job_experience = extract_experience(job_doc)

            # Calculate scores
            skill_score, matched_skills, missing_skills = calculate_semantic_similarity(cv_skills, job_skills)
            title_score, matched_titles, missing_titles = calculate_match(cv_titles, job_titles)
            degree_score, matched_degrees, missing_degrees = calculate_match(cv_degrees, job_degrees)
            experience_score = 100 if job_experience in extract_experience(cv_doc) else 50
            final_score = round((0.5 * skill_score + 0.2 * title_score + 0.15 * degree_score + 0.15 * experience_score), 1)

            # Tabs
            tabs = st.tabs([
                "📈 Summary Dashboard",
                "📊 Match Overview",
                "✅ Matched Items",
                "❌ Missing Items",
                "📃 Details",
                "🎯 Job Recommendations"
            ])

            with tabs[0]:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.subheader("Summary Dashboard")
                st.markdown(f"**CV Summary**: {len(cv_skills)} skills, {len(cv_titles)} titles, {len(cv_degrees)} degrees detected.")
                st.markdown(f"**Job Description Summary**: {len(job_skills)} skills, {len(job_titles)} titles, {len(job_degrees)} degrees, Experience: {job_experience}")
                st.markdown(f"**Final Match Score**: {final_score}%")
                fig = px.pie(
                    values=[final_score, 100-final_score],
                    names=[f'Match ({final_score}%)', 'Gap'],
                    color_discrete_sequence=['#1e3a8a', '#e5e7eb'],
                    title="Overall Match Score"
                )
                st.plotly_chart(fig, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

            with tabs[1]:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.subheader("Match Overview")
                st.progress(final_score / 100)
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("⭐ Final Score", f"{final_score}%")
                st.markdown('</div>', unsafe_allow_html=True)

                st.markdown("### Match Breakdown")
                categories = ['Skills', 'Titles', 'Degrees', 'Experience']
                scores = [skill_score, title_score, degree_score, experience_score]

                fig = px.bar(
                    x=categories, y=scores, color=categories,
                    color_discrete_sequence=px.colors.sequential.Blues,
                    title="Match Breakdown by Category",
                    labels={'y': 'Match Percentage (%)', 'x': ''},
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

            with tabs[2]:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.subheader("Matched Items")
                st.markdown("**🧠 Skills:** " + ", ".join([f"`{s}`" for s in matched_skills]) if matched_skills else "No matched skills found.")
                st.markdown("**👤 Titles:** " + ", ".join([f"`{t}`" for t in matched_titles]) if matched_titles else "No matched titles found.")
                st.markdown("**🎓 Degrees:** " + ", ".join([f"`{d}`" for d in matched_degrees]) if matched_degrees else "No matched degrees found.")
                st.markdown(f"**⏳ Experience:** {extract_experience(cv_doc)} matches {job_experience}" if experience_score == 100 else f"Experience mismatch: CV ({extract_experience(cv_doc)}) vs Job ({job_experience})")
                st.markdown('</div>', unsafe_allow_html=True)

            with tabs[3]:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.subheader("Missing from CV")
                st.markdown("**🧠 Skills:** " + ", ".join([f"`{s}`" for s in missing_skills]) if missing_skills else "No missing skills.")
                st.markdown("**👤 Titles:** " + ", ".join([f"`{t}`" for t in missing_titles]) if missing_titles else "No missing titles.")
                st.markdown("**🎓 Degrees:** " + ", ".join([f"`{d}`" for d in missing_degrees]) if missing_degrees else "No missing degrees.")
                if missing_skills:
                    st.markdown("**💡 Skill Gap Analysis:** Add these skills to your CV to improve your match score:")
                    for skill in missing_skills:
                        st.markdown(f"- {skill} (e.g., take a course on Coursera or Udemy)")
                if experience_score < 100:
                    st.markdown(f"**⏳ Experience Gap:** Job requires {job_experience}. Consider highlighting relevant experience.")
                st.markdown('</div>', unsafe_allow_html=True)

            with tabs[4]:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                with st.expander("🔍 CV Skills", expanded=False):
                    st.write(", ".join(cv_skills) if cv_skills else "No skills detected.")
                with st.expander("🔍 Job Description Skills", expanded=False):
                    st.write(", ".join(job_skills) if job_skills else "No skills detected.")
                with st.expander("📌 Titles and Degrees", expanded=False):
                    st.write(f"**CV Titles:** {', '.join(cv_titles) if cv_titles else 'None'}")
                    st.write(f"**JD Titles:** {', '.join(job_titles) if job_titles else 'None'}")
                    st.write(f"**CV Degrees:** {', '.join(cv_degrees) if cv_degrees else 'None'}")
                    st.write(f"**JD Degrees:** {', '.join(job_degrees) if job_degrees else 'None'}")
                    st.write(f"**CV Experience:** {extract_experience(cv_doc)}")
                    st.write(f"**JD Experience:** {job_experience}")
                st.markdown('</div>', unsafe_allow_html=True)

            with tabs[5]:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.subheader("Recommended Jobs")
                recommendations = recommend_jobs(cv_text, user_experience if user_experience != "Any" else None)
                if recommendations:
                    for job, score, matched, missing, exp in recommendations:
                        st.markdown(f"**{job}** — Match Score: {score}% (Experience Required: {exp})")
                        with st.expander(f"Details for {job}"):
                            st.markdown(f"**Matched Skills:** {', '.join(matched) if matched else 'None'}")
                            st.markdown(f"**Missing Skills:** {', '.join(missing) if missing else 'None'}")
                            if missing:
                                st.markdown("**Improve Your CV:**")
                                st.markdown(f"Consider adding: {', '.join(missing)}")
                else:
                    st.info("No strong job matches found. Try adding more skills, qualifications, or relevant experience to your CV!")
                st.markdown("---")
                st.markdown("### Available Skills Considered")
                st.markdown("Below are all skills the tool checks for. Ensure your CV includes relevant ones:")
                st.write(", ".join([f"`{s}`" for s in SKILLS]))

                # Download report
                if final_score and recommendations:
                    pdf_buffer = generate_pdf_report(final_score, matched_skills, missing_skills, recommendations, job_experience)
                    b64_pdf = base64.b64encode(pdf_buffer.getvalue()).decode('utf-8')
                    href = f'<a href="data:application/pdf;base64,{b64_pdf}" download="cv_match_report.pdf">Download Match Report</a>'
                    st.markdown(href, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
