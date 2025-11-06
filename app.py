import fitz  # PyMuPDF for PDF handling (optional fallback)
import docx
import pytesseract  # OCR for image extraction (optional fallback)
import joblib
import pandas as pd
import streamlit as st
from PIL import Image
from sklearn.pipeline import Pipeline
import numpy as np
import requests
import time
import plotly.graph_objects as go

# ----------- AFFINDA RESUME PARSING -----------
def affinda_parse_resume(file):
    AFFINDA_API_KEY = "aff_48d9ebd32229f35e0e91d9160bdb477c5a870503"
    url = "https://api.affinda.com/v2/resumes"
    headers = {
        "Authorization": f"Bearer {AFFINDA_API_KEY}"
    }
    files = {"file": (file.name, file, file.type)}
    response = requests.post(url, headers=headers, files=files)
    if response.status_code in (200, 201):
        data = response.json()
        if data.get("failed") or (data.get("error", {}).get("errorCode") or data.get("error", {}).get("errorDetail")):
            st.warning(f"Affinda Resume Parsing error: {data.get('error')}")
            return "", [], {}
        text = data.get("data", {}).get("text", "")
        skills = data.get("data", {}).get("skills", [])
        skills_list = [skill.get("name", "") for skill in skills]
        return text, skills_list, data.get("data", {})
    else:
        st.warning(f"Affinda Resume Parsing HTTP error: {response.text}")
        return "", [], {}

# ----------- FALLBACK PARSERS (IN CASE AFFINDA FAILS) -----------
def extract_text_from_pdf(uploaded_file):
    try:
        pdf_data = uploaded_file.read()
        doc = fitz.open(stream=pdf_data, filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text("text") + "\n"
        doc.close()
        return text.strip()
    except Exception as e:
        return f"Error extracting text: {e}"

def extract_text_from_docx(uploaded_file):
    doc = docx.Document(uploaded_file)
    text = "\n".join([para.text for para in doc.paragraphs])
    return text.strip()

def extract_text_from_image(uploaded_file):
    img = Image.open(uploaded_file)
    text = pytesseract.image_to_string(img)
    return text.strip()

def fallback_parse_resume(uploaded_file):
    ext = uploaded_file.name.split(".")[-1].lower()
    if ext == "pdf":
        text = extract_text_from_pdf(uploaded_file)
    elif ext == "docx":
        text = extract_text_from_docx(uploaded_file)
    elif ext in ["jpg", "jpeg", "png"]:
        text = extract_text_from_image(uploaded_file)
    else:
        st.error("Unsupported file format! Please upload PDF, DOCX, JPG, or PNG.")
        return "", []
    return text, []

# ----------- ML MODEL FOR JOB ROLE PREDICTION -----------
# CORRECTED: Load both pipeline and mlb
pipeline, mlb = joblib.load("model_full_data.pkl")

SKILL_SET = {
    "python", "java", "aws", "docker", "linux", "terraform", "kubernetes", "sql",
    "javascript", "html", "css", "git", "c++", "c#", "azure", "gcp", "django",
    "flask", "react", "node.js", "mongodb", "postgresql", "excel", "pandas", "numpy"
}

def normalize_skills(skills):
    skills_list = [skill.strip().lower() for skill in skills]
    skills_list = sorted(set(skills_list))
    return ', '.join(skills_list)

def extract_skills_from_text(text):
    found = set()
    text_lower = text.lower()
    for skill in SKILL_SET:
        if skill in text_lower:
            found.add(skill)
    return list(found)

# CORRECTED: Use mlb.classes_ and pipeline from tuple
def predict_job_roles(skills_list):
    if not skills_list:
        return [], []
    norm_skills = normalize_skills(skills_list)
    job_probs = pipeline.predict_proba([norm_skills])[0]
    max_prob = np.max(job_probs)
    best_indices = np.where(job_probs == max_prob)[0]
    best_roles = mlb.classes_[best_indices]
    top_indices = np.argsort(job_probs)[-6:][::-1]
    top_roles = mlb.classes_[top_indices]
    filtered_roles = [role for role in top_roles if role not in best_roles]
    return filtered_roles, best_roles

# ----------- ADZUNA JOB SCRAPING (INDIA ONLY, LOCATIONLESS) -----------
def scrape_jobs_adzuna(keyword):
    APP_ID = "de9c9858"
    APP_KEY = "bd41a93e1c5c6363cf40e9652d70e594"
    COUNTRY = "in"
    url = f"https://api.adzuna.com/v1/api/jobs/{COUNTRY}/search/1"
    params = {
        "app_id": APP_ID,
        "app_key": APP_KEY,
        "results_per_page": 20,
        "what": keyword,
        "content-type": "application/json"
    }
    resp = requests.get(url, params=params)
    jobs = []
    if resp.status_code == 200:
        api_results = resp.json().get("results", [])
        for job in api_results:
            if job.get("title") and job.get("redirect_url"):
                jobs.append({
                    "id": job.get("id", job.get("redirect_url")),
                    "title": job.get("title"),
                    "company": job.get("company", {}).get("display_name", "Unknown"),
                    "location": job.get("location", {}).get("display_name", "Unknown"),
                    "url": job.get("redirect_url")
                })
    else:
        st.warning(f"Error fetching jobs: {resp.text}")
    return jobs

def submit_applications(jobs, selected_job_ids, resume_text):
    # This is a mock; real automated application is not supported via API
    applications = []
    for job in jobs:
        if job["id"] in selected_job_ids:
            status = "Applied"
            applications.append({
                "job_id": job["id"],
                "title": job["title"],
                "company": job["company"],
                "status": status,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                "job_url": job["url"]
            })
    return applications

# ----------- STREAMLIT UI -----------
st.set_page_config(page_title="AI Resume & Job Portal", page_icon=":mag_right:", layout="wide")
st.title("🔍 AI Resume Screening & Live Job Portal")

tab1, tab2, tab3 = st.tabs([
    "Resume Parsing & Analysis (Affinda)",
    "Live Job Search & Application",
    "Real-Time Application Dashboard"
])

# ----------- TAB 1: RESUME PARSING & ANALYSIS -----------
with tab1:
    st.write("Upload your resume to get parsed and analyzed. We'll extract your skills, show the resume summary and recommend job roles for you.")
    uploaded_file = st.file_uploader("Upload Resume (PDF, DOCX, JPG, PNG)", type=["pdf", "docx", "jpg", "jpeg", "png"], key="resume_upload_affinda")
    resume_text, skills_list, resume_data = "", [], {}
    job_recommendations, best_job_roles = [], []
    if uploaded_file is not None:
        st.write("📄 **Uploaded File:**", uploaded_file.name)
        resume_text, skills_list, resume_data = affinda_parse_resume(uploaded_file)
        if not resume_text:
            uploaded_file.seek(0)
            resume_text, skills_list = fallback_parse_resume(uploaded_file)
            if not skills_list:
                skills_list = extract_skills_from_text(resume_text)
        st.write("🛠️ **Extracted Skills:**", ", ".join(skills_list) if skills_list else "None")
        if resume_text:
            st.write("📝 **Resume Summary:**")
            st.write(resume_text[:1000] + ("..." if len(resume_text) > 1000 else ""))
            job_recommendations, best_job_roles = predict_job_roles(skills_list)
            if len(job_recommendations) > 0 or len(best_job_roles) > 0:
                st.subheader("✨ Recommended Job Roles")
                for job in job_recommendations:
                    st.write(f"- {job}")
                if len(best_job_roles) > 0:
                    st.success(
                        f"🎯 **Best-Suited Job Role(s): {', '.join(best_job_roles)}**"
                    )
            else:
                st.warning("⚠️ No job recommendations could be generated. Try another resume or improve your skills section.")
        else:
            st.warning("⚠️ No text extracted from the resume. Please check the file and try again.")
        # Save state for use in job search tab
        st.session_state['parsed_resume'] = {
            'resume_text': resume_text,
            'skills_list': skills_list,
            'job_recommendations': job_recommendations,
            'best_job_roles': best_job_roles
        }
    st.write("---")
    st.write("Made with ❤️ for SmartSync!")

# ----------- TAB 2: JOB SCRAPING & APPLICATIONS (NO LOCATION) -----------
with tab2:
    st.subheader("Live Job Search & Application (Powered by Adzuna - India Only)")

    parsed_resume = st.session_state.get('parsed_resume', {})
    predicted_keyword = ""
    if parsed_resume.get('best_job_roles'):
        predicted_keyword = parsed_resume['best_job_roles'][0]
    elif parsed_resume.get('job_recommendations'):
        predicted_keyword = parsed_resume['job_recommendations'][0]
    else:
        predicted_keyword = "Developer"

    keyword = st.text_input("Job Role/Keyword", predicted_keyword, key="job_keyword")

    # Only fetch jobs if button is pressed
    if st.button("Fetch Jobs"):
        jobs = scrape_jobs_adzuna(keyword)
        st.session_state['latest_jobs'] = jobs
        st.session_state['selected_job_ids'] = []
    else:
        jobs = st.session_state.get('latest_jobs', [])

    selected_job_ids = st.session_state.get('selected_job_ids', [])

    if jobs:
        job_ids = [job["id"] for job in jobs]
        new_selected_job_ids = st.multiselect(
            "Select jobs to apply for",
            options=job_ids,
            default=selected_job_ids,
            format_func=lambda x: next((job["title"] + " at " + job["company"] for job in jobs if job["id"] == x), x),
            key="job_multiselect"
        )
        if new_selected_job_ids != selected_job_ids:
            st.session_state['selected_job_ids'] = new_selected_job_ids
            selected_job_ids = new_selected_job_ids

        if selected_job_ids and 'parsed_resume' in st.session_state:
            st.info("Click below to submit your application(s) (simulated).")
            if st.button("Submit Applications (Simulated)"):
                resume_text = st.session_state['parsed_resume'].get('resume_text', '')
                applications = submit_applications(jobs, selected_job_ids, resume_text)
                if 'applications' not in st.session_state:
                    st.session_state['applications'] = []
                existing_ids = {a['job_id'] for a in st.session_state['applications']}
                new_apps = [a for a in applications if a['job_id'] not in existing_ids]
                st.session_state['applications'].extend(new_apps)
                st.success("Applications submitted (simulated)!")
        else:
            st.warning("Select at least one job to apply.")
    else:
        st.info("No jobs fetched yet. Enter a keyword and click Fetch Jobs.")

# ----------- TAB 3: REAL-TIME APPLICATION DASHBOARD -----------
with tab3:
    st.title("Job Application Dashboard")
    df = pd.DataFrame(st.session_state.get('applications', []))

    if df.empty:
        st.info("No application data to display yet.")
    else:
        # --- FILTERS ---
        with st.container():
            col1, col2, col3 = st.columns(3)
            with col1:
                filter_status = st.selectbox(
                    "Filter by Status", ["All"] + sorted(df["status"].unique().tolist()), index=0
                )
            with col2:
                year_options = sorted(list(set([d[:4] for d in df['timestamp']])))
                filter_year = st.selectbox("Filter by Year", ["All"] + year_options, index=0)
            with col3:
                filter_company = st.selectbox(
                    "Filter by Company", ["All"] + sorted(df["company"].unique().tolist()), index=0
                )

            filtered_df = df.copy()
            if filter_status != "All":
                filtered_df = filtered_df[filtered_df["status"] == filter_status]
            if filter_year != "All":
                filtered_df = filtered_df[filtered_df["timestamp"].str.startswith(filter_year)]
            if filter_company != "All":
                filtered_df = filtered_df[filtered_df["company"] == filter_company]

        def kpi_card(color, value, label):
            return f"<div style='background:{color};padding:20px 0;border-radius:8px;text-align:center;color:white;'><h2>{value}</h2><p>{label}</p></div>"

        total_applied = df[df['status'].isin(['Applied', 'Interview', 'Offered', 'Rejected'])]['job_id'].nunique()
        total_interview = df[df['status'] == 'Interview']['job_id'].nunique()
        total_hired = df[df['status'] == 'Offered']['job_id'].nunique()
        total_rejected = df[df['status'] == 'Rejected']['job_id'].nunique()

        colA, colB, colC, colD = st.columns(4)
        colA.markdown(kpi_card("#1976d2", total_applied, "Total Roles Applied"), unsafe_allow_html=True)
        colB.markdown(kpi_card("#2e7d32", total_interview, "Roles in Interview"), unsafe_allow_html=True)
        colC.markdown(kpi_card("#ed6c02", total_hired, "Roles Hired"), unsafe_allow_html=True)
        colD.markdown(kpi_card("#c62828", total_rejected, "Roles Rejected"), unsafe_allow_html=True)

        st.write("### Application History")
        st.dataframe(filtered_df[["title", "company", "status", "timestamp", "job_url"]].rename(
            columns={
                "title": "Job Title",
                "company": "Company",
                "status": "Status",
                "timestamp": "Applied At",
                "job_url": "Job Link"
            }), use_container_width=True
        )

        st.write("### Trends & Distribution")
        colE, colF = st.columns([1,2])

        with colE:
            status_counts = df['status'].value_counts()
            fig1 = go.Figure(data=[go.Pie(labels=status_counts.index, values=status_counts.values, hole=.6)])
            fig1.update_traces(marker=dict(colors=["#1976d2", "#2e7d32", "#ed6c02", "#c62828"]))
            fig1.update_layout(showlegend=True, margin=dict(t=0, b=0, l=0, r=0), height=250)
            st.plotly_chart(fig1, use_container_width=True)

        with colF:
            df['date'] = pd.to_datetime(df['timestamp']).dt.date
            trend = df.groupby(['date', 'status'])['job_id'].count().unstack(fill_value=0)
            fig2 = go.Figure()
            for status in trend.columns:
                fig2.add_trace(go.Bar(x=trend.index, y=trend[status], name=status))
            fig2.update_layout(barmode='stack', height=250, margin=dict(t=20, b=0, l=0, r=0))
            st.plotly_chart(fig2, use_container_width=True)