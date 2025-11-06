import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
import numpy as np

def normalize_skills(skills):
    if pd.isna(skills):
        return ""
    skills_list = [skill.strip().lower() for skill in str(skills).split(',')]
    skills_list = sorted(set(skills_list))
    return ', '.join(skills_list)

def combine_resume_sections(row):
    sections = []
    for field in ['Skills', 'Projects', 'Experience', 'Certifications', 'Education', 'Summary', 'Achievements', 'Responsibilities', 'Interests']:
        val = row.get(field, "")
        if field == 'Skills':
            val = normalize_skills(val)
        if pd.notna(val) and str(val).strip():
            sections.append(str(val))
    return " ".join(sections)

def split_multilabels(label):
    if pd.isna(label):
        return []
    label = str(label)
    for sep in [',', ';']:
        if sep in label:
            return [l.strip() for l in label.split(sep) if l.strip()]
    return [label.strip()] if label.strip() else []

def train_model(base_csv="base_data.csv", model_path="model_full_data.pkl", top_n=3):
    df = pd.read_csv(base_csv)
    df['Combined'] = df.apply(combine_resume_sections, axis=1)
    X = df['Combined']
    y_raw = df['Job Role'].apply(split_multilabels)
    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(y_raw)
    pipeline = make_pipeline(
        TfidfVectorizer(stop_words='english', max_features=3500),
        OneVsRestClassifier(MultinomialNB())
    )
    pipeline.fit(X, y)
    joblib.dump((pipeline, mlb), model_path)
    print(f"✅ Model trained and saved as '{model_path}'")

    # --- Evaluation Section ---
    y_pred_prob = pipeline.predict_proba(X)
    y_pred = (y_pred_prob >= 0.5).astype(int)
    report = classification_report(y, y_pred, target_names=mlb.classes_, output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report).transpose().reset_index()
    report_df = report_df.rename(columns={'index': 'Job Role'})
    for col in ['precision', 'recall', 'f1-score']:
        if col in report_df.columns:
            report_df[col] = report_df[col].apply(lambda x: round(x, 2) if pd.api.types.is_number(x) else x)
    if 'support' in report_df.columns:
        report_df['support'] = report_df['support'].apply(lambda x: int(round(x)) if pd.api.types.is_number(x) else x)
    print("\n=== Classification Report ===")
    print(report_df[['Job Role', 'precision', 'recall', 'f1-score', 'support']].to_string(index=False))
    # --- End Evaluation Section ---

    return pipeline, mlb

def analyze_resume(resume_dict, model_path="model_full_data.pkl", top_n=3):
    pipeline, mlb = joblib.load(model_path)
    resume_text = combine_resume_sections(resume_dict)
    X_new = [resume_text]
    proba = pipeline.predict_proba(X_new)[0]
    top_indices = np.argsort(proba)[::-1][:top_n]
    top_roles = [mlb.classes_[i] for i in top_indices]
    top_scores = [proba[i] for i in top_indices]
    suitable_roles = [mlb.classes_[i] for i, p in enumerate(proba) if p >= 0.5]
    return {
        "top_n_roles": list(zip(top_roles, [round(s, 2) for s in top_scores])),
        "suitable_roles": suitable_roles
    }

if __name__ == "__main__":
    # Example usage: train on your dataset and show performance metrics
    train_model(base_csv="base_data.csv", model_path="model_full_data.pkl", top_n=3)
    # Example inference for a new resume (optional)
    test_resume = {
        "Skills": "Python, Machine Learning, Data Analysis, SQL",
        "Projects": "Built a machine learning model for sales prediction using scikit-learn.",
        "Experience": "Interned at Analytics Corp as a Data Analyst.",
        "Certifications": "Coursera ML Certificate",
        "Education": "B.Tech in Computer Science",
        "Summary": "Aspiring data scientist with strong analytical skills.",
        "Achievements": "Won hackathon for data-driven business solutions.",
        "Interests": "AI, Data Science, Cloud Computing"
    }
    result = analyze_resume(test_resume, model_path="model_full_data.pkl", top_n=3)
    print("\nTop-3 Predicted Roles (with scores):", result["top_n_roles"])
    print("Most Suitable Roles (probability >= 0.5):", result["suitable_roles"])