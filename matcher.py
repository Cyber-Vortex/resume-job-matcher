from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

RESUME = """
Python programmer familiar with pandas, data cleaning, SQL, Git, basic machine learning,
model evaluation, and building small dashboards.
"""

JOB_DESCRIPTION = """
We are looking for a junior data analyst with Python, SQL, pandas, data visualization,
cleaning messy datasets, Git version control, and clear communication skills.
"""

def main():
    documents = [RESUME, JOB_DESCRIPTION]
    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform(documents)
    score = cosine_similarity(matrix[0:1], matrix[1:2])[0][0] * 100

    resume_terms = set(vectorizer.build_analyzer()(RESUME))
    job_terms = set(vectorizer.build_analyzer()(JOB_DESCRIPTION))
    shared = sorted(resume_terms & job_terms)
    missing = sorted(job_terms - resume_terms)

    print(f"Text similarity: {score:.1f}%")
    print("Shared terms:", ", ".join(shared) if shared else "None")
    print("Terms in job description not found in resume:", ", ".join(missing) if missing else "None")
    print("\nNote: similarity is not the probability of getting a job.")

if __name__ == "__main__":
    main()
