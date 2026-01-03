from src.preproces import clean_text

# Example predefined skill list (expand later)
SKILL_DB = [
    "python", "java", "c++", "aws", "docker", "kubernetes", "terraform",
    "linux", "windows", "azure", "react", "node.js", "sql", "html", "css"
]

def extract_skills(resume_text):
    """
    Extract skills from a resume using a simple rule-based matching
    """
    resume_text = clean_text(resume_text)
    resume_words = set(resume_text.split())

    found_skills = [skill for skill in SKILL_DB if skill in resume_words]

    return found_skills

