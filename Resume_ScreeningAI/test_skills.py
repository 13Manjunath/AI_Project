from src.parser import extract_text
from src.skill_extractor import extract_skills

# Replace with your resume file name
resume_text = extract_text("data/resume/Manjunath_2025_Rolls.pdf")

skills = extract_skills(resume_text)

print("Skills found in resume:")
print(skills)

