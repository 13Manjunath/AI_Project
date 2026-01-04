from src.parser import extract_text
from src.preproces import clean_text
# Make sure your resume file name matches exactly
text = extract_text("data/resume/Manjunath_2025_Rolls.pdf")
cleaned = clean_text(text)

print("Cleaned text preview:")
print(cleaned[:500])
