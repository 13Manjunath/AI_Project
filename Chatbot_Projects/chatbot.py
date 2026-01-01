import json
import random
import re
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# -----------------------------
# 1. Load intents.json
# -----------------------------
with open("intents.json", "r") as f:
    data = json.load(f)

texts = []
labels = []
responses = {}

for intent in data["intents"]:
    tag = intent["tag"]
    responses[tag] = intent["responses"]
    for pattern in intent["patterns"]:
        texts.append(pattern.lower())
        labels.append(tag)

# -----------------------------
# 2. Train-test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.2, random_state=42
)

# -----------------------------
# 3. TF-IDF + SVM model
# -----------------------------
svm_model = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english")),
    ("clf", LinearSVC())
])

svm_model.fit(X_train, y_train)

# -----------------------------
# 4. Evaluate SVM
# -----------------------------
print("\nSVM Model Evaluation:\n")
print(classification_report(y_test, svm_model.predict(X_test)))

# -----------------------------
# 5. Load GPT-2 (DialoGPT-small)
# -----------------------------
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
gpt2_model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small")

# -----------------------------
# 6. Chatbot Response Function (Hybrid)
# -----------------------------
def chatbot_response(user_input, threshold=0.75):
    user_input = user_input.lower().strip()

    # -------------------------
    # 6a. Basic guards
    # -------------------------
    if not user_input:
        return "Please type a valid question."

    if user_input.isdigit():
        return "Please ask a question, not just a number."

    # -------------------------
    # 6b. Try SVM for known intents
    # -------------------------
    tfidf_vector = svm_model.named_steps["tfidf"].transform([user_input])
    if tfidf_vector.nnz != 0:  # has known words
        intent = svm_model.named_steps["clf"].predict(tfidf_vector)[0]
        # You can also use decision_function / probability if you want confidence check
        return random.choice(responses[intent])

    # -------------------------
    # 6c. Otherwise, use GPT-2 to generate response
    # -------------------------
    new_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors="pt")
    # Generate response
    chat_history_ids = gpt2_model.generate(
        new_input_ids,
        max_length=1000,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        top_p=0.92,
        top_k=50,
        no_repeat_ngram_size=3
    )
    response = tokenizer.decode(chat_history_ids[:, new_input_ids.shape[-1]:][0], skip_special_tokens=True)
    return response

# -----------------------------
# 7. Chat Loop
# -----------------------------
print("\n🤖 Hybrid Chatbot Ready! Type 'exit' to quit.\n")

while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        print("Chatbot: Goodbye 👋")
        break

    print("Chatbot:", chatbot_response(user_input))
