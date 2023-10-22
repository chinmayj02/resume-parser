from flask import Flask, render_template, jsonify,session, request, url_for, flash, redirect
import os
from extract import * 
from pyresparser import ResumeParser
import nltk
from nltk.corpus import words
from nltk.metrics.distance import edit_distance
nltk.download('words')

def filter_skills(skill_list):
    valid_words = set(words.words())
    filtered_skills = []

    for skill in skill_list:
        # Remove non-alphabetic characters and convert to lowercase for better matching
        cleaned_skill = ''.join(e for e in skill if e.isalnum()).lower()

        if any(edit_distance(cleaned_skill, word) <= 1 for word in valid_words):
            filtered_skills.append(skill)

    return filtered_skills

app = Flask(__name__)
app.secret_key=os.urandom(24)

education_keywords = ['Bsc', 'B. Pharmacy', 'B Pharmacy', 'Msc', 'M. Pharmacy', 'Ph.D', 'Bachelor','hssc', 'Master','b.e']


@app.route("/")
def index_page():
    return render_template('home.html', current_page='home')
@app.route("/upload/true",methods=['post'])
def resume_accept_page():
    request.files['resume'].save(os.path.join("uploads","resume.pdf"))
    resume_text=extract_text_from_pdf("uploads/resume.pdf")
    resume_data = ResumeParser('uploads/resume.pdf').get_extracted_data()

    # resume_vectorized = vectorizer.transform([resume_text])
    # skills_predicted = model.predict(resume_vectorized)
    # skills = [word for i, word in enumerate(resume_text.split()) if skills_predicted[0][i] == 1]

    name=extract_name(resume_text)
    contact=extract_contact_number_from_resume(resume_text)
    email=extract_email_from_resume(resume_text)
    # skills=resume_data['skills']
    skills=filter_skills(resume_data['skills'])
    education=extract_education_from_resume(resume_text, education_keywords)
    # education=resume_data['degree']
    print("\n##########################\n")
    print(resume_data)
    print("\n##########################\n")
    return render_template('output.html', current_page='output', name=name, contact=contact, email=email, skills=skills,education=education)

if __name__ == '__main__':
    # app.run(host='192.168.0.107', port=5000, debug=True)
    app.run(host='localhost',debug=True)
    # app.run(host='192.168.1.2', port=5000, debug=False)