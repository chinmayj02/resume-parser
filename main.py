from flask import Flask, jsonify, request
from flask_cors import CORS
from tika import parser as tika_parser
import os
import spacy
import tempfile
from spacy.matcher import Matcher
import re
import requests
import shutil


app = Flask(__name__)
app.secret_key = os.urandom(24)
CORS(app)

def extract_information(text):
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(text)

    name = None
    email = None
    phone_number = None
    extracted_skills = []

    # Define regex pattern for email
    email_regex = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'

    # Find email addresses using regex
    emails = re.findall(email_regex, text)
    if emails:
        email = emails[0]

    phone_pattern = [{"ORTH": {"REGEX": "(\\d[- .\\(]?){9,}(\\d{1})"}}, {"ORTH": {"IN": [")", "-"]}, "OP": "?"}]
    matcher = Matcher(nlp.vocab)
    matcher.add("PHONE_NUMBER", [phone_pattern])

    for match_id, start, end in matcher(doc):
        phone_number = doc[start:end].text
        break

    for ent in doc.ents:
        if ent.label_ == "PERSON" and not name:
            name = ent.text
        elif ent.label_ == "PHONE_NUMBER" and not phone_number:
            phone_number = ent.text

    # skills
    api_url = "http://localhost:8080/jobportal/api/skills"
    try:
        # Send GET request to the API
        response = requests.get(api_url)
        # Check if the request was successful (status code 200)
        if response.status_code == 200:
            # Parse JSON response
            data = response.json()
            # Ensure data is a list
            if isinstance(data, list):
                # Iterate over the list and extract skills
                for item in data:
                    # Extract skill name from each item
                    skill_name = item.get("skillName")
                    # Check if skill name exists and search for it in the text
                    if skill_name and skill_name.lower() in text.lower():
                        extracted_skills.append(skill_name)
            else:
                print("Unexpected response format: expected list, got", type(data))
        else:
            print("Failed to fetch data from the API. Status code:", response.status_code)
    except requests.exceptions.RequestException as e:
        print("Error occurred while fetching data:", e)

    return {
        "Full Name": name,
        "Email": email,
        "Mobile Number": phone_number,
        "Skills":extracted_skills
    }

@app.route('/parse_resume', methods=['POST'])
def resu_parser():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file and file.filename.endswith('.pdf'):
        temp_dir = tempfile.mkdtemp()
        file_path = os.path.join(temp_dir, file.filename)
        file.save(file_path)
        print(file_path)
        parsed = tika_parser.from_file(file_path)
        text = parsed['content']
        result = extract_information(text)
        # Cleanup temporary files
        shutil.rmtree(temp_dir)  # This will remove the entire temporary directory and its contents
        return jsonify(result)
    else:
        return jsonify({'error': 'Unsupported file format'}), 400

if __name__ == '__main__':
    app.run(host='localhost', debug=True)
