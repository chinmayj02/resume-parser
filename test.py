
import time
import os
import re
import json
import torch
import requests
import numpy as np
import pandas as pd
from flask import jsonify
import matplotlib.pyplot as plt
import gensim.downloader as api
from scipy.spatial import distance
from collections import defaultdict
from datetime import datetime, timezone
from sklearn.metrics import pairwise_distances
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# Model for extracting embeddings
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')


# get single candidate data
def get_candidate_data(sessionId):
    api_url = "http://localhost:8080/jobportal/profile?sessionId=" + sessionId
    try:
        # Send GET request to the API
        response = requests.get(api_url)
        # Check if the request was successful (status code 200)
        if response.status_code == 200:
            candidate_info = response.json()  # Assuming the response contains JSON data
            transformed_data = {
                'candidateId': candidate_info[1]['candidateId'],
                'gender': candidate_info[0]['gender'],
                # Calculate age based on date of birth
                'age': calculate_age(candidate_info[0]['dob']),
                'education': ', '.join(
                    [education['degreeName'] for education in candidate_info[1]['candidateEducations']]),
                'job_preferences': ', '.join(candidate_info[1]['jobPreferences']),
                'languages': ', '.join(candidate_info[1]['languages']),
                'skills': {skill['skillName']: (skill['proficiency'] or 1) for skill in candidate_info[1]['skills']},
                'previous_job_roles': {
                    experience['jobRole']: calculate_experience_years(experience['startDate'], experience['endDate'])
                    for experience in candidate_info[1]['candidateExperiences']}
            }
            if (transformed_data['gender'] == "M"):
                transformed_data["gender"] = "Male"
            elif (transformed_data["gender"] == "F"):
                transformed_data["gender"] = "Female"
            return transformed_data
        else:
            print("Failed to fetch data from the API. Status code:", response.status_code)
            return None
    except requests.exceptions.RequestException as e:
        print("Error occurred while fetching data:", e)


# Helper function to calculate years of experience
def calculate_experience_years(start_date, end_date):
    # Parse the start date string into a datetime object
    start_date_datetime = datetime.fromtimestamp(start_date / 1000, timezone.utc)
    start_date_year = start_date_datetime.year
    start_date_month = start_date_datetime.month

    # Parse the end date string into a datetime object
    end_date_datetime = datetime.fromtimestamp(end_date / 1000, timezone.utc)
    end_date_year = end_date_datetime.year
    end_date_month = end_date_datetime.month

    print(start_date_datetime)
    print(end_date_datetime)

    experience_years = end_date_year - start_date_year
    if end_date_month < start_date_month:
        experience_years -= 1  # Adjust for partial years

    return experience_years


# Helper function to calculate age from date of birth
def calculate_age(dob):
    # Convert the dob to a datetime object
    dob_datetime = datetime.fromtimestamp(dob / 1000, timezone.utc)
    # Get the current year
    current_year = datetime.now(timezone.utc).year
    # Get the year from the dob
    dob_year = dob_datetime.year
    # Calculate the age
    age = current_year - dob_year
    return age


# get single job data
def get_job_data(job_id):
    api_url = "http://localhost:8080/jobportal/api/job/detail/" + str(job_id)
    try:
        # Send GET request to the API
        response = requests.get(api_url)
        # Check if the request was successful (status code 200)
        if response.status_code == 200:
            job_details = response.json()
        else:
            print("Failed to fetch data from the API. Status code:", response.status_code)
            return None
    except requests.exceptions.RequestException as e:
        print("Error occurred while fetching data:", e)

    if (job_details['requiredGender'] == "M"):
        job_details["requiredGender"] = "Male"
    elif (job_details["requiredGender"] == "F"):
        job_details["requiredGender"] = "Female"
    else:
        job_details["requiredGender"] = "Male or Female"
    transformed_job = {
        "jobId": job_details["jobId"],
        "requiredExperienceMin": job_details["requiredExperienceMin"],
        "requiredExperienceMax": job_details["requiredExperienceMax"],
        "requiredAgeMin": job_details["requiredAgeMin"],
        "requiredAgeMax": job_details["requiredAgeMax"],
        "locations": [location["cityName"] for location in job_details["locations"]],
        "skills": [{"skillName": skill["skillName"], "proficiency": skill["proficiency"]} for skill in
                   job_details["skills"]],
        "languages": [language["language"] for language in job_details["languages"]],
        "preferences": [preference["preference"] for preference in job_details["preferences"]],
        "requiredHighestEducation": job_details["requiredHighestEducation"],
        "requiredGender": job_details["requiredGender"]
    }
    print(transformed_job)

    return transformed_job


# get job list
def get_jobs_list(gender, age):
    api_url = "http://localhost:8080/jobportal/job-list?categoryId=&isActive=1&isRecent=1&pageSize=-1"
    try:
        # Send GET request to the API
        response = requests.get(api_url)
        # Check if the request was successful (status code 200)
        if response.status_code == 200:
            jobs_data = response.json()
        else:
            print("Failed to fetch data from the API. Status code:", response.status_code)
            return None
    except requests.exceptions.RequestException as e:
        print("Error occurred while fetching data:", e)

    transformed_data = []

    for job in jobs_data:
        if (job['requiredGender'] == "M"):
            job["requiredGender"] = "Male"
        elif (job["requiredGender"] == "F"):
            job["requiredGender"] = "Female"
        else:
            job["requiredGender"] = "Male or Female"
        if gender == "Male or Female" or job['requiredGender'] == gender or job['requiredGender'] == "Male or Female":
            if age >= job["requiredAgeMin"]:
                transformed_job = {
                    "jobId": job["jobId"],
                    "requiredExperienceMin": job["requiredExperienceMin"],
                    "requiredExperienceMax": job["requiredExperienceMax"],
                    "requiredAgeMin": job["requiredAgeMin"],
                    "requiredAgeMax": job["requiredAgeMax"],
                    "locations": [location["cityName"] for location in job["locations"]],
                    "skills": [{"skillName": skill["skillName"], "proficiency": skill["proficiency"]} for skill in
                               job["skills"]],
                    "languages": [language["language"] for language in job["languages"]],
                    "preferences": [preference["preference"] for preference in job["preferences"]],
                    "requiredHighestEducation": job["requiredHighestEducation"],
                    "requiredGender": job["requiredGender"]
                }
                transformed_data.append(transformed_job)

    if not transformed_data:
        return None
    else:
        return transformed_data


# get candidate list
def get_candidate_list():
    api_url = "http://localhost:8080/jobportal/api/candidates"
    try:
        # Send GET request to the API
        response = requests.get(api_url)
        # Check if the request was successful (status code 200)
        if response.status_code == 200:
            candidate_list = response.json()
        else:
            print("Failed to fetch data from the API. Status code:", response.status_code)
            return None
    except requests.exceptions.RequestException as e:
        print("Error occurred while fetching data:", e)
        return None

    # Debugging: Print the structure of candidate_list
    print("Candidate List Structure:", candidate_list)

    # Create the list to store candidate data
    candidates_data = []

    for candidate_info in candidate_list:
        # Extracting candidate information
        user_info = candidate_info[0]
        candidate_details = candidate_info[1]

        candidate_id = user_info['userId']
        gender = "Male" if user_info['gender'] == "M" else "Female"
        dob = datetime.fromtimestamp(user_info['dob'] / 1000)
        age = datetime.now().year - dob.year

        # Extracting education information
        if candidate_details['candidateEducations']:
            education_info = candidate_details['candidateEducations'][0]
            degree_name = education_info['degreeName']
            education = degree_name
        else:
            education = "No education information"

        # Extracting job preferences and languages
        job_preferences = ', '.join(candidate_details['jobPreferences'])
        languages = ', '.join(candidate_details['languages'])

        # Extracting skills
        skills = {}
        for skill in candidate_details['skills']:
            if skill['proficiency']:
                skills[skill['skillName']] = int(skill['proficiency'])

        # Extracting previous job roles and calculating experience
        previous_job_roles = {}
        for experience in candidate_details['candidateExperiences']:
            job_role = experience['jobRole']
            start_date = datetime.fromtimestamp(experience['startDate'] / 1000)
            end_date = datetime.fromtimestamp(experience['endDate'] / 1000)
            experience_years = (end_date - start_date).days // 365
            if job_role in previous_job_roles:
                previous_job_roles[job_role] += experience_years
            else:
                previous_job_roles[job_role] = experience_years

        # Construct candidate data dictionary
        candidate_data = {
            'candidateId': candidate_id,
            'gender': gender,
            'age': age,
            'education': education,
            'job_preferences': job_preferences,
            'languages': languages,
            'skills': skills,
            'previous_job_roles': previous_job_roles
        }

        # Add to the candidates_data list
        candidates_data.append(candidate_data)

        # Debugging print statement
        print(candidate_data)

    return candidates_data


# Function to preprocess text : Converting to lowercase, removing punctuation marks
def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    return text


# Function to generate BERT Embeddings
def generate_bert_embeddings(text, tokenizer):
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings.numpy()


# Preprocess user data
def preprocess_user_data(user_data):

    print("User Details :")
    print(user_data)
    # Convert education, job preferences, and languages to lowercase and remove punctuation
    if isinstance(user_data['education'], list):
        user_data['education'] = [preprocess_text(edu) for edu in user_data['education']]
    else:
        user_data['education'] = preprocess_text(user_data['education'])

    if isinstance(user_data['job_preferences'], list):
        user_data['job_preferences'] = [preprocess_text(pref) for pref in user_data['job_preferences']]
    else:
        user_data['job_preferences'] = preprocess_text(user_data['job_preferences'])

    if isinstance(user_data['languages'], list):
        user_data['languages'] = [preprocess_text(lang) for lang in user_data['languages']]
    else:
        user_data['languages'] = preprocess_text(user_data['languages'])

    # Convert skills to lowercase and remove punctuation
    user_data['skills'] = {preprocess_text(skill): proficiency for skill, proficiency in user_data['skills'].items()}

    return user_data


# Function to preprocess jobs_data
def preprocess_job_data(job_data):
    if isinstance(job_data, dict):  # If job_data is a single dictionary, wrap it in a list
        job_data = [job_data]
    for job in job_data:
        # Convert proficiency to integer
        for skill in job['skills']:
            skill['skillName'] = preprocess_text(skill['skillName'])
            skill['proficiency'] = int(skill['proficiency'])

        # Convert age and experience to integer, handle empty cases
        job['requiredExperienceMin'] = int(job['requiredExperienceMin']) if job['requiredExperienceMin'] else None
        job['requiredExperienceMax'] = int(job['requiredExperienceMax']) if job['requiredExperienceMax'] else None
        job['requiredAgeMin'] = int(job['requiredAgeMin']) if job['requiredAgeMin'] else None
        job['requiredAgeMax'] = int(job['requiredAgeMax']) if job['requiredAgeMax'] else None

        # Preprocess locations, languages, and preferences, handle empty cases
        job['locations'] = [preprocess_text(location) for location in job['locations']] if job['locations'] else []
        job['languages'] = [preprocess_text(language) for language in job['languages']] if job['languages'] else []
        job['preferences'] = [preprocess_text(preference) for preference in job['preferences']] if job[
            'preferences'] else []

        # Preprocess highest education and gender, handle empty cases
        job['requiredHighestEducation'] = preprocess_text(job['requiredHighestEducation']) if job[
            'requiredHighestEducation'] else None
        job['requiredGender'] = preprocess_text(job['requiredGender']) if job['requiredGender'] else None

    return job_data


# Define functions to calculate similarity metrics
def cosine_similarity_score(candidate_vector, job_vectors):
    return cosine_similarity(candidate_vector.reshape(1, -1), job_vectors.reshape(len(job_vectors), -1))


def calculate_experience_match(candidate_previous_job_roles, required_experience_min, required_experience_max):
    # Calculate the total experience by summing up the durations of all previous job roles
    total_experience = sum(candidate_previous_job_roles.values())

    # Calculate the average experience by dividing the total experience by the number of job roles
    average_experience = total_experience / max(len(candidate_previous_job_roles),1)

    # Normalize the required experience range by dividing it by a constant (e.g., 10)
    normalized_required_experience_min = required_experience_min / 10  # Assuming 10 years as max experience
    normalized_required_experience_max = required_experience_max / 10

    # Calculate the match score based on the average experience and the normalized required experience range
    if average_experience < normalized_required_experience_min:
        return 0.0  # If the candidate's average experience is below the minimum required, return 0.0
    elif average_experience > normalized_required_experience_max:
        return 1.0  # If the candidate's average experience is above the maximum required, return 1.0
    else:
        # Otherwise, calculate the match score based on the range
        return (average_experience - normalized_required_experience_min) / (
                normalized_required_experience_max - normalized_required_experience_min)


def job_recommendation(sessionId):
    # Step 1: Extract & preprocess user data
    #--------------TESTING DATA---------------
    candidate_info = {'candidateId': 4,
                      'gender': 'Male',
                      'age': 39,
                      'education': 'graduate',
                      'job_preferences': 'contract', 'languages': 'english',
                      'skills': {'python': '9',
                                 'java': '5',
                                 'Scripting languages': '3',
                                 'Test automation tools': '4',
                                 'HTML':'9',
                                 'javascript':'8',
                                 'css':'10',
                                 'java ee':'10'
                                 },
                      'previous_job_roles': {'Software development trainee':4}}
    # Check if candidate has no skills
    if not candidate_info['skills']:
        return -1  # No jobs to be recommended

    preprocessed_candidate_info = preprocess_user_data(candidate_info)
    print("Pre processed canddate")
    print(preprocessed_candidate_info)
    # Step 2: Extract & preprocess jobs data
    jobs_data = jobs_data = [
        {'jobId': 3, 'requiredExperienceMin': 1, 'requiredExperienceMax': 10, 'requiredAgeMin': 21, 'requiredAgeMax': 40, 'locations': ['Banglore'], 'skills': [{'skillName': 'Data Analysis', 'proficiency': '7'}, {'skillName': 'Network performance analysis', 'proficiency': '5'}, {'skillName': 'Network monitoring tools', 'proficiency': '6'}, {'skillName': 'Troubleshooting', 'proficiency': '9'}, {'skillName': 'Capacity planning', 'proficiency': '6'}, {'skillName': 'Network protocols', 'proficiency': '5'}], 'languages': ['English'], 'preferences': ['Full-Time'], 'requiredHighestEducation': 'Graduate', 'requiredGender': 'Male'},
        {'jobId': 4, 'requiredExperienceMin': 3, 'requiredExperienceMax': 14, 'requiredAgeMin': 21, 'requiredAgeMax': 40, 'locations': [], 'skills': [{'skillName': 'Performance testing', 'proficiency': '2'}, {'skillName': 'Load testing', 'proficiency': '3'}, {'skillName': 'Stress testing', 'proficiency': '2'}, {'skillName': 'Test scenarios', 'proficiency': '8'}, {'skillName': 'Performance monitoring', 'proficiency': '6'}, {'skillName': 'Benchmarking', 'proficiency': '4'}, {'skillName': 'Performance analysis', 'proficiency': '5'}], 'languages': [], 'preferences': ['Contract'], 'requiredHighestEducation': 'Graduate', 'requiredGender': 'Male or Female'},
        {'jobId': 7, 'requiredExperienceMin': 3, 'requiredExperienceMax': 12, 'requiredAgeMin': 21, 'requiredAgeMax': 40, 'locations': [], 'skills': [{'skillName': 'Backend development', 'proficiency': '6'}, {'skillName': 'RESTful APIs', 'proficiency': '9'}, {'skillName': 'Database integration', 'proficiency': '6'}, {'skillName': 'Performance optimization', 'proficiency': '5'}, {'skillName': 'Version control', 'proficiency': '2'}, {'skillName': 'JAVA EE', 'proficiency': '7'}], 'languages': [], 'preferences': ['Intern'], 'requiredHighestEducation': 'Graduate', 'requiredGender': 'Male'},
        {'jobId': 8, 'requiredExperienceMin': 4, 'requiredExperienceMax': 15, 'requiredAgeMin': 21, 'requiredAgeMax': 40, 'locations': [], 'skills': [{'skillName': 'HTML', 'proficiency': '6'}, {'skillName': 'Javascript', 'proficiency': '4'}, {'skillName': 'CSS', 'proficiency': '5'}, {'skillName': 'Responsive design', 'proficiency': '6'}, {'skillName': 'Front-end web development', 'proficiency': '3'}, {'skillName': 'Web performance optimization', 'proficiency': '2'}, {'skillName': 'Cross-browser compatibility', 'proficiency': '8'}], 'languages': [], 'preferences': ['Full-Time'], 'requiredHighestEducation': 'Graduate', 'requiredGender': 'Male'},
        {'jobId': 9, 'requiredExperienceMin': 3, 'requiredExperienceMax': 15, 'requiredAgeMin': 21, 'requiredAgeMax': 40, 'locations': [], 'skills': [{'skillName': 'UI design principles and best practices', 'proficiency': '9'}, {'skillName': 'Graphic design tools', 'proficiency': '6'}, {'skillName': 'Typography and color theory', 'proficiency': '5'}, {'skillName': 'Visual design and layout', 'proficiency': '7'}, {'skillName': 'Responsive design', 'proficiency': '2'}], 'languages': [], 'preferences': ['Full-Time'], 'requiredHighestEducation': 'Post Graduate', 'requiredGender': 'Male'}, {'jobId': 10, 'requiredExperienceMin': 4, 'requiredExperienceMax': 12, 'requiredAgeMin': 21, 'requiredAgeMax': 40, 'locations': [], 'skills': [{'skillName': 'User-centered design principles', 'proficiency': '3'}, {'skillName': 'UX/UI design tools', 'proficiency': '2'}, {'skillName': 'Wireframing and prototyping', 'proficiency': '8'}, {'skillName': 'Usability testing and user research', 'proficiency': '6'}, {'skillName': 'Information architecture and user flows', 'proficiency': '4'}], 'languages': [], 'preferences': ['Full-Time'], 'requiredHighestEducation': 'Graduate', 'requiredGender': 'Male or Female'},
        {'jobId': 11, 'requiredExperienceMin': 3, 'requiredExperienceMax': 13, 'requiredAgeMin': 21, 'requiredAgeMax': 40, 'locations': [], 'skills': [{'skillName': 'HTML', 'proficiency': '6'}, {'skillName': 'Javascript', 'proficiency': '4'}, {'skillName': 'CSS', 'proficiency': '5'}, {'skillName': 'Responsive design', 'proficiency': '6'}, {'skillName': 'User interface (UI) design', 'proficiency': '5'}, {'skillName': 'User experience (UX) design', 'proficiency': '6'}, {'skillName': 'Web design principles', 'proficiency': '9'}, {'skillName': 'Prototyping and wireframing', 'proficiency': '6'}, {'skillName': 'Front-end development', 'proficiency': '5'}, {'skillName': 'Interaction design', 'proficiency': '7'}, {'skillName': 'User testing', 'proficiency': '2'}, {'skillName': 'Usability testing', 'proficiency': '3'}, {'skillName': 'Collaboration', 'proficiency': '2'}, {'skillName': 'Attention to detail', 'proficiency': '8'}], 'languages': [], 'preferences': ['Contract'], 'requiredHighestEducation': 'Post Graduate', 'requiredGender': 'Male or Female'},
        {'jobId': 5, 'requiredExperienceMin': 4, 'requiredExperienceMax': 14, 'requiredAgeMin': 21, 'requiredAgeMax': 40, 'locations': [], 'skills': [{'skillName': 'Security consulting', 'proficiency': '6'}, {'skillName': 'Risk assessment', 'proficiency': '9'}, {'skillName': 'Security audits', 'proficiency': '6'}, {'skillName': 'Security policy', 'proficiency': '5'}, {'skillName': 'development', 'proficiency': '7'}, {'skillName': 'Penetration testing', 'proficiency': '2'}, {'skillName': 'Client communication', 'proficiency': '3'}], 'languages': [], 'preferences': ['Contract'], 'requiredHighestEducation': 'Post Graduate', 'requiredGender': 'Male or Female'},
        {'jobId': 13, 'requiredExperienceMin': 4, 'requiredExperienceMax': 9, 'requiredAgeMin': 21, 'requiredAgeMax': 40, 'locations': [], 'skills': [{'skillName': 'Data integration', 'proficiency': '2'}, {'skillName': 'ETL (Extract, Transform, Load)', 'proficiency': '8'}, {'skillName': 'Big data technologies', 'proficiency': '6'}, {'skillName': 'Database management', 'proficiency': '4'}, {'skillName': 'Data warehousing', 'proficiency': '5'}], 'languages': [], 'preferences': ['Temporary'], 'requiredHighestEducation': 'Graduate', 'requiredGender': 'Male'},
        {'jobId': 14, 'requiredExperienceMin': 3, 'requiredExperienceMax': 11, 'requiredAgeMin': 21, 'requiredAgeMax': 40, 'locations': [], 'skills': [{'skillName': 'Troubleshooting', 'proficiency': '7'}, {'skillName': 'Embedded systems', 'proficiency': '6'}, {'skillName': 'Electronics design', 'proficiency': '9'}, {'skillName': 'PCB layout', 'proficiency': '6'}, {'skillName': 'Circuit analysis', 'proficiency': '5'}], 'languages': [], 'preferences': ['Contract'], 'requiredHighestEducation': 'Graduate', 'requiredGender': 'Male'},
        {'jobId': 15, 'requiredExperienceMin': 5, 'requiredExperienceMax': 8, 'requiredAgeMin': 21, 'requiredAgeMax': 40, 'locations': [], 'skills': [{'skillName': 'Python', 'proficiency': '5'}, {'skillName': 'Java', 'proficiency': '4'}, {'skillName': 'Scripting languages', 'proficiency': '2'}, {'skillName': 'Test automation tools', 'proficiency': '3'}, {'skillName': 'Test framework development', 'proficiency': '2'}, {'skillName': 'Continuous integration tools', 'proficiency': '8'}, {'skillName': 'Test data management', 'proficiency': '6'}, {'skillName': 'Selenium', 'proficiency': '6'}], 'languages': ['english','hindi', ], 'preferences': ['Contract'], 'requiredHighestEducation': 'Masters of Engineering', 'requiredGender': 'Male'},
        {'jobId': 16, 'requiredExperienceMin': 1, 'requiredExperienceMax': 13, 'requiredAgeMin': 21, 'requiredAgeMax': 40, 'locations': [], 'skills': [{'skillName': 'HTML', 'proficiency': '9'}, {'skillName': 'Javascript', 'proficiency': '6'}, {'skillName': 'CSS', 'proficiency': '5'}, {'skillName': 'Responsive design', 'proficiency': '7'}, {'skillName': 'Frontend frameworks', 'proficiency': '2'}, {'skillName': 'React JS', 'proficiency': '3'}, {'skillName': 'Angular JS', 'proficiency': '2'}], 'languages': [], 'preferences': ['Contract'], 'requiredHighestEducation': 'Graduate', 'requiredGender': 'Male or Female'}
    ]
    if jobs_data is None:
        return jsonify({'error': 'No Jobs'}), 204

    print("Jobs data:")
    print(jobs_data)

    preprocessed_jobs_data = preprocess_job_data(jobs_data)
    print("Pre Processed Jobs data:")
    print(preprocessed_jobs_data)

    # Step 5: Calculate similarity metric
    recommended_jobs = []
    for i in range(len(preprocessed_jobs_data)):
        job_info = preprocessed_jobs_data[i]

        # Calculate skill similarity
        # Extract the skill names from the job_info dictionary.
        # If the 'skills' key is not present or is empty, job_skills will be an empty list.
        job_skills = [skill['skillName'] for skill in job_info.get('skills', [])]

        # Create a dictionary mapping each skill name to its required proficiency level for the job.
        # If the 'skills' key is not present or is empty, job_proficiencies will be an empty dictionary.
        job_proficiencies = {skill['skillName']: int(skill['proficiency']) for skill in job_info.get('skills', [])}

        # Count the number of skills in the candidate's skill set that meet or
        # exceed the required proficiency level for the corresponding skill in the job.
        match_count = sum(int(preprocessed_candidate_info['skills'].get(skill, '0')) >=
                          job_proficiencies.get(skill, 0) for skill in job_skills) # +-1 range

        # Calculate the skill similarity as the ratio of matched skills to the total number of job skills.
        # Ensure the denominator is at least 1 to avoid division by zero.
        skill_similarity = match_count / max(len(job_skills), 1)

        if skill_similarity <= 0:
            continue

        # Calculate preference similarity
        # Extract the candidate's job preferences & the job's preferences from  candidate_info & job_info dictionaries.
        # If either of the preferences is not provided, set them to empty strings.
        candidate_preferences = candidate_info.get('job_preferences', '')
        job_preferences = ''.join(job_info.get('preferences', ''))

        # Check if either the candidate's preferences or the job's preferences are empty.
        # If either is empty, set preference_similarity to 0.0, indicating no similarity.
        if not candidate_preferences or not job_preferences:
            preference_similarity = 0.0
        else:
            # Generate BERT embeddings for the candidate's preferences and the job's preferences.
            candidate_preference_embedding = generate_bert_embeddings(candidate_preferences, tokenizer)
            job_preference_embedding = generate_bert_embeddings(job_preferences, tokenizer)

            # Calculate the cosine similarity between the embeddings of the candidate's preferences & job's preferences.
            preference_similarity = cosine_similarity_score(candidate_preference_embedding, job_preference_embedding)


        # Calculate education similarity
        # Extract the required highest education level for the job from the 'requiredHighestEducation' key in job_info.
        # If the key is not present, set job_education to an empty string.
        job_education = job_info.get('requiredHighestEducation', '')

        # Extract the education level of the candidate from the 'education' key in candidate_info.
        # If the key is not present, set candidate_education to an empty string.
        candidate_education = candidate_info.get('education', '')

        # Check if either the job's required education or the candidate's education is empty.
        # If either is empty, set education_similarity to 0.0, indicating no similarity.
        if not job_education or not candidate_education:
            education_similarity = 0.0
        else:
            # Generate BERT embeddings for the job's required education and the candidate's education.
            education_embedding = generate_bert_embeddings(job_education, tokenizer)
            candidate_education_embedding = generate_bert_embeddings(candidate_education, tokenizer)

            # Calculate the cosine similarity between the embeddings of job's required education &candidate's education.
            # This measures how similar the textual representations of the education levels are.
            education_similarity = cosine_similarity_score(candidate_education_embedding, education_embedding)


        # Calculate language similarity
        # Extract the languages required for the job and the languages known by the candidate from the 'languages' keys
        # in job_info and candidate_info dictionaries, respectively. If these keys are not present,
        # set job_languages and candidate_languages to empty lists.
        job_languages = job_info.get('languages', [])
        candidate_languages = candidate_info.get('languages', [])

        # Check if either the job's required languages or the candidate's languages are empty.
        # If either is empty, set language_similarity to 0.0, indicating no similarity.
        if not job_languages or not candidate_languages:
            language_similarity = 0.0
        else:
            # Concatenate the job's required languages into a single text string.
            # This is done to create a single embedding for all job languages.
            language_texts = [''.join(job_languages)]

            # Generate BERT embeddings for the concatenated job languages text and the candidate's languages.
            language_embedding = generate_bert_embeddings(language_texts[0], tokenizer)
            candidate_language_embedding = generate_bert_embeddings(candidate_languages, tokenizer)

            # Calculate the cosine similarity between the embeddings of the job languages and the candidate languages.
            language_similarity = cosine_similarity_score(candidate_language_embedding, language_embedding)


        # Calculate experience similarity
        if not candidate_info['previous_job_roles']:
            experience_similarity = 0.0
        experience_similarity = calculate_experience_match(candidate_info['previous_job_roles'],
                                                           job_info['requiredExperienceMin'],
                                                           job_info['requiredExperienceMax'])

        # Calculate overall similarity score with equal weightage for each attribute
        skill_weight = 0.40
        preference_weight = 0.20
        experience_weight = 0.10
        education_weight = 0.20
        language_weight = 0.10

        overall_similarity = float(skill_similarity * skill_weight + preference_similarity * preference_weight +
                                   education_similarity * education_weight + language_similarity * language_weight +
                                   experience_similarity * experience_weight) * 100  # Convert to percentage

        # Append job data along with similarity scores to the recommended jobs list
        recommended_jobs.append({
            "jobId": job_info['jobId'],
            "skillSimilarity": float(skill_similarity),
            "preferenceSimilarity": float(preference_similarity),
            "educationSimilarity": float(education_similarity),
            "languageSimilarity": float(language_similarity),
            "experienceMatch": float(experience_similarity),
            "overallSimilarity": float(overall_similarity)
        })
    # Sort recommended jobs by overall similarity score in descending order
    sorted_jobs = sorted(recommended_jobs, key=lambda x: x['overallSimilarity'], reverse=True)
    filtered_jobs = [job for job in sorted_jobs if job['overallSimilarity'] > 45]
    # print("Recommended Jobs:")
    # print(sorted_jobs)
    if not filtered_jobs:
        print("No jobs are there for you")
        return ""


    df = pd.DataFrame(filtered_jobs)
    sorted_df = df.sort_values(by='overallSimilarity', ascending=False)

    # Set pandas display options to show all rows and columns
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)

    # Print the sorted DataFrame
    print("Recommended Jobs:")
    print(sorted_df)

    return json.dumps(sorted_jobs)

job_recommendation("sessionId")

# Method returns list of recommended candidate
# def candidate_recommendation(job_id):
#     job_info = get_job_data(job_id)
#     if not job_info:
#         return jsonify({'error': 'Unsupported format'}), 400
#     candidate_embeddings = []
#     candidates_data = get_candidate_list(job_info["required_gender"], job_info["required_age_min"])
#     print(candidates_data)
#
#     if (candidates_data == None):
#         return jsonify({'error': 'No Candidates'}), 204
#     # job_info_text = ' '.join(map(str, job_info))
#     job_info_text = " ".join([str(val) for val in job_info.values()])
#     job_embedding = get_bert_embeddings(job_info_text)
#
#     for candidate_info in zip(candidates_data['gender'],
#                               candidates_data['age'],
#                               candidates_data['education'],
#                               candidates_data['job_preferences'],
#                               candidates_data['languages'],
#                               candidates_data['skills'],
#                               candidates_data['previous_job_roles']):
#         candidate_info_text = ' '.join(map(str, candidate_info))
#         candidate_embeddings.append(generate_bert_embeddings(candidate_info_text, tokenizer))
#     candidate_embeddings = np.array(candidate_embeddings)
#
#     # this line of code computes the cosine similarity between a single job embedding and multiple candidate embeddings, resulting in a similarity score for each candidate.
#     similarity_scores = cosine_similarity(job_embedding.reshape(1, -1),
#                                           candidate_embeddings.reshape(len(candidate_embeddings), -1))
#     ranked_candidates_indices = np.argsort(similarity_scores[0])[::-1]  # Descending order
#     ranked_candidates = [(candidates_data['candidate_id'][idx], similarity_scores[0][idx]) for idx in
#                          ranked_candidates_indices]
#
#     return log_and_return(ranked_candidates, job_id, "Job", "Candidate")


# ----------------------------------------------------------------------
# Define functions to calculate Jaccard Similarity
# def jaccard_similarity_score(candidate_vector, job_vectors):
#     intersection = np.sum(np.minimum(candidate_vector.reshape(1, -1), job_vectors.reshape(len(job_vectors), -1)),
#                           axis=1)
#     union = np.sum(np.maximum(candidate_vector.reshape(1, -1), job_vectors.reshape(len(job_vectors), -1)), axis=1)
#     return intersection / union
#
#
# # Define functions to calculate Euclidean Distance
# def euclidean_distance_score(candidate_vector, job_vectors):
#     return np.linalg.norm(candidate_vector.reshape(1, -1) - job_vectors.reshape(len(job_vectors), -1), axis=1)
#
#
# # Define functions to calculate Manhattan Distance
# def manhattan_distance_score(candidate_vector, job_vectors):
#     return np.sum(np.abs(candidate_vector.reshape(1, -1) - job_vectors.reshape(len(job_vectors), -1)), axis=1)
# def job_recommendation(sessionId):
#     # Step 1: Extract & preprocess user data
#     candidate_info = get_candidate_data(sessionId)
#     if candidate_info is None:
#         return []  # Return an empty list if user data is not available
#
#     print("Candidate Data: ")
#     print(candidate_info)
#
#     if(candidate_info['skills']) is None:
#         return -1
#
#     preprocessed_candidate_info = preprocess_user_data(candidate_info)
#     print("Preprocessed user Data: ")
#     print(preprocessed_candidate_info)
#
#     # Step 2: Extract & preprocess jobs data
#     jobs_data = get_jobs_list(candidate_info["gender"], candidate_info["age"])
#     if jobs_data is None:
#         return jsonify({'error': 'No Jobs'}), 204
#
#     print("Jobs data :")
#     print(jobs_data)
#
#     preprocessed_jobs_data = preprocess_job_data(jobs_data)
#     print("Pre Processed Jobs data :")
#     print(preprocessed_jobs_data)
#
#     candidate_info_order = ['skills', 'previous_job_roles', 'education', 'job_preferences', 'languages', 'gender',
#                             'age']
#     candidate_info_values = [candidate_info[key] for key in candidate_info_order]
#     candidate_info_text = "".join([str(val) for val in candidate_info_values])
#     candidate_embedding = generate_bert_embeddings(candidate_info_text, tokenizer)
#
#     # Step 4: Generate BERT embeddings for jobs
#     job_embeddings = []
#     for job_info in jobs_data:
#         job_info_values = [job_info.get('skills', []), job_info.get('preferences', []),
#                            job_info.get('requiredHighestEducation', ''), job_info.get('requiredGender', ''),
#                            job_info.get('languages', []), job_info.get('requiredAgeMin', 0),
#                            job_info.get('requiredAgeMax', 100), job_info.get('requiredJobRole', ''),
#                            job_info.get('requiredExperienceMin', 0), job_info.get('requiredExperienceMax', 100)]
#         job_info_text = "".join([str(val) for val in job_info_values])
#         job_embeddings.append(generate_bert_embeddings(job_info_text, tokenizer))
#     job_embeddings = np.array(job_embeddings)
#     print("Jobs embeddings Data:")
#     print(job_embeddings)
#
#     # Calculate similarity metric
#     # Calculate cosine similarity score
#     cosine_similarities = cosine_similarity_score(candidate_embedding, job_embeddings)
#     print("Cosine Similarity: ")
#     print(cosine_similarities)
#     print(len(cosine_similarities))
#
#     recommended_jobs = []
#     for i in range(len(preprocessed_jobs_data)):
#         job_info = preprocessed_jobs_data[i]
#
#         # Calculate skill similarity
#         job_skills = [skill['skillName'] for skill in job_info.get('skills', [])]
#         job_proficiencies = {skill['skillName']: int(skill['proficiency']) for skill in job_info.get('skills', [])}
#         match_count = sum(
#             int(candidate_info['skills'].get(skill, '0')) >= job_proficiencies.get(skill, 0) for skill in job_skills)
#         skill_similarity = match_count / max(len(job_skills), 1)
#
#         # Calculate preference similarity
#         job_preference_texts = [job_info['preferences']]
#         job_preference_embedding = generate_bert_embeddings(job_info['preferences'], tokenizer)
#         preference_similarity = cosine_similarity_score(candidate_embedding, job_preference_embedding)
#
#         # Calculate education similarity
#         education_text = job_info['requiredHighestEducation']
#         education_embedding = generate_bert_embeddings(education_text, tokenizer)
#         education_similarity = cosine_similarity_score(candidate_embedding, education_embedding)
#
#         # Calculate languages similarity
#         language_texts = [', '.join(job_info.get('languages', []))]
#         language_embedding = generate_bert_embeddings(language_texts[0], tokenizer)
#         language_similarity = cosine_similarity_score(candidate_embedding, language_embedding)
#
#         # Calculate experience similarity
#         experience_similarity = calculate_experience_match(candidate_info['previous_job_roles'],job_info['requiredExperienceMin'],job_info['requiredExperienceMax'])
#
#         # Calculate overall similarity score (you can adjust weights if needed)
#         overall_similarity = (skill_similarity + preference_similarity + education_similarity + language_similarity + experience_similarity) / 5
#
#         # Append job data along with similarity scores to the recommended jobs list
#         recommended_jobs.append({
#             "jobId": job_info['jobId'],
#             "skillSimilarity": skill_similarity,
#             "preferenceSimilarity": preference_similarity,
#             "educationSimilarity": education_similarity,
#             "languageSimilarity": language_similarity,
#             "experienceMatch": experience_similarity,
#             "overallSimilarity": overall_similarity
#         })
#     print(recommended_jobs)
#     normalised_jobs = normalize_scores(recommended_jobs)
#     recommended_jobs = sorted(normalised_jobs, key=lambda x: x['recommendedScore'], reverse=True)
#     # Sort recommended jobs by recommended score
#
#     # # Calculate Jaccard similarity scores
#     # jaccard_similarities = jaccard_similarity_score(candidate_embedding, job_embeddings)
#     # print("Jaccard Similarity: ")
#     # print(jaccard_similarities)
#     # print(len(jaccard_similarities))
#     #
#     # # Calculate Euclidean distance scores
#     # euclidean_distances = euclidean_distance_score(candidate_embedding, job_embeddings)
#     # print("Euclidean Distance: ")
#     # print(euclidean_distances)
#     # print(len(euclidean_distances))
#     #
#     # # Calculate Manhattan distance scores
#     # manhattan_distances = manhattan_distance_score(candidate_embedding, job_embeddings)
#     # print("Manhattan Distance: ")
#     # print(manhattan_distances)
#     # print(len(manhattan_distances))
#
#     # Prepare recommended jobs list with similarity scores
#     # recommended_jobs = []
#     # for i in range(len(preprocessed_jobs_data)):
#     #     job_info = preprocessed_jobs_data[i]
#     #     job_similarity = {
#     #         "jobId": job_info['jobId'],
#     #         "cosineSimilarity": float(cosine_similarities[0][i]),
#     #         # "jaccardSimilarity": float(jaccard_similarities[i]),
#     #         # "euclideanSimilarity": float(euclidean_distances[i]),
#     #         # "overallSimilarity": float(
#     #         #     (float(cosine_similarities[0][i]) + float(jaccard_similarities[i]) + float(euclidean_distances[i])) / 3)
#     #     }
#     #     recommended_jobs.append(job_similarity)
#     # # Sort recommended jobs by overall similarity score
#     # recommended_jobs = sorted(recommended_jobs, key=lambda x: x['cosineSimilarity'], reverse=True)
#     # print(recommended_jobs)
#     # # Code for generating and visualizing graphs
#     # top_n = 5  # Number of top jobs to display
#     # output_dir = "graphs"
#     # if not os.path.exists(output_dir):
#     #     os.makedirs(output_dir)
#     #
#     # for i in range(top_n):
#     #     job = recommended_jobs[i]
#     #     job_id = job['jobId']
#     #     cosine_similarity = job['cosineSimilarity']
#     #     jaccard_similarity = job['jaccardSimilarity']
#     #     euclidean_similarity = job['euclideanSimilarity']
#     #     overall_similarity = job['overallSimilarity']
#     #
#     #     plt.figure(figsize=(12, 6))
#     #
#     #     # Bar Chart for Similarity Scores for top 5 jobs individually
#     #     x = np.arange(4)
#     #     score_labels = ['Cosine', 'Jaccard', 'Euclidean', 'Overall']
#     #     scores = [cosine_similarity, jaccard_similarity, euclidean_similarity, overall_similarity]
#     #     colors = ['skyblue', 'lightcoral', 'lightgreen', 'orange']
#     #     for j in range(len(scores)):
#     #         plt.bar(x[j], scores[j], color=colors[j], label=score_labels[j])
#     #         plt.axhline(y=scores[j], color=colors[j], linestyle='--', linewidth=0.8)  # Dotted line for precise value
#     #
#     #     plt.xlabel('Similarity Metrics')
#     #     plt.ylabel('Similarity Score')
#     #     plt.title('Job ID: {}'.format(job_id))
#     #     plt.xticks(x, score_labels)
#     #     plt.ylim(-1, 1)  # Setting y-axis limit to -1 to 1 for consistency
#     #     plt.legend()
#     #     plt.tight_layout()
#     #     # Save the graph
#     #     plt.savefig(os.path.join(output_dir, 'job_{}_similarity_graph.png'.format(job_id)))
#     #
#     #     plt.show()
#     #
#     #     # Code for generating and visualizing the bar plot
#     #     top_n = 5  # Number of top jobs to display
#     #     top_jobs = recommended_jobs[:top_n]  # Assuming recommended_jobs is sorted by cosine similarity
#     #     job_ids = [job['jobId'] for job in top_jobs]
#     #     cosine_similarities = [job['cosineSimilarity'] for job in top_jobs]
#     #
#     #     # Sort job IDs and cosine similarities in descending order of cosine similarities
#     #     job_ids_sorted = [job_id for _, job_id in sorted(zip(cosine_similarities, job_ids), reverse=True)]
#     #     cosine_similarities_sorted = sorted(cosine_similarities, reverse=True)
#     #
#     #     # Define colors for different jobs
#     #     colors = ['lightcoral', 'lightskyblue', 'lightgreen', 'lightyellow', 'lightpink']
#     #
#     #     plt.figure(figsize=(12, 6))
#     #
#     #     # Bar Plot for Cosine Similarity Scores
#     #     x = np.arange(len(job_ids_sorted))
#     #     for i in range(len(job_ids_sorted)):
#     #         plt.bar(x[i], cosine_similarities_sorted[i], color=colors[i], label='Job {}'.format(job_ids_sorted[i]))
#     #         plt.text(x[i], cosine_similarities_sorted[i], '{}'.format(cosine_similarities_sorted[i]), ha='center', va='bottom')
#     #
#     #     plt.xlabel('Job IDs')
#     #     plt.ylabel('Cosine Similarity')
#     #     plt.title('Top 5 Jobs based on Cosine Similarity')
#     #     plt.xticks(x, job_ids_sorted)
#     #     plt.ylim(0, 1)  # Setting y-axis limit to 0-1 for cosine similarity
#     #     plt.legend()
#     #     plt.tight_layout()
#     #
#     #     # Save the bar plot
#     #     output_dir = "graphs"
#     #     if not os.path.exists(output_dir):
#     #         os.makedirs(output_dir)
#     #     plt.savefig(os.path.join(output_dir, 'top_5_jobs_cosine_similarity.png'))
#     #
#     #     plt.show()
#     #
#     # # 2. Histogram of Similarity Scores
#     # plt.figure(figsize=(12, 8))
#     # plt.hist([job['overallSimilarity'] for job in recommended_jobs], bins=10, color='lightcoral', edgecolor='black')
#     # plt.xlabel('Similarity Score')
#     # plt.ylabel('Frequency')
#     # plt.title('Distribution of Similarity Scores')
#     # plt.tight_layout()
#     # plt.savefig(os.path.join('graphs', 'histogram_similarity_scores.png'))
#     # plt.show()
#     #
#     # # 3. Box Plot of Similarity Scores
#     # plt.figure(figsize=(12, 8))
#     # plt.boxplot([job['overallSimilarity'] for job in recommended_jobs], vert=False)
#     # plt.xlabel('Similarity Score')
#     # plt.title('Box Plot of Similarity Scores')
#     # plt.tight_layout()
#     # plt.savefig(os.path.join('graphs', 'boxplot_similarity_scores.png'))
#     # plt.show()
#     #
#     # # 4. Scatter Plot of Similarity Scores vs. Job Attributes
#     # plt.figure(figsize=(12, 8))
#     # plt.scatter([job['requiredExperienceMin'] for job in jobs_data],
#     #             [job['cosineSimilarity'] for job in recommended_jobs])
#     # plt.xlabel('Required Experience (min)')
#     # plt.ylabel('Similarity Score')
#     # plt.title('Similarity Score vs. Required Experience')
#     # plt.tight_layout()
#     # plt.savefig(os.path.join('graphs', 'scatter_similarity_vs_experience.png'))
#     # plt.show()
#     #
#     # # 6. Heatmap of Similarity Scores
#     # similarity_matrix = np.array([[job['overallSimilarity'] for job in recommended_jobs]])
#     # plt.figure(figsize=(12, 8))
#     # plt.imshow(similarity_matrix, cmap='hot', interpolation='nearest')
#     # plt.colorbar()
#     # plt.title('Heatmap of Similarity Scores')
#     # plt.tight_layout()
#     # plt.savefig(os.path.join('graphs', 'heatmap_similarity_scores.png'))
#     # plt.show()
#     # Format the recommended jobs according to the desired response format
#     formatted_jobs = []
#     for job in recommended_jobs:
#         formatted_job = {
#             "jobId": job['jobId'],
#             "recommededScore": int(job['recommendedScore'] * 100)  # Converting similarity score to percentage
#         }
#         formatted_jobs.append(formatted_job)
#
#     return json.dumps(formatted_jobs)
#
# # Method to normalise scores
# def normalize_scores(recommended_jobs):
#     # Normalize similarity scores to [0, 1] range
#     max_skill_similarity = max(job['skillSimilarity'] for job in recommended_jobs)
#     max_preference_similarity = max(job['preferenceSimilarity'] for job in recommended_jobs)
#     max_education_similarity = max(job['educationSimilarity'] for job in recommended_jobs)
#     max_language_similarity = max(job['languageSimilarity'] for job in recommended_jobs)
#     max_experience_match = max(job['experienceMatch'] for job in recommended_jobs)
#
#     for job in recommended_jobs:
#         job['skillSimilarity'] /= max_skill_similarity if max_skill_similarity != 0 else 1
#         job['preferenceSimilarity'] /= max_preference_similarity if max_preference_similarity != 0 else 1
#         job['educationSimilarity'] /= max_education_similarity if max_education_similarity != 0 else 1
#         job['languageSimilarity'] /= max_language_similarity if max_language_similarity != 0 else 1
#         job['experienceMatch'] /= max_experience_match if max_experience_match != 0 else 1
#
#     # Calculate recommended score as weighted average of similarities
#     for job in recommended_jobs:
#         job['recommendedScore'] = (
#             0.3 * job['skillSimilarity'] +
#             0.2 * job['preferenceSimilarity'] +
#             0.2 * job['educationSimilarity'] +
#             0.2 * job['languageSimilarity'] +
#             0.1 * job['experienceMatch']
#         )
#
#     return recommended_jobs
# def preprocess_user_data(user_data):
#     # Convert education, job preferences, and languages to lowercase and remove punctuation
#     user_data['education'] = preprocess_text(user_data['education'])
#     user_data['job_preferences'] = preprocess_text(user_data['job_preferences'])
#     user_data['languages'] = preprocess_text(user_data['languages'])
#
#     # Convert skills to lowercase and remove punctuation
#     user_data['skills'] = {preprocess_text(skill): proficiency for skill, proficiency in user_data['skills'].items()}
#
#     return user_data
