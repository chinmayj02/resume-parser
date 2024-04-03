from transformers import BertTokenizer, BertModel
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import json
import requests
from flask import jsonify
from datetime import datetime
import time

# Cached job and candidate lists
cached_jobs = None
cached_candidates = None
last_job_fetch_time = 0
last_candidate_fetch_time = 0
cache_expiry_time = 3600  # Cache expiry time in seconds (1 hour)

def get_jobs_list_cached():
    global cached_jobs, last_job_fetch_time

    # Check if cache needs to be refreshed
    if not cached_jobs or time.time() - last_job_fetch_time > cache_expiry_time:
        cached_jobs = get_jobs_list()
        last_job_fetch_time = time.time()

    return cached_jobs

def get_candidate_list_cached():
    global cached_candidates, last_candidate_fetch_time

    # Check if cache needs to be refreshed
    if not cached_candidates or time.time() - last_candidate_fetch_time > cache_expiry_time:
        cached_candidates = get_candidate_list()
        last_candidate_fetch_time = time.time()

    return cached_candidates

# get single candidate data
def get_candidate_data(sessionId):
    api_url = "http://localhost:8080/jobportal/profile?sessionId="+sessionId
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
                'education': ', '.join([education['degreeName'] for education in candidate_info[1]['candidateEducations']]),
                'job_preferences': ', '.join(candidate_info[1]['jobPreferences']),
                'languages': ', '.join(candidate_info[1]['languages']),
                'skills': {skill['skillName']: (skill['proficiency'] or 1) for skill in candidate_info[1]['skills']},
                'previous_job_roles': {experience['jobRole']: calculate_experience_years(experience['startDate'], experience['endDate']) for experience in candidate_info[1]['candidateExperiences']}
            }
            return transformed_data
        else:
            print("Failed to fetch data from the API. Status code:", response.status_code)
            return None
    except requests.exceptions.RequestException as e:
        print("Error occurred while fetching data:", e)

# Helper function to calculate years of experience
def calculate_experience_years(start_date, end_date):
    # Parse the start date string into a datetime object
    start_date = datetime.strptime(str(start_date), "%Y%m%d")

    # Parse the end date string into a datetime object
    end_date = datetime.strptime(str(end_date), "%Y%m%d")
    return (end_date - start_date).days // 365 if end_date > start_date else 0

# Helper function to calculate age from date of birth
def calculate_age(dob):
    # Implementation of calculate_age depends on your specific date format and timezone considerations
    # For example, if dob is in milliseconds since epoch, you can calculate age like this:
    # from datetime import datetime
    # current_year = datetime.now().year
    # dob_year = datetime.utcfromtimestamp(dob / 1000).year
    # return current_year - dob_year
    pass  # Implement according to your requirements

# get single job data
def get_job_data(job_id):
    api_url = "http://localhost:8080/jobportal/api/job/detail/"+str(job_id)
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

    job_data = {
        'job_id': job_details['jobId'],
        'required_education': [degree['degreeName'] for degree in job_details['degrees']],
        'required_job_preferences': [preference['preference'] for preference in job_details['preferences']],
        'required_languages': [language['language'] for language in job_details['languages']],
        'required_skills': {skill['skillName']: skill['proficiency'] for skill in job_details['skills']},
        'required_age_min': job_details['requiredAgeMin'],
        'required_age_max': job_details['requiredAgeMax'],
        'required_job_role': job_details['jobRole'],
        'required_experience_min': job_details['requiredExperienceMin'],
        'required_experience_max': job_details['requiredExperienceMax'],
        'required_gender':job_details['requiredGender']
    }
    if job_details.get('requiredGender') is None:
        job_data['required_gender'] = ["M", "F"]
    return job_data

# get job list
def get_jobs_list():
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

    transformed_data = {
        'job_id': [],
        'required_education': [],
        'required_job_preferences': [],
        'required_languages': [],
        'required_skills': {},
        'required_gender': [],
        'required_age_min': [],
        'required_age_max': [],
        'required_job_role': [],
        'required_experience_min': [],
        'required_experience_max': []
    }

    for job in jobs_data:
        transformed_data['job_id'].append(job['jobId'])
        transformed_data['required_education'].append(', '.join([degree['degreeName'] for degree in job['degrees']]))
        transformed_data['required_job_preferences'].append(', '.join([preference['preference'] for preference in job['preferences']]))
        transformed_data['required_languages'].append(', '.join([language['language'] for language in job['languages']]))

        # Extract skills and their proficiencies
        for skill in job['skills']:
            skill_name = skill['skillName']
            proficiency = skill.get('proficiency', 1)  # Default proficiency to 0 if missing
            transformed_data['required_skills'][skill_name] = proficiency

        transformed_data['required_gender'].append(job['requiredGender'])
        transformed_data['required_age_min'].append(job['requiredAgeMin'])
        transformed_data['required_age_max'].append(job['requiredAgeMax'])
        transformed_data['required_job_role'].append(job['jobRole'])
        transformed_data['required_experience_min'].append(job['requiredExperienceMin'])
        transformed_data['required_experience_max'].append(job['requiredExperienceMax'])

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

    candidates_data = {
        'candidate_id': [],
        'gender': [],
        'age': [],
        'education': [],
        'job_preferences': [],
        'languages': [],
        'skills': {},
        'previous_job_roles': {}
    }

    for candidate_info, candidate_details in candidate_list:
        # Extracting candidate information
        candidate_id = candidate_info['userId']
        gender = candidate_info['gender']
        dob = datetime.fromtimestamp(candidate_info['dob'] / 1000)
        age = datetime.now().year - dob.year
        
        # Extracting education information
        education_info = candidate_details['candidateEducations'][0]
        degree_name = education_info['degreeName']
        institute_name = education_info['instituteName']
        education = f"{degree_name} from {institute_name}"

        # Extracting job preferences and languages
        job_preferences = candidate_details['jobPreferences']
        languages = ', '.join(candidate_details['languages'])

        # Extracting skills
        skills = {}
        for skill in candidate_details['skills']:
            if skill['proficiency']:
                skills[skill['skillName']] = int(skill['proficiency'])

        # Updating candidates_data
        candidates_data['candidate_id'].append(candidate_id)
        candidates_data['gender'].append(gender)
        candidates_data['age'].append(age)
        candidates_data['education'].append(education)
        candidates_data['job_preferences'].append(job_preferences)
        candidates_data['languages'].append(languages)

        # Updating skills
        for skill, proficiency in skills.items():
            if skill in candidates_data['skills']:
                candidates_data['skills'][skill] = max(candidates_data['skills'][skill], proficiency)
            else:
                candidates_data['skills'][skill] = proficiency
    return candidates_data

# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Define function to generate BERT embeddings
def get_bert_embeddings(text):
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings.numpy()

def job_recommendation(sessionId):
    candidate_info = get_candidate_data(sessionId)
    print(candidate_info)
    if not candidate_info:
        return jsonify({'error': 'Unsupported format'}), 400
    candidate_info_text = ' '.join(map(str, candidate_info))
    candidate_embedding = get_bert_embeddings(candidate_info_text)

    job_embeddings = []
    jobs_data=get_jobs_list()
    for job_info in zip(jobs_data['required_education'], jobs_data['required_job_preferences'], jobs_data['required_languages']):
        job_info_text = ' '.join(map(str, job_info))
        job_embeddings.append(get_bert_embeddings(job_info_text))
    job_embeddings = np.array(job_embeddings)

    similarity_scores = cosine_similarity(candidate_embedding.reshape(1, -1), job_embeddings.reshape(len(job_embeddings), -1))
    ranked_jobs_indices = np.argsort(similarity_scores[0])[::-1]  # Descending order
    ranked_jobs = [(jobs_data['job_id'][idx], similarity_scores[0][idx]) for idx in ranked_jobs_indices]
    return log_and_return(ranked_jobs, candidate_info["candidateId"], "Candidate","Job")

def candidate_recommendation(job_id):
    job_info = get_job_data(job_id)
    if not job_info:
        return jsonify({'error': 'Unsupported format'}), 400
    job_info_text = ' '.join(map(str, job_info))
    job_embedding = get_bert_embeddings(job_info_text)

    candidate_embeddings = []
    candidates_data = get_candidate_list_cached()
    for candidate_info in zip(candidates_data['gender'], candidates_data['age'], candidates_data['education'], 
                              candidates_data['job_preferences'], candidates_data['languages']):
        candidate_info_text = ' '.join(map(str, candidate_info))
        candidate_embeddings.append(get_bert_embeddings(candidate_info_text))
    candidate_embeddings = np.array(candidate_embeddings)

    similarity_scores = cosine_similarity(job_embedding.reshape(1, -1), candidate_embeddings.reshape(len(candidate_embeddings), -1))
    ranked_candidates_indices = np.argsort(similarity_scores[0])[::-1]  # Descending order
    ranked_candidates = [(candidates_data['candidate_id'][idx], similarity_scores[0][idx]) for idx in ranked_candidates_indices]

    return log_and_return(ranked_candidates, job_id, "Job","Candidate")


def log_and_return(matches, entity_id, entity_type, matched_entity_type):
    result = {
        entity_type: entity_id,
        matched_entity_type: []
    }
    for item_id, similarity_score in matches:
        result[matched_entity_type].append({matched_entity_type + " ID": item_id, "Similarity Score": float(similarity_score)})
    return json.dumps(result, indent=4)

