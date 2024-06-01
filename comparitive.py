import time
import os
import re
import json
import torch
import requests
import numpy as np
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


# Function to check whether the candidate falls in the experience range
def calculate_experience_match(candidate_previous_job_roles, required_experience_min, required_experience_max):
    total_experience = sum(candidate_previous_job_roles.values())
    average_experience = total_experience / max(len(candidate_previous_job_roles), 1)

    # Normalize the required experience range
    normalized_required_experience_min = required_experience_min / 10  # Assuming 10 years as max experience
    normalized_required_experience_max = required_experience_max / 10

    # Calculate the match score
    if average_experience < normalized_required_experience_min:
        return 0.0
    elif average_experience > normalized_required_experience_max:
        return 1.0
    else:
        return (average_experience - normalized_required_experience_min) / (
                normalized_required_experience_max - normalized_required_experience_min)


# Method returns list of recommended jobs for a given candidate along with the recommendedScore
def job_recommendation(sessionId):
    # Step 1: Extract & preprocess user data
    candidate_info = get_candidate_data(sessionId)
    if candidate_info is None:
        return []  # Return an empty list if user data is not available

    print("Candidate Data:")
    print(candidate_info)

    # Check if candidate has no skills
    if not candidate_info['skills']:
        return -1  # No jobs to be recommended

    preprocessed_candidate_info = preprocess_user_data(candidate_info)
    print("Preprocessed user Data:")
    print(preprocessed_candidate_info)

    # Step 2: Extract & preprocess jobs data
    jobs_data = get_jobs_list(candidate_info["gender"], candidate_info["age"])
    if jobs_data is None:
        return jsonify({'error': 'No Jobs'}), 204

    print("Jobs data:")
    print(jobs_data)

    preprocessed_jobs_data = preprocess_job_data(jobs_data)
    print("Pre Processed Jobs data:")
    print(preprocessed_jobs_data)

    # Step 3: Calculate similarity metric
    recommended_jobs = []
    for i in range(len(preprocessed_jobs_data)):
        job_info = preprocessed_jobs_data[i]

        job_skills = [skill['skillName'] for skill in job_info.get('skills', [])]
        job_proficiencies = {skill['skillName']: int(skill['proficiency']) for skill in job_info.get('skills', [])}
        match_count = sum(int(preprocessed_candidate_info['skills'].get(skill, '0')) >=
                          job_proficiencies.get(skill, 0) for skill in job_skills)  # +-1 range
        skill_similarity = match_count / max(len(job_skills), 1)
        if skill_similarity <= 0:
            continue

        # Calculate preference similarity
        candidate_preferences = candidate_info.get('job_preferences', '')
        job_preferences = job_info.get('preferences', '')
        if not candidate_preferences or not job_preferences:
            preference_similarity = 0.0
        else:
            candidate_preference_embedding = generate_bert_embeddings(candidate_preferences, tokenizer)
            job_preference_embedding = generate_bert_embeddings(job_preferences, tokenizer)
            preference_similarity = cosine_similarity_score(candidate_preference_embedding, job_preference_embedding)

        # Calculate education similarity
        job_education = job_info.get('requiredHighestEducation', '')
        candidate_education = candidate_info.get('education', '')
        if not job_education or not candidate_education:
            education_similarity = 0.0
        else:
            education_embedding = generate_bert_embeddings(job_education, tokenizer)
            candidate_education_embedding = generate_bert_embeddings(candidate_education, tokenizer)
            education_similarity = cosine_similarity_score(candidate_education_embedding, education_embedding)

        # Calculate language similarity
        job_languages = job_info.get('languages', [])
        candidate_languages = candidate_info.get('languages', [])
        if not job_languages or not candidate_languages:
            language_similarity = 0.0
        else:
            language_texts = [''.join(job_languages)]
            language_embedding = generate_bert_embeddings(language_texts[0], tokenizer)
            candidate_language_embedding = generate_bert_embeddings(candidate_languages, tokenizer)
            language_similarity = cosine_similarity_score(candidate_language_embedding, language_embedding)

        # Calculate experience similarity
        if not candidate_info['previous_job_roles']:
            experience_similarity = 0.0
        else:
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
    if not filtered_jobs:
        print("No jobs available for the given candidate")
        return jsonify({'error': 'No jobs are available for the candidate'}), 204
    formatted_jobs = []
    for job in filtered_jobs:
        formatted_job = {
            "jobId": job['jobId'],
            "recommendationScore": round(job['overallSimilarity'], 2)
        }
        formatted_jobs.append(formatted_job)
    print("Recommended jobs are:")
    print(formatted_jobs)
    return json.dumps(formatted_jobs)


# Method returns list of recommended candidates for a given job along with the recommendedScore
def candidate_recommendation(job_id):
    # Step 1: Extract & preprocess job data
    job_info = get_job_data(job_id)
    if not job_info:
        return jsonify({'error': 'Unsupported format'}), 400

    print("Job Data:")
    print(job_info)

    preprocessed_jobs_data = preprocess_job_data(job_info)
    print("Pre Processed Jobs data:")
    print(preprocessed_jobs_data)

    # Step 2: Extract & preprocess candidates data
    candidates_data = get_candidate_list()
    if not candidates_data:
        return jsonify({'error': 'No Candidates'}), 204

    print("Candidates Data:")
    print(candidates_data)

    preprocessed_candidates_data = [preprocess_user_data(candidate) for candidate in candidates_data]
    print("Preprocessed Candidates Data:")
    print(preprocessed_candidates_data)

    # Step 3: Calculate similarity metric for each candidate
    recommended_candidates = []

    for candidate_info in preprocessed_candidates_data:
        # Check if candidate has no skills
        if not candidate_info['skills']:
            continue  # Skip this candidate

        # Calculate skill similarity
        candidate_skills = candidate_info.get('skills', {})
        job_skills = [skill['skillName'] for skill in job_info.get('skills', [])]
        job_proficiencies = {skill['skillName']: int(skill['proficiency']) for skill in job_info.get('skills', [])}
        match_count = sum(
            int(candidate_skills.get(skill, '0')) >= job_proficiencies.get(skill, 0) for skill in job_skills)
        skill_similarity = match_count / max(len(job_skills), 1)
        if skill_similarity <= 0:
            continue

        # Calculate preference similarity
        candidate_preferences = candidate_info.get('job_preferences', '')
        job_preferences = job_info.get('preferences', '')
        if not candidate_preferences or not job_preferences:
            preference_similarity = 0.0
        else:
            candidate_preference_embedding = generate_bert_embeddings(candidate_info['job_preferences'], tokenizer)
            job_preference_embedding = generate_bert_embeddings(job_info['preferences'], tokenizer)
            preference_similarity = cosine_similarity_score(candidate_preference_embedding, job_preference_embedding)

        # Calculate education similarity
        job_education = job_info.get('requiredHighestEducation', '')
        candidate_education = candidate_info.get('education', '')
        if not job_education or not candidate_education:
            education_similarity = 0.0
        else:
            education_embedding = generate_bert_embeddings(job_education, tokenizer)
            candidate_education_embedding = generate_bert_embeddings(candidate_education, tokenizer)
            education_similarity = cosine_similarity_score(candidate_education_embedding, education_embedding)

        # Calculate languages similarity
        job_languages = job_info.get('languages', [])
        candidate_languages = candidate_info.get('languages', [])
        if not job_languages or not candidate_languages:
            language_similarity = 0.0
        else:
            language_texts = [''.join(job_info.get('languages', []))]
            language_embedding = generate_bert_embeddings(language_texts[0], tokenizer)
            candidate_language_embedding = generate_bert_embeddings(candidate_info['languages'], tokenizer)
            language_similarity = cosine_similarity_score(candidate_language_embedding, language_embedding)

        # Calculate experience similarity
        if not candidate_info['previous_job_roles']:
            experience_similarity = 0.0
        else:
            experience_similarity = calculate_experience_match(
                candidate_info['previous_job_roles'],
                job_info['requiredExperienceMin'],
                job_info['requiredExperienceMax']
            )

        # Calculate overall similarity score with equal weightage for each attribute
        skill_weight = 0.40
        preference_weight = 0.20
        experience_weight = 0.10
        education_weight = 0.20
        language_weight = 0.10

        overall_similarity = float(
            skill_similarity * skill_weight +
            preference_similarity * preference_weight +
            education_similarity * education_weight +
            language_similarity * language_weight +
            experience_similarity * experience_weight
        ) * 100  # Convert to percentage

        # Append candidate data along with similarity scores to the recommended candidates list
        recommended_candidates.append({
            "candidateId": candidate_info['candidateId'],
            "skillSimilarity": skill_similarity,
            "preferenceSimilarity": preference_similarity,
            "educationSimilarity": education_similarity,
            "languageSimilarity": language_similarity,
            "experienceMatch": experience_similarity,
            "overallSimilarity": overall_similarity
        })

    # Log the recommended candidates for debugging
    for candidate in recommended_candidates:
        print(candidate)

    sorted_candidates = sorted(recommended_candidates, key=lambda x:x['overallSimilarity'], reverse=True)
    filtered_candidates = [candidate for candidate in sorted_candidates if candidate['overallSimilarity'] > 45]
    if not filtered_candidates:
        print("No candidates available for the given job")
        return jsonify({'error': 'No candidates are available for the job'}), 204
    print("Recommended candidates are:")
    print(filtered_candidates)
    formatted_candidates = []
    for candidate in filtered_candidates:
        formatted_candidate = {
            "candidateId": candidate['candidateId'],
            "recommendationScore": round(candidate['overallSimilarity'], 2)
        }
        formatted_candidates.append(formatted_candidate)
    return json.dumps(formatted_candidates)