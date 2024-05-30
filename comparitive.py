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
        'required_gender': job_details['requiredGender']
    }
    print(job_data)
    if job_details.get('requiredGender') is None:
        job_data['required_gender'] = ["M", "F"]
    return job_data


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
        print(candidates_data)
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
    # Convert education, job preferences, and languages to lowercase and remove punctuation
    user_data['education'] = preprocess_text(user_data['education'])
    user_data['job_preferences'] = preprocess_text(user_data['job_preferences'])
    user_data['languages'] = preprocess_text(user_data['languages'])

    # Convert skills to lowercase and remove punctuation
    user_data['skills'] = {preprocess_text(skill): proficiency for skill, proficiency in user_data['skills'].items()}

    return user_data


# Function to preprocess jobs_data
def preprocess_job_data(job_data):
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


# Method returns list of recommended jobs along with the similarities scores
import json

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

    # Step 3: Generate Candidate Embeddings
    candidate_info_order = ['skills', 'previous_job_roles', 'education', 'job_preferences', 'languages', 'gender',
                            'age']
    candidate_info_values = [candidate_info[key] for key in candidate_info_order]
    candidate_info_text = "".join([str(val) for val in candidate_info_values])
    candidate_embedding = generate_bert_embeddings(candidate_info_text, tokenizer)

    # Step 4: Generate BERT embeddings for jobs
    job_embeddings = []
    for job_info in jobs_data:
        job_info_values = [job_info.get('skills', []), job_info.get('preferences', []),
                           job_info.get('requiredHighestEducation', ''), job_info.get('requiredGender', ''),
                           job_info.get('languages', []), job_info.get('requiredAgeMin', 0),
                           job_info.get('requiredAgeMax', 100), job_info.get('requiredJobRole', ''),
                           job_info.get('requiredExperienceMin', 0), job_info.get('requiredExperienceMax', 100)]
        job_info_text = "".join([str(val) for val in job_info_values])
        job_embeddings.append(generate_bert_embeddings(job_info_text, tokenizer))
    job_embeddings = np.array(job_embeddings)
    print("Jobs embeddings Data:")
    print(job_embeddings)

    # Calculate similarity metric
    recommended_jobs = []
    for i in range(len(preprocessed_jobs_data)):
        job_info = preprocessed_jobs_data[i]

        # Calculate skill similarity
        job_skills = [skill['skillName'] for skill in job_info.get('skills', [])]
        job_proficiencies = {skill['skillName']: int(skill['proficiency']) for skill in job_info.get('skills', [])}
        match_count = sum(
            int(candidate_info['skills'].get(skill, '0')) >= job_proficiencies.get(skill, 0) for skill in job_skills)
        skill_similarity = match_count / max(len(job_skills), 1)

        # Calculate preference similarity
        candidate_preference_embedding = generate_bert_embeddings(candidate_info['job_preferences'], tokenizer)
        job_preference_embedding = generate_bert_embeddings(job_info['preferences'], tokenizer)
        preference_similarity = cosine_similarity_score(candidate_preference_embedding, job_preference_embedding)

        # Calculate education similarity
        education_text = job_info['requiredHighestEducation']
        education_embedding = generate_bert_embeddings(education_text, tokenizer)
        candidate_education_embedding = generate_bert_embeddings(candidate_info['education'], tokenizer)
        education_similarity = cosine_similarity_score(candidate_education_embedding, education_embedding)

        # Calculate languages similarity
        language_texts = [''.join(job_info.get('languages', []))]
        language_embedding = generate_bert_embeddings(language_texts[0], tokenizer)
        candidate_language_embedding = generate_bert_embeddings(candidate_info['languages'], tokenizer)
        language_similarity = cosine_similarity_score(candidate_language_embedding, language_embedding)

        # Calculate experience similarity
        experience_similarity = calculate_experience_match(candidate_info['previous_job_roles'],job_info['requiredExperienceMin'],job_info['requiredExperienceMax'])

        # Calculate overall similarity score (you can adjust weights if needed)
        overall_similarity = (skill_similarity + preference_similarity + education_similarity + language_similarity + experience_similarity) / 5

        # Append job data along with similarity scores to the recommended jobs list
        recommended_jobs.append({
            "jobId": job_info['jobId'],
            "skillSimilarity": skill_similarity,
            "preferenceSimilarity": preference_similarity,
            "educationSimilarity": education_similarity,
            "languageSimilarity": language_similarity,
            "experienceMatch": experience_similarity,
            "overallSimilarity": overall_similarity
        })

    print("Recommended Jobs:")
    print(recommended_jobs)

    # Filter recommended jobs based on test cases
    filtered_jobs = []
    for job in recommended_jobs:
        if job['overallSimilarity'] >= 0.5:  # Only consider jobs with overall similarity score >= 50%
            filtered_jobs.append(job)

    if not filtered_jobs:
        return []  # No jobs to recommend

    filtered_jobs = normalize_scores(filtered_jobs)
    # Sort recommended jobs by overall similarity score in descending order
    sorted_jobs = sorted(filtered_jobs, key=lambda x: x['recommendedScore'], reverse=True)

    formatted_jobs = []
    for job in sorted_jobs:
        formatted_job = {
            "jobId": job['jobId'],
            "overallSimilarity": int(job['recommendedScore'] * 100)  # Converting similarity score to percentage
        }
        formatted_jobs.append(formatted_job)
    sorted_jobs = sorted(formatted_jobs, key=lambda x: x['overallSimilarity'], reverse=True)

    return json.dumps(sorted_jobs)

# Method to normalise scores
def normalize_scores(recommended_jobs):
    # Normalize similarity scores to [0, 1] range
    max_skill_similarity = max(job['skillSimilarity'] for job in recommended_jobs)
    max_preference_similarity = max(job['preferenceSimilarity'] for job in recommended_jobs)
    max_education_similarity = max(job['educationSimilarity'] for job in recommended_jobs)
    max_language_similarity = max(job['languageSimilarity'] for job in recommended_jobs)
    max_experience_match = max(job['experienceMatch'] for job in recommended_jobs)

    for job in recommended_jobs:
        job['skillSimilarity'] /= max_skill_similarity if max_skill_similarity != 0 else 1
        job['preferenceSimilarity'] /= max_preference_similarity if max_preference_similarity != 0 else 1
        job['educationSimilarity'] /= max_education_similarity if max_education_similarity != 0 else 1
        job['languageSimilarity'] /= max_language_similarity if max_language_similarity != 0 else 1
        job['experienceMatch'] /= max_experience_match if max_experience_match != 0 else 1

    # Calculate recommended score as weighted average of similarities
    for job in recommended_jobs:
        job['recommendedScore'] = (
            0.3 * job['skillSimilarity'] +
            0.2 * job['preferenceSimilarity'] +
            0.2 * job['educationSimilarity'] +
            0.2 * job['languageSimilarity'] +
            0.1 * job['experienceMatch']
        )

    return recommended_jobs


def calculate_experience_match(candidate_previous_job_roles, required_experience_min, required_experience_max):
    """
    Calculate the experience match between candidate's previous job roles and the required experience for the job.

    Args:
        candidate_previous_job_roles (dict): A dictionary containing candidate's previous job roles.
        required_experience_min (int): Minimum required experience for the job.
        required_experience_max (int): Maximum required experience for the job.

    Returns:
        float: The experience match score, normalized between 0 and 1.
    """
    total_experience = sum(candidate_previous_job_roles.values())
    average_experience = total_experience / len(candidate_previous_job_roles)

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

# Method returns list of recommended candidate
def candidate_recommendation(job_id):
    job_info = get_job_data(job_id)
    if not job_info:
        return jsonify({'error': 'Unsupported format'}), 400
    candidate_embeddings = []
    candidates_data = get_candidate_list(job_info["required_gender"], job_info["required_age_min"])
    print(candidates_data)

    if (candidates_data == None):
        return jsonify({'error': 'No Candidates'}), 204
    # job_info_text = ' '.join(map(str, job_info))
    job_info_text = " ".join([str(val) for val in job_info.values()])
    job_embedding = get_bert_embeddings(job_info_text)

    for candidate_info in zip(candidates_data['gender'],
                              candidates_data['age'],
                              candidates_data['education'],
                              candidates_data['job_preferences'],
                              candidates_data['languages'],
                              candidates_data['skills'],
                              candidates_data['previous_job_roles']):
        candidate_info_text = ' '.join(map(str, candidate_info))
        candidate_embeddings.append(generate_bert_embeddings(candidate_info_text, tokenizer))
    candidate_embeddings = np.array(candidate_embeddings)

    # this line of code computes the cosine similarity between a single job embedding and multiple candidate embeddings, resulting in a similarity score for each candidate.
    similarity_scores = cosine_similarity(job_embedding.reshape(1, -1),
                                          candidate_embeddings.reshape(len(candidate_embeddings), -1))
    ranked_candidates_indices = np.argsort(similarity_scores[0])[::-1]  # Descending order
    ranked_candidates = [(candidates_data['candidate_id'][idx], similarity_scores[0][idx]) for idx in
                         ranked_candidates_indices]

    return log_and_return(ranked_candidates, job_id, "Job", "Candidate")


#----------------------------------------------------------------------
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