import time
import re
import json
import torch
import requests
import numpy as np
from flask import jsonify
import gensim.downloader as api
from scipy.spatial import distance
from collections import defaultdict
from datetime import datetime, timezone
from sklearn.metrics import pairwise_distances
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer






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
import requests
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
                    "skills": [{"skillName": skill["skillName"], "proficiency": skill["proficiency"]} for skill in job["skills"]],
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

# Function to preprocess user_data
# def preprocess_user_data(user_data):
#     # Preprocess education and job preferences
#     user_data['education'] = preprocess_text(user_data['education'])
#     user_data['job_preferences'] = preprocess_text(user_data['job_preferences'])
#
#     # Preprocess languages
#     user_data['languages'] = [preprocess_text(language) for language in user_data['languages'].split(',')]
#
#     # Convert skills proficiency to integer
#     user_data['skills'] = {skill: int(proficiency) for skill, proficiency in user_data['skills'].items()}
#
#     # Convert previous job roles experience to integer
#     user_data['previous_job_roles'] = {role: int(experience) for role, experience in user_data['previous_job_roles'].items()}
#
#     return user_data


# Function to preprocess user data
def preprocess_user_data(user_data):
    # Preprocess education and job preferences
    user_data['education'] = preprocess_text(user_data['education'])
    user_data['job_preferences'] = preprocess_text(user_data['job_preferences'])

    # Preprocess languages
    user_data['languages'] = preprocess_text(','.join(user_data['languages']))

    # Convert skills proficiency to integer
    user_data['skills'] = {skill: int(proficiency) for skill, proficiency in user_data['skills'].items()}

    # Convert previous job roles experience to integer
    user_data['previous_job_roles'] = {role: int(experience) for role, experience in user_data['previous_job_roles'].items()}

    return user_data



# Function to preprocess jobs_data
# def preprocess_job_data(job_data):
#     for job in job_data:
#         # Convert proficiency to integer
#         for skill in job['skills']:
#             skill['proficiency'] = int(skill['proficiency'])
#
#         # Convert age and experience to integer, handle empty cases
#         job['requiredExperienceMin'] = int(job['requiredExperienceMin']) if job['requiredExperienceMin'] else None
#         job['requiredExperienceMax'] = int(job['requiredExperienceMax']) if job['requiredExperienceMax'] else None
#         job['requiredAgeMin'] = int(job['requiredAgeMin']) if job['requiredAgeMin'] else None
#         job['requiredAgeMax'] = int(job['requiredAgeMax']) if job['requiredAgeMax'] else None
#
#         # Preprocess locations, languages, and preferences, handle empty cases
#         job['locations'] = [preprocess_text(location) for location in job['locations']] if job['locations'] else []
#         job['languages'] = [preprocess_text(language) for language in job['languages']] if job['languages'] else []
#         job['preferences'] = [preprocess_text(preference) for preference in job['preferences']] if job[
#             'preferences'] else []
#
#         # Preprocess highest education and gender, handle empty cases
#         job['requiredHighestEducation'] = preprocess_text(job['requiredHighestEducation']) if job[
#             'requiredHighestEducation'] else None
#         job['requiredGender'] = preprocess_text(job['requiredGender']) if job['requiredGender'] else None
#
#     return job_data

# Function to preprocess jobs data
def preprocess_job_data(job_data):
    for job in job_data:
        # Preprocess skills proficiency to integer
        for skill in job['skills']:
            skill['proficiency'] = int(skill['proficiency'])

        # Preprocess age and experience to integer, handle empty cases
        job['requiredExperienceMin'] = int(job['requiredExperienceMin']) if job['requiredExperienceMin'] else None
        job['requiredExperienceMax'] = int(job['requiredExperienceMax']) if job['requiredExperienceMax'] else None
        job['requiredAgeMin'] = int(job['requiredAgeMin']) if job['requiredAgeMin'] else None
        job['requiredAgeMax'] = int(job['requiredAgeMax']) if job['requiredAgeMax'] else None

        # Preprocess locations, languages, and preferences, handle empty cases
        job['locations'] = preprocess_text(','.join(job['locations']))
        job['languages'] = preprocess_text(','.join(job['languages']))
        job['preferences'] = preprocess_text(','.join(job['preferences']))

        # Preprocess highest education and gender, handle empty cases
        job['requiredHighestEducation'] = preprocess_text(job['requiredHighestEducation']) if job['requiredHighestEducation'] else None
        job['requiredGender'] = preprocess_text(job['requiredGender']) if job['requiredGender'] else None

    return job_data

# Function to Convert text data to TF-IDF vectors
def convert_to_tfidf(user_data):
    # Concatenate text features (education, job preferences, languages) into a single string
    text_features = ' '.join([user_data['education'], user_data['job_preferences'], *user_data['languages']])

    # Initialize TF-IDF vectorizer
    vectorizer = TfidfVectorizer()

    # Fit and transform the text features to TF-IDF vectors
    tfidf_vectors = vectorizer.fit_transform([text_features])

    return tfidf_vectors
def convert_jobs_to_tfidf(jobs_data):
    text_features = ""

    # Concatenate job preferences into text_features
    if 'preferences' in jobs_data:
        text_features += ' '.join(jobs_data['preferences']) + ' '

    # Concatenate languages into text_features
    if 'languages' in jobs_data:
        text_features += ' '.join(jobs_data['languages']) + ' '

    # Concatenate skills and proficiency into text_features
    if 'skills' in jobs_data:
        for skill_dict in jobs_data['skills']:
            text_features += skill_dict['skillName'] + ' '

    if text_features:
        # Initialize TF-IDF vectorizer
        vectorizer = TfidfVectorizer()
        # Fit and transform the text features to TF-IDF vectors
        tfidf_vectors = vectorizer.fit_transform([text_features])
        return tfidf_vectors
    else:
        return None


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')
def generate_bert_embeddings_for_jobs(text):
    # Tokenize and encode the text
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)

    # Generate BERT embeddings
    with torch.no_grad():
        outputs = model(**inputs)

    # Calculate the mean of the BERT embeddings
    embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

    return embeddings

# Function to Generate Word2Vec embeddings
# def generate_word2vec_embeddings(user_data):
#     # Load pre-trained Word2Vec model
#     model_path = "word2vec-google-news-300.gz"  # Assuming the model file is in the current directory
#     word2vec_model = api.load(model_path)
#
#     # Initialize an empty list to store word vectors
#     word_vectors = []
#
#     # Get word vectors for each word in the text features
#     for word in user_data['education'].split() + user_data['job_preferences'].split() + user_data['languages']:
#         if word in word2vec_model:
#             word_vectors.append(word2vec_model[word])
#
#     # Calculate the average word vector
#     if word_vectors:
#         embeddings = np.mean(word_vectors, axis=0)
#     else:
#         embeddings = np.zeros_like(word2vec_model['computer'])  # Use a zero vector if no word vectors found
#
#     return embeddings

# Function to Generate BERT embeddings
# Load pre-trained BERT model and tokenize
def generate_bert_embeddings(text, tokenizer):
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings.numpy()

# Function to calculate similarity among a candidate & jobs
def calculate_similarity(candidate_vector, job_vector, method):
    if method == "cosine":
        return cosine_similarity(candidate_vector, job_vector)[0][0]
    elif method == "jaccard":
        intersection = len(set(candidate_vector.nonzero()[1]) & set(job_vector.nonzero()[1]))
        union = len(set(candidate_vector.nonzero()[1]) | set(job_vector.nonzero()[1]))
        return intersection / union if union != 0 else 0
    elif method == "euclidean":
        return 1 / (1 + euclidean_distances(candidate_vector, job_vector))
    elif method == "manhattan":
        return 1 / (1 + manhattan_distances(candidate_vector, job_vector))


# Function which will return list of recommend jobs with their similarity score
def job_recommendation(sessionId):
    # Step 1: Extract user data
    user_data = get_candidate_data(sessionId)
    if user_data is None:
        return []  # Return an empty list if user data is not available

    # Step 2: Preprocess user data
    user_data = preprocess_user_data(user_data)

    # Step 3: Convert text data to TF-IDF vectors
    user_data_tfidf_vector = convert_to_tfidf(user_data)
    print("User TFIDF Data:")
    print(user_data_tfidf_vector)

    # Step 4: Generate BERT embeddings for candidate
    user_data_bert_embeddings = generate_bert_embeddings(
        ' '.join(map(str, user_data['skills'].values())) + ' ' + user_data['education'] + ' ' + user_data['job_preferences'] + ' ' + ' '.join(user_data['languages']), tokenizer)
    print("BERT Embeddings User Data:")
    print(user_data_bert_embeddings)

    # Step 5: Fetch job details
    jobs_data = get_jobs_list(user_data["gender"], user_data["age"])
    if jobs_data is None:
        return jsonify({'error': 'No Jobs'}), 204

    # Step 7: Preprocess jobs data
    preprocessed_jobs_data = preprocess_job_data(jobs_data)

    # Step 8: Convert jobs text data to TF-IDF vectors
    jobs_data_tfidf_vectors = []
    for job_data in preprocessed_jobs_data:
        tfidf_vector = convert_jobs_to_tfidf(job_data)
        jobs_data_tfidf_vectors.append(tfidf_vector)

    print("Jobs TFIDF Data:")
    print(jobs_data_tfidf_vectors)

    # Step 9: Generate BERT embeddings for jobs
    job_embeddings = []
    for job_data in preprocessed_jobs_data:
        job_info_text = " ".join([str(val) for val in job_data.values()])
        job_embeddings.append(generate_bert_embeddings_for_jobs(job_info_text))
    job_embeddings = np.array(job_embeddings)
    print("Jobs embeddings Data:")
    print(job_embeddings)

    # Step 10: Calculate similarity scores
    recommended_jobs = []
    for i, job_data in enumerate(preprocessed_jobs_data):
        # # TF-IDF similarity
        tfidf_similarity_cosine = calculate_similarity(user_data_tfidf_vector, jobs_data_tfidf_vectors[i], method="cosine")
        tfidf_similarity_jaccard = calculate_similarity(user_data_tfidf_vector, jobs_data_tfidf_vectors[i],
                                                       method="jaccard")
        tfidf_similarity_manhattan = calculate_similarity(user_data_tfidf_vector, jobs_data_tfidf_vectors[i],
                                                        method="manhattan")
        tfidf_similarity_manhattan = calculate_similarity(user_data_tfidf_vector, jobs_data_tfidf_vectors[i],
                                                          method="manhattan")

        # BERT embeddings similarity
        bert_similarity = cosine_similarity(user_data_bert_embeddings.reshape(1, -1),
                                            job_embeddings[i].reshape(1, -1))[0][0]

        # Overall similarity score (average of TF-IDF and BERT similarities)
        overall_similarity_score = (tfidf_similarity + bert_similarity) / 2

        # Append job recommendation with similarity scores
        recommended_job = {
            "jobId": job_data["jobId"],
            "CosineSimilarity_TFIDF": tfidf_similarity,
            "CosineSimilarity_BERT": bert_similarity,
            "OverallSimilarityScore": overall_similarity_score
        }
        recommended_jobs.append(recommended_job)

    # Sort recommended jobs by overall similarity score in descending order
    recommended_jobs = sorted(recommended_jobs, key=lambda x: x["OverallSimilarityScore"], reverse=True)
    print("Recommended Jobs")
    print(recommended_jobs)
    # return recommended_jobs