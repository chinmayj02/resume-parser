from transformers import BertTokenizer, BertModel
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import json
import requests
from flask import jsonify

# def get_candidate_data(candidate_id):
#     api_url = "http://localhost:8080/jobportal/api/candidate-data?"+candidate_id
#     try:
#         # Send GET request to the API
#         response = requests.get(api_url)
#         # Check if the request was successful (status code 200)
#         if response.status_code == 200:
#             # Parse JSON response
#             return response.json()
#         else:
#             print("Failed to fetch data from the API. Status code:", response.status_code)
#             return -1
#     except requests.exceptions.RequestException as e:
#         print("Error occurred while fetching data:", e)

def get_candidate_data(candidate_id):
    candidate_index = candidates_data['candidate_id'].index(candidate_id)
    if candidate_index != -1:
        candidate_info = {
            'gender': candidates_data['gender'][candidate_index],
            'age': candidates_data['age'][candidate_index],
            'education': candidates_data['education'][candidate_index],
            'job_preferences': candidates_data['job_preferences'][candidate_index],
            'languages': candidates_data['languages'][candidate_index],
            'skills': candidates_data['skills'],
            'previous_job_roles': candidates_data['previous_job_roles']
        }
        return candidate_info
    else:
        return None

def get_job_data(job_id):
    job_index = job_id - 101  # Adjusting index
    if 0 <= job_index < len(jobs_data['job_id']):
        job_info = {
            'job_id': jobs_data['job_id'][job_index],
            'required_education': jobs_data['required_education'][job_index],
            'required_job_preferences': jobs_data['required_job_preferences'][job_index],
            'required_languages': jobs_data['required_languages'][job_index],
            'required_skills': jobs_data['required_skills'],
            'required_gender': jobs_data['required_gender'][job_index],
            'required_age_min': jobs_data['required_age_min'][job_index],
            'required_age_max': jobs_data['required_age_max'][job_index],
            'required_job_role': jobs_data['required_job_role'][job_index],
            'required_experience_min': jobs_data['required_experience_min'][job_index],
            'required_experience_max': jobs_data['required_experience_max'][job_index]
        }
        return job_info
    else:
        return None

# Sample data

# sample candidate
candidates_data = {
    'candidate_id': [1, 2, 3],
    'gender': ['Male', 'Female', 'Male'],
    'age': [30, 25, 35],
    'education': ['Masters in Computer Science, Data Science', 'Bachelor in Computer Engineering', 'PhD in Statistics'],
    'job_preferences': ['Full Time', 'Part Time', 'Full Time'],
    'languages': ['English, Spanish', 'English, French', 'English, Chinese'],
    'skills': {'Python': 9, 'Java': 7, 'SQL': 8},
    'previous_job_roles': {'Data Scientist': 3, 'Software Engineer': 5}
}

jobs_data = {
    'job_id': [101, 102, 103],
    'required_education': ['Masters in Computer Science', 'Bachelor in Computer Science', 'PhD in Statistics'],
    'required_job_preferences': ['Full Time', 'Part Time', 'Full Time'],
    'required_languages': ['English', 'French', 'Chinese'],
    'required_skills': {'Python': 8, 'Java': 6, 'SQL': 7},
    'required_gender': ['Male', 'Female', 'Male'],
    'required_age_min': [25, 20, 30],
    'required_age_max': [35, 30, 40],
    'required_job_role': ['Data Scientist', 'Software Engineer', 'Statistician'],
    'required_experience_min': [2, 1, 3],
    'required_experience_max': [5, 3, 7]
}

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

def job_recommendation(candidate_id):
    print(candidate_id)
    candidate_info = get_candidate_data(candidate_id)
    print(candidate_info)
    if not candidate_info:
        return jsonify({'error': 'Unsupported file format'}), 400
    candidate_info_text = ' '.join(map(str, candidate_info))
    candidate_embedding = get_bert_embeddings(candidate_info_text)

    job_embeddings = []
    for job_info in zip(jobs_data['required_education'], jobs_data['required_job_preferences'], jobs_data['required_languages']):
        job_info_text = ' '.join(map(str, job_info))
        job_embeddings.append(get_bert_embeddings(job_info_text))
    job_embeddings = np.array(job_embeddings)

    similarity_scores = cosine_similarity(candidate_embedding.reshape(1, -1), job_embeddings.reshape(len(job_embeddings), -1))
    ranked_jobs_indices = np.argsort(similarity_scores[0])[::-1]  # Descending order
    ranked_jobs = [(jobs_data['job_id'][idx], similarity_scores[0][idx]) for idx in ranked_jobs_indices]
    
    return log_and_return(ranked_jobs, candidate_id, "Candidate","Job")

def candidate_recommendation(job_id):
    print(job_id)
    job_info = get_job_data(job_id)
    if not job_info:
        return jsonify({'error': 'Unsupported file format'}), 400
    job_info_text = ' '.join(map(str, job_info))
    job_embedding = get_bert_embeddings(job_info_text)

    candidate_embeddings = []
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

