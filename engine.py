from transformers import BertTokenizer, BertModel
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import json
import requests
from flask import jsonify

# Sample data
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
def get_candidate_data(candidate_id):
    api_url = "http://localhost:8080/jobportal/api/candidate-data?"+candidate_id
    try:
        # Send GET request to the API
        response = requests.get(api_url)
        # Check if the request was successful (status code 200)
        if response.status_code == 200:
            # Parse JSON response
            return response.json()
        else:
            print("Failed to fetch data from the API. Status code:", response.status_code)
            return -1
    except requests.exceptions.RequestException as e:
        print("Error occurred while fetching data:", e)

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
    candidate_info = get_candidate_data(candidate_id)
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
    
    return log_and_return(ranked_jobs, candidate_id, "Candidate")

def candidate_recommendation(job_id):
    job_info = jobs_data[job_id - 101]  # Adjusting index
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

    return log_and_return(ranked_candidates, job_id, "Job")


# Define generic log and return function
def log_and_return(matches, entity_id, entity_type):
    result = []
    for item_id, similarity_score in matches:
        result.append({f"{entity_type} ID": entity_id, "Matched Entity ID": item_id, "Similarity Score": similarity_score})
    print(f"{entity_type} {entity_id} matches:")
    for item in result:
        print(item)
    return json.dumps(result)