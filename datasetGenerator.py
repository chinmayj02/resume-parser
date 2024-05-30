import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
from nltk.stem import PorterStemmer
import nltk

nltk.download('stopwords')
from nltk.corpus import stopwords

# Define mapping of broader skills to specific related skills
skill_mapping = {
    'backend development': ['node.js', 'python', 'php'],  # Example mapping, add more as needed
    'data analysis': ['statistics', 'data visualization', 'machine learning'],
    # Add more mappings as needed
}

# Example data
candidate_info = {
    'candidateId': 4,
    'gender': 'Male',
    'age': 39,
    'education': 'BSc',
    'job_preferences': 'fulltime day shift hybrid',
    'languages': 'hindi konkani marathi english',
    'skills': {
        'backend development': 1,
        'data analysis': 2
    },
    'previous_job_roles': {
        'Software Developer': 2
    }
}

jobs_data = [
    {
        'jobId': 3, 'requiredExperienceMin': 1, 'requiredExperienceMax': 10, 'requiredAgeMin': 21, 'requiredAgeMax': 40,
        'locations': ['banglore'], 'skills': [{'skillName': 'data analysis', 'proficiency': 7}, {'skillName': 'network performance analysis', 'proficiency': 5}],
        'languages': ['english'], 'preferences': ['fulltime'], 'requiredHighestEducation': 'graduate', 'requiredGender': 'male'
    },
    {
        'jobId': 4, 'requiredExperienceMin': 3, 'requiredExperienceMax': 14, 'requiredAgeMin': 21, 'requiredAgeMax': 40,
        'locations': [], 'skills': [{'skillName': 'performance testing', 'proficiency': 2}, {'skillName': 'load testing', 'proficiency': 3}],
        'languages': [], 'preferences': ['contract'], 'requiredHighestEducation': 'graduate', 'requiredGender': 'male or female'
    },
    {
        'jobId': 7, 'requiredExperienceMin': 3, 'requiredExperienceMax': 12, 'requiredAgeMin': 21, 'requiredAgeMax': 40,
        'locations': [], 'skills': [{'skillName': 'backend development', 'proficiency': 6}, {'skillName': 'restful apis', 'proficiency': 9}],
        'languages': [], 'preferences': ['intern'], 'requiredHighestEducation': 'graduate', 'requiredGender': 'male'
    }
    # Add more job data here...
]

# Preprocess candidate data
candidate_df = pd.DataFrame([candidate_info])
job_df = pd.DataFrame(jobs_data)

# Convert categorical data
label_encoders = {}
for column in ['gender', 'education']:
    le = LabelEncoder()
    candidate_df[column] = le.fit_transform(candidate_df[column])
    le.fit(list(job_df['requiredGender'].unique()) + ['male', 'female'])  # Ensure all possible values are encoded
    job_df['requiredGender'] = job_df['requiredGender'].apply(lambda x: le.transform([x])[0] if x in le.classes_ else -1)
    le.fit(list(job_df['requiredHighestEducation'].unique()) + ['BSc', 'graduate', 'post graduate'])  # Ensure all possible values are encoded
    job_df['requiredHighestEducation'] = job_df['requiredHighestEducation'].apply(lambda x: le.transform([x])[0] if x in le.classes_ else -1)
    label_encoders[column] = le

# Combine age features for scaling
candidate_age = candidate_df[['age']].values
job_ages = job_df[['requiredAgeMin', 'requiredAgeMax']].values

# Stack ages correctly
combined_ages = np.vstack((candidate_age, job_ages[:, 0].reshape(-1, 1), job_ages[:, 1].reshape(-1, 1)))

scaler = StandardScaler()
scaled_ages = scaler.fit_transform(combined_ages)

# Split scaled ages back to respective dataframes
candidate_df[['age']] = scaled_ages[:candidate_age.shape[0], :]
job_df[['requiredAgeMin', 'requiredAgeMax']] = scaled_ages[candidate_age.shape[0]:].reshape(job_ages.shape)

# Extract skills and job preferences
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    tokens = text.lower().split()
    tokens = [stemmer.stem(token) for token in tokens if token not in stop_words]
    return ' '.join(tokens)

candidate_df['job_preferences'] = candidate_df['job_preferences'].apply(preprocess_text)
job_df['preferences'] = job_df['preferences'].apply(lambda x: ' '.join(x)).apply(preprocess_text)

# Expand candidate skills based on skill mapping
for broad_skill, specific_skills in skill_mapping.items():
    if broad_skill in candidate_info['skills']:
        for specific_skill in specific_skills:
            candidate_info['skills'][specific_skill] = candidate_info['skills'].get(specific_skill, 0) + candidate_info['skills'][broad_skill]

# Vectorize text features
vectorizer = CountVectorizer()
candidate_preferences_vector = vectorizer.fit_transform(candidate_df['job_preferences']).toarray()
job_preferences_vector = vectorizer.transform(job_df['preferences']).toarray()

# Combine features
candidate_features = np.hstack((candidate_df[['age', 'gender', 'education']].values, candidate_preferences_vector))
job_features = np.hstack((job_df[['requiredAgeMin', 'requiredAgeMax', 'requiredGender', 'requiredHighestEducation']].values, job_preferences_vector))

# Clustering candidates and jobs
num_clusters = 1
candidate_clusters = KMeans(n_clusters=num_clusters, random_state=42).fit_predict(candidate_features)
job_clusters = KMeans(n_clusters=num_clusters, random_state=42).fit_predict(job_features)

candidate_df['cluster'] = candidate_clusters
job_df['cluster'] = job_clusters

# Recommend jobs based on clusters
def recommend_jobs(candidate_id):
    candidate_cluster = candidate_df[candidate_df['candidateId'] == candidate_id]['cluster'].values[0]
    recommended_jobs = job_df[job_df['cluster'] == candidate_cluster]
    return recommended_jobs

# Example recommendation
recommended_jobs = recommend_jobs(4)
print(recommended_jobs)
