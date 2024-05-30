import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# User data
user_data = {
    "candidateId": 4,
    "gender": "Male",
    "age": 39,
    "education": "Masters of Engineering",
    "job_preferences": "Full-Time, Day Shift, Hybrid",
    "languages": "Hindi, Konkani, Marathi, English",
    "skills": {
        "Python": 8,
        "Php": 8,
        "Java": 6,
        "fortran": 7,
        "perl": 6,
        "C#": 6
    },
    "previous_job_roles": {
        "Software Developer": 2
    }
}

# Convert user data to DataFrame
user_df = pd.DataFrame([user_data])

# Job data
job_data = [
    {
        "jobId": 3,
        "requiredExperienceMin": 1,
        "requiredExperienceMax": 10,
        "requiredAgeMin": 21,
        "requiredAgeMax": 40,
        "locations": ["Banglore"],
        "skills": {
            "Data Analysis": 7,
            "Network performance analysis": 5,
            "Network monitoring tools": 6,
            "Troubleshooting": 9,
            "Capacity planning": 6,
            "Network protocols": 5
        },
        "languages": ["English"],
        "preferences": ["Full-Time"],
        "requiredHighestEducation": "Graduate",
        "requiredGender": "Male"
    }
]

# Convert job data to DataFrame
job_df = pd.DataFrame(job_data)

def preprocess_skills(skills):
    if isinstance(skills, dict):
        # If skills is already in the correct format, just return it
        return skills
    else:
        # If skills is a list, convert it to a dictionary
        return {str(skill["skillName"]): int(skill["proficiency"]) for skill in skills}


def preprocess_languages(languages):
    return len(languages)

user_df["languages"] = user_df["languages"].apply(lambda x: len(x.split(',')))
job_df["languages"] = job_df["languages"].apply(preprocess_languages)

# Preprocess job preferences
def preprocess_job_preferences(preferences):
    return len(preferences.split(',')) if isinstance(preferences, str) else 0

job_df["preferences"] = job_df["preferences"].apply(preprocess_job_preferences)

# Print preprocessed data
print("User Data:")
print(user_df)
print("\nJob Data:")
print(job_df)

# Combine user and job data for clustering
combined_df = pd.concat([user_df.drop(columns=["candidateId", "gender", "education", "previous_job_roles"]),
                         job_df.drop(columns=["jobId", "locations"])])

# Check if combined_df is empty
if combined_df.empty:
    raise ValueError("Combined DataFrame is empty. Cannot proceed with clustering.")

# Add a placeholder numeric column for scaling (e.g., using "age" from the user data)
combined_df["placeholder_numeric_column"] = user_df["age"]

# Select numeric columns for scaling
numeric_columns = combined_df.select_dtypes(include=['int64', 'float64']).columns

# Impute missing values
imputer = SimpleImputer(strategy='mean')
imputed_data = imputer.fit_transform(combined_df[numeric_columns])

# Scale the numeric data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(imputed_data)

# Clustering
kmeans = KMeans(n_clusters=2, random_state=42)
clusters = kmeans.fit_predict(scaled_data)

# Add cluster labels to the DataFrames
user_df["cluster"] = clusters[:len(user_df)]
job_df["cluster"] = clusters[len(user_df):]

# Print cluster assignments
print("\nUser Clusters:")
print(user_df[["candidateId", "cluster"]])
print("\nJob Clusters:")
print(job_df[["jobId", "cluster"]])

