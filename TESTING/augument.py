import pandas as pd
import numpy as np
import ast
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.utils import resample

# Load the existing dataset
df = pd.read_csv('output1.csv')

# Function to reduce candidate data
def reduce_candidate_data(df, target_size=100):
    reduced_df = df.copy()

    # If current dataset size is less than or equal to target size, return as is
    if len(reduced_df['candidateId'].unique()) <= target_size:
        print(f"Current dataset size ({len(reduced_df['candidateId'].unique())}) is less than or equal to target size ({target_size}). Returning original dataset.")
        return reduced_df

    # Reduce dataset to target size by sampling
    candidate_ids = reduced_df['candidateId'].unique()[:target_size]
    reduced_df = reduced_df[reduced_df['candidateId'].isin(candidate_ids)]

    return reduced_df

# Reduce the dataset to 100 candidates
reduced_df = reduce_candidate_data(df, target_size=1000)

# Evaluation loop
for k in range(0, 17):
    print(f"Processing for k={k}...")

    true_labels = []
    predicted_labels = []
    all_job_ids = set()

    # Collect all job IDs and ground truth preferences
    for _, row in reduced_df.iterrows():
        recommended_job_id = f"j{row['jobId']}"
        all_job_ids.add(recommended_job_id)

        ground_truth = row['candidateRankedJobs']
        ground_truth_jobs = [item['jobId'] for item in ast.literal_eval(ground_truth)]
        all_job_ids.update(ground_truth_jobs)

    for candidate_id in reduced_df['candidateId'].unique():
        candidate_data = reduced_df[reduced_df['candidateId'] == candidate_id]

        recommended_jobs = [f"j{job_id}" for job_id in candidate_data['jobId'].tolist()]

        ground_truth = candidate_data['candidateRankedJobs'].iloc[0]
        ground_truth_jobs = [item['jobId'] for item in ast.literal_eval(ground_truth)]

        top_k_ground_truth_jobs = ground_truth_jobs[:k]

        for job in top_k_ground_truth_jobs:
            if job in recommended_jobs:
                true_labels.append(1)
                predicted_labels.append(1)
            else:
                true_labels.append(1)
                predicted_labels.append(0)

        for job in recommended_jobs:
            if job not in top_k_ground_truth_jobs:
                true_labels.append(0)
                predicted_labels.append(1)

        for job in all_job_ids:
            if job not in top_k_ground_truth_jobs and job not in recommended_jobs:
                true_labels.append(0)
                predicted_labels.append(0)

    cm = confusion_matrix(true_labels, predicted_labels)
    tn, fp, fn, tp = cm.ravel()

    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels, zero_division=0)
    recall = recall_score(true_labels, predicted_labels, zero_division=0)
    f1 = f1_score(true_labels, predicted_labels, zero_division=0)

    print(f"Confusion Matrix:\n{cm}")
    print(f"TN: {tn}, FP: {fp}, FN: {fn}, TP: {tp}")
    print(f"Accuracy: {accuracy:.2f}, Precision: {precision:.2f}, Recall: {recall:.2f}, F1: {f1:.2f}\n")
