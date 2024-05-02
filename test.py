# # FY PROJECT : APPROACH USING SIMILARITY METRICS
# # WORKING OF THE CODE :
# # When recommending a job to a candidate, the code matches the candidate's skills, languages, job preferences, education,
# # and experiences with the requirements and details of the job posting. Here's a breakdown of the parameters being matched:
# # Skills: The code compares the skills listed by the candidate with the skills required for the job.
# # It considers the similarity between the skill names and, if available, the proficiency levels.
# # Languages: It checks if the candidate's spoken languages match the languages preferred or required for the job.
# # Job Preferences: The code verifies if the candidate's job preferences (e.g., Full Time, Part Time, Work From Home)
# # align with the job posting's preferences.
# # Education: It examines the candidate's educational background to see if it meets the job's educational requirements or preferences.
# # Experiences: The code evaluates the candidate's job experiences to determine if they have relevant experience for the job role.
# # By comparing these parameters between the candidate profile and the job posting,
# # the code identifies the most suitable jobs for the candidate and recommends them based on the similarity scores calculated using various similarity metrics.
#
# # Candidate and job data
# candidate_data = [
#     {
#         "userId": 1,
#         "fullName": "Vadiraj Gururaj Inamdar",
#         "contact": "7083491368",
#         "email": "vadirajinamdar6@gmail.com",
#         "dob": 1019500200000,
#         "gender": "M",
#         "userGroup": "CANDIDATE",
#         "skills": [
#             {"skillName": "HTML", "proficiency": 3},
#             {"skillName": "Python", "proficiency": 3},
#             {"skillName": "CSS", "proficiency": 3},
#             {"skillName": "C", "proficiency": None}
#         ],
#         "languages": ["Hindi", "Kannada", "Konkani", "Marathi", "English"],
#         "jobPreferences": ["Full Time", "Part Time", "Work From Home"],
#         "candidateEducations": [
#             {
#                 "degreeName": "Bachelor of Engineering",
#                 "courseName": "Computer Engineering",
#                 "instituteName": "Don Bosco College of Engineering",
#                 "startDate": 1605744000000,
#                 "endDate": 1715385600000,
#                 "highest": False
#             }
#         ],
#         "candidateExperiences": [
#             {
#                 "jobRole": "Senior Web Developer",
#                 "salary": 28000.00,
#                 "startDate": 1585679400000,
#                 "endDate": 1713897000000,
#                 "location": {
#                     "cityName": "Banglore",
#                     "zipcode": "560100",
#                     "state": "Karnataka"
#                 },
#                 "company": "Google"
#             }
#         ]
#     }
# ]
#
# job_data = [
#     {
#         "jobId": 6,
#         "requiredExperienceMin": 0,
#         "requiredExperienceMax": 5,
#         "requiredAgeMin": 18,
#         "requiredAgeMax": 28,
#         "locations": [
#             {
#                 "cityName": "Banglore",
#                 "zipcode": "560100",
#                 "state": "Karnataka"
#             }
#         ],
#         "skills": [
#             {"skillName": "HTML", "proficiency": 3},
#             {"skillName": "Javascript", "proficiency": 2},
#             {"skillName": "CSS", "proficiency": 3}
#         ],
#         "languages": ["Hindi", "English"],
#         "preferences": ["Full Time"],
#         "degrees": ["Bachelor of Engineering"],
#         "jobRole": "Senior Web Developer",
#         "about": "Notice Period - Immediate or who can join in within 15 days\r\n\r\n Proven experience as a Web App Developer, with expertise in React.JS and.NET technologies. - Strong knowledge of JavaScript, HTML, CSS, and related web technologies. - \r\n\r\nExperience with Azure Tech Stack, including Azure App Service, Azure Functions, Azure DevOps, and Azure SQL Database. - \r\n\r\nProficiency in.NET framework, C#, and ASP.NET MVC. - Familiarity with database systems such as SQL Server or Azure SQL Database. \r\n\r\n- Experience with CICD tools and practices, such as Git, Jenkins, or Azure DevOps. - Strong problem-solving skills and attention to detail. \r\n\r\n- Excellent communication and leadership abilities. If you are a motivated and skilled Web App Developer with expertise in React.JS,.NET, Azure Tech Stack, CICD, and Team Leading experience, we would love to hear from you. Join our team and contribute to the development of innovative web applications while leading a talented group of developer",
#         "noOfOpenings": 5,
#         "salaryMin": 10000.00,
#         "salaryMax": 20000.00,
#         "recruiterId": 3,
#         "closingDate": None,
#         "jobCategory": "Engineering",
#         "requiredHighestEducation": "Graduate",
#         "postingDate": 1714282858000,
#         "requiredGender": None,
#         "company": "Google",
#         "companyId": 5
#     },
#     {
#         "jobId": 8,
#         "requiredExperienceMin": 0,
#         "requiredExperienceMax": 5,
#         "requiredAgeMin": 18,
#         "requiredAgeMax": 28,
#         "locations": [
#             {
#                 "cityName": "Banglore",
#                 "zipcode": "560100",
#                 "state": "Karnataka"
#             }
#         ],
#         "skills": [
#             {"skillName": "HTML", "proficiency": 2},
#             {"skillName": "Javascript", "proficiency": 2},
#             {"skillName": "CSS", "proficiency": 2}
#         ],
#         "languages": ["Hindi", "English"],
#         "preferences": ["Full Time"],
#         "degrees": ["Bachelor of Engineering"],
#         "jobRole": "Senior Web Developer",
#         "about": "Notice Period - Immediate or who can join in within 15 days\r\n\r\n Proven experience as a Web App Developer, with expertise in React.JS and.NET technologies. - Strong knowledge of JavaScript, HTML, CSS, and related web technologies. - \r\n\r\nExperience with Azure Tech Stack, including Azure App Service, Azure Functions, Azure DevOps, and Azure SQL Database. - \r\n\r\nProficiency in.NET framework, C#, and ASP.NET MVC. - Familiarity with database systems such as SQL Server or Azure SQL Database. \r\n\r\n- Experience with CICD tools and practices, such as Git, Jenkins, or Azure DevOps. - Strong problem-solving skills and attention to detail. \r\n\r\n- Excellent communication and leadership abilities. If you are a motivated and skilled Web App Developer with expertise in React.JS,.NET, Azure Tech Stack, CICD, and Team Leading experience, we would love to hear from you. Join our team and contribute to the development of innovative web applications while leading a talented group of developer",
#         "noOfOpenings": 5,
#         "salaryMin": 10000.00,
#         "salaryMax": 20000.00,
#         "recruiterId": 3,
#         "closingDate": None,
#         "jobCategory": "Engineering",
#         "requiredHighestEducation": "Graduate",
#         "postingDate": 1714282858000,
#         "requiredGender": None,
#         "company": "Google",
#         "companyId": 5
#     },
#     {
#         "jobId": 7,
#         "requiredExperienceMin": 0,
#         "requiredExperienceMax": 5,
#         "requiredAgeMin": 18,
#         "requiredAgeMax": 28,
#         "locations": [
#             {
#                 "cityName": "Banglore",
#                 "zipcode": "560100",
#                 "state": "Karnataka"
#             }
#         ],
#         "skills": [
#             {"skillName": "HTML", "proficiency": 1},
#             {"skillName": "Javascript", "proficiency": 1},
#             {"skillName": "CSS", "proficiency": 1}
#         ],
#         "languages": ["Hindi", "English"],
#         "preferences": ["Full Time"],
#         "degrees": ["Bachelor of Engineering"],
#         "jobRole": "Senior Web Developer",
#         "about": "Notice Period - Immediate or who can join in within 15 days\r\n\r\n Proven experience as a Web App Developer, with expertise in React.JS and.NET technologies. - Strong knowledge of JavaScript, HTML, CSS, and related web technologies. - \r\n\r\nExperience with Azure Tech Stack, including Azure App Service, Azure Functions, Azure DevOps, and Azure SQL Database. - \r\n\r\nProficiency in.NET framework, C#, and ASP.NET MVC. - Familiarity with database systems such as SQL Server or Azure SQL Database. \r\n\r\n- Experience with CICD tools and practices, such as Git, Jenkins, or Azure DevOps. - Strong problem-solving skills and attention to detail. \r\n\r\n- Excellent communication and leadership abilities. If you are a motivated and skilled Web App Developer with expertise in React.JS,.NET, Azure Tech Stack, CICD, and Team Leading experience, we would love to hear from you. Join our team and contribute to the development of innovative web applications while leading a talented group of developer",
#         "noOfOpenings": 5,
#         "salaryMin": 10000.00,
#         "salaryMax": 20000.00,
#         "recruiterId": 3,
#         "closingDate": None,
#         "jobCategory": "Engineering",
#         "requiredHighestEducation": "Graduate",
#         "postingDate": 1714282858000,
#         "requiredGender": None,
#         "company": "Google",
#         "companyId": 5
#     }
# ]
# import json
# import numpy as np
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
# from scipy.spatial import distance
# from transformers import BertTokenizer, BertModel
# import torch
#
# # Define similarity functions
# def calculate_cosine_similarity(candidate, job):
#     candidate_skills = ' '.join([skill['skillName'] + ' ' + str(skill['proficiency'] or 0) for skill in candidate])
#     job_skills = ' '.join([skill['skillName'] + ' ' + str(skill['proficiency'] or 0) for skill in job])
#
#     tfidf = TfidfVectorizer()
#     tfidf_matrix = tfidf.fit_transform([candidate_skills, job_skills])
#     return np.float64(cosine_similarity(tfidf_matrix)[0, 1])
#
#
# def calculate_bert_similarity(candidate, job):
#     candidate_skills = ' '.join([skill['skillName'] + ' ' + str(skill['proficiency'] or 0) for skill in candidate])
#     job_skills = ' '.join([skill['skillName'] + ' ' + str(skill['proficiency'] or 0) for skill in job])
#
#     tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
#     model = BertModel.from_pretrained('bert-base-uncased')
#
#     candidate_tokens = tokenizer(candidate_skills, return_tensors='pt')
#     job_tokens = tokenizer(job_skills, return_tensors='pt')
#
#     with torch.no_grad():
#         candidate_output = model(**candidate_tokens)[0][:, 0, :].numpy()
#         job_output = model(**job_tokens)[0][:, 0, :].numpy()
#
#     return np.float64(cosine_similarity(candidate_output, job_output)[0, 0])
#
#
# def calculate_jaccard_similarity(candidate, job):
#     candidate_skills = set([skill['skillName'] for skill in candidate])
#     job_skills = set([skill['skillName'] for skill in job])
#
#     intersection = len(candidate_skills & job_skills)
#     union = len(candidate_skills | job_skills)
#     return np.float64(intersection / union)
#
#
# def calculate_manhattan_similarity(candidate, job):
#     candidate_skills = [skill['proficiency'] or 0 for skill in candidate]
#     job_skills = [skill['proficiency'] or 0 for skill in job]
#
#     # Pad the skill lists with zeros to make them equal length
#     max_len = max(len(candidate_skills), len(job_skills))
#     candidate_skills += [0] * (max_len - len(candidate_skills))
#     job_skills += [0] * (max_len - len(job_skills))
#
#     return np.float64(distance.cityblock(candidate_skills, job_skills))
#
#
# # Calculate similarities for each job
# def calculate_job_similarities(candidate, job):
#     cosine_sim = calculate_cosine_similarity(candidate["skills"], job["skills"])
#     bert_sim = calculate_bert_similarity(candidate["skills"], job["skills"])
#     jaccard_sim = calculate_jaccard_similarity(candidate["skills"], job["skills"])
#     manhattan_sim = calculate_manhattan_similarity(candidate["skills"], job["skills"])
#     return {
#         "jobId": job["jobId"],
#         "cosineSimilarity": cosine_sim,
#         "bertEmbeddingsSimilarity": bert_sim,
#         "jaccardSimilarity": jaccard_sim,
#         "manhattanSimilarity": manhattan_sim
#     }
#
#
# # Recommended jobs for a particular candidate
# def recommend_jobs_for_candidate(candidate, job_data):
#     similarities = []
#     for job in job_data:
#         similarities.append(calculate_job_similarities(candidate, job))
#
#     # Sort jobs by cosine similarity in descending order
#     similarities.sort(key=lambda x: x["cosineSimilarity"], reverse=True)
#     return similarities
#
#
# # Example usage
# recommended_jobs = recommend_jobs_for_candidate(candidate_data[0], job_data)
# print(json.dumps(recommended_jobs, indent=4, default=str))

# ##################

# FY PROJECT : APPROACH USING SIMILARITY METRICS

import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial import distance
from transformers import BertTokenizer, BertModel
import torch

# Define similarity functions for attributes



# Candidate and job data
candidate_data = [
    {
        "userId": 1,
        "fullName": "Vadiraj Gururaj Inamdar",
        "contact": "7083491368",
        "email": "vadirajinamdar6@gmail.com",
        "dob": 1019500200000,
        "gender": "M",
        "userGroup": "CANDIDATE",
        "skills": [
            {"skillName": "HTML", "proficiency": 3},
            {"skillName": "Python", "proficiency": 3},
            {"skillName": "CSS", "proficiency": 3},
            {"skillName": "C", "proficiency": None}
        ],
        "languages": ["Hindi", "Kannada", "Konkani", "Marathi", "English"],
        "jobPreferences": ["Full Time", "Part Time", "Work From Home"],
        "candidateEducations": [
            {
                "degreeName": "Bachelor of Engineering",
                "courseName": "Computer Engineering",
                "instituteName": "Don Bosco College of Engineering",
                "startDate": 1605744000000,
                "endDate": 1715385600000,
                "highest": False
            }
        ],
        "candidateExperiences": [
            {
                "jobRole": "Senior Web Developer",
                "salary": 28000.00,
                "startDate": 1585679400000,
                "endDate": 1713897000000,
                "location": {
                    "cityName": "Banglore",
                    "zipcode": "560100",
                    "state": "Karnataka"
                },
                "company": "Google"
            }
        ]
    }
]

job_data = [
    {
        "jobId": 6,
        "requiredExperienceMin": 0,
        "requiredExperienceMax": 5,
        "requiredAgeMin": 18,
        "requiredAgeMax": 28,
        "locations": [
            {
                "cityName": "Banglore",
                "zipcode": "560100",
                "state": "Karnataka"
            }
        ],
        "skills": [
            {"skillName": "HTML", "proficiency": 3},
            {"skillName": "Javascript", "proficiency": 2},
            {"skillName": "CSS", "proficiency": 3}
        ],
        "languages": ["Hindi", "English"],
        "preferences": ["Full Time"],
        "degrees": ["Bachelor of Engineering"],
        "jobRole": "Senior Web Developer",
        "about": "Notice Period - Immediate or who can join in within 15 days\r\n\r\n Proven experience as a Web App Developer, with expertise in React.JS and.NET technologies. - Strong knowledge of JavaScript, HTML, CSS, and related web technologies. - \r\n\r\nExperience with Azure Tech Stack, including Azure App Service, Azure Functions, Azure DevOps, and Azure SQL Database. - \r\n\r\nProficiency in.NET framework, C#, and ASP.NET MVC. - Familiarity with database systems such as SQL Server or Azure SQL Database. \r\n\r\n- Experience with CICD tools and practices, such as Git, Jenkins, or Azure DevOps. - Strong problem-solving skills and attention to detail. \r\n\r\n- Excellent communication and leadership abilities. If you are a motivated and skilled Web App Developer with expertise in React.JS,.NET, Azure Tech Stack, CICD, and Team Leading experience, we would love to hear from you. Join our team and contribute to the development of innovative web applications while leading a talented group of developer",
        "noOfOpenings": 5,
        "salaryMin": 10000.00,
        "salaryMax": 20000.00,
        "recruiterId": 3,
        "closingDate": None,
        "jobCategory": "Engineering",
        "requiredHighestEducation": "Graduate",
        "postingDate": 1714282858000,
        "requiredGender": None,
        "company": "Google",
        "companyId": 5
    },
    {
        "jobId": 8,
        "requiredExperienceMin": 0,
        "requiredExperienceMax": 5,
        "requiredAgeMin": 18,
        "requiredAgeMax": 28,
        "locations": [
            {
                "cityName": "Banglore",
                "zipcode": "560100",
                "state": "Karnataka"
            }
        ],
        "skills": [
            {"skillName": "HTML", "proficiency": 2},
            {"skillName": "Javascript", "proficiency": 2},
            {"skillName": "CSS", "proficiency": 2}
        ],
        "languages": ["Hindi", "English"],
        "preferences": ["Full Time"],
        "degrees": ["Bachelor of Engineering"],
        "jobRole": "Senior Web Developer",
        "about": "Notice Period - Immediate or who can join in within 15 days\r\n\r\n Proven experience as a Web App Developer, with expertise in React.JS and.NET technologies. - Strong knowledge of JavaScript, HTML, CSS, and related web technologies. - \r\n\r\nExperience with Azure Tech Stack, including Azure App Service, Azure Functions, Azure DevOps, and Azure SQL Database. - \r\n\r\nProficiency in.NET framework, C#, and ASP.NET MVC. - Familiarity with database systems such as SQL Server or Azure SQL Database. \r\n\r\n- Experience with CICD tools and practices, such as Git, Jenkins, or Azure DevOps. - Strong problem-solving skills and attention to detail. \r\n\r\n- Excellent communication and leadership abilities. If you are a motivated and skilled Web App Developer with expertise in React.JS,.NET, Azure Tech Stack, CICD, and Team Leading experience, we would love to hear from you. Join our team and contribute to the development of innovative web applications while leading a talented group of developer",
        "noOfOpenings": 5,
        "salaryMin": 10000.00,
        "salaryMax": 20000.00,
        "recruiterId": 3,
        "closingDate": None,
        "jobCategory": "Engineering",
        "requiredHighestEducation": "Graduate",
        "postingDate": 1714282858000,
        "requiredGender": None,
        "company": "Google",
        "companyId": 5
    },
    {
        "jobId": 7,
        "requiredExperienceMin": 0,
        "requiredExperienceMax": 5,
        "requiredAgeMin": 18,
        "requiredAgeMax": 28,
        "locations": [
            {
                "cityName": "Banglore",
                "zipcode": "560100",
                "state": "Karnataka"
            }
        ],
        "skills": [
            {"skillName": "HTML", "proficiency": 1},
            {"skillName": "Javascript", "proficiency": 1},
            {"skillName": "CSS", "proficiency": 1}
        ],
        "languages": ["Hindi", "English"],
        "preferences": ["Full Time"],
        "degrees": ["Bachelor of Engineering"],
        "jobRole": "Senior Web Developer",
        "about": "Notice Period - Immediate or who can join in within 15 days\r\n\r\n Proven experience as a Web App Developer, with expertise in React.JS and.NET technologies. - Strong knowledge of JavaScript, HTML, CSS, and related web technologies. - \r\n\r\nExperience with Azure Tech Stack, including Azure App Service, Azure Functions, Azure DevOps, and Azure SQL Database. - \r\n\r\nProficiency in.NET framework, C#, and ASP.NET MVC. - Familiarity with database systems such as SQL Server or Azure SQL Database. \r\n\r\n- Experience with CICD tools and practices, such as Git, Jenkins, or Azure DevOps. - Strong problem-solving skills and attention to detail. \r\n\r\n- Excellent communication and leadership abilities. If you are a motivated and skilled Web App Developer with expertise in React.JS,.NET, Azure Tech Stack, CICD, and Team Leading experience, we would love to hear from you. Join our team and contribute to the development of innovative web applications while leading a talented group of developer",
        "noOfOpenings": 5,
        "salaryMin": 10000.00,
        "salaryMax": 20000.00,
        "recruiterId": 3,
        "closingDate": None,
        "jobCategory": "Engineering",
        "requiredHighestEducation": "Graduate",
        "postingDate": 1714282858000,
        "requiredGender": None,
        "company": "Google",
        "companyId": 5
    }
]


def calculate_skill_similarity(candidate_skills, job_skills):
    cosine_sim = calculate_cosine_similarity(candidate_skills, job_skills)
    bert_sim = calculate_bert_similarity(candidate_skills, job_skills)
    jaccard_sim = calculate_jaccard_similarity(candidate_skills, job_skills)
    manhattan_sim = calculate_manhattan_similarity(candidate_skills, job_skills)
    return {
        "cosineSimilarity": cosine_sim,
        "bertEmbeddingsSimilarity": bert_sim,
        "jaccardSimilarity": jaccard_sim,
        "manhattanSimilarity": manhattan_sim
    }

def calculate_language_similarity(candidate_languages, job_languages):
    candidate_lang_set = set(candidate_languages)
    job_lang_set = set(job_languages)

    intersection = len(candidate_lang_set & job_lang_set)
    union = len(candidate_lang_set | job_lang_set)

    jaccard_sim = np.float64(intersection / union)
    return {
        "cosineSimilarity": jaccard_sim,  # Since languages are categorical, we use Jaccard similarity
        "bertEmbeddingsSimilarity": jaccard_sim,
        "jaccardSimilarity": jaccard_sim,
        "manhattanSimilarity": 1 - jaccard_sim  # Manhattan distance as a dissimilarity measure
    }

def calculate_location_similarity(candidate_location, job_location):
    if candidate_location == job_location:
        return {
            "cosineSimilarity": 1.0,
            "bertEmbeddingsSimilarity": 1.0,
            "jaccardSimilarity": 1.0,
            "manhattanSimilarity": 0.0
        }
    else:
        return {
            "cosineSimilarity": 0.0,
            "bertEmbeddingsSimilarity": 0.0,
            "jaccardSimilarity": 0.0,
            "manhattanSimilarity": 1.0
        }

def calculate_preference_similarity(candidate_preferences, job_preferences):
    candidate_pref_set = set(candidate_preferences)
    job_pref_set = set(job_preferences)

    intersection = len(candidate_pref_set & job_pref_set)
    union = len(candidate_pref_set | job_pref_set)

    jaccard_sim = np.float64(intersection / union)
    return {
        "cosineSimilarity": jaccard_sim,  # Since preferences are categorical, we use Jaccard similarity
        "bertEmbeddingsSimilarity": jaccard_sim,
        "jaccardSimilarity": jaccard_sim,
        "manhattanSimilarity": 1 - jaccard_sim  # Manhattan distance as a dissimilarity measure
    }

def calculate_cosine_similarity(text1, text2):
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform([text1, text2])
    return np.float64(cosine_similarity(tfidf_matrix)[0, 1])

def calculate_bert_similarity(text1, text2):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')

    tokens = tokenizer([text1, text2], return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**tokens)
        embeddings = outputs.last_hidden_state[:, 0, :]

    return np.float64(cosine_similarity(embeddings)[0, 1])

def calculate_jaccard_similarity(set1, set2):
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return np.float64(intersection / union)

def calculate_manhattan_similarity(list1, list2):
    return np.float64(distance.cityblock(list1, list2))

# Calculate similarities for each job
def calculate_job_similarities(candidate, job):
    skill_similarity = calculate_skill_similarity([skill['skillName'] for skill in candidate["skills"]],
                                                  [skill['skillName'] for skill in job["skills"]])

    language_similarity = calculate_language_similarity(candidate["languages"], job["languages"])

    location_similarity = calculate_location_similarity(candidate["candidateExperiences"][0]["location"],
                                                        job["locations"][0])

    preference_similarity = calculate_preference_similarity(candidate["jobPreferences"], job["preferences"])

    overall_similarity = np.mean(list(skill_similarity.values()) + list(language_similarity.values()) +
                                  list(location_similarity.values()) + list(preference_similarity.values()))

    return {
        "jobId": job["jobId"],
        "skillSimilarityScore": skill_similarity,
        "languageSimilarityScore": language_similarity,
        "locationSimilarityScore": location_similarity,
        "preferenceSimilarityScore": preference_similarity,
        "overallSimilarityScore": round(overall_similarity, 5)
    }

# Recommended jobs for a particular candidate
def recommend_jobs_for_candidate(candidate, job_data):
    similarities = []
    for job in job_data:
        similarities.append(calculate_job_similarities(candidate, job))

    # Sort jobs by overall similarity in descending order
    similarities.sort(key=lambda x: x["overallSimilarityScore"], reverse=True)
    return similarities

# Example usage

recommended_jobs = recommend_jobs_for_candidate(candidate_data[0], job_data)
print(json.dumps(recommended_jobs, indent=4, default=str))

