# # import json
# # import numpy as np
# # from sklearn.feature_extraction.text import TfidfVectorizer
# # from sklearn.metrics.pairwise import cosine_similarity
# # from scipy.spatial import distance
# # from transformers import BertTokenizer, BertModel
# # import torch
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
#             {"skillName": "REACT", "proficiency": 3},
#             {"skillName": "PYTHON", "proficiency": 2},
#             {"skillName": "C", "proficiency": 3}
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
#             {"skillName": "React", "proficiency": 3}
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
#             {"skillName": "Notion", "proficiency": 2},
#             {"skillName": "Vue", "proficiency": 2}
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
# #
# #
# # # Define similarity functions for attributes
# # def calculate_skill_similarity_with_proficiency(candidate_skills, job_skills):
# #     skill_similarities = []
# #     for candidate_skill in candidate_skills:
# #         for job_skill in job_skills:
# #             if candidate_skill['skillName'] == job_skill['skillName']:
# #                 # Calculate similarity based on proficiency level
# #                 if candidate_skill['proficiency'] is None or job_skill['proficiency'] is None:
# #                     # If proficiency level is missing for either candidate or job, skip comparison
# #                     continue
# #                 proficiency_diff = abs(candidate_skill['proficiency'] - job_skill['proficiency'])
# #                 similarity_score = 1 / (1 + proficiency_diff)  # Higher proficiency leads to higher similarity
# #                 skill_similarities.append(similarity_score)
# #                 break  # Move to the next candidate skill
# #
# #     if not skill_similarities:
# #         # If no common skills found, return 0 similarity for all metrics
# #         return {
# #             "cosineSimilarity": 0.0,
# #             "bertEmbeddingsSimilarity": 0.0,
# #             "jaccardSimilarity": 0.0,
# #             "manhattanSimilarity": 0.0
# #         }
# #
# #     # Calculate mean similarity score
# #     mean_similarity_score = np.mean(skill_similarities)
# #     return mean_similarity_score
# #
# #
# # def calculate_language_similarity(candidate_languages, job_languages):
# #     candidate_lang_set = set(candidate_languages)
# #     job_lang_set = set(job_languages)
# #
# #     intersection = len(candidate_lang_set & job_lang_set)
# #     union = len(candidate_lang_set | job_lang_set)
# #
# #     jaccard_sim = np.float64(intersection / union)
# #     return {
# #         "cosineSimilarity": jaccard_sim,  # Since languages are categorical, we use Jaccard similarity
# #         "bertEmbeddingsSimilarity": jaccard_sim,
# #         "jaccardSimilarity": jaccard_sim,
# #         "manhattanSimilarity": 1 - jaccard_sim  # Manhattan distance as a dissimilarity measure
# #     }
# #
# #
# # def calculate_location_similarity(candidate_location, job_location):
# #     if candidate_location == job_location:
# #         return {
# #             "cosineSimilarity": 1.0,
# #             "bertEmbeddingsSimilarity": 1.0,
# #             "jaccardSimilarity": 1.0,
# #             "manhattanSimilarity": 0.0
# #         }
# #     else:
# #         return {
# #             "cosineSimilarity": 0.0,
# #             "bertEmbeddingsSimilarity": 0.0,
# #             "jaccardSimilarity": 0.0,
# #             "manhattanSimilarity": 1.0
# #         }
# #
# #
# # def calculate_preference_similarity(candidate_preferences, job_preferences):
# #     candidate_pref_set = set(candidate_preferences)
# #     job_pref_set = set(job_preferences)
# #
# #     intersection = len(candidate_pref_set & job_pref_set)
# #     union = len(candidate_pref_set | job_pref_set)
# #
# #     jaccard_sim = np.float64(intersection / union)
# #     return {
# #         "cosineSimilarity": jaccard_sim,
# #         "bertEmbeddingsSimilarity": jaccard_sim,
# #         "jaccardSimilarity": jaccard_sim,
# #         "manhattanSimilarity": 1 - jaccard_sim
# #     }
# #
# #
# # def calculate_cosine_similarity(text1, text2):
# #     text1_str = ' '.join(text1)
# #     text2_str = ' '.join(text2)
# #
# #     tfidf = TfidfVectorizer()
# #     tfidf_matrix = tfidf.fit_transform([text1_str, text2_str])
# #     return np.float64(cosine_similarity(tfidf_matrix)[0, 1])
# #
# #
# # def calculate_bert_similarity(text1, text2=None):
# #     tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# #     model = BertModel.from_pretrained('bert-base-uncased')
# #
# #     if text2 is None:
# #         tokens = tokenizer(text1, return_tensors='pt', padding=True, truncation=True)
# #     else:
# #         tokens = tokenizer(text1, text2, return_tensors='pt', padding=True, truncation=True)
# #
# #     with torch.no_grad():
# #         outputs = model(**tokens)
# #         embeddings = outputs.last_hidden_state[:, 0, :]
# #
# #     return embeddings
# #
# # # Calculate similarities for each job
# # def calculate_job_similarities(candidate, job):
# #     candidate_skills = [skill['skillName'] for skill in candidate["skills"]]
# #     job_skills = [skill['skillName'] for skill in job["skills"]]
# #
# #     candidate_embeddings = calculate_bert_similarity(candidate_skills)
# #     job_embeddings = calculate_bert_similarity(job_skills)
# #
# #     skill_similarity1 = calculate_skill_similarity_with_proficiency(candidate["skills"], job["skills"])
# #     language_similarity = calculate_language_similarity(candidate["languages"], job["languages"])["cosineSimilarity"]
# #     location_similarity = calculate_location_similarity(candidate["candidateExperiences"][0]["location"], job["locations"][0])["cosineSimilarity"]
# #     preference_similarity = calculate_preference_similarity(candidate["jobPreferences"], job["preferences"])["cosineSimilarity"]
# #     overall_similarity = np.mean([skill_similarity1, language_similarity, location_similarity, preference_similarity])
# #
# #     return {
# #         "jobId": job["jobId"],
# #         "skillSimilarityScore": skill_similarity1,
# #         "languageSimilarityScore": language_similarity,
# #         "locationSimilarityScore": location_similarity,
# #         "preferenceSimilarityScore": preference_similarity,
# #         "overallSimilarityScore": round(overall_similarity, 5)
# #     }
# #
# #
# # # Recommended jobs for a particular candidate
# # def recommend_jobs_for_candidate(candidate, job_data):
# #     similarities = []
# #     for job in job_data:
# #         similarities.append(calculate_job_similarities(candidate, job))
# #
# #     # Sort jobs by overall similarity in descending order
# #     similarities.sort(key=lambda x: x["overallSimilarityScore"], reverse=True)
# #     return similarities
# #
# #
# # recommended_jobs = recommend_jobs_for_candidate(candidate_data[0], job_data)
# # print(json.dumps(recommended_jobs, indent=4, default=str))
#
#
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
# from transformers import BertTokenizer, BertModel
# import torch
# import numpy as np
# import json
#
#
# def calculate_skill_similarity_with_proficiency(candidate_skills, job_skills):
#     skill_similarities = []
#     for candidate_skill in candidate_skills:
#         for job_skill in job_skills:
#             if candidate_skill['skillName'] == job_skill['skillName']:
#                 # Calculate similarity based on proficiency level
#                 if candidate_skill['proficiency'] is None or job_skill['proficiency'] is None:
#                     # If proficiency level is missing for either candidate or job, skip comparison
#                     continue
#                 proficiency_diff = abs(candidate_skill['proficiency'] - job_skill['proficiency'])
#                 similarity_score = 1 / (1 + proficiency_diff)  # Higher proficiency leads to higher similarity
#                 skill_similarities.append(similarity_score)
#                 break  # Move to the next candidate skill
#
#     if not skill_similarities:
#         # If no common skills found, return 0 similarity for all metrics
#         return {
#             "cosineSimilarity": 0.0,
#             "bertEmbeddingsSimilarity": 0.0,
#             "jaccardSimilarity": 0.0,
#             "manhattanSimilarity": 0.0
#         }
#
#     # Calculate mean similarity score
#     mean_similarity_score = np.mean(skill_similarities)
#     return {
#         "cosineSimilarity": mean_similarity_score,
#         "bertEmbeddingsSimilarity": mean_similarity_score,
#         "jaccardSimilarity": mean_similarity_score,
#         "manhattanSimilarity": mean_similarity_score
#     }
#
#
# def calculate_language_similarity(candidate_languages, job_languages):
#     candidate_lang_set = set(candidate_languages)
#     job_lang_set = set(job_languages)
#
#     intersection = len(candidate_lang_set & job_lang_set)
#     union = len(candidate_lang_set | job_lang_set)
#
#     jaccard_sim = np.float64(intersection / union)
#     return {
#         "cosineSimilarity": jaccard_sim,  # Since languages are categorical, we use Jaccard similarity
#         "bertEmbeddingsSimilarity": jaccard_sim,
#         "jaccardSimilarity": jaccard_sim,
#         "manhattanSimilarity": 1 - jaccard_sim  # Manhattan distance as a dissimilarity measure
#     }
#
#
# def calculate_location_similarity(candidate_location, job_location):
#     if candidate_location == job_location:
#         return {
#             "cosineSimilarity": 1.0,
#             "bertEmbeddingsSimilarity": 1.0,
#             "jaccardSimilarity": 1.0,
#             "manhattanSimilarity": 0.0
#         }
#     else:
#         return {
#             "cosineSimilarity": 0.0,
#             "bertEmbeddingsSimilarity": 0.0,
#             "jaccardSimilarity": 0.0,
#             "manhattanSimilarity": 1.0
#         }
#
#
# def calculate_preference_similarity(candidate_preferences, job_preferences):
#     candidate_pref_set = set(candidate_preferences)
#     job_pref_set = set(job_preferences)
#
#     intersection = len(candidate_pref_set & job_pref_set)
#     union = len(candidate_pref_set | job_pref_set)
#
#     jaccard_sim = np.float64(intersection / union)
#     return {
#         "cosineSimilarity": jaccard_sim,
#         "bertEmbeddingsSimilarity": jaccard_sim,
#         "jaccardSimilarity": jaccard_sim,
#         "manhattanSimilarity": 1 - jaccard_sim
#     }
#
#
# def calculate_cosine_similarity(text1, text2):
#     text1_str = ' '.join(text1)
#     text2_str = ' '.join(text2)
#
#     tfidf = TfidfVectorizer()
#     tfidf_matrix = tfidf.fit_transform([text1_str, text2_str])
#     return np.float64(cosine_similarity(tfidf_matrix)[0, 1])
#
#
# def calculate_bert_similarity(text1, text2=None):
#     tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
#     model = BertModel.from_pretrained('bert-base-uncased')
#
#     if text2 is None:
#         tokens = tokenizer(text1, return_tensors='pt', padding=True, truncation=True)
#     else:
#         tokens = tokenizer(text1, text2, return_tensors='pt', padding=True, truncation=True)
#
#     with torch.no_grad():
#         outputs = model(**tokens)
#         embeddings = outputs.last_hidden_state[:, 0, :]
#
#     return embeddings
#
#
# def calculate_job_similarities(candidate, job):
#     candidate_skills = [skill['skillName'] for skill in candidate["skills"]]
#     job_skills = [skill['skillName'] for skill in job["skills"]]
#
#     candidate_embeddings = calculate_bert_similarity(candidate_skills)
#     job_embeddings = calculate_bert_similarity(job_skills)
#
#     skill_similarity = calculate_skill_similarity_with_proficiency(candidate["skills"], job["skills"])
#     language_similarity = calculate_language_similarity(candidate["languages"], job["languages"])
#     location_similarity = calculate_location_similarity(candidate["candidateExperiences"][0]["location"], job["locations"][0])
#     preference_similarity = calculate_preference_similarity(candidate["jobPreferences"], job["preferences"])
#
#     # Print similarity scores for each attribute
#     print("Skill Similarity Scores:", skill_similarity)
#     print("Language Similarity Scores:", language_similarity)
#     print("Location Similarity Scores:", location_similarity)
#     print("Preference Similarity Scores:", preference_similarity)
#
#     # Calculate overall similarity score
#     overall_similarity = np.mean([
#         skill_similarity["cosineSimilarity"],
#         language_similarity["cosineSimilarity"],
#         location_similarity["cosineSimilarity"],
#         preference_similarity["cosineSimilarity"]
#     ])
#
#     return {
#         "jobId": job["jobId"],
#         "skillSimilarityScore": skill_similarity,
#         "languageSimilarityScore": language_similarity,
#         "locationSimilarityScore": location_similarity,
#         "preferenceSimilarityScore": preference_similarity,
#         "overallSimilarityScore": round(overall_similarity, 5)
#     }
#
#
# # Recommended jobs for a particular candidate
# def recommend_jobs_for_candidate(candidate, job_data):
#     similarities = []
#     for job in job_data:
#         similarities.append(calculate_job_similarities(candidate, job))
#
#     # Sort jobs by overall similarity in descending order
#     similarities.sort(key=lambda x: x["overallSimilarityScore"], reverse=True)
#     return similarities
#
#
# recommended_jobs = recommend_jobs_for_candidate(candidate_data[0], job_data)
# print(json.dumps(recommended_jobs, indent=4, default=str))
#


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertTokenizer, BertModel
import torch
import numpy as np
import json
import matplotlib.pyplot as plt
from wordcloud import WordCloud

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
            {"skillName": "REACT", "proficiency": 3},
            {"skillName": "PYTHON", "proficiency": 2},
            {"skillName": "C", "proficiency": 3}
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
            {"skillName": "React", "proficiency": 3}
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
            {"skillName": "Notion", "proficiency": 2},
            {"skillName": "Vue", "proficiency": 2}
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

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertTokenizer, BertModel
import torch
import numpy as np
import json


def calculate_skill_similarity_with_proficiency(candidate_skills, job_skills):
    skill_similarities = []
    for candidate_skill in candidate_skills:
        for job_skill in job_skills:
            if candidate_skill['skillName'] == job_skill['skillName']:
                # Calculate similarity based on proficiency level
                if candidate_skill['proficiency'] is None or job_skill['proficiency'] is None:
                    # If proficiency level is missing for either candidate or job, skip comparison
                    continue
                proficiency_diff = abs(candidate_skill['proficiency'] - job_skill['proficiency'])
                similarity_score = 1 / (1 + proficiency_diff)  # Higher proficiency leads to higher similarity
                skill_similarities.append(similarity_score)
                break  # Move to the next candidate skill

    if not skill_similarities:
        # If no common skills found, return 0 similarity for all metrics
        return {
            "cosineSimilarity": 0.0,
            "bertEmbeddingsSimilarity": 0.0,
            "jaccardSimilarity": 0.0,
            "manhattanSimilarity": 0.0
        }

    # Calculate mean similarity score
    mean_similarity_score = np.mean(skill_similarities)
    return {
        "cosineSimilarity": mean_similarity_score,
        "bertEmbeddingsSimilarity": mean_similarity_score,
        "jaccardSimilarity": mean_similarity_score,
        "manhattanSimilarity": mean_similarity_score
    }


def calculate_language_similarity(candidate_languages, job_languages):
    candidate_lang_set = set(candidate_languages)
    job_lang_set = set(job_languages)

    intersection = len(candidate_lang_set & job_lang_set)
    union = len(candidate_lang_set | job_lang_set)

    jaccard_sim = np.float64(intersection / union)

    # Cosine similarity calculation
    cosine_sim = calculate_cosine_similarity(candidate_languages, job_languages)

    # BERT embeddings similarity calculation
    bert_sim = calculate_bert_similarity(candidate_languages)

    # Manhattan similarity calculation
    manhattan_sim = calculate_manhattan_similarity(candidate_languages, job_languages)

    return {
        "cosineSimilarity": cosine_sim,
        "bertEmbeddingsSimilarity": bert_sim,
        "jaccardSimilarity": jaccard_sim,
        "manhattanSimilarity": manhattan_sim
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

    # Cosine similarity calculation
    cosine_sim = calculate_cosine_similarity(candidate_preferences, job_preferences)

    # BERT embeddings similarity calculation
    bert_sim = calculate_bert_similarity(candidate_preferences)

    # Manhattan similarity calculation
    manhattan_sim = calculate_manhattan_similarity(candidate_preferences, job_preferences)

    return {
        "cosineSimilarity": cosine_sim,
        "bertEmbeddingsSimilarity": bert_sim,
        "jaccardSimilarity": jaccard_sim,
        "manhattanSimilarity": manhattan_sim
    }


def calculate_job_similarities(candidate, job):
    candidate_skills = [skill['skillName'] for skill in candidate["skills"]]
    job_skills = [skill['skillName'] for skill in job["skills"]]

    skill_similarity = calculate_skill_similarity_with_proficiency(candidate["skills"], job["skills"])
    language_similarity = calculate_language_similarity(candidate["languages"], job["languages"])
    location_similarity = calculate_location_similarity(candidate["candidateExperiences"][0]["location"],
                                                        job["locations"][0])
    preference_similarity = calculate_preference_similarity(candidate["jobPreferences"], job["preferences"])

    # Calculate overall similarity score
    overall_similarity = np.mean([
        skill_similarity["cosineSimilarity"],
        language_similarity["cosineSimilarity"],
        location_similarity["cosineSimilarity"],
        preference_similarity["cosineSimilarity"]
    ])

    return {
        "jobId": job["jobId"],
        "skillSimilarityScore": skill_similarity,
        "languageSimilarityScore": language_similarity,
        "locationSimilarityScore": location_similarity,
        "preferenceSimilarityScore": preference_similarity,
        "overallSimilarityScore": round(overall_similarity, 5)
    }


def calculate_cosine_similarity(text1, text2):
    text1_str = ' '.join(text1)
    text2_str = ' '.join(text2)

    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform([text1_str, text2_str])
    return np.float64(cosine_similarity(tfidf_matrix)[0, 1])


def calculate_bert_similarity(texts):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')

    tokens = tokenizer(texts, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**tokens)
        embeddings = outputs.last_hidden_state[:, 0, :]

    # Convert tensor to numpy array
    embeddings = embeddings.numpy()
    # Calculate cosine similarity between embeddings
    similarity_score = cosine_similarity(embeddings)[0, 1]

    return similarity_score


def calculate_manhattan_similarity(text1, text2):
    # Convert texts to sets of characters
    set1 = set(text1)
    set2 = set(text2)

    # Compute Manhattan distance
    distance = len(set1.union(set2)) - len(set1.intersection(set2))

    # Normalize distance to a similarity score
    similarity_score = 1 - distance / max(len(set1), len(set2))

    return similarity_score


# Recommended jobs for a particular candidate
def recommend_jobs_for_candidate(candidate, job_data):
    similarities = []
    for job in job_data:
        similarities.append(calculate_job_similarities(candidate, job))

    # Sort jobs by overall similarity in descending order
    similarities.sort(key=lambda x: x["overallSimilarityScore"], reverse=True)
    return similarities


recommended_jobs = recommend_jobs_for_candidate(candidate_data[0], job_data)
print(json.dumps(recommended_jobs, indent=4, default=str))



def plot_bar_chart(similarity_scores, title):
    labels = [score['jobId'] for score in similarity_scores]
    scores = [score['overallSimilarityScore'] for score in similarity_scores]

    plt.figure(figsize=(10, 6))
    plt.bar(labels, scores, color='skyblue')
    plt.xlabel('Job ID')
    plt.ylabel('Overall Similarity Score')
    plt.title(title)
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()


import matplotlib.pyplot as plt

def plot_comparison_bar_chart(similarity_metrics, title):
    metrics = list(similarity_metrics.keys())
    scores = list(similarity_metrics.values())

    plt.figure(figsize=(8, 6))
    plt.bar(metrics, scores, color='lightgreen')
    plt.xlabel('Similarity Metrics')
    plt.ylabel('Similarity Score')
    plt.title(title)
    plt.ylim(0, 1)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()



def plot_language_wordcloud(languages, title):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(languages))

    plt.figure(figsize=(10, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_scatter_plot(candidate_embeddings_sim, job_embeddings_sim):
    plt.figure(figsize=(8, 6))
    plt.scatter(candidate_embeddings_sim, job_embeddings_sim, color='orange')
    plt.xlabel('Candidate Embeddings Similarity')
    plt.ylabel('Job Embeddings Similarity')
    plt.title('Scatter Plot of BERT Embeddings Similarity')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()


def plot_preference_pie_chart(preference_similarity, title):
    labels = list(preference_similarity.keys())
    values = list(preference_similarity.values())

    plt.figure(figsize=(8, 6))
    plt.pie(values, labels=labels, autopct='%1.1f%%', startangle=140)
    plt.axis('equal')
    plt.title(title)
    plt.tight_layout()
    plt.show()


plot_bar_chart(recommended_jobs, 'Overall Similarity Scores of Recommended Jobs')
plot_comparison_bar_chart(recommended_jobs[0]['skillSimilarityScore'], 'Comparison of Skill Similarity Metrics')
plot_scatter_plot(recommended_jobs[0]['skillSimilarityScore']['bertEmbeddingsSimilarity'],
                  recommended_jobs[1]['skillSimilarityScore']['bertEmbeddingsSimilarity'])
plot_preference_pie_chart(recommended_jobs[0]['preferenceSimilarityScore'], 'Preference Similarity Distribution')
