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

# Candidate and job data
candidate_data = [
    {
        "userId": 4,
        "fullName": "Amey Kerkar",
        "contact": "9860687151",
        "email": "amecop47@gmail.com",
        "dob": 474661800000,
        "gender": "M",
        "userGroup": "CANDIDATE"
    },
    {
        "candidateId": 4,
        "skills": [
            {
                "skillId": 7,
                "skillName": "Python",
                "proficiency": "8"
            },
            {
                "skillId": 10,
                "skillName": "Php",
                "proficiency": "8"
            },
            {
                "skillId": 20,
                "skillName": "Java",
                "proficiency": "6"
            },
            {
                "skillId": 61,
                "skillName": "fortran",
                "proficiency": "7"
            },
            {
                "skillId": 62,
                "skillName": "CSS",
                "proficiency": "6"
            },
            {
                "skillId": 63,
                "skillName": "HTML",
                "proficiency": "6"
            }
        ],
        "languages": [
            "Hindi",
            "Konkani",
            "Marathi",
            "English"
        ],
        "jobPreferences": [
            "Full-Time",
            "Day Shift",
            "Hybrid"
        ],
        "candidateEducations": [
            {
                "degreeName": "Masters of Engineering",
                "courseName": "Information Technology",
                "instituteName": "Padre Conceicao College of Engineering Goa",
                "startDate": 1212710400000,
                "endDate": 1286668800000,
                "highest": "true"
            }
        ],
        "candidateExperiences": [
            {
                "jobRole": "Software Developer",
                "salary": 25000.00,
                "startDate": 1212258600000,
                "endDate": 1304188200000,
                "location": {
                    "cityName": "Panjim",
                    "zipcode": "403001",
                    "state": "Goa"
                },
                "company": "Anant Infomedia"
            }
        ]
    }
]

job_data =[
        {
            "jobId": 1,
            "requiredExperienceMin": 4,
            "requiredExperienceMax": 13,
            "requiredAgeMin": 21,
            "requiredAgeMax": 40,
            "locations": [],
            "skills": [
                {
                    "skillId": 72,
                    "skillName": "Engineering design",
                    "proficiency": "9"
                },
                {
                    "skillId": 73,
                    "skillName": "CAD software",
                    "proficiency": "6"
                },
                {
                    "skillId": 74,
                    "skillName": "Problem-solving",
                    "proficiency": "5"
                },
                {
                    "skillId": 75,
                    "skillName": "Technical knowledge",
                    "proficiency": "7"
                },
                {
                    "skillId": 76,
                    "skillName": "Communication skills",
                    "proficiency": "2"
                }
            ],
            "languages": [],
            "preferences": [
                {
                    "preferencesId": 1,
                    "preference": "Full-Time"
                }
            ],
            "degrees": [],
            "jobRole": "Design Engineer",
            "about": "A Design Engineer creates and develops product designs and specifications, using engineering principles and design software to bring innovative products to market.",
            "noOfOpenings": 5,
            "salaryMin": 62000.00,
            "salaryMax": 93000.00,
            "recruiterId": 12,
            "closingDate": 1714501800000,
            "jobCategory": "Structural Engineering",
            "requiredHighestEducation": "Graduate",
            "postingDate": 1714415400000,
            "requiredGender": "F",
            "company": "Aditya Birla Fashion and Retail Limited",
            "companyId": 10
        },
        {
            "jobId": 2,
            "requiredExperienceMin": 1,
            "requiredExperienceMax": 14,
            "requiredAgeMin": 21,
            "requiredAgeMax": 40,
            "locations": [],
            "skills": [
                {
                    "skillId": 77,
                    "skillName": "Manufacturing processes",
                    "proficiency": "3"
                },
                {
                    "skillId": 78,
                    "skillName": "CAD/CAM software",
                    "proficiency": "2"
                },
                {
                    "skillId": 79,
                    "skillName": "Lean manufacturing",
                    "proficiency": "8"
                },
                {
                    "skillId": 80,
                    "skillName": "Quality control and assurance",
                    "proficiency": "6"
                },
                {
                    "skillId": 81,
                    "skillName": "Six Sigma certification",
                    "proficiency": "4"
                }
            ],
            "languages": [],
            "preferences": [
                {
                    "preferencesId": 10,
                    "preference": "Intern"
                }
            ],
            "degrees": [],
            "jobRole": "Manufacturing Engineer",
            "about": "Manufacturing Engineers optimize manufacturing processes, improve production efficiency, and ensure product quality in manufacturing environments.",
            "noOfOpenings": 6,
            "salaryMin": 64000.00,
            "salaryMax": 120000.00,
            "recruiterId": 13,
            "closingDate": 1714501800000,
            "jobCategory": "Mechanical Engineering",
            "requiredHighestEducation": "Phd",
            "postingDate": 1714415400000,
            "requiredGender": "F",
            "company": "India Post",
            "companyId": 11
        },
        {
            "jobId": 3,
            "requiredExperienceMin": 1,
            "requiredExperienceMax": 10,
            "requiredAgeMin": 21,
            "requiredAgeMax": 40,
            "locations": [],
            "skills": [
                {
                    "skillId": 17,
                    "skillName": "Data Analysis",
                    "proficiency": "7"
                },
                {
                    "skillId": 82,
                    "skillName": "Network performance analysis",
                    "proficiency": "5"
                },
                {
                    "skillId": 83,
                    "skillName": "Network monitoring tools",
                    "proficiency": "6"
                },
                {
                    "skillId": 84,
                    "skillName": "Troubleshooting",
                    "proficiency": "9"
                },
                {
                    "skillId": 85,
                    "skillName": "Capacity planning",
                    "proficiency": "6"
                },
                {
                    "skillId": 86,
                    "skillName": "Network protocols",
                    "proficiency": "5"
                }
            ],
            "languages": [],
            "preferences": [
                {
                    "preferencesId": 1,
                    "preference": "Full-Time"
                }
            ],
            "degrees": [],
            "jobRole": "Network Performance Analyst",
            "about": "Network Performance Analysts monitor and optimize network performance. They collect and analyze network data, identify issues, and implement solutions to enhance network speed, reliability, and efficiency.",
            "noOfOpenings": 7,
            "salaryMin": 64000.00,
            "salaryMax": 130000.00,
            "recruiterId": 14,
            "closingDate": 1717785000000,
            "jobCategory": "Network Analysis",
            "requiredHighestEducation": "Graduate",
            "postingDate": 1714415400000,
            "requiredGender": "M",
            "company": "Reliance Industries Limited",
            "companyId": 12
        },
        {
            "jobId": 4,
            "requiredExperienceMin": 3,
            "requiredExperienceMax": 14,
            "requiredAgeMin": 21,
            "requiredAgeMax": 40,
            "locations": [],
            "skills": [
                {
                    "skillId": 87,
                    "skillName": "Performance testing",
                    "proficiency": "2"
                },
                {
                    "skillId": 88,
                    "skillName": "Load testing",
                    "proficiency": "3"
                },
                {
                    "skillId": 89,
                    "skillName": "Stress testing",
                    "proficiency": "2"
                },
                {
                    "skillId": 90,
                    "skillName": "Test scenarios",
                    "proficiency": "8"
                },
                {
                    "skillId": 91,
                    "skillName": "Performance monitoring",
                    "proficiency": "6"
                },
                {
                    "skillId": 92,
                    "skillName": "Benchmarking",
                    "proficiency": "4"
                },
                {
                    "skillId": 93,
                    "skillName": "Performance analysis",
                    "proficiency": "5"
                }
            ],
            "languages": [],
            "preferences": [
                {
                    "preferencesId": 6,
                    "preference": "Contract"
                }
            ],
            "degrees": [],
            "jobRole": "Performance Testing Specialist",
            "about": "Performance Testing Specialists assess the performance of software applications by conducting load, stress, and scalability tests. They identify bottlenecks, optimize performance, and ensure software can handle user demands effectively.",
            "noOfOpenings": 8,
            "salaryMin": 63000.00,
            "salaryMax": 95000.00,
            "recruiterId": 15,
            "closingDate": 1717180200000,
            "jobCategory": "Quality Assuarance",
            "requiredHighestEducation": "Graduate",
            "postingDate": 1714415400000,
            "requiredGender": "A",
            "company": "HDFC Bank",
            "companyId": 13
        },
        {
            "jobId": 6,
            "requiredExperienceMin": 5,
            "requiredExperienceMax": 13,
            "requiredAgeMin": 21,
            "requiredAgeMax": 40,
            "locations": [],
            "skills": [
                {
                    "skillId": 101,
                    "skillName": "UI design principles and best practices",
                    "proficiency": "2"
                },
                {
                    "skillId": 102,
                    "skillName": "Graphic design tools",
                    "proficiency": "8"
                },
                {
                    "skillId": 103,
                    "skillName": "Typography and color theory",
                    "proficiency": "6"
                },
                {
                    "skillId": 104,
                    "skillName": "Visual design and layout",
                    "proficiency": "4"
                },
                {
                    "skillId": 105,
                    "skillName": "Responsive design",
                    "proficiency": "5"
                }
            ],
            "languages": [],
            "preferences": [
                {
                    "preferencesId": 1,
                    "preference": "Full-Time"
                }
            ],
            "degrees": [],
            "jobRole": "User Interface Designer",
            "about": "User Interface Designers focus on the visual and interactive aspects of digital interfaces. They design layouts, buttons, and other elements to ensure a cohesive and visually appealing user interface.",
            "noOfOpenings": 5,
            "salaryMin": 64000.00,
            "salaryMax": 123000.00,
            "recruiterId": 17,
            "closingDate": 1714501800000,
            "jobCategory": "UX/UI",
            "requiredHighestEducation": "Post Graduate",
            "postingDate": 1714415400000,
            "requiredGender": "F",
            "company": "Tata Consultancy Services (TCS)",
            "companyId": 15
        },
        {
            "jobId": 7,
            "requiredExperienceMin": 3,
            "requiredExperienceMax": 12,
            "requiredAgeMin": 21,
            "requiredAgeMax": 40,
            "locations": [],
            "skills": [
                {
                    "skillId": 106,
                    "skillName": "Backend development",
                    "proficiency": "6"
                },
                {
                    "skillId": 107,
                    "skillName": "RESTful APIs",
                    "proficiency": "9"
                },
                {
                    "skillId": 108,
                    "skillName": "Database integration",
                    "proficiency": "6"
                },
                {
                    "skillId": 109,
                    "skillName": "Performance optimization",
                    "proficiency": "5"
                },
                {
                    "skillId": 110,
                    "skillName": "Version control",
                    "proficiency": "2"
                },
                {
                    "skillId": 144,
                    "skillName": "JAVA EE",
                    "proficiency": "7"
                }
            ],
            "languages": [],
            "preferences": [
                {
                    "preferencesId": 10,
                    "preference": "Intern"
                }
            ],
            "degrees": [],
            "jobRole": "Java Backend Developer",
            "about": "Java Backend Developers specialize in building the server-side components of software applications using Java. They work on database integration, server logic, and performance optimization to ensure efficient and reliable backend functionality.",
            "noOfOpenings": 3,
            "salaryMin": 63000.00,
            "salaryMax": 126000.00,
            "recruiterId": 18,
            "closingDate": 1714674600000,
            "jobCategory": "Java Developement",
            "requiredHighestEducation": "Graduate",
            "postingDate": 1714415400000,
            "requiredGender": "M",
            "company": "Tata Motors",
            "companyId": 16
        },
        {
            "jobId": 8,
            "requiredExperienceMin": 4,
            "requiredExperienceMax": 15,
            "requiredAgeMin": 21,
            "requiredAgeMax": 40,
            "locations": [],
            "skills": [
                {
                    "skillId": 6,
                    "skillName": "HTML",
                    "proficiency": "6"
                },
                {
                    "skillId": 8,
                    "skillName": "Javascript",
                    "proficiency": "4"
                },
                {
                    "skillId": 9,
                    "skillName": "CSS",
                    "proficiency": "5"
                },
                {
                    "skillId": 105,
                    "skillName": "Responsive design",
                    "proficiency": "6"
                },
                {
                    "skillId": 111,
                    "skillName": "Front-end web development",
                    "proficiency": "3"
                },
                {
                    "skillId": 112,
                    "skillName": "Web performance optimization",
                    "proficiency": "2"
                },
                {
                    "skillId": 113,
                    "skillName": "Cross-browser compatibility",
                    "proficiency": "8"
                }
            ],
            "languages": [],
            "preferences": [
                {
                    "preferencesId": 1,
                    "preference": "Full-Time"
                }
            ],
            "degrees": [],
            "jobRole": "Front-End Developer",
            "about": "A Front-End Developer specializes in creating the user interface and user experience of websites or applications. They code and design elements that users interact with directly.",
            "noOfOpenings": 9,
            "salaryMin": 60000.00,
            "salaryMax": 124000.00,
            "recruiterId": 19,
            "closingDate": 1714588200000,
            "jobCategory": "UI Development",
            "requiredHighestEducation": "Graduate",
            "postingDate": 1714415400000,
            "requiredGender": "M",
            "company": "Oil and Natural Gas Corporation (ONGC)",
            "companyId": 17
        },
        {
            "jobId": 9,
            "requiredExperienceMin": 3,
            "requiredExperienceMax": 15,
            "requiredAgeMin": 21,
            "requiredAgeMax": 40,
            "locations": [],
            "skills": [
                {
                    "skillId": 101,
                    "skillName": "UI design principles and best practices",
                    "proficiency": "9"
                },
                {
                    "skillId": 102,
                    "skillName": "Graphic design tools",
                    "proficiency": "6"
                },
                {
                    "skillId": 103,
                    "skillName": "Typography and color theory",
                    "proficiency": "5"
                },
                {
                    "skillId": 104,
                    "skillName": "Visual design and layout",
                    "proficiency": "7"
                },
                {
                    "skillId": 105,
                    "skillName": "Responsive design",
                    "proficiency": "2"
                }
            ],
            "languages": [],
            "preferences": [
                {
                    "preferencesId": 1,
                    "preference": "Full-Time"
                }
            ],
            "degrees": [],
            "jobRole": "User Interface Designer",
            "about": "User Interface Designers focus on the visual and interactive aspects of digital interfaces. They design layouts, buttons, and other elements to ensure a cohesive and visually appealing user interface.",
            "noOfOpenings": 10,
            "salaryMin": 63000.00,
            "salaryMax": 127000.00,
            "recruiterId": 20,
            "closingDate": 1714501800000,
            "jobCategory": "UX/UI Design",
            "requiredHighestEducation": "Post Graduate",
            "postingDate": 1714415400000,
            "requiredGender": "M",
            "company": "Infosys",
            "companyId": 18
        },
        {
            "jobId": 10,
            "requiredExperienceMin": 4,
            "requiredExperienceMax": 12,
            "requiredAgeMin": 21,
            "requiredAgeMax": 40,
            "locations": [],
            "skills": [
                {
                    "skillId": 114,
                    "skillName": "User-centered design principles",
                    "proficiency": "3"
                },
                {
                    "skillId": 115,
                    "skillName": "UX/UI design tools",
                    "proficiency": "2"
                },
                {
                    "skillId": 116,
                    "skillName": "Wireframing and prototyping",
                    "proficiency": "8"
                },
                {
                    "skillId": 117,
                    "skillName": "Usability testing and user research",
                    "proficiency": "6"
                },
                {
                    "skillId": 118,
                    "skillName": "Information architecture and user flows",
                    "proficiency": "4"
                }
            ],
            "languages": [],
            "preferences": [
                {
                    "preferencesId": 1,
                    "preference": "Full-Time"
                }
            ],
            "degrees": [],
            "jobRole": "User Experience Designer",
            "about": "User Experience Designers create intuitive and user-friendly digital interfaces. They conduct user research, design prototypes, and work to enhance the overall user experience of websites and applications.",
            "noOfOpenings": 2,
            "salaryMin": 62000.00,
            "salaryMax": 93000.00,
            "recruiterId": 21,
            "closingDate": 1714415400000,
            "jobCategory": "UX/UI Design",
            "requiredHighestEducation": "Graduate",
            "postingDate": 1711737000000,
            "requiredGender": "A",
            "company": "Indian Railways",
            "companyId": 19
        },
        {
            "jobId": 11,
            "requiredExperienceMin": 3,
            "requiredExperienceMax": 13,
            "requiredAgeMin": 21,
            "requiredAgeMax": 40,
            "locations": [],
            "skills": [
                {
                    "skillId": 6,
                    "skillName": "HTML",
                    "proficiency": "6"
                },
                {
                    "skillId": 8,
                    "skillName": "Javascript",
                    "proficiency": "4"
                },
                {
                    "skillId": 9,
                    "skillName": "CSS",
                    "proficiency": "5"
                },
                {
                    "skillId": 105,
                    "skillName": "Responsive design",
                    "proficiency": "6"
                },
                {
                    "skillId": 119,
                    "skillName": "User interface (UI) design",
                    "proficiency": "5"
                },
                {
                    "skillId": 120,
                    "skillName": "User experience (UX) design",
                    "proficiency": "6"
                },
                {
                    "skillId": 121,
                    "skillName": "Web design principles",
                    "proficiency": "9"
                },
                {
                    "skillId": 122,
                    "skillName": "Prototyping and wireframing",
                    "proficiency": "6"
                },
                {
                    "skillId": 123,
                    "skillName": "Front-end development",
                    "proficiency": "5"
                },
                {
                    "skillId": 124,
                    "skillName": "Interaction design",
                    "proficiency": "7"
                },
                {
                    "skillId": 125,
                    "skillName": "User testing",
                    "proficiency": "2"
                },
                {
                    "skillId": 126,
                    "skillName": "Usability testing",
                    "proficiency": "3"
                },
                {
                    "skillId": 127,
                    "skillName": "Collaboration",
                    "proficiency": "2"
                },
                {
                    "skillId": 128,
                    "skillName": "Attention to detail",
                    "proficiency": "8"
                }
            ],
            "languages": [],
            "preferences": [
                {
                    "preferencesId": 6,
                    "preference": "Contract"
                }
            ],
            "degrees": [],
            "jobRole": "UI/UX Developer",
            "about": "A UI/UX Developer combines design and technical skills to create user-friendly interfaces for digital products. They ensure a seamless and visually appealing user experience.",
            "noOfOpenings": 3,
            "salaryMin": 59000.00,
            "salaryMax": 114000.00,
            "recruiterId": 22,
            "closingDate": 1714329000000,
            "jobCategory": "Front-End Developement",
            "requiredHighestEducation": "Post Graduate",
            "postingDate": 1710441000000,
            "requiredGender": "A",
            "company": "Hindustan Aeronautics Limited (HAL)",
            "companyId": 20
        },
        {
            "jobId": 5,
            "requiredExperienceMin": 4,
            "requiredExperienceMax": 14,
            "requiredAgeMin": 21,
            "requiredAgeMax": 40,
            "locations": [],
            "skills": [
                {
                    "skillId": 94,
                    "skillName": "Security consulting",
                    "proficiency": "6"
                },
                {
                    "skillId": 95,
                    "skillName": "Risk assessment",
                    "proficiency": "9"
                },
                {
                    "skillId": 96,
                    "skillName": "Security audits",
                    "proficiency": "6"
                },
                {
                    "skillId": 97,
                    "skillName": "Security policy",
                    "proficiency": "5"
                },
                {
                    "skillId": 98,
                    "skillName": "development",
                    "proficiency": "7"
                },
                {
                    "skillId": 99,
                    "skillName": "Penetration testing",
                    "proficiency": "2"
                },
                {
                    "skillId": 100,
                    "skillName": "Client communication",
                    "proficiency": "3"
                }
            ],
            "languages": [],
            "preferences": [
                {
                    "preferencesId": 6,
                    "preference": "Contract"
                }
            ],
            "degrees": [],
            "jobRole": "Security Consultant",
            "about": "A Security Consultant is responsible for assessing an organizations security infrastructure, identifying vulnerabilities, and recommending solutions to enhance security. They conduct risk assessments, implement security measures, and provide guidance on security policies and procedures.",
            "noOfOpenings": 4,
            "salaryMin": 58000.00,
            "salaryMax": 112000.00,
            "recruiterId": 16,
            "closingDate": 1714415400000,
            "jobCategory": "Network Security",
            "requiredHighestEducation": "Post Graduate",
            "postingDate": 1709231400000,
            "requiredGender": "A",
            "company": "Nestlé India",
            "companyId": 14
        },
        {
            "jobId": 12,
            "requiredExperienceMin": 0,
            "requiredExperienceMax": 9,
            "requiredAgeMin": 21,
            "requiredAgeMax": 40,
            "locations": [],
            "skills": [
                {
                    "skillId": 17,
                    "skillName": "Data Analysis",
                    "proficiency": "3"
                },
                {
                    "skillId": 82,
                    "skillName": "Network performance analysis",
                    "proficiency": "9"
                },
                {
                    "skillId": 83,
                    "skillName": "Network monitoring tools",
                    "proficiency": "6"
                },
                {
                    "skillId": 84,
                    "skillName": "Troubleshooting",
                    "proficiency": "5"
                },
                {
                    "skillId": 85,
                    "skillName": "Capacity planning",
                    "proficiency": "7"
                },
                {
                    "skillId": 86,
                    "skillName": "Network protocols",
                    "proficiency": "2"
                }
            ],
            "languages": [],
            "preferences": [
                {
                    "preferencesId": 2,
                    "preference": "Part-Time"
                }
            ],
            "degrees": [],
            "jobRole": "Network Performance Analyst",
            "about": "Network Performance Analysts monitor and optimize network performance. They collect and analyze network data, identify issues, and implement solutions to enhance network speed, reliability, and efficiency.",
            "noOfOpenings": 9,
            "salaryMin": 58000.00,
            "salaryMax": 93000.00,
            "recruiterId": 23,
            "closingDate": 1714242600000,
            "jobCategory": "Network Analysis",
            "requiredHighestEducation": "Graduate",
            "postingDate": 1709145000000,
            "requiredGender": "F",
            "company": "Uflex Ltd",
            "companyId": 21
        },
        {
            "jobId": 13,
            "requiredExperienceMin": 4,
            "requiredExperienceMax": 9,
            "requiredAgeMin": 21,
            "requiredAgeMax": 40,
            "locations": [],
            "skills": [
                {
                    "skillId": 129,
                    "skillName": "Data integration",
                    "proficiency": "2"
                },
                {
                    "skillId": 130,
                    "skillName": "ETL (Extract, Transform, Load)",
                    "proficiency": "8"
                },
                {
                    "skillId": 131,
                    "skillName": "Big data technologies",
                    "proficiency": "6"
                },
                {
                    "skillId": 132,
                    "skillName": "Database management",
                    "proficiency": "4"
                },
                {
                    "skillId": 133,
                    "skillName": "Data warehousing",
                    "proficiency": "5"
                }
            ],
            "languages": [],
            "preferences": [
                {
                    "preferencesId": 9,
                    "preference": "Temporary"
                }
            ],
            "degrees": [],
            "jobRole": "Data Engineer",
            "about": "Data Engineers design and maintain data pipelines, ensuring data availability and quality for analysis and reporting purposes.",
            "noOfOpenings": 5,
            "salaryMin": 59000.00,
            "salaryMax": 104000.00,
            "recruiterId": 24,
            "closingDate": 1714156200000,
            "jobCategory": "Data Science",
            "requiredHighestEducation": "Graduate",
            "postingDate": 1707849000000,
            "requiredGender": "M",
            "company": "State Bank of India (SBI)",
            "companyId": 22
        },
        {
            "jobId": 14,
            "requiredExperienceMin": 3,
            "requiredExperienceMax": 11,
            "requiredAgeMin": 21,
            "requiredAgeMax": 40,
            "locations": [],
            "skills": [
                {
                    "skillId": 84,
                    "skillName": "Troubleshooting",
                    "proficiency": "7"
                },
                {
                    "skillId": 134,
                    "skillName": "Embedded systems",
                    "proficiency": "6"
                },
                {
                    "skillId": 135,
                    "skillName": "Electronics design",
                    "proficiency": "9"
                },
                {
                    "skillId": 136,
                    "skillName": "PCB layout",
                    "proficiency": "6"
                },
                {
                    "skillId": 137,
                    "skillName": "Circuit analysis",
                    "proficiency": "5"
                }
            ],
            "languages": [],
            "preferences": [
                {
                    "preferencesId": 6,
                    "preference": "Contract"
                }
            ],
            "degrees": [],
            "jobRole": "Electronics Hardware Engineer",
            "about": "Electronics Hardware Engineers develop and design electronic components and systems, from circuit boards to hardware prototypes, ensuring functionality and performance.",
            "noOfOpenings": 7,
            "salaryMin": 65000.00,
            "salaryMax": 130000.00,
            "recruiterId": 25,
            "closingDate": 1714069800000,
            "jobCategory": "Electrical Engineering",
            "requiredHighestEducation": "Graduate",
            "postingDate": 1706553000000,
            "requiredGender": "M",
            "company": "Wipro Limited",
            "companyId": 23
        },
        {
            "jobId": 15,
            "requiredExperienceMin": 5,
            "requiredExperienceMax": 8,
            "requiredAgeMin": 21,
            "requiredAgeMax": 40,
            "locations": [],
            "skills": [
                {
                    "skillId": 7,
                    "skillName": "Python",
                    "proficiency": "5"
                },
                {
                    "skillId": 20,
                    "skillName": "Java",
                    "proficiency": "4"
                },
                {
                    "skillId": 138,
                    "skillName": "Scripting languages",
                    "proficiency": "2"
                },
                {
                    "skillId": 139,
                    "skillName": "Test automation tools",
                    "proficiency": "3"
                },
                {
                    "skillId": 140,
                    "skillName": "Test framework development",
                    "proficiency": "2"
                },
                {
                    "skillId": 141,
                    "skillName": "Continuous integration tools",
                    "proficiency": "8"
                },
                {
                    "skillId": 142,
                    "skillName": "Test data management",
                    "proficiency": "6"
                },
                {
                    "skillId": 145,
                    "skillName": "Selenium",
                    "proficiency": "6"
                }
            ],
            "languages": [],
            "preferences": [
                {
                    "preferencesId": 6,
                    "preference": "Contract"
                }
            ],
            "degrees": [],
            "jobRole": "Automation Tester",
            "about": "An Automation Tester uses automated testing tools to verify the functionality and performance of software applications. They create and execute automated test scripts.",
            "noOfOpenings": 6,
            "salaryMin": 64000.00,
            "salaryMax": 84000.00,
            "recruiterId": 26,
            "closingDate": 1713983400000,
            "jobCategory": "Software Testing",
            "requiredHighestEducation": "Graduate",
            "postingDate": 1705257000000,
            "requiredGender": "M",
            "company": "ICICI Bank",
            "companyId": 24
        },
        {
            "jobId": 16,
            "requiredExperienceMin": 1,
            "requiredExperienceMax": 13,
            "requiredAgeMin": 21,
            "requiredAgeMax": 40,
            "locations": [],
            "skills": [
                {
                    "skillId": 6,
                    "skillName": "HTML",
                    "proficiency": "9"
                },
                {
                    "skillId": 8,
                    "skillName": "Javascript",
                    "proficiency": "6"
                },
                {
                    "skillId": 9,
                    "skillName": "CSS",
                    "proficiency": "5"
                },
                {
                    "skillId": 105,
                    "skillName": "Responsive design",
                    "proficiency": "7"
                },
                {
                    "skillId": 143,
                    "skillName": "Frontend frameworks",
                    "proficiency": "2"
                },
                {
                    "skillId": 146,
                    "skillName": "React JS",
                    "proficiency": "3"
                },
                {
                    "skillId": 147,
                    "skillName": "Angular JS",
                    "proficiency": "2"
                }
            ],
            "languages": [],
            "preferences": [
                {
                    "preferencesId": 6,
                    "preference": "Contract"
                }
            ],
            "degrees": [],
            "jobRole": "Frontend Web Developer",
            "about": "Frontend Web Developers design and implement user interfaces for websites, ensuring they are visually appealing and user-friendly. They collaborate with designers and backend developers to create seamless web experiences for users.",
            "noOfOpenings": 8,
            "salaryMin": 57000.00,
            "salaryMax": 90000.00,
            "recruiterId": 27,
            "closingDate": 1713897000000,
            "jobCategory": "Web Developement",
            "requiredHighestEducation": "Graduate",
            "postingDate": 1703961000000,
            "requiredGender": "A",
            "company": "ITC Limited",
            "companyId": 25
        }
    ]

# Import necessary libraries
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertTokenizer, BertModel
import torch
import numpy as np
import json

# Function to calculate skill similarity with proficiency
def calculate_skill_similarity_with_proficiency(candidate_skills, job_skills):
    skill_similarities = []
    for candidate_skill in candidate_skills:
        for job_skill in job_skills:
            if candidate_skill['skillName'] == job_skill['skillName']:
                # Check for missing proficiency levels
                if candidate_skill['proficiency'] is None or job_skill['proficiency'] is None:
                    continue
                # Convert proficiency levels to integers
                candidate_proficiency = int(candidate_skill['proficiency'])
                job_proficiency = int(job_skill['proficiency'])
                # Calculate similarity based on proficiency level
                proficiency_diff = abs(candidate_proficiency - job_proficiency)
                similarity_score = 1 / (1 + proficiency_diff)  # Higher proficiency leads to higher similarity
                skill_similarities.append(similarity_score)
                break
    # If no common skills found, return 0 similarity for all metrics
    if not skill_similarities:
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

# Function to calculate language similarity
def calculate_language_similarity(candidate_languages, job_languages):
    if not candidate_languages or not job_languages:
        return {
            "cosineSimilarity": 0.0,
            "bertEmbeddingsSimilarity": 0.0,
            "jaccardSimilarity": 0.0,
            "manhattanSimilarity": 0.0
        }
    candidate_lang_set = set(candidate_languages)
    job_lang_set = set(job_languages)
    intersection = len(candidate_lang_set & job_lang_set)
    union = len(candidate_lang_set | job_lang_set)
    jaccard_sim = np.float64(intersection / union)
    return {
        "cosineSimilarity": 0.0,  # Placeholder, cosine similarity calculation is not provided
        "bertEmbeddingsSimilarity": 0.0,  # Placeholder, BERT embeddings similarity calculation is not provided
        "jaccardSimilarity": jaccard_sim,
        "manhattanSimilarity": 0.0  # Placeholder, Manhattan similarity calculation is not provided
    }

# Function to calculate preference similarity
def calculate_preference_similarity(candidate_preferences, job_preferences):
    if not candidate_preferences or not job_preferences:
        return {
            "cosineSimilarity": 0.0,
            "bertEmbeddingsSimilarity": 0.0,
            "jaccardSimilarity": 0.0,
            "manhattanSimilarity": 0.0
        }
    candidate_pref_set = set(candidate_preferences)
    job_pref_set = set(job_preferences)
    intersection = len(candidate_pref_set & job_pref_set)
    union = len(candidate_pref_set | job_pref_set)
    jaccard_sim = np.float64(intersection / union)
    return {
        "cosineSimilarity": 0.0,  # Placeholder, cosine similarity calculation is not provided
        "bertEmbeddingsSimilarity": 0.0,  # Placeholder, BERT embeddings similarity calculation is not provided
        "jaccardSimilarity": jaccard_sim,
        "manhattanSimilarity": 0.0  # Placeholder, Manhattan similarity calculation is not provided
    }

# Function to calculate job similarities
def calculate_job_similarities(candidate, job):
    candidate_skills = [skill['skillName'] for skill in candidate["skills"]]
    job_skills = [skill['skillName'] for skill in job["skills"]]
    skill_similarity = calculate_skill_similarity_with_proficiency(candidate["skills"], job["skills"])
    language_similarity = calculate_language_similarity(candidate["languages"], job["languages"])
    # preference_similarity = calculate_preference_similarity(candidate["jobPreferences"], job["preferences"])
    # Calculate overall similarity score
    overall_similarity = np.mean([
        skill_similarity["cosineSimilarity"],
        language_similarity["cosineSimilarity"],
        # preference_similarity["cosineSimilarity"]
    ])
    return {
        "jobId": job["jobId"],
        "skillSimilarityScore": skill_similarity,
        "languageSimilarityScore": language_similarity,
        # "preferenceSimilarityScore": preference_similarity,
        "overallSimilarityScore": round(overall_similarity, 5)
    }

# Main function to recommend jobs for a candidate
def recommend_jobs_for_candidate(candidate, job_data):
    similarities = []
    for job in job_data:
        similarities.append(calculate_job_similarities(candidate, job))
    # Sort jobs by overall similarity in descending order
    similarities.sort(key=lambda x: x["overallSimilarityScore"], reverse=True)
    return similarities

recommended_jobs = recommend_jobs_for_candidate(candidate_data[1], job_data)
print(json.dumps(recommended_jobs, indent=4, default=str))
