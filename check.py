import time
import os
import re
import json
import torch
import requests
import numpy as np
import pandas as pd
from flask import jsonify
from scipy.spatial import distance
from collections import defaultdict
from datetime import datetime, timezone
from sklearn.metrics import pairwise_distances
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import csv

data_objects = []
# DATASET
job_data = [
    {
        'jobId': 3,
        'requiredExperienceMin': 1,
        'requiredExperienceMax': 10,
        'requiredAgeMin': 21,
        'requiredAgeMax': 40,
        'locations': ['Banglore'],
        'skills': [
            {'skillName': 'Data Analysis', 'proficiency': '7'},
            {'skillName': 'Network performance analysis', 'proficiency': '5'},
            {'skillName': 'Network monitoring tools', 'proficiency': '6'},
            {'skillName': 'Troubleshooting', 'proficiency': '9'},
            {'skillName': 'Capacity planning', 'proficiency': '6'},
            {'skillName': 'Network protocols', 'proficiency': '5'}
        ],
        'languages': ['English'],
        'preferences': ['Full-Time'],
        'requiredHighestEducation': 'Graduate',
        'requiredGender': 'Male'
    },

    {
        'jobId': 4,
        'requiredExperienceMin': 3,
        'requiredExperienceMax': 14,
        'requiredAgeMin': 21,
        'requiredAgeMax': 40,
        'locations': [],
        'skills': [
            {'skillName': 'Performance testing', 'proficiency': '2'},
            {'skillName': 'Load testing', 'proficiency': '3'},
            {'skillName': 'Stress testing', 'proficiency': '2'},
            {'skillName': 'Test scenarios', 'proficiency': '8'},
            {'skillName': 'Performance monitoring', 'proficiency': '6'},
            {'skillName': 'Benchmarking', 'proficiency': '4'},
            {'skillName': 'Performance analysis', 'proficiency': '5'}
        ],
        'languages': [],
        'preferences': ['Contract'],
        'requiredHighestEducation': 'Graduate',
        'requiredGender': 'Male or Female'
    },

    {
        'jobId': 7,
        'requiredExperienceMin': 3,
        'requiredExperienceMax': 12,
        'requiredAgeMin': 21,
        'requiredAgeMax': 40,
        'locations': [],
        'skills': [
            {'skillName': 'Backend development', 'proficiency': '6'},
            {'skillName': 'RESTful APIs', 'proficiency': '9'},
            {'skillName': 'Database integration', 'proficiency': '6'},
            {'skillName': 'Performance optimization', 'proficiency': '5'},
            {'skillName': 'Version control', 'proficiency': '2'},
            {'skillName': 'JAVA EE', 'proficiency': '7'}
        ],
        'languages': [],
        'preferences': ['Intern'],
        'requiredHighestEducation': 'Graduate',
        'requiredGender': 'Male'
    },

    {
        'jobId': 8,
        'requiredExperienceMin': 4,
        'requiredExperienceMax': 15,
        'requiredAgeMin': 21,
        'requiredAgeMax': 40,
        'locations': [],
        'skills': [
            {'skillName': 'HTML', 'proficiency': '6'},
            {'skillName': 'Javascript', 'proficiency': '4'},
            {'skillName': 'CSS', 'proficiency': '5'},
            {'skillName': 'Responsive design', 'proficiency': '6'},
            {'skillName': 'Front-end web development', 'proficiency': '3'},
            {'skillName': 'Web performance optimization', 'proficiency': '2'},
            {'skillName': 'Cross-browser compatibility', 'proficiency': '8'}
        ],
        'languages': [],
        'preferences': ['Full-Time'],
        'requiredHighestEducation': 'Graduate',
        'requiredGender': 'Male'
    },

    {
        'jobId': 9,
        'requiredExperienceMin': 3,
        'requiredExperienceMax': 15,
        'requiredAgeMin': 21,
        'requiredAgeMax': 40,
        'locations': [],
        'skills': [
            {'skillName': 'UI design principles and best practices', 'proficiency': '9'},
            {'skillName': 'Graphic design tools', 'proficiency': '6'},
            {'skillName': 'Typography and color theory', 'proficiency': '5'},
            {'skillName': 'Visual design and layout', 'proficiency': '7'},
            {'skillName': 'Responsive design', 'proficiency': '2'}
        ],
        'languages': [],
        'preferences': ['Full-Time'],
        'requiredHighestEducation': 'Post Graduate',
        'requiredGender': 'Male'
    },

    {
        'jobId': 10,
        'requiredExperienceMin': 4,
        'requiredExperienceMax': 12,
        'requiredAgeMin': 21,
        'requiredAgeMax': 40,
        'locations': [],
        'skills': [
            {'skillName': 'User-centered design principles', 'proficiency': '3'},
            {'skillName': 'UX/UI design tools', 'proficiency': '2'},
            {'skillName': 'Wireframing and prototyping', 'proficiency': '8'},
            {'skillName': 'Usability testing and user research', 'proficiency': '6'},
            {'skillName': 'Information architecture and user flows', 'proficiency': '4'}
        ],
        'languages': [],
        'preferences': ['Full-Time'],
        'requiredHighestEducation': 'Graduate',
        'requiredGender': 'Male or Female'
    },

    {'jobId': 11,
     'requiredExperienceMin': 3,
     'requiredExperienceMax': 13,
     'requiredAgeMin': 21,
     'requiredAgeMax': 40,
     'locations': [],
     'skills': [
         {'skillName': 'HTML', 'proficiency': '6'},
         {'skillName': 'Javascript', 'proficiency': '4'},
         {'skillName': 'CSS', 'proficiency': '5'},
         {'skillName': 'Responsive design', 'proficiency': '6'},
         {'skillName': 'User interface (UI) design', 'proficiency': '5'},
         {'skillName': 'User experience (UX) design', 'proficiency': '6'},
         {'skillName': 'Web design principles', 'proficiency': '9'},
         {'skillName': 'Prototyping and wireframing', 'proficiency': '6'},
         {'skillName': 'Front-end development', 'proficiency': '5'},
         {'skillName': 'Interaction design', 'proficiency': '7'},
         {'skillName': 'User testing', 'proficiency': '2'},
         {'skillName': 'Usability testing', 'proficiency': '3'},
         {'skillName': 'Collaboration', 'proficiency': '2'},
         {'skillName': 'Attention to detail', 'proficiency': '8'}
     ],
     'languages': [],
     'preferences': ['Contract'],
     'requiredHighestEducation': 'Post Graduate',
     'requiredGender': 'Male or Female'
     },

    {
        'jobId': 5,
        'requiredExperienceMin': 4,
        'requiredExperienceMax': 14,
        'requiredAgeMin': 21,
        'requiredAgeMax': 40,
        'locations': [],
        'skills': [
            {'skillName': 'Security consulting', 'proficiency': '6'},
            {'skillName': 'Risk assessment', 'proficiency': '9'},
            {'skillName': 'Security audits', 'proficiency': '6'},
            {'skillName': 'Security policy', 'proficiency': '5'},
            {'skillName': 'development', 'proficiency': '7'},
            {'skillName': 'Penetration testing', 'proficiency': '2'},
            {'skillName': 'Client communication', 'proficiency': '3'}
        ],
        'languages': [],
        'preferences': ['Contract'],
        'requiredHighestEducation': 'Post Graduate',
        'requiredGender': 'Male or Female'
    },

    {
        'jobId': 13,
        'requiredExperienceMin': 4,
        'requiredExperienceMax': 9,
        'requiredAgeMin': 21,
        'requiredAgeMax': 40,
        'locations': [],
        'skills': [
            {'skillName': 'Data integration', 'proficiency': '2'},
            {'skillName': 'ETL (Extract, Transform, Load)', 'proficiency': '8'},
            {'skillName': 'Big data technologies', 'proficiency': '6'},
            {'skillName': 'Database management', 'proficiency': '4'},
            {'skillName': 'Data warehousing', 'proficiency': '5'}
        ],
        'languages': [],
        'preferences': ['Temporary'],
        'requiredHighestEducation': 'Graduate',
        'requiredGender': 'Male'
    },

    {
        'jobId': 14,
        'requiredExperienceMin': 3,
        'requiredExperienceMax': 11,
        'requiredAgeMin': 21,
        'requiredAgeMax': 40,
        'locations': [],
        'skills': [
            {'skillName': 'Troubleshooting', 'proficiency': '7'},
            {'skillName': 'Embedded systems', 'proficiency': '6'},
            {'skillName': 'Electronics design', 'proficiency': '9'},
            {'skillName': 'PCB layout', 'proficiency': '6'},
            {'skillName': 'Circuit analysis', 'proficiency': '5'}
        ],
        'languages': [],
        'preferences': ['Contract'],
        'requiredHighestEducation': 'Graduate',
        'requiredGender': 'Male'
    },

    {
        'jobId': 15,
        'requiredExperienceMin': 5,
        'requiredExperienceMax': 8,
        'requiredAgeMin': 21,
        'requiredAgeMax': 40,
        'locations': [],
        'skills': [
            {'skillName': 'Python', 'proficiency': '5'},
            {'skillName': 'Java', 'proficiency': '4'},
            {'skillName': 'Scripting languages', 'proficiency': '2'},
            {'skillName': 'Test automation tools', 'proficiency': '3'},
            {'skillName': 'Test framework development', 'proficiency': '2'},
            {'skillName': 'Continuous integration tools', 'proficiency': '8'},
            {'skillName': 'Test data management', 'proficiency': '6'},
            {'skillName': 'Selenium', 'proficiency': '6'}
        ],
        'languages': [],
        'preferences': ['Contract'],
        'requiredHighestEducation': 'Graduate',
        'requiredGender': 'Male'
    },

    {
        'jobId': 16,
        'requiredExperienceMin': 1,
        'requiredExperienceMax': 13,
        'requiredAgeMin': 21,
        'requiredAgeMax': 40,
        'locations': [],
        'skills': [
            {'skillName': 'HTML', 'proficiency': '9'},
            {'skillName': 'Javascript', 'proficiency': '6'},
            {'skillName': 'CSS', 'proficiency': '5'},
            {'skillName': 'Responsive design', 'proficiency': '7'},
            {'skillName': 'Frontend frameworks', 'proficiency': '2'},
            {'skillName': 'React JS', 'proficiency': '3'},
            {'skillName': 'Angular JS', 'proficiency': '2'}
        ],
        'languages': [],
        'preferences': ['Contract'],
        'requiredHighestEducation': 'Graduate',
        'requiredGender': 'Male or Female'
    }
]
candidate_data = [
    {'candidateId': 1,
     'candidateName': 'Vadiraj Inamdar',
     'gender': 'Male', 'age': 22,
     'education': "Bachelor's Computer Engineering ",
     'job_preferences': 'Full time, Day Shift, Work from office, Work from home',
     'languages': 'English',
     'skills': {
         'C': '9', 'C++': '9', 'Java': '6', 'Python': '6', 'HTML': '8', 'CSS': '8', 'JavaScript': '8', 'Php': '4',
         'MySQL': '8'},
     'previous_job_roles': {'Software Developer': 2},
     'groundTruth': [{'jobId': 'j8', 'preference': 10.0}, {'jobId': 'j9', 'preference': 10.0},
                     {'jobId': 'j10', 'preference': 10.0}, {'jobId': 'j11', 'preference': 10.0},
                     {'jobId': 'j16', 'preference': 10.0}, {'jobId': 'j7', 'preference': 9.0},
                     {'jobId': 'j15', 'preference': 9.0}, {'jobId': 'j6', 'preference': 8.0}
                     ]
     },
    {'candidateId': 2, 'candidateName': 'Ivan Azim', 'gender': 'Male', 'age': 25, 'education': "Bachelor's Computer Engineering", 'job_preferences': 'Full time, Day Shift', 'languages': 'English', 'skills': {'Laravel': '6', 'Php': '6', 'Html': '9', 'CSS': '9', 'JavaScript': '7', 'Bootstrap': '8', 'C++': '5', 'GitHub': '6'}, 'previous_job_roles': {'Software Developer': 2}, 'groundTruth': [{'jobId': 'j6', 'preference': 9.0}, {'jobId': 'j8', 'preference': 9.0}, {'jobId': 'j9', 'preference': 9.0}, {'jobId': 'j10', 'preference': 9.0}, {'jobId': 'j3', 'preference': 8.0}, {'jobId': 'j7', 'preference': 8.0}, {'jobId': 'j11', 'preference': 8.0}, {'jobId': 'j16', 'preference': 8.0}, {'jobId': 'j1', 'preference': 7.0}, {'jobId': 'j4', 'preference': 7.0}, {'jobId': 'j12', 'preference': 6.0}, {'jobId': 'j2', 'preference': 4.0}, {'jobId': 'j5', 'preference': 4.0}, {'jobId': 'j13', 'preference': 4.0}, {'jobId': 'j15', 'preference': 4.0}, {'jobId': 'j14', 'preference': 1.0}]}, {'candidateId': 3, 'candidateName': 'Chinmay Joshi', 'gender': 'Male', 'age': 21, 'education': "Bachelor's Computer Engineering", 'job_preferences': 'Full time, Day Shift, Work from office, Work from home', 'languages': 'English', 'skills': {'Java': '8', 'Python': '5', 'CSS': '5', 'HTML': '10', 'JavaScript': '5'}, 'previous_job_roles': {'Software Developer': 2}, 'groundTruth': [{'jobId': 'j7', 'preference': 10.0}, {'jobId': 'j15', 'preference': 10.0}, {'jobId': 'j8', 'preference': 9.0}, {'jobId': 'j10', 'preference': 8.0}, {'jobId': 'j11', 'preference': 8.0}, {'jobId': 'j16', 'preference': 8.0}, {'jobId': 'j4', 'preference': 7.0}, {'jobId': 'j12', 'preference': 7.0}, {'jobId': 'j13', 'preference': 6.0}, {'jobId': 'j3', 'preference': 5.0}, {'jobId': 'j5', 'preference': 4.0}, {'jobId': 'j6', 'preference': 3.0}, {'jobId': 'j9', 'preference': 3.0}]}, {'candidateId': 4, 'candidateName': 'Luke costa', 'gender': 'Male', 'age': 22, 'education': "Bachelor's Computer Engineering ", 'job_preferences': 'Full time, Work from office, Work from home', 'languages': 'English', 'skills': {'Java': '6', 'Python': '3', 'AutoCAD': '3', 'CSS': '3', 'Ms.Excel': '4', 'C++': '6', 'Angular': '5'}, 'previous_job_roles': {'Software Developer': 2}, 'groundTruth': [{'jobId': 'j6', 'preference': 10.0}, {'jobId': 'j7', 'preference': 10.0}, {'jobId': 'j8', 'preference': 10.0}, {'jobId': 'j11', 'preference': 10.0}, {'jobId': 'j16', 'preference': 10.0}, {'jobId': 'j4', 'preference': 9.0}, {'jobId': 'j5', 'preference': 9.0}, {'jobId': 'j9', 'preference': 9.0}, {'jobId': 'j12', 'preference': 9.0}, {'jobId': 'j2', 'preference': 8.0}, {'jobId': 'j3', 'preference': 8.0}, {'jobId': 'j10', 'preference': 8.0}, {'jobId': 'j13', 'preference': 8.0}, {'jobId': 'j14', 'preference': 8.0}, {'jobId': 'j15', 'preference': 8.0}, {'jobId': 'j1', 'preference': 7.0}]}, {'candidateId': 5, 'candidateName': 'shreyaa', 'gender': 'Female', 'age': 20, 'education': "Bachelor's electronics ", 'job_preferences': 'Work from office', 'languages': 'English', 'skills': {'esp32': '10', 'Arduino': '7', 'c': '7', 'c++': '5'}, 'previous_job_roles': {'Software Developer': 2}, 'groundTruth': [{'jobId': 'j2', 'preference': 9.0}, {'jobId': 'j8', 'preference': 9.0}, {'jobId': 'j9', 'preference': 9.0}, {'jobId': 'j11', 'preference': 9.0}, {'jobId': 'j13', 'preference': 9.0}, {'jobId': 'j15', 'preference': 9.0}, {'jobId': 'j1', 'preference': 8.0}, {'jobId': 'j3', 'preference': 8.0}, {'jobId': 'j6', 'preference': 8.0}, {'jobId': 'j10', 'preference': 8.0}, {'jobId': 'j12', 'preference': 8.0}, {'jobId': 'j14', 'preference': 8.0}, {'jobId': 'j16', 'preference': 8.0}, {'jobId': 'j4', 'preference': 7.0}, {'jobId': 'j7', 'preference': 7.0}, {'jobId': 'j5', 'preference': 6.0}]}, {'candidateId': 6, 'candidateName': 'Jonas Chris Ferrao', 'gender': 'Male', 'age': 21, 'education': "Bachelor's Computer Engineering", 'job_preferences': 'Full time, Part time, Internship, Night shift, Work from home', 'languages': 'English', 'skills': {'MachineLearning': '9', 'DataScience': '7', 'DeepLearning': '9', 'DataStructureandAlgorithms': '7', 'WebDevelopment': '6', 'Python': '8'}, 'previous_job_roles': {'Software Developer': 2}, 'groundTruth': [{'jobId': 'j13', 'preference': 9.0}, {'jobId': 'j4', 'preference': 7.0}, {'jobId': 'j8', 'preference': 7.0}, {'jobId': 'j3', 'preference': 6.0}, {'jobId': 'j5', 'preference': 6.0}, {'jobId': 'j6', 'preference': 6.0}, {'jobId': 'j7', 'preference': 6.0}, {'jobId': 'j9', 'preference': 6.0}, {'jobId': 'j10', 'preference': 6.0}, {'jobId': 'j11', 'preference': 6.0}, {'jobId': 'j12', 'preference': 6.0}, {'jobId': 'j15', 'preference': 6.0}, {'jobId': 'j16', 'preference': 5.0}, {'jobId': 'j1', 'preference': 4.0}, {'jobId': 'j14', 'preference': 4.0}, {'jobId': 'j2', 'preference': 1.0}]}, {'candidateId': 7, 'candidateName': 'Yagnesh Karwarker', 'gender': 'Male', 'age': 21, 'education': "Bachelor's Computer engineering", 'job_preferences': 'Full time, Internship', 'languages': 'English', 'skills': {'Python': '7', 'Mysql': '9', 'JavaScript': '9', 'Php': '7'}, 'previous_job_roles': {'Software Developer': 2}, 'groundTruth': [{'jobId': 'j12', 'preference': 10.0}, {'jobId': 'j4', 'preference': 9.0}, {'jobId': 'j13', 'preference': 9.0}, {'jobId': 'j15', 'preference': 9.0}, {'jobId': 'j16', 'preference': 9.0}, {'jobId': 'j3', 'preference': 8.0}, {'jobId': 'j5', 'preference': 8.0}, {'jobId': 'j6', 'preference': 8.0}, {'jobId': 'j8', 'preference': 8.0}, {'jobId': 'j9', 'preference': 8.0}, {'jobId': 'j10', 'preference': 8.0}, {'jobId': 'j11', 'preference': 8.0}, {'jobId': 'j7', 'preference': 7.0}, {'jobId': 'j1', 'preference': 5.0}, {'jobId': 'j14', 'preference': 5.0}, {'jobId': 'j2', 'preference': 4.0}]}, {'candidateId': 8, 'candidateName': 'Andrea Treeza Fernandes ', 'gender': 'Female', 'age': 20, 'education': "Bachelor's Computer Engineering ", 'job_preferences': 'Full time, Part time, Day Shift, Work from office, Internship, Work from home', 'languages': 'English', 'skills': {'Excel': '8'}, 'previous_job_roles': {'Software Developer': 2}, 'groundTruth': [{'jobId': 'j2', 'preference': 8.0}, {'jobId': 'j1', 'preference': 7.0}, {'jobId': 'j3', 'preference': 7.0}, {'jobId': 'j13', 'preference': 7.0}, {'jobId': 'j11', 'preference': 6.0}, {'jobId': 'j12', 'preference': 6.0}, {'jobId': 'j4', 'preference': 5.0}, {'jobId': 'j8', 'preference': 5.0}, {'jobId': 'j9', 'preference': 5.0}, {'jobId': 'j16', 'preference': 5.0}, {'jobId': 'j5', 'preference': 4.0}, {'jobId': 'j6', 'preference': 4.0}, {'jobId': 'j10', 'preference': 4.0}, {'jobId': 'j15', 'preference': 4.0}, {'jobId': 'j7', 'preference': 3.0}, {'jobId': 'j14', 'preference': 3.0}]}, {'candidateId': 9, 'candidateName': 'Nidi Nair ', 'gender': 'Female', 'age': 2019, 'education': "Bachelor's Computer Engineering ", 'job_preferences': 'Work from home', 'languages': 'English', 'skills': {}, 'previous_job_roles': {'Software Developer': 2}, 'groundTruth': [{'jobId': 'j5', 'preference': 4.0}, {'jobId': 'j13', 'preference': 4.0}, {'jobId': 'j1', 'preference': 3.0}, {'jobId': 'j3', 'preference': 3.0}, {'jobId': 'j4', 'preference': 3.0}, {'jobId': 'j12', 'preference': 3.0}, {'jobId': 'j6', 'preference': 2.0}, {'jobId': 'j7', 'preference': 2.0}, {'jobId': 'j8', 'preference': 2.0}, {'jobId': 'j9', 'preference': 2.0}, {'jobId': 'j10', 'preference': 2.0}, {'jobId': 'j15', 'preference': 2.0}, {'jobId': 'j2', 'preference': 1.0}, {'jobId': 'j11', 'preference': 1.0}, {'jobId': 'j14', 'preference': 1.0}, {'jobId': 'j16', 'preference': 1.0}]}, {'candidateId': 10, 'candidateName': 'Samuel Thomas Mesquita', 'gender': 'Male', 'age': 21, 'education': "Bachelor's Computer Engineering", 'job_preferences': 'Full time, Internship, Work from home', 'languages': 'English', 'skills': {'Python': '7', 'MsExcel': '4', 'Tableau': '1'}, 'previous_job_roles': {'Software Developer': 2}, 'groundTruth': [{'jobId': 'j13', 'preference': 8.0}, {'jobId': 'j3', 'preference': 5.0}, {'jobId': 'j5', 'preference': 5.0}, {'jobId': 'j12', 'preference': 5.0}, {'jobId': 'j15', 'preference': 4.0}, {'jobId': 'j4', 'preference': 2.0}, {'jobId': 'j1', 'preference': 1.0}, {'jobId': 'j2', 'preference': 1.0}, {'jobId': 'j6', 'preference': 1.0}, {'jobId': 'j7', 'preference': 1.0}, {'jobId': 'j8', 'preference': 1.0}, {'jobId': 'j9', 'preference': 1.0}, {'jobId': 'j10', 'preference': 1.0}, {'jobId': 'j11', 'preference': 1.0}, {'jobId': 'j14', 'preference': 1.0}, {'jobId': 'j16', 'preference': 1.0}]}, {'candidateId': 11, 'candidateName': 'Nikhil Rao', 'gender': 'Male', 'age': 21, 'education': "Bachelor's Computer Engineering", 'job_preferences': 'Full time, Work from office, Internship, Work from home', 'languages': 'English', 'skills': {'C++': '5', 'Python': '1'}, 'previous_job_roles': {'Software Developer': 2}, 'groundTruth': [{'jobId': 'j16', 'preference': 10.0}, {'jobId': 'j3', 'preference': 9.0}, {'jobId': 'j11', 'preference': 9.0}, {'jobId': 'j8', 'preference': 8.0}, {'jobId': 'j10', 'preference': 8.0}, {'jobId': 'j6', 'preference': 7.0}, {'jobId': 'j7', 'preference': 6.0}, {'jobId': 'j9', 'preference': 6.0}, {'jobId': 'j1', 'preference': 5.0}, {'jobId': 'j5', 'preference': 5.0}, {'jobId': 'j4', 'preference': 4.0}, {'jobId': 'j12', 'preference': 4.0}, {'jobId': 'j2', 'preference': 1.0}, {'jobId': 'j13', 'preference': 1.0}, {'jobId': 'j14', 'preference': 1.0}, {'jobId': 'j15', 'preference': 1.0}]}, {'candidateId': 12, 'candidateName': 'Nirupam Samant ', 'gender': 'Male', 'age': 24, 'education': "Master's Education ", 'job_preferences': 'Full time, Part time, Day Shift, Work from office, Internship, Night shift, Work from home', 'languages': 'English', 'skills': {}, 'previous_job_roles': {'Software Developer': 2}, 'groundTruth': [{'jobId': 'j8', 'preference': 9.0}, {'jobId': 'j3', 'preference': 6.0}, {'jobId': 'j4', 'preference': 6.0}, {'jobId': 'j1', 'preference': 5.0}, {'jobId': 'j10', 'preference': 5.0}, {'jobId': 'j2', 'preference': 4.0}, {'jobId': 'j5', 'preference': 4.0}, {'jobId': 'j9', 'preference': 4.0}, {'jobId': 'j14', 'preference': 4.0}, {'jobId': 'j15', 'preference': 4.0}, {'jobId': 'j6', 'preference': 3.0}, {'jobId': 'j7', 'preference': 3.0}, {'jobId': 'j13', 'preference': 3.0}, {'jobId': 'j11', 'preference': 1.0}, {'jobId': 'j12', 'preference': 1.0}, {'jobId': 'j16', 'preference': 1.0}]}, {'candidateId': 13, 'candidateName': 'Anushka Da Silva', 'gender': 'Female', 'age': 20, 'education': "Bachelor's Computer Engineering ", 'job_preferences': 'Part time', 'languages': 'English', 'skills': {'C++': '7', 'HTML': '5', 'CSS': '5', 'Python': '5', 'Figma': '5', 'C': '8'}, 'previous_job_roles': {'Software Developer': 2}, 'groundTruth': [{'jobId': 'j8', 'preference': 6.0}, {'jobId': 'j16', 'preference': 6.0}, {'jobId': 'j5', 'preference': 5.0}, {'jobId': 'j6', 'preference': 5.0}, {'jobId': 'j9', 'preference': 5.0}, {'jobId': 'j10', 'preference': 5.0}, {'jobId': 'j11', 'preference': 5.0}, {'jobId': 'j12', 'preference': 5.0}, {'jobId': 'j13', 'preference': 5.0}, {'jobId': 'j3', 'preference': 4.0}, {'jobId': 'j4', 'preference': 4.0}, {'jobId': 'j1', 'preference': 3.0}, {'jobId': 'j2', 'preference': 3.0}, {'jobId': 'j7', 'preference': 2.0}, {'jobId': 'j14', 'preference': 1.0}, {'jobId': 'j15', 'preference': 1.0}]}, {'candidateId': 14, 'candidateName': 'Atharv Batule ', 'gender': 'Male', 'age': 18, 'education': "Bachelor's Computer Engineering ", 'job_preferences': 'Full time', 'languages': 'English', 'skills': {'C': '8', 'Python': '5', 'Php': '5', 'Css': '4', 'PowerPoint': '8', 'Word': '8', 'MsExcel': '8'}, 'previous_job_roles': {'Software Developer': 2}, 'groundTruth': [{'jobId': 'j8', 'preference': 8.0}, {'jobId': 'j15', 'preference': 8.0}, {'jobId': 'j16', 'preference': 8.0}, {'jobId': 'j3', 'preference': 7.0}, {'jobId': 'j7', 'preference': 7.0}, {'jobId': 'j1', 'preference': 6.0}, {'jobId': 'j5', 'preference': 6.0}, {'jobId': 'j6', 'preference': 6.0}, {'jobId': 'j4', 'preference': 5.0}, {'jobId': 'j10', 'preference': 5.0}, {'jobId': 'j12', 'preference': 5.0}, {'jobId': 'j2', 'preference': 4.0}, {'jobId': 'j9', 'preference': 4.0}, {'jobId': 'j11', 'preference': 4.0}, {'jobId': 'j13', 'preference': 3.0}, {'jobId': 'j14', 'preference': 1.0}]}, {'candidateId': 15, 'candidateName': 'Aryan Kotru', 'gender': 'Male', 'age': 22, 'education': "Bachelor's Computer Engineering", 'job_preferences': 'Full time', 'languages': 'English', 'skills': {'Python': '8', 'C++': '6', 'Unity': '7', 'MySQL': '8', 'Neo4j': '5'}, 'previous_job_roles': {'Software Developer': 2}, 'groundTruth': [{'jobId': 'j3', 'preference': 8.0}, {'jobId': 'j5', 'preference': 8.0}, {'jobId': 'j12', 'preference': 8.0}, {'jobId': 'j4', 'preference': 6.0}, {'jobId': 'j11', 'preference': 6.0}, {'jobId': 'j6', 'preference': 5.0}, {'jobId': 'j10', 'preference': 5.0}, {'jobId': 'j13', 'preference': 5.0}, {'jobId': 'j9', 'preference': 4.0}, {'jobId': 'j16', 'preference': 4.0}, {'jobId': 'j1', 'preference': 3.0}, {'jobId': 'j7', 'preference': 3.0}, {'jobId': 'j2', 'preference': 2.0}, {'jobId': 'j8', 'preference': 2.0}, {'jobId': 'j14', 'preference': 1.0}, {'jobId': 'j15', 'preference': 1.0}]}, {'candidateId': 16, 'candidateName': 'Luke Fernandes ', 'gender': 'Male', 'age': 18, 'education': "Bachelor's Computer Engineering", 'job_preferences': 'Part time, Day Shift, Night shift, Work from home', 'languages': 'English', 'skills': {}, 'previous_job_roles': {'Software Developer': 2}, 'groundTruth': [{'jobId': 'j3', 'preference': 10.0}, {'jobId': 'j6', 'preference': 10.0}, {'jobId': 'j1', 'preference': 9.0}, {'jobId': 'j2', 'preference': 9.0}, {'jobId': 'j5', 'preference': 9.0}, {'jobId': 'j7', 'preference': 9.0}, {'jobId': 'j8', 'preference': 9.0}, {'jobId': 'j9', 'preference': 9.0}, {'jobId': 'j12', 'preference': 9.0}, {'jobId': 'j16', 'preference': 9.0}, {'jobId': 'j4', 'preference': 8.0}, {'jobId': 'j10', 'preference': 8.0}, {'jobId': 'j11', 'preference': 8.0}, {'jobId': 'j13', 'preference': 8.0}, {'jobId': 'j14', 'preference': 8.0}, {'jobId': 'j15', 'preference': 8.0}]}, {'candidateId': 17, 'candidateName': 'Veron Azim', 'gender': 'Male', 'age': 20, 'education': "Bachelor's Civil Engineering ", 'job_preferences': 'Part time, Work from office, Internship, Work from home', 'languages': 'English', 'skills': {'Python': '8'}, 'previous_job_roles': {'Software Developer': 2}, 'groundTruth': [{'jobId': 'j1', 'preference': 9.0}, {'jobId': 'j2', 'preference': 9.0}, {'jobId': 'j5', 'preference': 9.0}, {'jobId': 'j6', 'preference': 9.0}, {'jobId': 'j7', 'preference': 9.0}, {'jobId': 'j10', 'preference': 9.0}, {'jobId': 'j11', 'preference': 9.0}, {'jobId': 'j3', 'preference': 8.0}, {'jobId': 'j4', 'preference': 8.0}, {'jobId': 'j8', 'preference': 8.0}, {'jobId': 'j9', 'preference': 8.0}, {'jobId': 'j12', 'preference': 8.0}, {'jobId': 'j16', 'preference': 8.0}, {'jobId': 'j13', 'preference': 7.0}, {'jobId': 'j15', 'preference': 7.0}, {'jobId': 'j14', 'preference': 5.0}]}, {'candidateId': 18, 'candidateName': 'Vanshika Gupta ', 'gender': 'Female', 'age': 20, 'education': "Bachelor's Electronics and computer science ", 'job_preferences': 'Part time, Internship, Work from home', 'languages': 'English', 'skills': {'Java': '5', 'Python': '8', 'C': '8', 'C++': '7', 'Ms.Excel': '7'}, 'previous_job_roles': {'Software Developer': 2}, 'groundTruth': [{'jobId': 'j5', 'preference': 10.0}, {'jobId': 'j3', 'preference': 9.0}, {'jobId': 'j7', 'preference': 9.0}, {'jobId': 'j13', 'preference': 9.0}, {'jobId': 'j14', 'preference': 9.0}, {'jobId': 'j16', 'preference': 9.0}, {'jobId': 'j4', 'preference': 8.0}, {'jobId': 'j9', 'preference': 8.0}, {'jobId': 'j15', 'preference': 8.0}, {'jobId': 'j1', 'preference': 7.0}, {'jobId': 'j6', 'preference': 7.0}, {'jobId': 'j8', 'preference': 7.0}, {'jobId': 'j12', 'preference': 7.0}, {'jobId': 'j2', 'preference': 6.0}, {'jobId': 'j10', 'preference': 5.0}, {'jobId': 'j11', 'preference': 5.0}]}, {'candidateId': 19, 'candidateName': 'Elrich Afroy Colaco', 'gender': 'Male', 'age': 18, 'education': "Bachelor's Computer Engineering", 'job_preferences': 'Part time, Internship, Night shift, Work from home', 'languages': 'English', 'skills': {}, 'previous_job_roles': {'Software Developer': 2}, 'groundTruth': [{'jobId': 'j8', 'preference': 8.0}, {'jobId': 'j11', 'preference': 7.0}, {'jobId': 'j16', 'preference': 7.0}, {'jobId': 'j15', 'preference': 6.0}, {'jobId': 'j1', 'preference': 3.0}, {'jobId': 'j2', 'preference': 2.0}, {'jobId': 'j3', 'preference': 2.0}, {'jobId': 'j4', 'preference': 2.0}, {'jobId': 'j5', 'preference': 2.0}, {'jobId': 'j6', 'preference': 2.0}, {'jobId': 'j7', 'preference': 2.0}, {'jobId': 'j12', 'preference': 2.0}, {'jobId': 'j13', 'preference': 2.0}, {'jobId': 'j14', 'preference': 2.0}, {'jobId': 'j9', 'preference': 1.0}, {'jobId': 'j10', 'preference': 1.0}]}, {'candidateId': 20, 'candidateName': 'Hayden Cassiano Fernandes', 'gender': 'Male', 'age': 18, 'education': "Bachelor's Computer Engineering", 'job_preferences': 'Part time, Internship, Work from home', 'languages': 'English', 'skills': {'C': '9', 'C++': '10', 'Python': '5', 'HTML': '7', 'CSS': '5', 'IOT': '3', 'Canva': '7'}, 'previous_job_roles': {'Software Developer': 2}, 'groundTruth': [{'jobId': 'j4', 'preference': 8.0}, {'jobId': 'j8', 'preference': 7.0}, {'jobId': 'j11', 'preference': 7.0}, {'jobId': 'j16', 'preference': 7.0}, {'jobId': 'j3', 'preference': 6.0}, {'jobId': 'j12', 'preference': 6.0}, {'jobId': 'j1', 'preference': 5.0}, {'jobId': 'j5', 'preference': 4.0}, {'jobId': 'j15', 'preference': 4.0}, {'jobId': 'j9', 'preference': 3.0}, {'jobId': 'j10', 'preference': 3.0}, {'jobId': 'j6', 'preference': 2.0}, {'jobId': 'j13', 'preference': 2.0}, {'jobId': 'j2', 'preference': 1.0}, {'jobId': 'j7', 'preference': 1.0}, {'jobId': 'j14', 'preference': 1.0}]}, {'candidateId': 21, 'candidateName': 'TRIPURARI SINGH ', 'gender': 'Male', 'age': 19, 'education': "Bachelor's Civil engineering ", 'job_preferences': 'Full time, Work from office', 'languages': 'English', 'skills': {'Autocad': '7'}, 'previous_job_roles': {'Software Developer': 2}, 'groundTruth': [{'jobId': 'j2', 'preference': 7.0}, {'jobId': 'j1', 'preference': 6.0}, {'jobId': 'j4', 'preference': 6.0}, {'jobId': 'j6', 'preference': 4.0}, {'jobId': 'j13', 'preference': 3.0}, {'jobId': 'j3', 'preference': 2.0}, {'jobId': 'j5', 'preference': 2.0}, {'jobId': 'j10', 'preference': 2.0}, {'jobId': 'j12', 'preference': 2.0}, {'jobId': 'j16', 'preference': 2.0}, {'jobId': 'j7', 'preference': 1.0}, {'jobId': 'j8', 'preference': 1.0}, {'jobId': 'j9', 'preference': 1.0}, {'jobId': 'j11', 'preference': 1.0}, {'jobId': 'j14', 'preference': 1.0}, {'jobId': 'j15', 'preference': 1.0}]}, {'candidateId': 22, 'candidateName': 'Elivia Sweema Silva ', 'gender': 'Female', 'age': 18, 'education': "Bachelor's Electronic and computer science ", 'job_preferences': 'Part time, Internship, Work from home', 'languages': 'English', 'skills': {'C': '8', 'C++': '8'}, 'previous_job_roles': {'Software Developer': 2}, 'groundTruth': [{'jobId': 'j14', 'preference': 10.0}, {'jobId': 'j7', 'preference': 9.0}, {'jobId': 'j8', 'preference': 9.0}, {'jobId': 'j11', 'preference': 9.0}, {'jobId': 'j3', 'preference': 8.0}, {'jobId': 'j5', 'preference': 8.0}, {'jobId': 'j6', 'preference': 8.0}, {'jobId': 'j9', 'preference': 8.0}, {'jobId': 'j15', 'preference': 8.0}, {'jobId': 'j4', 'preference': 7.0}, {'jobId': 'j10', 'preference': 7.0}, {'jobId': 'j12', 'preference': 7.0}, {'jobId': 'j16', 'preference': 7.0}, {'jobId': 'j13', 'preference': 5.0}, {'jobId': 'j1', 'preference': 4.0}, {'jobId': 'j2', 'preference': 3.0}]}, {'candidateId': 23, 'candidateName': 'Jason Rodrigues ', 'gender': 'Male', 'age': 22, 'education': "Bachelor's Computer Engineering ", 'job_preferences': 'Part time, Day Shift, Work from office, Night shift, Work from home', 'languages': 'English', 'skills': {'Html': '6', 'C++': '5', 'Python': '5', 'Flutter': '6', 'Java': '5'}, 'previous_job_roles': {'Software Developer': 2}, 'groundTruth': [{'jobId': 'j1', 'preference': 7.0}, {'jobId': 'j3', 'preference': 7.0}, {'jobId': 'j15', 'preference': 7.0}, {'jobId': 'j5', 'preference': 6.0}, {'jobId': 'j7', 'preference': 6.0}, {'jobId': 'j9', 'preference': 6.0}, {'jobId': 'j14', 'preference': 6.0}, {'jobId': 'j2', 'preference': 5.0}, {'jobId': 'j6', 'preference': 5.0}, {'jobId': 'j8', 'preference': 5.0}, {'jobId': 'j10', 'preference': 5.0}, {'jobId': 'j12', 'preference': 5.0}, {'jobId': 'j13', 'preference': 5.0}, {'jobId': 'j16', 'preference': 5.0}, {'jobId': 'j4', 'preference': 4.0}, {'jobId': 'j11', 'preference': 3.0}]}, {'candidateId': 24, 'candidateName': 'RITESH DILKUSH CHARI ', 'gender': 'Male', 'age': 19, 'education': "Bachelor's Computer Engineering ", 'job_preferences': 'Part time, Internship, Work from home', 'languages': 'English', 'skills': {'Python': '8'}, 'previous_job_roles': {'Software Developer': 2}, 'groundTruth': [{'jobId': 'j1', 'preference': 10.0}, {'jobId': 'j2', 'preference': 10.0}, {'jobId': 'j4', 'preference': 10.0}, {'jobId': 'j5', 'preference': 10.0}, {'jobId': 'j6', 'preference': 10.0}, {'jobId': 'j7', 'preference': 10.0}, {'jobId': 'j8', 'preference': 10.0}, {'jobId': 'j9', 'preference': 10.0}, {'jobId': 'j10', 'preference': 10.0}, {'jobId': 'j11', 'preference': 10.0}, {'jobId': 'j12', 'preference': 10.0}, {'jobId': 'j13', 'preference': 10.0}, {'jobId': 'j14', 'preference': 10.0}, {'jobId': 'j15', 'preference': 10.0}, {'jobId': 'j16', 'preference': 10.0}, {'jobId': 'j3', 'preference': 9.0}]}, {'candidateId': 25, 'candidateName': 'Faizan Farid Akabani ', 'gender': 'Male', 'age': 19, 'education': "Bachelor's Computer Engineering ", 'job_preferences': 'Part time, Night shift, Work from home', 'languages': 'English', 'skills': {'Videography': '7'}, 'previous_job_roles': {'Software Developer': 2}, 'groundTruth': [{'jobId': 'j11', 'preference': 6.0}, {'jobId': 'j2', 'preference': 5.0}, {'jobId': 'j4', 'preference': 5.0}, {'jobId': 'j8', 'preference': 5.0}, {'jobId': 'j9', 'preference': 5.0}, {'jobId': 'j16', 'preference': 5.0}, {'jobId': 'j10', 'preference': 4.0}, {'jobId': 'j14', 'preference': 4.0}, {'jobId': 'j15', 'preference': 4.0}, {'jobId': 'j1', 'preference': 2.0}, {'jobId': 'j13', 'preference': 2.0}, {'jobId': 'j3', 'preference': 1.0}, {'jobId': 'j5', 'preference': 1.0}, {'jobId': 'j6', 'preference': 1.0}, {'jobId': 'j7', 'preference': 1.0}, {'jobId': 'j12', 'preference': 1.0}]}, {'candidateId': 26, 'candidateName': 'Shridevi Bharne ', 'gender': 'Female', 'age': 21, 'education': "Bachelor's Computer engineering ", 'job_preferences': 'Full time, Day Shift, Work from office, Work from home', 'languages': 'English', 'skills': {'Python': '8', 'Css': '9', 'Html': '8', 'Javascript': '6', 'C++': '9'}, 'previous_job_roles': {'Software Developer': 2}, 'groundTruth': [{'jobId': 'j2', 'preference': 8.0}, {'jobId': 'j3', 'preference': 7.0}, {'jobId': 'j6', 'preference': 7.0}, {'jobId': 'j8', 'preference': 7.0}, {'jobId': 'j4', 'preference': 6.0}, {'jobId': 'j11', 'preference': 6.0}, {'jobId': 'j12', 'preference': 6.0}, {'jobId': 'j1', 'preference': 5.0}, {'jobId': 'j5', 'preference': 5.0}, {'jobId': 'j9', 'preference': 5.0}, {'jobId': 'j13', 'preference': 5.0}, {'jobId': 'j14', 'preference': 5.0}, {'jobId': 'j16', 'preference': 5.0}, {'jobId': 'j7', 'preference': 4.0}, {'jobId': 'j10', 'preference': 4.0}, {'jobId': 'j15', 'preference': 4.0}]}, {'candidateId': 27, 'candidateName': 'SHRINIVAS.G.INAMDAR', 'gender': 'Male', 'age': 18, 'education': "Bachelor's Computer Engineering", 'job_preferences': 'Full time, Day Shift, Work from office, Work from home', 'languages': 'English', 'skills': {'C++': '8', 'Python': '5', 'HTML': '8', 'CSS': '8'}, 'previous_job_roles': {'Software Developer': 2}, 'groundTruth': [{'jobId': 'j6', 'preference': 8.0}, {'jobId': 'j5', 'preference': 7.0}, {'jobId': 'j8', 'preference': 7.0}, {'jobId': 'j11', 'preference': 4.0}]}, {'candidateId': 28, 'candidateName': 'Pranav Naik ', 'gender': 'Male', 'age': 20, 'education': "Bachelor's Computer Engineering ", 'job_preferences': 'Internship', 'languages': 'English', 'skills': {'Python': '9', 'AI/ML': '8', 'Cybersecurity': '7', 'C++': '10'}, 'previous_job_roles': {'Software Developer': 2}, 'groundTruth': [{'jobId': 'j5', 'preference': 8.0}, {'jobId': 'j13', 'preference': 8.0}, {'jobId': 'j15', 'preference': 7.0}, {'jobId': 'j3', 'preference': 5.0}, {'jobId': 'j4', 'preference': 5.0}, {'jobId': 'j7', 'preference': 5.0}, {'jobId': 'j6', 'preference': 4.0}, {'jobId': 'j8', 'preference': 4.0}, {'jobId': 'j10', 'preference': 4.0}, {'jobId': 'j12', 'preference': 4.0}, {'jobId': 'j9', 'preference': 3.0}, {'jobId': 'j11', 'preference': 2.0}, {'jobId': 'j16', 'preference': 2.0}, {'jobId': 'j1', 'preference': 1.0}, {'jobId': 'j2', 'preference': 1.0}, {'jobId': 'j14', 'preference': 1.0}]}, {'candidateId': 29, 'candidateName': 'Navin Morajkar ', 'gender': 'Male', 'age': 22, 'education': "Bachelor's Computer Engineering ", 'job_preferences': 'Full time', 'languages': 'English', 'skills': {'HTML': '10', 'CSS': '9', 'Javascript': '8'}, 'previous_job_roles': {'Software Developer': 2}, 'groundTruth': [{'jobId': 'j8', 'preference': 10.0}, {'jobId': 'j16', 'preference': 10.0}, {'jobId': 'j6', 'preference': 9.0}, {'jobId': 'j7', 'preference': 9.0}, {'jobId': 'j9', 'preference': 9.0}, {'jobId': 'j10', 'preference': 9.0}, {'jobId': 'j11', 'preference': 9.0}, {'jobId': 'j4', 'preference': 8.0}, {'jobId': 'j15', 'preference': 8.0}, {'jobId': 'j13', 'preference': 6.0}, {'jobId': 'j3', 'preference': 5.0}, {'jobId': 'j5', 'preference': 5.0}, {'jobId': 'j12', 'preference': 5.0}, {'jobId': 'j1', 'preference': 4.0}, {'jobId': 'j14', 'preference': 4.0}, {'jobId': 'j2', 'preference': 1.0}]}, {'candidateId': 30, 'candidateName': 'Skyla Barreto ', 'gender': 'Female', 'age': 21, 'education': "Bachelor's Computer Engineering ", 'job_preferences': 'Full time, Part time, Work from home', 'languages': 'English', 'skills': {'Css': '9', 'Java': '6', 'python': '7', 'React': '9'}, 'previous_job_roles': {'Software Developer': 2}, 'groundTruth': [{'jobId': 'j1', 'preference': 6.0}, {'jobId': 'j4', 'preference': 6.0}, {'jobId': 'j5', 'preference': 6.0}, {'jobId': 'j6', 'preference': 6.0}, {'jobId': 'j7', 'preference': 6.0}, {'jobId': 'j8', 'preference': 6.0}, {'jobId': 'j9', 'preference': 6.0}, {'jobId': 'j10', 'preference': 6.0}, {'jobId': 'j11', 'preference': 6.0}, {'jobId': 'j15', 'preference': 6.0}, {'jobId': 'j16', 'preference': 6.0}, {'jobId': 'j3', 'preference': 5.0}, {'jobId': 'j12', 'preference': 5.0}, {'jobId': 'j13', 'preference': 5.0}, {'jobId': 'j2', 'preference': 2.0}, {'jobId': 'j14', 'preference': 2.0}]}, {'candidateId': 31, 'candidateName': 'Aayush Sandeep Parab', 'gender': 'Male', 'age': 22, 'education': "Bachelor's Computer Engineering", 'job_preferences': 'Full time, Part time, Day Shift, Work from home', 'languages': 'English', 'skills': {'Java': '5'}, 'previous_job_roles': {'Software Developer': 2}, 'groundTruth': [{'jobId': 'j2', 'preference': 7.0}, {'jobId': 'j3', 'preference': 7.0}, {'jobId': 'j6', 'preference': 7.0}, {'jobId': 'j7', 'preference': 7.0}, {'jobId': 'j12', 'preference': 7.0}, {'jobId': 'j1', 'preference': 6.0}, {'jobId': 'j4', 'preference': 6.0}, {'jobId': 'j5', 'preference': 6.0}, {'jobId': 'j8', 'preference': 6.0}, {'jobId': 'j9', 'preference': 6.0}, {'jobId': 'j10', 'preference': 6.0}, {'jobId': 'j11', 'preference': 6.0}, {'jobId': 'j13', 'preference': 5.0}, {'jobId': 'j14', 'preference': 5.0}, {'jobId': 'j15', 'preference': 5.0}, {'jobId': 'j16', 'preference': 5.0}]}, {'candidateId': 32, 'candidateName': 'Gauravi Gajanan Kamat', 'gender': 'Female', 'age': 21, 'education': "Bachelor's Computer Engineering ", 'job_preferences': 'Full time, Part time, Day Shift, Work from office, Internship, Work from home', 'languages': 'English', 'skills': {}, 'previous_job_roles': {'Software Developer': 2}, 'groundTruth': [{'jobId': 'j6', 'preference': 10.0}, {'jobId': 'j8', 'preference': 10.0}, {'jobId': 'j9', 'preference': 10.0}, {'jobId': 'j10', 'preference': 10.0}, {'jobId': 'j16', 'preference': 10.0}, {'jobId': 'j11', 'preference': 9.0}, {'jobId': 'j15', 'preference': 9.0}, {'jobId': 'j7', 'preference': 7.0}, {'jobId': 'j3', 'preference': 5.0}, {'jobId': 'j4', 'preference': 5.0}, {'jobId': 'j12', 'preference': 5.0}, {'jobId': 'j13', 'preference': 5.0}, {'jobId': 'j5', 'preference': 3.0}, {'jobId': 'j1', 'preference': 2.0}, {'jobId': 'j2', 'preference': 2.0}, {'jobId': 'j14', 'preference': 2.0}]}, {'candidateId': 33, 'candidateName': 'Tejas', 'gender': 'Male', 'age': 22, 'education': "Bachelor's Computer ", 'job_preferences': 'Full time', 'languages': 'English', 'skills': {'Python': '5'}, 'previous_job_roles': {'Software Developer': 2}, 'groundTruth': [{'jobId': 'j1', 'preference': 8.0}, {'jobId': 'j2', 'preference': 6.0}, {'jobId': 'j3', 'preference': 6.0}, {'jobId': 'j4', 'preference': 6.0}, {'jobId': 'j5', 'preference': 6.0}, {'jobId': 'j6', 'preference': 6.0}, {'jobId': 'j7', 'preference': 6.0}, {'jobId': 'j8', 'preference': 6.0}, {'jobId': 'j9', 'preference': 6.0}, {'jobId': 'j10', 'preference': 6.0}, {'jobId': 'j11', 'preference': 6.0}, {'jobId': 'j12', 'preference': 6.0}, {'jobId': 'j13', 'preference': 6.0}, {'jobId': 'j14', 'preference': 6.0}, {'jobId': 'j15', 'preference': 6.0}, {'jobId': 'j16', 'preference': 6.0}]}, {'candidateId': 34, 'candidateName': 'Jaden Mascarenhas ', 'gender': 'Male', 'age': 21, 'education': "Bachelor's Computer ", 'job_preferences': 'Full time, Part time, Internship, Work from home', 'languages': 'English', 'skills': {}, 'previous_job_roles': {'Software Developer': 2}, 'groundTruth': [{'jobId': 'j12', 'preference': 8.0}, {'jobId': 'j5', 'preference': 7.0}, {'jobId': 'j15', 'preference': 7.0}, {'jobId': 'j1', 'preference': 6.0}, {'jobId': 'j2', 'preference': 6.0}, {'jobId': 'j8', 'preference': 6.0}, {'jobId': 'j11', 'preference': 6.0}, {'jobId': 'j14', 'preference': 6.0}, {'jobId': 'j3', 'preference': 5.0}, {'jobId': 'j6', 'preference': 5.0}, {'jobId': 'j7', 'preference': 5.0}, {'jobId': 'j9', 'preference': 5.0}, {'jobId': 'j10', 'preference': 5.0}, {'jobId': 'j13', 'preference': 5.0}, {'jobId': 'j16', 'preference': 5.0}, {'jobId': 'j4', 'preference': 3.0}]}, {'candidateId': 35, 'candidateName': 'Jayden Ferrao ', 'gender': 'Male', 'age': 21, 'education': "Bachelor's Computer engineering ", 'job_preferences': 'Full time, Day Shift, Work from office', 'languages': 'English', 'skills': {}, 'previous_job_roles': {'Software Developer': 2}, 'groundTruth': [{'jobId': 'j8', 'preference': 10.0}, {'jobId': 'j11', 'preference': 10.0}, {'jobId': 'j13', 'preference': 10.0}, {'jobId': 'j5', 'preference': 9.0}, {'jobId': 'j6', 'preference': 9.0}, {'jobId': 'j7', 'preference': 9.0}, {'jobId': 'j9', 'preference': 9.0}, {'jobId': 'j10', 'preference': 9.0}, {'jobId': 'j16', 'preference': 9.0}, {'jobId': 'j12', 'preference': 8.0}, {'jobId': 'j1', 'preference': 7.0}, {'jobId': 'j3', 'preference': 7.0}, {'jobId': 'j4', 'preference': 6.0}, {'jobId': 'j2', 'preference': 4.0}, {'jobId': 'j14', 'preference': 3.0}, {'jobId': 'j15', 'preference': 3.0}]}, {'candidateId': 36, 'candidateName': 'Iti sawami muti swami bal gopal iyer', 'gender': 'Male', 'age': 23, 'education': "Bachelor's Computer engineering", 'job_preferences': 'Full time', 'languages': 'English', 'skills': {}, 'previous_job_roles': {'Software Developer': 2}, 'groundTruth': [{'jobId': 'j6', 'preference': 10.0}, {'jobId': 'j8', 'preference': 10.0}, {'jobId': 'j10', 'preference': 8.0}, {'jobId': 'j11', 'preference': 8.0}, {'jobId': 'j9', 'preference': 7.0}, {'jobId': 'j16', 'preference': 6.0}, {'jobId': 'j1', 'preference': 1.0}, {'jobId': 'j2', 'preference': 1.0}, {'jobId': 'j3', 'preference': 1.0}, {'jobId': 'j4', 'preference': 1.0}, {'jobId': 'j5', 'preference': 1.0}, {'jobId': 'j7', 'preference': 1.0}, {'jobId': 'j12', 'preference': 1.0}, {'jobId': 'j13', 'preference': 1.0}, {'jobId': 'j14', 'preference': 1.0}, {'jobId': 'j15', 'preference': 1.0}]}, {'candidateId': 37, 'candidateName': 'Rawdon Arvino Jobais Noronha ', 'gender': 'Male', 'age': 21, 'education': "Bachelor's Computer Engineering ", 'job_preferences': 'Full time, Part time, Work from home', 'languages': 'English', 'skills': {}, 'previous_job_roles': {'Software Developer': 2}, 'groundTruth': [{'jobId': 'j6', 'preference': 10.0}, {'jobId': 'j7', 'preference': 10.0}, {'jobId': 'j8', 'preference': 10.0}, {'jobId': 'j16', 'preference': 10.0}, {'jobId': 'j3', 'preference': 9.0}, {'jobId': 'j11', 'preference': 9.0}, {'jobId': 'j13', 'preference': 9.0}, {'jobId': 'j5', 'preference': 8.0}, {'jobId': 'j9', 'preference': 8.0}, {'jobId': 'j12', 'preference': 8.0}, {'jobId': 'j15', 'preference': 8.0}, {'jobId': 'j4', 'preference': 7.0}, {'jobId': 'j10', 'preference': 6.0}, {'jobId': 'j1', 'preference': 5.0}, {'jobId': 'j14', 'preference': 3.0}, {'jobId': 'j2', 'preference': 1.0}]}, {'candidateId': 38, 'candidateName': 'Sanjana Vernekar ', 'gender': 'Female', 'age': 22, 'education': "Bachelor's Computer engineering ", 'job_preferences': 'Full time', 'languages': 'English', 'skills': {}, 'previous_job_roles': {'Software Developer': 2}, 'groundTruth': [{'jobId': 'j11', 'preference': 10.0}, {'jobId': 'j16', 'preference': 10.0}, {'jobId': 'j8', 'preference': 9.0}, {'jobId': 'j7', 'preference': 8.0}, {'jobId': 'j9', 'preference': 8.0}, {'jobId': 'j13', 'preference': 8.0}, {'jobId': 'j10', 'preference': 7.0}, {'jobId': 'j12', 'preference': 7.0}, {'jobId': 'j3', 'preference': 6.0}, {'jobId': 'j4', 'preference': 6.0}, {'jobId': 'j5', 'preference': 6.0}, {'jobId': 'j6', 'preference': 6.0}, {'jobId': 'j14', 'preference': 6.0}, {'jobId': 'j15', 'preference': 6.0}, {'jobId': 'j1', 'preference': 1.0}, {'jobId': 'j2', 'preference': 1.0}]}, {'candidateId': 39, 'candidateName': 'Rudra Kande ', 'gender': 'Male', 'age': 24, 'education': "Bachelor's Computer engineering ", 'job_preferences': 'Full time, Work from office, Work from home', 'languages': 'English', 'skills': {}, 'previous_job_roles': {'Software Developer': 2}, 'groundTruth': [{'jobId': 'j8', 'preference': 9.0}, {'jobId': 'j16', 'preference': 8.0}, {'jobId': 'j10', 'preference': 7.0}, {'jobId': 'j11', 'preference': 7.0}, {'jobId': 'j9', 'preference': 6.0}, {'jobId': 'j12', 'preference': 4.0}, {'jobId': 'j6', 'preference': 3.0}, {'jobId': 'j7', 'preference': 3.0}, {'jobId': 'j5', 'preference': 2.0}, {'jobId': 'j1', 'preference': 1.0}, {'jobId': 'j2', 'preference': 1.0}, {'jobId': 'j3', 'preference': 1.0}, {'jobId': 'j4', 'preference': 1.0}, {'jobId': 'j13', 'preference': 1.0}, {'jobId': 'j14', 'preference': 1.0}, {'jobId': 'j15', 'preference': 1.0}]}, {'candidateId': 40, 'candidateName': 'Priti Sahebgowda Desai', 'gender': 'Female', 'age': 25, 'education': "Bachelor's Computer Engineering", 'job_preferences': 'Full time, Day Shift', 'languages': 'English', 'skills': {'Python': '6'}, 'previous_job_roles': {'Software Developer': 2}, 'groundTruth': [{'jobId': 'j6', 'preference': 9.0}, {'jobId': 'j7', 'preference': 9.0}, {'jobId': 'j8', 'preference': 9.0}, {'jobId': 'j1', 'preference': 8.0}, {'jobId': 'j11', 'preference': 8.0}, {'jobId': 'j3', 'preference': 7.0}, {'jobId': 'j9', 'preference': 7.0}, {'jobId': 'j10', 'preference': 7.0}, {'jobId': 'j16', 'preference': 7.0}, {'jobId': 'j4', 'preference': 6.0}, {'jobId': 'j12', 'preference': 6.0}, {'jobId': 'j13', 'preference': 6.0}, {'jobId': 'j14', 'preference': 6.0}, {'jobId': 'j15', 'preference': 6.0}, {'jobId': 'j2', 'preference': 5.0}, {'jobId': 'j5', 'preference': 5.0}]}, {'candidateId': 41, 'candidateName': 'Duval Gomes', 'gender': 'Male', 'age': 24, 'education': "Bachelor's Computer Engineering ", 'job_preferences': 'Full time, Part time, Day Shift, Work from office, Internship, Night shift, Work from home', 'languages': 'English', 'skills': {}, 'previous_job_roles': {'Software Developer': 2}, 'groundTruth': [{'jobId': 'j3', 'preference': 8.0}, {'jobId': 'j12', 'preference': 8.0}, {'jobId': 'j4', 'preference': 6.0}, {'jobId': 'j1', 'preference': 5.0}, {'jobId': 'j5', 'preference': 5.0}, {'jobId': 'j6', 'preference': 5.0}, {'jobId': 'j11', 'preference': 5.0}, {'jobId': 'j13', 'preference': 5.0}, {'jobId': 'j15', 'preference': 5.0}, {'jobId': 'j2', 'preference': 4.0}, {'jobId': 'j7', 'preference': 4.0}, {'jobId': 'j8', 'preference': 4.0}, {'jobId': 'j9', 'preference': 4.0}, {'jobId': 'j10', 'preference': 4.0}, {'jobId': 'j14', 'preference': 4.0}, {'jobId': 'j16', 'preference': 4.0}]}, {'candidateId': 42, 'candidateName': 'Nadia Fernandes ', 'gender': 'Female', 'age': 21, 'education': "Bachelor's Computer Engineering ", 'job_preferences': 'Full time', 'languages': 'English', 'skills': {}, 'previous_job_roles': {'Software Developer': 2}, 'groundTruth': [{'jobId': 'j8', 'preference': 10.0}, {'jobId': 'j9', 'preference': 10.0}, {'jobId': 'j6', 'preference': 8.0}, {'jobId': 'j7', 'preference': 8.0}, {'jobId': 'j10', 'preference': 8.0}, {'jobId': 'j11', 'preference': 8.0}, {'jobId': 'j15', 'preference': 8.0}, {'jobId': 'j16', 'preference': 8.0}, {'jobId': 'j12', 'preference': 7.0}, {'jobId': 'j13', 'preference': 7.0}, {'jobId': 'j4', 'preference': 6.0}, {'jobId': 'j3', 'preference': 5.0}, {'jobId': 'j5', 'preference': 5.0}, {'jobId': 'j1', 'preference': 4.0}, {'jobId': 'j14', 'preference': 4.0}, {'jobId': 'j2', 'preference': 2.0}]}, {'candidateId': 43, 'candidateName': 'Anish Anup Naik Mule ', 'gender': 'Male', 'age': 21, 'education': "Bachelor's computer engineering ", 'job_preferences': 'Internship, Work from home', 'languages': 'English', 'skills': {}, 'previous_job_roles': {'Software Developer': 2}, 'groundTruth': [{'jobId': 'j16', 'preference': 9.0}]}, {'candidateId': 44, 'candidateName': 'Khushi shirsat ', 'gender': 'Female', 'age': 19, 'education': 'PhD Computer science ', 'job_preferences': 'Day Shift, Work from home', 'languages': 'English', 'skills': {}, 'previous_job_roles': {'Software Developer': 2}, 'groundTruth': []}, {'candidateId': 45, 'candidateName': 'Jenny Gilbert ', 'gender': 'Male', 'age': 21, 'education': "Bachelor's Electronics and Telecommunications ", 'job_preferences': 'Full time, Work from office', 'languages': 'English', 'skills': {}, 'previous_job_roles': {'Software Developer': 2}, 'groundTruth': [{'jobId': 'j14', 'preference': 9.0}, {'jobId': 'j1', 'preference': 5.0}, {'jobId': 'j15', 'preference': 5.0}, {'jobId': 'j6', 'preference': 3.0}, {'jobId': 'j2', 'preference': 1.0}, {'jobId': 'j3', 'preference': 1.0}, {'jobId': 'j4', 'preference': 1.0}, {'jobId': 'j5', 'preference': 1.0}, {'jobId': 'j7', 'preference': 1.0}, {'jobId': 'j8', 'preference': 1.0}, {'jobId': 'j9', 'preference': 1.0}, {'jobId': 'j10', 'preference': 1.0}, {'jobId': 'j11', 'preference': 1.0}, {'jobId': 'j12', 'preference': 1.0}, {'jobId': 'j13', 'preference': 1.0}, {'jobId': 'j16', 'preference': 1.0}]}, {'candidateId': 46, 'candidateName': 'Amogh kunkolienkar ', 'gender': 'Male', 'age': 27, 'education': "Master's MBA IN FINANCE ", 'job_preferences': 'Full time', 'languages': 'English', 'skills': {'CSS': '7'}, 'previous_job_roles': {'Software Developer': 2}, 'groundTruth': [{'jobId': 'j1', 'preference': 10.0}, {'jobId': 'j2', 'preference': 10.0}, {'jobId': 'j3', 'preference': 10.0}, {'jobId': 'j4', 'preference': 10.0}, {'jobId': 'j5', 'preference': 10.0}, {'jobId': 'j6', 'preference': 10.0}, {'jobId': 'j7', 'preference': 10.0}, {'jobId': 'j8', 'preference': 10.0}, {'jobId': 'j9', 'preference': 10.0}, {'jobId': 'j10', 'preference': 10.0}, {'jobId': 'j11', 'preference': 10.0}, {'jobId': 'j12', 'preference': 10.0}, {'jobId': 'j13', 'preference': 10.0}, {'jobId': 'j14', 'preference': 10.0}, {'jobId': 'j15', 'preference': 10.0}, {'jobId': 'j16', 'preference': 10.0}]}, {'candidateId': 47, 'candidateName': 'Damodar', 'gender': 'Male', 'age': 43, 'education': "Master's MBA", 'job_preferences': 'Full time', 'languages': 'English', 'skills': {}, 'previous_job_roles': {'Software Developer': 2}, 'groundTruth': [{'jobId': 'j1', 'preference': 1.0}, {'jobId': 'j2', 'preference': 1.0}, {'jobId': 'j3', 'preference': 1.0}, {'jobId': 'j4', 'preference': 1.0}, {'jobId': 'j5', 'preference': 1.0}, {'jobId': 'j6', 'preference': 1.0}, {'jobId': 'j7', 'preference': 1.0}, {'jobId': 'j8', 'preference': 1.0}, {'jobId': 'j9', 'preference': 1.0}, {'jobId': 'j10', 'preference': 1.0}, {'jobId': 'j11', 'preference': 1.0}, {'jobId': 'j12', 'preference': 1.0}, {'jobId': 'j13', 'preference': 1.0}, {'jobId': 'j14', 'preference': 1.0}, {'jobId': 'j15', 'preference': 1.0}, {'jobId': 'j16', 'preference': 1.0}]}, {'candidateId': 48, 'candidateName': 'Shailesh Majukar ', 'gender': 'Male', 'age': 26, 'education': "Bachelor's Mechanical engineer ", 'job_preferences': 'Full time, Part time, Work from home', 'languages': 'English', 'skills': {}, 'previous_job_roles': {'Software Developer': 2}, 'groundTruth': [{'jobId': 'j2', 'preference': 10.0}, {'jobId': 'j3', 'preference': 10.0}, {'jobId': 'j7', 'preference': 10.0}, {'jobId': 'j8', 'preference': 10.0}, {'jobId': 'j15', 'preference': 10.0}, {'jobId': 'j16', 'preference': 10.0}, {'jobId': 'j1', 'preference': 9.0}, {'jobId': 'j6', 'preference': 9.0}, {'jobId': 'j9', 'preference': 9.0}, {'jobId': 'j10', 'preference': 9.0}, {'jobId': 'j11', 'preference': 9.0}, {'jobId': 'j12', 'preference': 9.0}, {'jobId': 'j13', 'preference': 9.0}, {'jobId': 'j4', 'preference': 8.0}, {'jobId': 'j5', 'preference': 8.0}, {'jobId': 'j14', 'preference': 8.0}]}, {'candidateId': 49, 'candidateName': 'Melfy correia ', 'gender': 'Female', 'age': 25, 'education': "Bachelor's Mechanical engineering ", 'job_preferences': 'Work from home', 'languages': 'English', 'skills': {}, 'previous_job_roles': {'Software Developer': 2}, 'groundTruth': [{'jobId': 'j1', 'preference': 6.0}, {'jobId': 'j2', 'preference': 6.0}, {'jobId': 'j3', 'preference': 5.0}, {'jobId': 'j12', 'preference': 5.0}, {'jobId': 'j4', 'preference': 4.0}, {'jobId': 'j8', 'preference': 3.0}, {'jobId': 'j11', 'preference': 3.0}, {'jobId': 'j5', 'preference': 2.0}, {'jobId': 'j6', 'preference': 2.0}, {'jobId': 'j7', 'preference': 2.0}, {'jobId': 'j9', 'preference': 2.0}, {'jobId': 'j10', 'preference': 2.0}, {'jobId': 'j13', 'preference': 2.0}, {'jobId': 'j14', 'preference': 2.0}, {'jobId': 'j15', 'preference': 2.0}, {'jobId': 'j16', 'preference': 2.0}]}]

# Model for extracting embeddings
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

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
    # print("User Details :")
    # print(user_data)
    # Convert education, job preferences, and languages to lowercase and remove punctuation
    if isinstance(user_data['education'], list):
        user_data['education'] = [preprocess_text(edu) for edu in user_data['education']]
    else:
        user_data['education'] = preprocess_text(user_data['education'])

    if isinstance(user_data['job_preferences'], list):
        user_data['job_preferences'] = [preprocess_text(pref) for pref in user_data['job_preferences']]
    else:
        user_data['job_preferences'] = preprocess_text(user_data['job_preferences'])

    if isinstance(user_data['languages'], list):
        user_data['languages'] = [preprocess_text(lang) for lang in user_data['languages']]
    else:
        user_data['languages'] = preprocess_text(user_data['languages'])

    # Convert skills to lowercase and remove punctuation
    user_data['skills'] = {preprocess_text(skill): proficiency for skill, proficiency in user_data['skills'].items()}

    return user_data


# Function to preprocess jobs_data
def preprocess_job_data(job_data):
    if isinstance(job_data, dict):  # If job_data is a single dictionary, wrap it in a list
        job_data = [job_data]
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


def cosine_similarity_score(candidate_vector, job_vectors):
    return cosine_similarity(candidate_vector.reshape(1, -1), job_vectors.reshape(len(job_vectors), -1))


# Function to check whether the candidate falls in the experience range
def calculate_experience_match(candidate_previous_job_roles, required_experience_min, required_experience_max):
    total_experience = sum(candidate_previous_job_roles.values())
    average_experience = total_experience / max(len(candidate_previous_job_roles), 1)

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


# Method returns list of recommended jobs for a given candidate along with the recommendedScore
# Method returns list of recommended jobs for a given candidate along with the recommendedScore
def job_recommendation(candidate_info, job_info):
    global data_objects
    # Step 1: Extract & preprocess user data
    if candidate_info is None:
        return json.dumps({'error': 'Candidate data is not available'})
    print(f"Candidate ID: {candidate_info['candidateId']} \nCandidate Name: {candidate_info['candidateName']}")

    # print("Candidate Data:")
    # print(candidate_info)

    # # Check if candidate has no skills
    # if not candidate_info['skills']:
    #     return json.dumps({'error': 'Candidate has no skills'})

    preprocessed_candidate_info = preprocess_user_data(candidate_info)
    # print("Preprocessed user Data:")
    # print(preprocessed_candidate_info)

    # Step 2: Extract & preprocess jobs data
    jobs_data = job_info
    if not jobs_data:
        return json.dumps({'error': 'No jobs available'})

    # print("Jobs data:")
    # print(jobs_data)

    preprocessed_jobs_data = preprocess_job_data(jobs_data)
    # print("Pre Processed Jobs data:")
    # print(preprocessed_jobs_data)
    # job_skills = ", ".join(skill['skillName'] for skill in preprocessed_jobs_data['skills'])
    # print(f"Job Skills: {job_skills}")
    flag = False
    # Step 3: Calculate similarity metric
    recommended_jobs = []
    for i in range(len(preprocessed_jobs_data)):
        job_info = preprocessed_jobs_data[i]
        job_skills = [skill['skillName'] for skill in job_info.get('skills', [])]
        job_proficiencies = {skill['skillName']: int(skill['proficiency']) for skill in job_info.get('skills', [])}
        match_count = sum(int(preprocessed_candidate_info['skills'].get(skill, '0')) >=
                          job_proficiencies.get(skill, 0) for skill in job_skills)  # +-1 range
        skill_similarity = match_count / max(len(job_skills), 1)
        if skill_similarity <= 0:
            continue

        # Calculate preference similarity
        candidate_preferences = candidate_info.get('job_preferences', '')
        job_preferences = job_info.get('preferences', '')
        if not candidate_preferences or not job_preferences:
            preference_similarity = 0.0
        else:
            candidate_preference_embedding = generate_bert_embeddings(candidate_preferences, tokenizer)
            job_preference_embedding = generate_bert_embeddings(job_preferences, tokenizer)
            preference_similarity = cosine_similarity_score(candidate_preference_embedding, job_preference_embedding)
            if isinstance(preference_similarity, np.ndarray):
                preference_similarity = preference_similarity[0][0]  # Extract scalar from array
            # print("Preference Similarity")
            # print(preference_similarity)

        # Calculate education similarity
        job_education = job_info.get('requiredHighestEducation', '')
        candidate_education = candidate_info.get('education', '')
        if not job_education or not candidate_education:
            education_similarity = 0.0
        else:
            education_embedding = generate_bert_embeddings(job_education, tokenizer)
            candidate_education_embedding = generate_bert_embeddings(candidate_education, tokenizer)
            education_similarity = cosine_similarity_score(candidate_education_embedding, education_embedding)
            if isinstance(education_similarity, np.ndarray):
                education_similarity = education_similarity[0][0]  # Extract scalar from array
            # print("Education Similarity")
            # print(education_similarity)

        # Calculate language similarity
        job_languages = job_info.get('languages', [])
        candidate_languages = candidate_info.get('languages', [])
        if not job_languages or not candidate_languages:
            language_similarity = 0.0
        else:
            language_texts = [''.join(job_languages)]
            language_embedding = generate_bert_embeddings(language_texts[0], tokenizer)
            candidate_language_embedding = generate_bert_embeddings(candidate_languages, tokenizer)
            language_similarity = cosine_similarity_score(candidate_language_embedding, language_embedding)
            if isinstance(language_similarity, np.ndarray):
                language_similarity = language_similarity[0][0]  # Extract scalar from array
            # print("Language Similarity")
            # print(language_similarity)

        # Calculate experience similarity
        if not candidate_info.get('previous_job_roles'):
            experience_similarity = 0.0
        else:
            experience_similarity = calculate_experience_match(candidate_info['previous_job_roles'],
                                                               job_info.get('requiredExperienceMin', 0),
                                                               job_info.get('requiredExperienceMax', 0))

        # Calculate overall similarity score with equal weightage for each attribute
        skill_weight = 0.40
        preference_weight = 0.20
        experience_weight = 0.10
        education_weight = 0.20
        language_weight = 0.10

        overall_similarity = float(skill_similarity * skill_weight + preference_similarity * preference_weight +
                                   education_similarity * education_weight + language_similarity * language_weight +
                                   experience_similarity * experience_weight) * 100  # Convert to percentage

        # Append job data along with similarity scores to the recommended jobs list
        recommended_jobs.append({
            "jobId": job_info['jobId'],
            "skillSimilarity": float(skill_similarity),
            "preferenceSimilarity": float(preference_similarity),
            "educationSimilarity": float(education_similarity),
            "languageSimilarity": float(language_similarity),
            "experienceMatch": float(experience_similarity),
            "overallSimilarity": float(overall_similarity)
        })
        data_objects.append({
            'candidateId': candidate_info['candidateId'],
            'candidateName': candidate_info['candidateName'],
            'jobId': job_info['jobId'],
            'skillSimilarity': float(skill_similarity),
            'preferenceSimilarity': float(preference_similarity),
            'educationSimilarity': float(education_similarity),
            'languageSimilarity': float(language_similarity),
            'experienceMatch': float(experience_similarity),
            'overallSimilarity': float(overall_similarity),
            'candidateRankedJobs':candidate_info['groundTruth']
        })
    # Sort recommended jobs by overall similarity score in descending order
    sorted_jobs = sorted(recommended_jobs, key=lambda x: x['overallSimilarity'], reverse=True)
    filtered_jobs = sorted_jobs
    if not filtered_jobs:
        print("No jobs available for the given candidate")
        return json.dumps({'error': 'No jobs are available for the candidate'})

    formatted_jobs = []
    for job in filtered_jobs:
        formatted_job = {
            "jobId": job['jobId'],
            "recommendationScore": round(job['overallSimilarity'], 2)
        }
        formatted_jobs.append(formatted_job)
    print("Recommended jobs are:")
    # Extracting data into a list of tuples
    data = [(job['jobId'], round(job['overallSimilarity'], 2)) for job in sorted_jobs]

    # Creating a DataFrame
    df = pd.DataFrame(data, columns=['JOB ID', 'RECOMMENDATION SCORE'])

    # Printing the DataFrame
    print(df)
    return json.dumps(formatted_jobs)


for i in range(0, 49):
    print("----------------------------")
    print(f"ITERATION {i+1}")
    job_recommendation(candidate_data[i], job_data)
    print("----------------------------\n")

# Specify the path where you want to save the CSV file
csv_file = 'output.csv'

# Extract headers from the first dictionary in data_objects
if data_objects:
    headers = list(data_objects[0].keys())

    # Writing data to CSV file
    with open(csv_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers)
        writer.writeheader()
        for data in data_objects:
            writer.writerow(data)

        print(f"CSV file '{csv_file}' has been created successfully.")
else:
    print("No data to write to CSV.")