# import torch
# from transformers import BertTokenizer, BertModel
# from sklearn.metrics.pairwise import cosine_similarity
#
# # Model for extracting embeddings
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# model = BertModel.from_pretrained('bert-base-uncased')
#
# # Function to generate BERT Embeddings
# def generate_bert_embeddings(text, tokenizer, model):
#     inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
#     with torch.no_grad():
#         outputs = model(**inputs)
#     embeddings = outputs.last_hidden_state.mean(dim=1)
#     return embeddings.numpy()
#
# def cosine_similarity_score(text1, text2):
#     embedding1 = generate_bert_embeddings(text1, tokenizer, model)
#     embedding2 = generate_bert_embeddings(text2, tokenizer, model)
#     return cosine_similarity(embedding1, embedding2)
# similarity_score = cosine_similarity_score(embedding1, embedding2)
#
# print("Similarity Score:", round(float(similarity_score[0][0]), 2))
#
# # skill = "PYTHON DEVELOPER,FULL STACK DEVELOPER"
# # education = "Bachelor of Computer Engineering"
# # language = "English Hindi Kannada"
# #
# # job_skill = "Flask,Springboot"
# # job_education = "Graduate"
# # job_language = "English"
#
# # print(f"Skill Similarity: {round(float(cosine_similarity_score(skill, job_skill)[0][0]), 2)}")
# # print(f"Education Similarity: {round(float(cosine_similarity_score(education, job_education)[0][0]), 2)}")
# # print(f"Language Similarity: {round(float(cosine_similarity_score(language, job_language)[0][0]), 2)}")

import torch
from transformers import BertTokenizer, BertModel
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import nltk

# Download the NLTK punkt tokenizer data
nltk.download('punkt')

# Load BERT tokenizer and model
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

# Load Sentence-BERT model
sentence_model = SentenceTransformer('bert-base-nli-mean-tokens')

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Function to generate BERT embeddings
def generate_bert_embeddings(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings.numpy()

# Function to generate Sentence-BERT embeddings
def generate_sentence_embeddings(text, model):
    embeddings = model.encode(text)
    return embeddings

# Preprocess text: tokenize, lemmatize, and join
def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return ' '.join(lemmatized_tokens)

# Generate combined embeddings from BERT and Sentence-BERT
def generate_combined_embeddings(text, bert_tokenizer, bert_model, sentence_model):
    bert_embeddings = generate_bert_embeddings(text, bert_tokenizer, bert_model)
    sentence_embeddings = generate_sentence_embeddings(text, sentence_model)
    combined_embeddings = np.concatenate((bert_embeddings, [sentence_embeddings]), axis=1)
    return combined_embeddings

# Calculate cosine similarity score
def cosine_similarity_score(text1, text2, bert_tokenizer, bert_model, sentence_model):
    embedding1 = generate_combined_embeddings(text1, bert_tokenizer, bert_model, sentence_model)
    embedding2 = generate_combined_embeddings(text2, bert_tokenizer, bert_model, sentence_model)
    return cosine_similarity(embedding1, embedding2)

# Input texts with added context and related terms
embedding1 = preprocess_text("Goa")
embedding2 = preprocess_text("India")


similarity_score = cosine_similarity_score(embedding1, embedding2)

print("Similarity Score:", round(float(similarity_score[0][0]), 2))
