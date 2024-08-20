import fitz  # PyMuPDF
import os
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from transformers import BertTokenizer, BertModel
import torch

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def extract_texts_from_folder(folder_path):
    texts = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            file_path = os.path.join(folder_path, filename)
            texts.append(extract_text_from_pdf(file_path))
    return " ".join(texts)

def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    text = re.sub(r'[^a-zA-Z0-9\s\-\.]', '', text)
    return text

def get_bert_embeddings(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

# Convert Windows paths to WSL paths
gri_text = extract_text_from_pdf('/mnt/c/Users/joesc/OneDrive - University of Waterloo/MES_Thesis_Data/Consolidated_GRI_Standards.pdf')
sasb_text = extract_texts_from_folder('/mnt/c/Users/joesc/OneDrive - University of Waterloo/MES_Thesis_Data/SASB_Standards')

gri_cleaned = clean_text(gri_text)
sasb_cleaned = clean_text(sasb_text)

# Combine texts for consistent TF-IDF vectorization
combined_texts = [gri_cleaned, sasb_cleaned]
vectorizer = TfidfVectorizer(max_features=1000)
combined_tfidf = vectorizer.fit_transform(combined_texts)

# Extract keywords
gri_tfidf = vectorizer.transform([gri_cleaned])
sasb_tfidf = vectorizer.transform([sasb_cleaned])

gri_keywords = vectorizer.get_feature_names_out()[gri_tfidf.toarray().flatten() > 0]
sasb_keywords = vectorizer.get_feature_names_out()[sasb_tfidf.toarray().flatten() > 0]

gri_set = set(gri_keywords)
sasb_set = set(sasb_keywords)

common_keywords = gri_set & sasb_set
unique_gri_keywords = gri_set - sasb_set
unique_sasb_keywords = sasb_set - gri_set

# Print unique keywords
print("Unique GRI Keywords:", unique_gri_keywords)
print("Unique SASB Keywords:", unique_sasb_keywords)

# Visualize cosine similarities
cosine_similarities = cosine_similarity(combined_tfidf)

labels = ['GRI', 'SASB']
fig, ax = plt.subplots()
cax = ax.matshow(cosine_similarities, cmap='coolwarm')

plt.xticks(np.arange(len(labels)), labels, rotation=90)
plt.yticks(np.arange(len(labels)), labels)

fig.colorbar(cax)
plt.title("Cosine Similarities using TF-IDF")
plt.savefig('/mnt/c/Users/joesc/OneDrive - University of Waterloo/MES_Thesis_Data/cosine_similarities_tfidf.png')
plt.close()

# BERT processing
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

gri_bert_embeddings = get_bert_embeddings(gri_cleaned)
sasb_bert_embeddings = get_bert_embeddings(sasb_cleaned)

# For TF-IDF visualization
pca_tfidf = PCA(n_components=2)
reduced_tfidf = pca_tfidf.fit_transform(combined_tfidf.toarray())

plt.figure(figsize=(10, 6))
plt.scatter(reduced_tfidf[0, 0], reduced_tfidf[0, 1], color='blue', label='GRI')
plt.scatter(reduced_tfidf[1, 0], reduced_tfidf[1, 1], color='red', label='SASB')
plt.legend()
plt.title("GRI vs SASB Keywords in Vector Space (TF-IDF)")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.savefig('/mnt/c/Users/joesc/OneDrive - University of Waterloo/MES_Thesis_Data/tfidf_visualization.png')
plt.close()

# For BERT visualization
all_bert_embeddings = np.array([gri_bert_embeddings, sasb_bert_embeddings])
pca_bert = PCA(n_components=2)
reduced_bert = pca_bert.fit_transform(all_bert_embeddings)

plt.figure(figsize=(10, 6))
plt.scatter(reduced_bert[0, 0], reduced_bert[0, 1], color='blue', label='GRI')
plt.scatter(reduced_bert[1, 0], reduced_bert[1, 1], color='red', label='SASB')
plt.legend()
plt.title("GRI vs SASB Keywords in Vector Space (BERT)")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.savefig('/mnt/c/Users/joesc/OneDrive - University of Waterloo/MES_Thesis_Data/bert_visualization.png')
plt.close()
