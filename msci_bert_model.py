import os
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances, manhattan_distances
from scipy.spatial.distance import jaccard
from scipy.stats import pearsonr
from transformers import BertModel, BertTokenizer
import torch
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Paths
unclassified_txt_dir = "/mnt/c/Users/joesc/OneDrive - University of Waterloo/MES_Thesis_Data/TSX_ESG_Clean"
output_dir = "/mnt/c/Users/joesc/OneDrive - University of Waterloo/MES_Thesis_Data/TSX_ESG_Embeddings_2"
csv_output_path = "/mnt/c/Users/joesc/OneDrive - University of Waterloo/MES_Thesis_Data/classification_results_2.csv"

# Function to read text from a file
def read_text_from_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
        return text
    except Exception as e:
        print(f"Error reading text from {file_path}: {e}")
        return ""

# Function to chunk text
def chunk_text(text, chunk_size=512, overlap=256):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        chunk = ' '.join(words[start:start + chunk_size])
        chunks.append(chunk)
        start += chunk_size - overlap  # overlap for continuity
    return chunks

# Function to generate embeddings for chunks
def get_embeddings_for_chunks(chunks, model, tokenizer):
    embeddings = []
    for chunk in chunks:
        inputs = tokenizer(chunk, return_tensors='pt', max_length=512, truncation=True, padding='max_length')
        with torch.no_grad():
            outputs = model(**inputs)
        embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
        embeddings.append(embedding)
    return np.array(embeddings)

# Function to classify text based on embeddings
def classify_text(unclassified_embeddings, gri_embeddings, sasb_embeddings):
    gri_cosine_similarity = cosine_similarity(unclassified_embeddings, gri_embeddings).mean()
    sasb_cosine_similarity = cosine_similarity(unclassified_embeddings, sasb_embeddings).mean()
    
    gri_euclidean_distance = euclidean_distances(unclassified_embeddings, gri_embeddings).mean()
    sasb_euclidean_distance = euclidean_distances(unclassified_embeddings, sasb_embeddings).mean()
    
    gri_manhattan_distance = manhattan_distances(unclassified_embeddings, gri_embeddings).mean()
    sasb_manhattan_distance = manhattan_distances(unclassified_embeddings, sasb_embeddings).mean()
    
    gri_jaccard_similarity = np.mean([1 - jaccard(u, g) for u in unclassified_embeddings for g in gri_embeddings])
    sasb_jaccard_similarity = np.mean([1 - jaccard(u, s) for u in unclassified_embeddings for s in sasb_embeddings])
    
    gri_pearson_correlation = np.mean([pearsonr(u, g)[0] for u in unclassified_embeddings for g in gri_embeddings])
    sasb_pearson_correlation = np.mean([pearsonr(u, s)[0] for u in unclassified_embeddings for s in sasb_embeddings])
    
    results = {
        "cosine_similarity": (gri_cosine_similarity, sasb_cosine_similarity),
        "euclidean_distance": (gri_euclidean_distance, sasb_euclidean_distance),
        "manhattan_distance": (gri_manhattan_distance, sasb_manhattan_distance),
        "jaccard_similarity": (gri_jaccard_similarity, sasb_jaccard_similarity),
        "pearson_correlation": (gri_pearson_correlation, sasb_pearson_correlation)
    }
    
    gri_score = sum([v[0] > v[1] for k, v in results.items() if k != 'euclidean_distance' and k != 'manhattan_distance'])
    sasb_score = sum([v[1] > v[0] for k, v in results.items() if k != 'euclidean_distance' and k != 'manhattan_distance'])
    
    return "GRI" if gri_score > sasb_score else "SASB", results

# Function to visualize embeddings
def visualize_embeddings(embeddings, labels, unclassified_embeddings, company, year, output_path):
    combined_embeddings = np.concatenate((embeddings, unclassified_embeddings), axis=0)
    min_perplexity = min(30, len(combined_embeddings) - 1)
    
    # TSNE for combined embeddings
    tsne = TSNE(n_components=2, random_state=42, perplexity=min_perplexity)
    reduced_embeddings_tsne = tsne.fit_transform(combined_embeddings)
    
    # PCA for combined embeddings
    pca = PCA(n_components=2)
    reduced_embeddings_pca = pca.fit_transform(combined_embeddings)
    
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    
    unique_labels = set(labels)
    colors = {'GRI': 'blue', 'SASB': 'red', 'Unclassified': 'green'}
    
    # TSNE plot
    for label in unique_labels:
        idx = np.where(labels == label)
        axs[0, 0].scatter(reduced_embeddings_tsne[idx, 0], reduced_embeddings_tsne[idx, 1], label=label, color=colors[label])
    
    unclassified_idx = range(len(embeddings), len(combined_embeddings))
    axs[0, 0].scatter(reduced_embeddings_tsne[unclassified_idx, 0], reduced_embeddings_tsne[unclassified_idx, 1], label='Unclassified', color=colors['Unclassified'], marker='x')
    axs[0, 0].set_title(f"t-SNE Visualization ({company} {year})")
    axs[0, 0].legend()
    axs[0, 0].set_xlabel("Dimension 1")
    axs[0, 0].set_ylabel("Dimension 2")
    
    # PCA plot
    for label in unique_labels:
        idx = np.where(labels == label)
        axs[0, 1].scatter(reduced_embeddings_pca[idx, 0], reduced_embeddings_pca[idx, 1], label=label, color=colors[label])
    
    axs[0, 1].scatter(reduced_embeddings_pca[unclassified_idx, 0], reduced_embeddings_pca[unclassified_idx, 1], label='Unclassified', color=colors['Unclassified'], marker='x')
    axs[0, 1].set_title(f"PCA Visualization ({company} {year})")
    axs[0, 1].legend()
    axs[0, 1].set_xlabel("Dimension 1")
    axs[0, 1].set_ylabel("Dimension 2")
    
    # TSNE with clusters
    kmeans_tsne = KMeans(n_clusters=2, random_state=0).fit(reduced_embeddings_tsne)
    clusters_tsne = kmeans_tsne.labels_
    colors_tsne = np.where(clusters_tsne == 0, 'blue', 'red')
    
    axs[1, 0].scatter(reduced_embeddings_tsne[:, 0], reduced_embeddings_tsne[:, 1], c=colors_tsne, alpha=0.5)
    axs[1, 0].set_title(f"t-SNE with KMeans Clusters ({company} {year})")
    axs[1, 0].set_xlabel("Dimension 1")
    axs[1, 0].set_ylabel("Dimension 2")
    
    # PCA with clusters
    kmeans_pca = KMeans(n_clusters=2, random_state=0).fit(reduced_embeddings_pca)
    clusters_pca = kmeans_pca.labels_
    colors_pca = np.where(clusters_pca == 0, 'blue', 'red')
    
    axs[1, 1].scatter(reduced_embeddings_pca[:, 0], reduced_embeddings_pca[:, 1], c=colors_pca, alpha=0.5)
    axs[1, 1].set_title(f"PCA with KMeans Clusters ({company} {year})")
    axs[1, 1].set_xlabel("Dimension 1")
    axs[1, 1].set_ylabel("Dimension 2")
    
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Combined plot saved to {output_path}")

# Load pre-trained BERT model and tokenizer
model_name = 'bert-base-uncased'
model = BertModel.from_pretrained(model_name)
tokenizer = BertTokenizer.from_pretrained(model_name)

# Load saved embeddings for GRI and SASB
gri_embeddings = np.load(os.path.join(output_dir, "gri_embeddings.npy"))
sasb_embeddings = np.load(os.path.join(output_dir, "sasb_embeddings.npy"))

# Process all unclassified text files and classify them
unclassified_files = [f for f in os.listdir(unclassified_txt_dir) if f.endswith('_clean.txt')]
classification_results = []

for txt_file in unclassified_files:
    print(f"Processing {txt_file}...")
    text = read_text_from_file(os.path.join(unclassified_txt_dir, txt_file))
    if text:
        chunks = chunk_text(text)
        embeddings = get_embeddings_for_chunks(chunks, model, tokenizer)
        np.save(os.path.join(output_dir, f"{os.path.basename(txt_file)}_embeddings.npy"), embeddings)
        
        classification, metrics = classify_text(embeddings, gri_embeddings, sasb_embeddings)
        ticker, year = txt_file.split('_')[0], txt_file.split('_')[2].split('.')[0]
        
        classification_results.append({
            "txt_file": txt_file,
            "ticker": ticker,
            "year": year,
            "classification": classification,
            **{f"gri_{k}": v[0] for k, v in metrics.items()},
            **{f"sasb_{k}": v[1] for k, v in metrics.items()}
        })
        
        visualize_embeddings(np.concatenate((gri_embeddings, sasb_embeddings)), 
                             np.concatenate((["GRI"] * len(gri_embeddings), ["SASB"] * len(sasb_embeddings))), 
                             unclassified_embeddings=embeddings, 
                             company=ticker, 
                             year=year, 
                             output_path=os.path.join(output_dir, f"{ticker}_{year}_combined_plot.png"))
        
        # Save classification results to CSV after each report is analyzed
        df = pd.DataFrame(classification_results)
        df.to_csv(csv_output_path, index=False)
        print(f"Classification results saved to {csv_output_path}")
    else:
        print(f"Failed to read text from {txt_file}")

