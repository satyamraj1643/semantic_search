import os
import numpy as np
import torch
from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity

def find_most_similar(query):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    embeddings_path = os.path.join(script_dir,"core", "files", "embeddings")
    
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    embedding_model = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={'device': device}
    )

    doc_embeddings = {}
    for filename in os.listdir(embeddings_path):
        if filename.endswith(".npy"):
            filepath = os.path.join(embeddings_path, filename)
            doc_embeddings[filename] = np.load(filepath)

    query_embedding = embedding_model.embed_query(query)

    doc_names = list(doc_embeddings.keys())
    doc_vectors = np.array(list(doc_embeddings.values()))
    query_vector = np.array([query_embedding])

    # Calculate similarity between the query and all documents
    similarity_scores = cosine_similarity(query_vector, doc_vectors)[0]

    # Find the index of the document with the highest score
    max_score_index = np.argmax(similarity_scores)
    
    # Get the best matching document name and its score
    best_doc_name = doc_names[max_score_index]
    max_score = similarity_scores[max_score_index]

    # Print only the single best result
    print("\n--- Most Similar Document ---")
    clean_name = best_doc_name.replace('.npy', '.txt')
    print(f"Document: {clean_name}\nSimilarity Score: {max_score:.4f}\n")


if __name__ == '__main__':
    user_query = "What was the economic impact of the Dutch East India Company?"
    find_most_similar(user_query)

