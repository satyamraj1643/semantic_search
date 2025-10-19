import os
import numpy as np
import torch
from langchain_huggingface import HuggingFaceEmbeddings

def generate_embeddings():
    script_dir = os.path.dirname(os.path.abspath(__file__))  #get the directory of the script
    docs_path = os.path.join(script_dir, "core" , "files", "documents") #path to the documents folder
    embeddings_path = os.path.join(script_dir, "core",  "files", "embeddings") #path to the embeddings folder

    model_name = "sentence-transformers/all-MiniLM-L6-v2" # Pre-trained model name
    device = "cuda" if torch.cuda.is_available() else "cpu" # Use GPU if available
    
    embedding_model = HuggingFaceEmbeddings( 
        model_name=model_name, 
        model_kwargs={'device': device} 
    )

    os.makedirs(embeddings_path, exist_ok=True)  #Ensure embeddings directory exists

    document_files = [f for f in os.listdir(docs_path) if f.endswith(".txt")]  #List all text files in the documents folder

    for doc_name in document_files:
        input_path = os.path.join(docs_path, doc_name)   #Full path to the document file
        
        with open(input_path, 'r', encoding='utf-8') as f:
            text = f.read()    #Read the content of the document

        embedding_vector = embedding_model.embed_query(text)   #Generate the embedding vector

        output_name = os.path.splitext(doc_name)[0] + ".npy"   #Change file extension to .npy
        output_path = os.path.join(embeddings_path, output_name)  #Full path to save the embedding

        np.save(output_path, np.array(embedding_vector))  #Save the embedding as a .npy file
        print(f"Processed: {doc_name}")   

if __name__ == '__main__':
    print("Starting embedding generation...")
    generate_embeddings()
    print("Embedding generation complete.")

