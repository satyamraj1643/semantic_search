# Document Similarity using Sentence Transformers

A simple framework to generate vector embeddings for text documents and perform semantic similarity searches.

## Project Structure

```
.
├── core
│   └── files
│       ├── documents/  # <-- Place your .txt files here
│       └── embeddings/ # <-- Output .npy embeddings are saved here
├── .gitignore
├── document_similarity.py    # <-- Script to generate embeddings
├── find_similar_documents.py # <-- Script to search embeddings
├── readme.md
└── requirements.txt
```

## Setup

1. Clone the repository and navigate into it.

2. Create and activate a virtual environment:

```bash
# Create a virtual environment
python -m venv env

# Activate it (Windows)
.\env\Scripts\activate
# Or (macOS/Linux)
source env/bin/activate
```

3. Install dependencies:

**Install PyTorch:**

- For CPU:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```
- For NVIDIA GPU (CUDA 12.1):
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**Install remaining packages:**

```bash
pip install -r requirements.txt
```

## Usage

There are two main scripts, both run from the project's root directory:

### 1. Generate Embeddings

Place your `.txt` files in the `core/files/documents/` folder. Then run:

```bash
python document_similarity.py
```

Embeddings will be saved in `core/files/embeddings/`.

### 2. Find Similar Documents

Once embeddings are generated, run:

```bash
python find_similar_documents.py
```

You can modify the `user_query` variable inside `find_similar_documents.py` to search for different topics.
