# scripts/warm_model.py
from sentence_transformers import SentenceTransformer
m = SentenceTransformer("BAAI/bge-base-en-v1.5")
print("Model loaded. Embedding dim:", m.get_sentence_embedding_dimension())