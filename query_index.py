#%%
import faiss
import numpy as np
import pandas as pd
import sqlite3
import torch
from torch.cuda.amp import autocast
from transformers import AutoTokenizer, AutoModel
from transformers import DPRContextEncoder, DPRContextEncoderTokenizer
from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer

def encode_query(query, tokenizer, model, device):
    """
    Encode a query into embeddings using the specified tokenizer and model.

    Parameters:
    - query (str): The input query string.
    - tokenizer (AutoTokenizer): The tokenizer for processing the query.
    - model (AutoModel): The model to generate embeddings.
    - device (str): The device to run the model on ('cpu' or 'cuda').

    Returns:
    - np.ndarray: The encoded query embeddings as a numpy array.
    """
    with autocast():
        inputs = tokenizer(query, return_tensors="pt", padding=True, truncation=True, max_length=256).to(device)
        with torch.no_grad():
            embeddings = model(**inputs).pooler_output
    return embeddings.cpu().numpy()

def load_index_memory_mapped(file_path):
    """
    Load a memory-mapped FAISS index.

    Parameters:
    - file_path (str): The file path to the FAISS index.

    Returns:
    - faiss.Index: The loaded FAISS index.
    """
    return faiss.read_index(file_path, faiss.IO_FLAG_MMAP)


def search_index(faiss_index, query_embeddings, k):
    """
    Search the FAISS index for the top `k` most similar items.

    Parameters:
    - faiss_index (faiss.Index): The FAISS index to search.
    - query_embeddings (np.ndarray or torch.Tensor): The embeddings of the query.
    - k (int): The number of top similar items to retrieve.

    Returns:
    - tuple: (distances, indices) where distances are the distances to the top items and indices are the indices of the top items.
    """
    if isinstance(query_embeddings, torch.Tensor):
        query_embeddings = query_embeddings.numpy().astype(np.float32)
    elif isinstance(query_embeddings, np.ndarray) and query_embeddings.dtype != np.float32:
        query_embeddings = query_embeddings.astype(np.float32)
    
    distances, indices = faiss_index.search(query_embeddings, k)
    return distances, indices

def retrieve_documents_from_db(db_path, indices):
    """
    Retrieve documents from a SQLite database based on provided indices.

    Parameters:
    - db_path (str): Path to the SQLite database.
    - indices (np.ndarray): Array of indices to retrieve from the database.

    Returns:
    - pd.DataFrame: DataFrame containing the retrieved documents.
    """
    conn = sqlite3.connect(db_path)
    indices_str = ', '.join([str(idx) for idx in indices[0]])
    query_passage_id = f"SELECT * FROM article_passages WHERE passage_id IN ({indices_str})"
    query_passages_df = pd.read_sql_query(query_passage_id, conn)
    matching_article_ids = list(query_passages_df['article_id'].astype(str))
    article_ids_str = ', '.join(matching_article_ids)
    query_article_id = f"SELECT * FROM articles WHERE id IN ({article_ids_str})"
    documents_df = pd.read_sql_query(query_article_id, conn)
    conn.close()
    return documents_df

def query_index(query, index_file_path = 'disk_index.index', 
                db_path = '/media/david/WDBLUE8TB/data/wikipedia_articles.db', 
                model_name='facebook/dpr-question_encoder-single-nq-base', k=10):
    """
    Encodes a query, searches a FAISS index, and retrieves matching documents from a database.

    Parameters:
    - query (str): The input query string.
    - index_file_path (str): Path to the FAISS index file.
    - db_path (str): Path to the SQLite database containing the documents.
    - model_name (str): The name of the pre-trained model for encoding (default is 'facebook/dpr-question_encoder-single-nq-base').
    - k (int): The number of top similar items to retrieve (default is 10).

    Returns:
    - pd.DataFrame: DataFrame containing the retrieved documents.
    """
    
    # Set up the device and load the model and tokenizer
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    question_encoder = DPRQuestionEncoder.from_pretrained(model_name).to(device)
    question_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(model_name)

    # Encode the query
    query_embedding = encode_query(query, question_tokenizer, question_encoder, device)

    # Load the FAISS index
    index = load_index_memory_mapped(index_file_path)

    # Search the index
    distances, indices = search_index(index, query_embedding, k)

    # Retrieve documents from the database
    results_df = retrieve_documents_from_db(db_path, indices)

    return results_df
# %%
