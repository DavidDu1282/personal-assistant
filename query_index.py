#%%
import faiss
import numpy as np
import os
import pandas as pd
import re
import sqlite3
import torch

from torch.cuda.amp import autocast
from transformers import AutoTokenizer, AutoModel
from transformers import DPRContextEncoder, DPRContextEncoderTokenizer
from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer
#%%
def encode_query(query, tokenizer, model, device):
    with autocast():
        inputs = tokenizer(query, return_tensors="pt", padding=True, truncation=True, max_length=256).to(device)
        with torch.no_grad():
            embeddings = model(**inputs).pooler_output
    return embeddings.cpu().numpy()

def load_index_memory_mapped(file_path):
    index = faiss.read_index(file_path, faiss.IO_FLAG_MMAP)
    return index


def search_index(faiss_index, query_embeddings, k):
    """ Search the FAISS index for the top `k` most similar items. """
    if isinstance(query_embeddings, torch.Tensor):
        query_embeddings = query_embeddings.numpy().astype(np.float32)
    elif isinstance(query_embeddings, np.ndarray) and query_embeddings.dtype != np.float32:
        query_embeddings = query_embeddings.astype(np.float32)
    
    distances, indices = faiss_index.search(query_embeddings, k)
    return distances, indices

def retrieve_documents_from_db(db_path, indices):
    """ Retrieve documents from a SQLite database based on indices. """
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
#%%
if __name__ == '__main__':

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    question_encoder = DPRQuestionEncoder.from_pretrained('facebook/dpr-question_encoder-single-nq-base').to(device)
    question_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained('facebook/dpr-question_encoder-single-nq-base')

    query = "Where is Alabama"
    query_embedding = encode_query(query, question_tokenizer, question_encoder, device)

    index_file_path = 'disk_index.index'
    index = load_index_memory_mapped(index_file_path)

    k = 10  
    distances, indices = search_index(index, query_embedding, k)
    
    db_path = '/media/david/WDBLUE8TB/data/wikipedia_articles.db'
    results_df = retrieve_documents_from_db(db_path, indices)
    
    for index, row in results_df.iterrows():
        print(row['title'], row['text'])


#%%
    # db_path = '/media/david/WDBLUE8TB/data/wikipedia_articles.db'
    # conn = sqlite3.connect(db_path)
    # article_df = pd.read_sql_query("SELECT * FROM articles", conn)
# %%
