#%%
import faiss
import gc
import numpy as np
import pandas as pd
import sqlite3
import torch
from faiss import StandardGpuResources 
from torch.cuda.amp import autocast
from transformers import DPRContextEncoder, DPRContextEncoderTokenizer
from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer
#%%
# Define functions for processing data and queries
def batch_encode_dataframe(df, column, tokenizer, model, batch_size=64, device='cuda'):
    all_embeddings = []
    df = df.dropna(subset=[column])  # Ensure no NaN values in the column to be processed

    for i in range(0, len(df), batch_size):
        batch_texts = df[column].iloc[i:i + batch_size].tolist()
        batch_texts = [str(text) for text in batch_texts if text]
        if not batch_texts:
            continue

        with autocast():
            encoded = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=256).to(device)
            with torch.no_grad():
                embeddings = model(**encoded).pooler_output
        all_embeddings.append(embeddings.cpu())

    return torch.cat(all_embeddings, dim=0).to(torch.float16) if all_embeddings else torch.tensor([])

def process_df_in_chunks(df, column, tokenizer, model, batch_size=64, chunk_size=10000, device='cuda'):
    num_chunks = (len(df) + chunk_size - 1) // chunk_size
    for chunk_idx in range(num_chunks):
        start_idx = chunk_idx * chunk_size
        end_idx = min((chunk_idx + 1) * chunk_size, len(df))
        chunk = df.iloc[start_idx:end_idx]
        embeddings = batch_encode_dataframe(chunk, column, tokenizer, model, batch_size, device)
        save_embeddings_to_disk(embeddings, f"embeddings_chunk_{chunk_idx}.pt")
        print(f"Processed {chunk_idx+1} of {num_chunks} chunks.")

def save_embeddings_to_disk(embeddings, filename):
    torch.save(embeddings, filename)

def split_into_passages(text):
    passages = text.split("\n\n")
    passages = [passage.strip() for passage in passages if passage.strip()]
    return passages

# Main execution block to prevent automatic execution on import
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    context_encoder = DPRContextEncoder.from_pretrained('facebook/dpr-ctx_encoder-single-nq-base').to(device)
    context_tokenizer = DPRContextEncoderTokenizer.from_pretrained('facebook/dpr-ctx_encoder-single-nq-base')

    # Connect to SQLite and load data into DataFrame
    db_path = '/media/david/WDBLUE8TB/data/wikipedia_articles.db'
    conn = sqlite3.connect(db_path)
    passage_df = pd.read_sql_query("SELECT * FROM article_passages", conn)
    conn.close()
    # passage_df.drop('id', axis=1, inplace=True)
    # passage_df['passages'] = passage_df['text'].apply(split_into_passages)
    # passage_df_exploded = passage_df.explode('passages')
    # passage_df_exploded['passage_text'] = passage_df_exploded['passages']
    # passage_df_exploded.drop('passages', axis=1, inplace=True)
    
    # del passage_df  
    # gc.collect()
    
    process_df_in_chunks(passage_df, 'text', context_tokenizer, context_encoder, device=device, batch_size=500, chunk_size=100000)
#%%

