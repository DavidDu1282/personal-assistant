#%%
import faiss
import gc
import pandas as pd
import sqlite3
import torch

from faiss import StandardGpuResources 
from torch.cuda.amp import autocast
from transformers import DPRContextEncoder, DPRContextEncoderTokenizer
from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer
#%%
#grab Dense Passage Retrieval models 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

context_encoder = DPRContextEncoder.from_pretrained('facebook/dpr-ctx_encoder-single-nq-base').to(device)
context_tokenizer = DPRContextEncoderTokenizer.from_pretrained('facebook/dpr-ctx_encoder-single-nq-base')
question_encoder = DPRQuestionEncoder.from_pretrained('facebook/dpr-question_encoder-single-nq-base').to(device)
question_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained('facebook/dpr-question_encoder-single-nq-base')

# get sqlite database where articles are stored, and import into pandas
db_path = '/media/david/WDBLUE8TB/data/wikipedia_articles.db'
conn = sqlite3.connect(db_path)
article_df = pd.read_sql_query("SELECT * FROM articles", conn)
conn.close()
article_df.drop('id', axis = 1, inplace = True)
#%%

def batch_encode_dataframe(df, column, tokenizer, model, batch_size=64, device='cuda'):
    all_embeddings = []
    df = df.dropna(subset=[column])  # Ensure no NaN values in the column to be processed

    for i in range(0, len(df), batch_size):
        batch_texts = df[column].iloc[i:i + batch_size].tolist()
        batch_texts = [str(text) for text in batch_texts]
        if not batch_texts:
            continue
        
        with autocast():
            encoded = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=256).to(device)
            with torch.no_grad():
                embeddings = model(**encoded).pooler_output
        all_embeddings.append(embeddings.cpu())

    return torch.cat(all_embeddings, dim=0).to(torch.float16) if all_embeddings else torch.tensor([])

def process_df_in_chunks(df, column, tokenizer, model, batch_size=64, chunk_size=10000, device='cuda'):
    # Create an index range for chunk processing
    num_chunks = (len(df) + chunk_size - 1) // chunk_size  # Calculate number of chunks needed

    for chunk_idx in range(num_chunks):
        # Define the start and end of the chunk
        start_idx = chunk_idx * chunk_size
        end_idx = min((chunk_idx + 1) * chunk_size, len(df))
        chunk = df.iloc[start_idx:end_idx]

        # Process each chunk using the batch encoding function
        embeddings = batch_encode_dataframe(chunk, column, tokenizer, model, batch_size, device)

        # Optionally save embeddings to disk or handle them as needed
        save_embeddings_to_disk(embeddings, f"embeddings_chunk_{chunk_idx}.pt")
        print(f"Processed {chunk_idx+1} chunks!")

def save_embeddings_to_disk(embeddings, filename):
    torch.save(embeddings, filename)
#%%
def split_into_passages(text):
    passages = text.split("\n\n")
    passages = [passage.strip() for passage in passages if passage.strip()]
    return passages

# Apply the function
article_df['passages'] = article_df['text'].apply(split_into_passages)

# Explode the DataFrame
article_df_exploded = article_df.explode('passages')
article_df_exploded['passage_text'] = article_df_exploded['passages']
article_df_exploded.drop('passages', axis=1, inplace=True)
#%%
try:
    del article_df
except NameError:
    print("Already deleted article df!")
gc.collect()
#%%
# Example data - large dataset
process_df_in_chunks(article_df_exploded, 'passage_text', context_tokenizer, context_encoder, device=device, batch_size=500, chunk_size = 1000000)
#%%










# Initialize resources and GPU index
gpu_resources = StandardGpuResources()
gpu_index = faiss.IndexFlatL2(passage_embeddings.size(1))
gpu_index = faiss.index_cpu_to_gpu(gpu_resources, 0, gpu_index)

# Add embeddings to the index
gpu_index.add(passage_embeddings.cpu().numpy())  # Move embeddings to CPU before adding them to FAISS GPU index

questions = ["What is the example query?"] * 10  # Assuming multiple questions
question_embeddings = batch_encode(questions, question_tokenizer, question_encoder)

# Search on GPU index
distances, indices = gpu_index.search(question_embeddings.cpu().numpy(), k = 5)

for i, idxs in enumerate(indices):
    print(f"Query {i+1} results:")
    for rank, idx in enumerate(idxs):
        print(f"  Rank {rank+1}: Passage {idx} retrieved with distance {distances[i][rank]}")
