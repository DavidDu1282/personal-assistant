#%%
import faiss
import gc
import numpy as np
import os
import re
import torch
#%%
def list_embedding_files(directory, pattern = r"embeddings_chunk_\d+\.pt$"):
    all_files = os.listdir(directory)
    matched_files = [filename for filename in all_files if re.match(pattern, filename)]
    return sorted(os.path.join(directory, filename) for filename in matched_files)

def load_and_concatenate_embeddings(files):
    all_embeddings = []
    for file in files:
        embeddings = torch.load(file)
        all_embeddings.append(embeddings)
    concatenated_embeddings = torch.cat(all_embeddings, dim = 0)
    return concatenated_embeddings

def setup_disk_based_ivfpq_index(file_path, data, nlist, m):
    dimension = data.shape[1]
    quantizer = faiss.IndexFlatL2(dimension)  # the quantizer
    index = faiss.IndexIVFPQ(quantizer, dimension, nlist, m, 8)  # m is the number of subquantizers
    
    if isinstance(data, torch.Tensor):
        data = data.numpy()
    if data.dtype !=  np.float32:
        data = data.astype(np.float32)

    index.train(data)
    index.add(data)
    faiss.write_index(index, file_path)
    return index

def add_data_to_index_in_chunks(index, data, file_path, chunk_size = 10000):
    num_chunks = (data.shape[0] + chunk_size - 1) // chunk_size
    for chunk_idx in range(num_chunks):
        start_idx = chunk_idx * chunk_size
        end_idx = min((chunk_idx + 1) * chunk_size, data.shape[0])
        chunk = data[start_idx:end_idx]

        if isinstance(chunk, torch.Tensor):
            chunk = chunk.numpy()  # Convert to numpy array first if it's a tensor
        if chunk.dtype !=  np.float32:
            chunk = chunk.astype(np.float32)  # Now use astype safely

        ids = np.arange(start_idx, end_idx)
        index.add_with_ids(chunk, ids)
        print(f"Chunk {chunk_idx} added to index.")
        faiss.write_index(index, file_path)
        gc.collect()
#%%
if __name__  ==  '__main__':
    embedding_directory = os.getcwd()  # Modify as needed
    embedding_files = list_embedding_files(embedding_directory)
    if not embedding_files:
        print('No embedding chunks were found!')
        exit()

    full_embeddings = load_and_concatenate_embeddings(embedding_files)
    print("Combined shape of embeddings:", full_embeddings.shape)

    index_file_path = 'disk_index.index'
    index = setup_disk_based_ivfpq_index(index_file_path, nlist = 100, data = full_embeddings, m = 16)
    add_data_to_index_in_chunks(index, full_embeddings, index_file_path, chunk_size = 1000000)
    gc.collect()
#%%