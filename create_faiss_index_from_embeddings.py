#%%
import faiss
import gc
import numpy as np
import os
import re
import torch
#%%

def list_embedding_files(directory, pattern=r"embeddings_chunk_\d+\.pt$"):
    all_files = os.listdir(directory)
    matched_files = [filename for filename in all_files if re.match(pattern, filename)]
    return sorted(os.path.join(directory, filename) for filename in matched_files)

def load_and_concatenate_embeddings(files):
    all_embeddings = []
    for file in files:
        # Load the tensor stored in each file
        embeddings = torch.load(file)
        all_embeddings.append(embeddings)
    
    # Concatenate all loaded tensors into one
    concatenated_embeddings = torch.cat(all_embeddings, dim=0)
    return concatenated_embeddings

# Example directory where your files are stored
embedding_directory = os.getcwd() #'/path/to/your/embedding/files'

# List embedding files
embedding_files = list_embedding_files(embedding_directory)

# Load and concatenate embeddings
if len(embedding_files) > 0:
    full_embeddings = load_and_concatenate_embeddings(embedding_files)
    print("Combined shape of embeddings:", full_embeddings.shape)
else:
    print('No embedding chunks were found!')
    exit()
#%%
def setup_disk_based_ivfflat_index(file_path, nlist, data):
    dimension = data.shape[1]
    # Define a quantizer index; this part stays in RAM
    quantizer = faiss.IndexFlatL2(dimension)

    # Create the on-disk index
    index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_L2)
    index = faiss.index_factory(dimension, f"IVF{nlist},Flat", faiss.IO_FLAG_ONDISK_SAME_DIR)
    index = faiss.IndexIDMap2(index)
    index.io_flags = faiss.IO_FLAG_ONDISK_SAME_DIR
    
    # Prepare the index file (remove it first if it exists)
    faiss.write_index(index, file_path)

    assert not index.is_trained
    if isinstance(data, torch.Tensor):
        data = data.numpy()
        gc.collect()
    if data.dtype != np.float32:
        data = data.astype(np.float32)
        gc.collect()
        print("convert successful!")

    index.train(data)
    assert index.is_trained
    faiss.write_index(index, file_path)
    return index

# ids = np.arange(full_embeddings.shape[0])
index = setup_disk_based_ivfflat_index('disk_index.index', nlist = 100, data = full_embeddings)
gc.collect()
#%%
def add_data_to_index_in_chunks(index, data, chunk_size = 10000):

    num_chunks = (data.shape[0] + chunk_size - 1) // chunk_size  # Calculate number of chunks needed
    for chunk_idx in range(num_chunks):
        # Define the start and end of the chunk
        start_idx = chunk_idx * chunk_size
        end_idx = min((chunk_idx + 1) * chunk_size, data.shape[0])
        chunk = data[start_idx:end_idx]

        if isinstance(chunk, torch.Tensor):
            chunk = chunk.numpy()
        if chunk.dtype != np.float32:
            chunk = chunk.astype(np.float32)

        # Use add_with_ids if you need to keep track of IDs
        ids = np.arange(chunk.shape[0]) + chunk_idx * chunk.shape[0]  # Example ID generation
        index.add_with_ids(chunk, ids)
        print(f"Chunk {chunk_idx} added to index.")
        gc.collect()



# Assuming chunks is a generator or list of chunks
add_data_to_index_in_chunks(index, full_embeddings, chunk_size = 1000000)
gc.collect()


#%%

