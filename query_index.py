#%%
import faiss
import gc
import numpy as np
import os
import re
import torch

#%%
def load_index_memory_mapped(file_path):
    # The flag `faiss.IO_FLAG_MMAP` enables memory-mapping
    index = faiss.read_index(file_path, faiss.IO_FLAG_MMAP)
    return index

index_file_path = 'disk_index.index'
index = load_index_memory_mapped(index_file_path)

def search_index(index, queries, k=5):
    # Ensure queries are in the correct format
    if isinstance(queries, torch.Tensor):
        queries = queries.numpy()
    if queries.dtype != np.float32:
        queries = queries.astype(np.float32)
    
    distances, indices = index.search(queries, k)
    return distances, indices