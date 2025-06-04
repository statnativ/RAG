# storage/vector_store.py
import pickle
import faiss
import numpy as np


class VectorStore:
    def __init__(self, dim):
        self.index = faiss.IndexFlatL2(dim)
        self.metadata = []

    def add(self, vectors, metadata):
        self.index.add(np.array(vectors).astype("float32"))
        self.metadata.extend(metadata)

    def save(self, path):
        faiss.write_index(self.index, path + ".index")
        with open(path + ".meta", "wb") as f:
            pickle.dump(self.metadata, f)

    def load(self, path):
        self.index = faiss.read_index(path + ".index")
        with open(path + ".meta", "rb") as f:
            self.metadata = pickle.load(f)
