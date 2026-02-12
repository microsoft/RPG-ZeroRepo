"""
Faiss-based vector database for feature tree management.
Compatible with the existing ZeroRepo FaissDocDB implementation.
Provides efficient similarity search for feature paths and content.
"""

import json
import logging
import os
from typing import Any, Dict, List, Optional, Union
import numpy as np
import faiss
import torch
from sentence_transformers import SentenceTransformer
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
import math
from tqdm import tqdm

class FaissDocDB:
    """
    A Faiss-based document database for efficient similarity search.
    Compatible with ZeroRepo's FaissDocDB implementation with enhanced features.
    
    Stores metadata in the format: {"key": query_text, "doc": document_content}
    Uses L2 normalized embeddings with IndexIDMap for custom ID mapping.
    """
    
    def __init__(self, 
        model_name: str = "infly/inf-retriever-v1", 
        use_gpu: bool = True,
        use_parallel_encoding: bool = True,
        default_num_workers: int = 4
    ):
        """
        Initialize the FaissDocDB.
        
        Args:
            model_name: Name of the sentence transformer model (default: infly/inf-retriever-v1)
            use_gpu: Whether to use GPU for embedding and Faiss index
            use_parallel_encoding: Whether to use parallel encoding for large text batches
            default_num_workers: Default number of workers for parallel encoding
        """
        self.model_name = model_name
        self.use_gpu = use_gpu
        self.use_parallel_encoding = use_parallel_encoding
        self.default_num_workers = default_num_workers
        self.index = None
        self.id2doc = {}  # Format: {id: {"key": key, "doc": doc}}
        self.next_id = 0
        
        # Initialize sentence transformer
        device = 'cuda' if self.use_gpu and torch.cuda.is_available() else 'cpu'
        try:
            self.model = SentenceTransformer(self.model_name, device=device)
            self.model.max_seq_length = 512
        except Exception as e:
            logging.warning(f"Failed to load model '{model_name}', falling back to 'all-MiniLM-L6-v2': {e}")
            self.model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
            self.model.max_seq_length = 512
            
        logging.info(f"Initialized FaissDocDB with model '{self.model_name}' on {device}")
    
    def _encode(self, texts: List[str], batch_size: int = 32, num_workers: Optional[int] = None, 
                show_progress: bool = True) -> np.ndarray:
        """
        Encode texts into embeddings using sentence transformer with parallel processing.
        
        Args:
            texts: List of text strings to encode
            batch_size: Batch size for encoding
            num_workers: Number of parallel workers for encoding
            show_progress: Whether to show progress bar for encoding
            
        Returns:
            Numpy array of embeddings
        """
        # Set default num_workers if None
        if num_workers is None:
            num_workers = self.default_num_workers
        
        # Ensure num_workers is valid (not None and > 0)
        if num_workers is None or num_workers <= 0:
            num_workers = 1
            
        try:
            # Ensure all texts are strings and not empty
            cleaned_texts = []
            for text in texts:
                if text is None:
                    cleaned_texts.append("")
                elif isinstance(text, dict):
                    # Convert dict to string representation
                    cleaned_texts.append(str(text))
                else:
                    cleaned_texts.append(str(text))
            
            # For very large datasets (>1M), disable parallel encoding to avoid memory issues
            if len(cleaned_texts) > 1000000:
                logging.warning(f"Large dataset detected ({len(cleaned_texts)} texts). Using single-threaded encoding to avoid memory issues.")
                # Process in chunks to avoid memory issues
                all_embeddings = []
                chunk_size = 10000
                
                if show_progress:
                    with tqdm(total=len(cleaned_texts), desc="🚀 Sequential Encoding", unit="texts",
                             bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]',
                             dynamic_ncols=True, leave=True) as pbar:
                        
                        for i in range(0, len(cleaned_texts), chunk_size):
                            chunk = cleaned_texts[i:i + chunk_size]
                            chunk_embeddings = self.model.encode(
                                chunk,
                                batch_size=batch_size,
                                convert_to_numpy=True,
                                show_progress_bar=False
                            )
                            all_embeddings.append(chunk_embeddings)
                            pbar.update(len(chunk))
                            pbar.set_postfix({'Chunk': f"{i//chunk_size + 1}/{math.ceil(len(cleaned_texts)/chunk_size)}"})
                else:
                    for i in range(0, len(cleaned_texts), chunk_size):
                        chunk = cleaned_texts[i:i + chunk_size]
                        chunk_embeddings = self.model.encode(
                            chunk,
                            batch_size=batch_size,
                            convert_to_numpy=True,
                            show_progress_bar=False
                        )
                        all_embeddings.append(chunk_embeddings)
                        if i % 100000 == 0 and i > 0:
                            logging.info(f"Encoding progress: {i}/{len(cleaned_texts)} texts completed")
                
                return np.vstack(all_embeddings)
            
            # If parallel encoding is disabled or texts are small, use single thread
            if not self.use_parallel_encoding or len(cleaned_texts) <= batch_size * 2:
                embeddings = self.model.encode(
                    cleaned_texts,
                    batch_size=batch_size,
                    convert_to_numpy=True,
                    show_progress_bar=show_progress
                )
                return embeddings
            
            # Use cleaned_texts for the rest of the function
            texts = cleaned_texts
            
            # For moderately sized datasets, use limited parallel processing
            logging.info(f"Using parallel encoding with {num_workers} workers for {len(texts)} texts")
            
            # Limit workers for larger datasets to avoid thread contention
            if len(texts) > 100000:
                num_workers = min(num_workers, 2)
                logging.info(f"Limiting to {num_workers} workers for large dataset")
            
            # Calculate number of batches per worker
            total_batches = math.ceil(len(texts) / batch_size)
            batches_per_worker = max(1, total_batches // num_workers)
            
            # Create text chunks for each worker
            text_chunks = []
            chunk_size = batches_per_worker * batch_size
            for i in range(0, len(texts), chunk_size):
                chunk = texts[i:i + chunk_size]
                if chunk:  # Only add non-empty chunks
                    text_chunks.append((i, chunk))
            
            # Process chunks in parallel
            embeddings_list = [None] * len(text_chunks)
            
            def encode_chunk(idx: int, chunk: List[str]) -> tuple:
                """Encode a chunk of texts"""
                chunk_embeddings = self.model.encode(
                    chunk,
                    batch_size=batch_size,
                    convert_to_numpy=True,
                    show_progress_bar=False  # Disable per-chunk progress bars
                )
                return idx, chunk_embeddings
            
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                # Submit all tasks and create mapping
                future_to_info = {}
                for idx, (_, chunk) in enumerate(text_chunks):
                    future = executor.submit(encode_chunk, idx, chunk)
                    future_to_info[future] = (idx, len(chunk))
                
                # Collect results with optional real-time progress bar
                if show_progress:
                    with tqdm(total=len(texts), desc="🚀 Parallel Encoding", unit="texts", 
                             bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]',
                             dynamic_ncols=True, leave=True) as pbar:
                        
                        completed_chunks = 0
                        for future in as_completed(future_to_info):
                            idx, chunk_len = future_to_info[future]
                            try:
                                _, chunk_embeddings = future.result()
                                embeddings_list[idx] = chunk_embeddings
                                completed_chunks += 1
                                
                                # Update progress bar with real-time feedback
                                pbar.update(chunk_len)
                                pbar.set_postfix({
                                    'Workers': num_workers, 
                                    'Chunks': f"{completed_chunks}/{len(text_chunks)}",
                                    'Batch': chunk_len
                                }, refresh=True)
                                
                            except Exception as e:
                                logging.error(f"Error encoding chunk {idx}: {e}")
                                raise
                else:
                    # No progress bar, just collect results
                    completed = 0
                    for future in as_completed(future_to_info):
                        idx, chunk_len = future_to_info[future]
                        try:
                            _, chunk_embeddings = future.result()
                            embeddings_list[idx] = chunk_embeddings
                            completed += chunk_len
                            
                            # Log progress periodically without progress bar
                            if completed % max(1000, len(texts) // 10) == 0:
                                logging.info(f"Encoding progress: {completed}/{len(texts)} texts completed")
                                
                        except Exception as e:
                            logging.error(f"Error encoding chunk {idx}: {e}")
                            raise
            
            # Concatenate all embeddings in the correct order
            all_embeddings = np.vstack(embeddings_list)
            
            logging.info(f"Parallel encoding completed. Shape: {all_embeddings.shape}")
            return all_embeddings
            
        except Exception as e:
            logging.error(f"Failed to encode texts: {e}")
            raise
    
    def build(self, keys: List[str], docs: List[Any], batch_size: int = 32, num_workers: Optional[int] = None, 
              show_progress: bool = True) -> None:
        """
        Build the Faiss index from keys and documents with optimized processing.
        
        Args:
            keys: List of text keys to encode and index
            docs: List of associated documents (can be any JSON-serializable data)
            batch_size: Batch size for embedding generation
            num_workers: Number of parallel workers for encoding
            show_progress: Whether to show progress bar during encoding
        """
        assert len(keys) == len(docs), "keys and docs must be same length"
        
        if not keys:
            raise ValueError("Cannot build index from empty key list")
        
        logging.info(f"Building Faiss index for {len(keys)} documents...")
        
        # Reset state
        self.id2doc = {}
        self.next_id = 0
        
        # Optimize batch size and workers based on dataset size
        optimal_batch_size, optimal_workers = self._optimize_processing_params(
            len(keys), batch_size, num_workers
        )
        
        # Encode keys to embeddings with optimized parameters
        embeddings = self._encode(keys, optimal_batch_size, optimal_workers, show_progress)
        
        # L2 normalize embeddings (for cosine similarity)
        faiss.normalize_L2(embeddings)
        
        # Create Faiss index with optimized parameters
        dim = embeddings.shape[1]
        index_flat = faiss.IndexFlatL2(dim)
        index = faiss.IndexIDMap(index_flat)
        
        # Move to GPU if requested and beneficial
        if self.use_gpu and torch.cuda.is_available() and len(keys) > 1000:
            try:
                index = faiss.index_cpu_to_all_gpus(index)
                logging.info("Using GPU for Faiss index")
            except Exception as e:
                logging.warning(f"Failed to use GPU for Faiss index: {e}. Using CPU.")
        
        self.index = index
        
        # Add embeddings with IDs in batches for better memory management
        ids = np.arange(len(keys)).astype(np.int64)
        
        if len(keys) > 50000:  # For large datasets, add in chunks
            chunk_size = 10000
            logging.info(f"Adding embeddings in chunks of {chunk_size} for memory efficiency")
            
            if show_progress:
                with tqdm(
                    total=len(keys), 
                    desc="🔗 Adding to Index", 
                    unit="docs",
                    bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]',
                    leave=False
                ) as pbar:
                    
                    for i in range(0, len(keys), chunk_size):
                        end_idx = min(i + chunk_size, len(keys))
                        chunk_embeddings = embeddings[i:end_idx]
                        chunk_ids = ids[i:end_idx]
                        
                        self.index.add_with_ids(chunk_embeddings, chunk_ids)
                        pbar.update(end_idx - i)
            else:
                for i in range(0, len(keys), chunk_size):
                    end_idx = min(i + chunk_size, len(keys))
                    chunk_embeddings = embeddings[i:end_idx]
                    chunk_ids = ids[i:end_idx]
                    self.index.add_with_ids(chunk_embeddings, chunk_ids)
        else:
            # For smaller datasets, add all at once
            self.index.add_with_ids(embeddings, ids)
        
        # Store documents with metadata
        for i, (key, doc) in enumerate(zip(keys, docs)):
            # Ensure key is a string
            key_str = str(key) if key is not None else ""
            # Doc can be a list or string - preserve its type
            self.id2doc[i] = {"key": key_str, "doc": doc}
        
        self.next_id = len(keys)
        logging.info(f"Successfully built Faiss index with {self.index.ntotal} documents")
    
    def _optimize_processing_params(
        self, 
        dataset_size: int, 
        batch_size: int, 
        num_workers: Optional[int]
    ) -> tuple:
        """
        Optimize batch size and number of workers based on dataset size.
        
        Args:
            dataset_size: Number of documents to process
            batch_size: Original batch size
            num_workers: Original number of workers
            
        Returns:
            Tuple of (optimized_batch_size, optimized_num_workers)
        """
        if num_workers is None:
            num_workers = self.default_num_workers
        
        # Optimize batch size based on dataset size
        if dataset_size < 1000:
            # Small dataset: smaller batch, fewer workers
            optimal_batch_size = min(batch_size, 16)
            optimal_workers = min(num_workers, 2)
        elif dataset_size < 10000:
            # Medium dataset: moderate batch and workers
            optimal_batch_size = min(batch_size, 32)
            optimal_workers = min(num_workers, 4)
        elif dataset_size < 100000:
            # Large dataset: larger batch, more workers
            optimal_batch_size = max(batch_size, 64)
            optimal_workers = min(num_workers, 8)
        elif dataset_size < 1000000:
            # Very large dataset: larger batch but limit workers to avoid contention
            optimal_batch_size = max(batch_size, 128)
            optimal_workers = min(num_workers, 4)
        else:
            # Extremely large dataset: disable parallel encoding
            optimal_batch_size = max(batch_size, 256)
            optimal_workers = 1  # Force single-threaded for very large datasets
        
        logging.info(
            f"Optimized processing: batch_size={optimal_batch_size}, "
            f"num_workers={optimal_workers} for dataset_size={dataset_size}"
        )
        
        return optimal_batch_size, optimal_workers
    
    def add(self, keys: List[str], docs: List[Any], batch_size: int = 32, num_workers: Optional[int] = None,
            show_progress: bool = True) -> None:
        """
        Add new keys and documents to existing index.
        
        Args:
            keys: List of new text keys to add
            docs: List of associated documents
            batch_size: Batch size for embedding generation
            num_workers: Number of parallel workers for encoding
            show_progress: Whether to show progress bar during encoding
        """
        if not keys:
            return
            
        if len(keys) != len(docs):
            raise ValueError("Number of keys and documents must match")
            
        if self.index is None:
            self.build(keys, docs, batch_size, num_workers, show_progress)
            return
            
        # Encode new keys
        embeddings = self._encode(keys, batch_size, num_workers, show_progress)
        faiss.normalize_L2(embeddings)
        
        # Add to index with new IDs
        new_ids = np.arange(self.next_id, self.next_id + len(keys)).astype(np.int64)
        self.index.add_with_ids(embeddings, new_ids)
        
        # Update documents and ID mapping
        for i, (key, doc) in enumerate(zip(keys, docs)):
            self.id2doc[self.next_id + i] = {"key": key, "doc": doc}
            
        self.next_id += len(keys)
        logging.info(f"Added {len(keys)} documents to index. Total: {self.index.ntotal}")
    
    def search(
        self, 
        queries: Union[str, List[str]], 
        score_threshold: float = 0.5,
        top_k: int = 10,
        num_workers: Optional[int] = None,
        show_progress: bool = False
    ) -> List[List[Dict[str, Any]]]:
        """
        Search for similar documents using cosine similarity.
        
        Args:
            queries: Query string(s) to search for
            score_threshold: Minimum similarity score (not used in current implementation)
            top_k: Number of top results to return per query
            num_workers: Number of parallel workers for encoding queries
            
        Returns:
            List of results for each query. Each result is a list of dicts with:
            - 'id': Document ID
            - 'key': The original key text
            - 'doc': The document content
            - 'score': L2 distance score (lower is better)
        """
        if self.index is None:
            raise RuntimeError("Index not built. Call build() first.")
            
        # Ensure queries is a list
        if isinstance(queries, str):
            queries = [queries]
            
        # Encode queries
        query_vec = self._encode(queries, batch_size=32, num_workers=num_workers)
        faiss.normalize_L2(query_vec)  # L2 normalize query vectors
        
        if query_vec.ndim == 1:
            query_vec = np.expand_dims(query_vec, axis=0)
        
        # Perform search
        k_search = min(len(self.id2doc), 2048)  # Limit search to avoid memory issues
        distances, indices = self.index.search(query_vec, k_search)
        
        # Format results for each query
        batched_results = []
        
        for q_idx in range(len(queries)):
            query_results = []
            
            for dist, idx in zip(distances[q_idx], indices[q_idx]):
                if idx == -1:  # Faiss uses -1 for missing results
                    continue
                    
                score = float(dist)
                doc_metadata = self.id2doc.get(idx, {"key": "", "doc": ""})
                
                query_results.append({
                    "id": int(idx),
                    "key": doc_metadata["key"],
                    "doc": doc_metadata["doc"],
                    "score": score
                })
            
            # Sort by score (lower is better for L2 distance) and limit to top_k
            query_results.sort(key=lambda x: x["score"])
            batched_results.append(query_results[:top_k])
        
        return batched_results
    
    def save(self, index_path: str = "index.faiss", doc_path: str = "id2doc.json") -> None:
        """
        Save the index and documents to disk.
        
        Args:
            index_path: Path to save the Faiss index
            doc_path: Path to save the document mapping (JSON)
        """
        if self.index is None:
            raise RuntimeError("No index to save. Build index first.")
        
        # Create directories if needed
        dir_path = os.path.dirname(index_path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
            
        doc_dir_path = os.path.dirname(doc_path)
        if doc_dir_path:
            os.makedirs(doc_dir_path, exist_ok=True)
            
        # Save Faiss index (move to CPU first if on GPU)
        if self.use_gpu and hasattr(self.index, '__class__') and 'Gpu' in self.index.__class__.__name__:
            cpu_index = faiss.index_gpu_to_cpu(self.index)
            faiss.write_index(cpu_index, index_path)
        else:
            faiss.write_index(self.index, index_path)
            
        # Save document mapping
        with open(doc_path, "w", encoding="utf-8") as f:
            json.dump(self.id2doc, f, ensure_ascii=False, indent=2)
            
        logging.info(f"Saved index to '{index_path}' and documents to '{doc_path}'")
    
    def load(self, index_path: str = "index.faiss", doc_path: str = "id2doc.json") -> None:
        """
        Load index and documents from disk.
        
        Args:
            index_path: Path to the Faiss index file
            doc_path: Path to the document mapping JSON file
        """
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"Index file not found: {index_path}")
        if not os.path.exists(doc_path):
            raise FileNotFoundError(f"Document file not found: {doc_path}")
            
        # Load Faiss index
        cpu_index = faiss.read_index(index_path)
        
        # Move to GPU if requested
        if self.use_gpu and torch.cuda.is_available():
            try:
                self.index = faiss.index_cpu_to_all_gpus(cpu_index)
                logging.info("Moved loaded index to GPU")
            except Exception as e:
                logging.warning(f"Failed to move index to GPU: {e}")
                self.index = cpu_index
        else:
            self.index = cpu_index
                
        # Load document mapping
        with open(doc_path, "r", encoding="utf-8") as f:
            self.id2doc = json.load(f)

        # Convert string keys back to integers (JSON serialization converts int keys to strings)
        self.id2doc = {int(k): v for k, v in self.id2doc.items()}
        self.next_id = max(self.id2doc.keys(), default=-1) + 1
        
        logging.info(f"Loaded index with {self.index.ntotal} documents from '{index_path}'")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the index."""
        if self.index is None:
            return {"status": "not_built"}
            
        return {
            "status": "ready",
            "total_documents": self.index.ntotal,
            "model_name": self.model_name,
            "use_gpu": self.use_gpu,
            "next_id": self.next_id
        }


def _process_subtree_worker(args: tuple) -> Dict[str, List[str]]:
    """
    Worker function to process a subtree and return hash table entries.
    This function needs to be at module level for multiprocessing.
    
    Args:
        args: Tuple of (key, value, current_path)
        
    Returns:
        Dictionary mapping feature names to lists of paths for this subtree
    """
    key, value, current_path = args
    local_hash_table = {}
    
    def _build_recursive_worker(node: Union[Dict[str, Any], List[str]], path: str):
        if isinstance(node, dict):
            for node_key, node_value in node.items():
                new_path = f"{path}/{node_key}" if path else node_key
                
                # Add this feature name to local hash table
                if node_key not in local_hash_table:
                    local_hash_table[node_key] = []
                local_hash_table[node_key].append(new_path)
                
                # Recurse into children
                _build_recursive_worker(node_value, new_path)
                
        elif isinstance(node, list):
            for item in node:
                new_path = f"{path}/{item}" if path else item
                
                # Add this feature name to local hash table
                if item not in local_hash_table:
                    local_hash_table[item] = []
                local_hash_table[item].append(new_path)
    
    # Process the current key-value pair
    new_path = f"{current_path}/{key}" if current_path else key
    if key not in local_hash_table:
        local_hash_table[key] = []
    local_hash_table[key].append(new_path)
    
    # Process the value (subtree)
    _build_recursive_worker(value, new_path)
    
    return local_hash_table


def build_feature_path_hash_table(
    feature_tree: Dict[str, Any], 
    prefix: str = "",
    use_multiprocessing: bool = True,
    num_workers: Optional[int] = None,
    show_progress: bool = True
) -> Dict[str, List[str]]:
    """
    Build a hash table mapping feature names to their full paths in the feature tree.
    Uses multiprocessing for faster processing of large feature trees.
    
    Args:
        feature_tree: Nested dictionary representing the feature tree
        prefix: Current path prefix (used internally for recursion)
        use_multiprocessing: Whether to use multiprocessing for parallel processing
        num_workers: Number of worker processes (default: CPU count)
        show_progress: Whether to show progress bar
        
    Returns:
        Dictionary mapping feature names to lists of full paths containing that feature
    """
    if not isinstance(feature_tree, dict) or not feature_tree:
        return {}
    
    # For small trees or when multiprocessing is disabled, use sequential processing
    if not use_multiprocessing or len(feature_tree) < 4:
        return _build_feature_path_hash_table_sequential(feature_tree, prefix)
    
    if num_workers is None:
        num_workers = min(cpu_count(), len(feature_tree))
    
    # Prepare work items for parallel processing
    work_items = [(key, value, prefix) for key, value in feature_tree.items()]
    
    hash_table = {}
    
    try:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            # Submit all subtree processing tasks
            future_to_key = {
                executor.submit(_process_subtree_worker, item): item[0] 
                for item in work_items
            }
            
            # Process results with progress bar
            if show_progress:
                with tqdm(
                    total=len(work_items), 
                    desc="🌲 Building Hash Table", 
                    unit="subtrees",
                    bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]',
                    dynamic_ncols=True,
                    leave=True
                ) as pbar:
                    
                    for future in as_completed(future_to_key):
                        key = future_to_key[future]
                        try:
                            local_hash_table = future.result()
                            
                            # Merge local results into global hash table
                            for feature_name, paths in local_hash_table.items():
                                if feature_name not in hash_table:
                                    hash_table[feature_name] = []
                                hash_table[feature_name].extend(paths)
                            
                            pbar.update(1)
                            pbar.set_postfix({
                                'Workers': num_workers,
                                'Features': len(hash_table)
                            }, refresh=True)
                            
                        except Exception as e:
                            logging.error(f"Error processing subtree '{key}': {e}")
                            # Fall back to sequential processing for this subtree
                            local_result = _build_feature_path_hash_table_sequential({key: feature_tree[key]}, prefix)
                            for feature_name, paths in local_result.items():
                                if feature_name not in hash_table:
                                    hash_table[feature_name] = []
                                hash_table[feature_name].extend(paths)
                            pbar.update(1)
            else:
                # Process without progress bar
                for future in as_completed(future_to_key):
                    key = future_to_key[future]
                    try:
                        local_hash_table = future.result()
                        
                        # Merge local results into global hash table
                        for feature_name, paths in local_hash_table.items():
                            if feature_name not in hash_table:
                                hash_table[feature_name] = []
                            hash_table[feature_name].extend(paths)
                        
                    except Exception as e:
                        logging.error(f"Error processing subtree '{key}': {e}")
                        # Fall back to sequential processing for this subtree
                        local_result = _build_feature_path_hash_table_sequential({key: feature_tree[key]}, prefix)
                        for feature_name, paths in local_result.items():
                            if feature_name not in hash_table:
                                hash_table[feature_name] = []
                            hash_table[feature_name].extend(paths)
        
        logging.info(f"Hash table built with {len(hash_table)} unique features using {num_workers} workers")
        
    except Exception as e:
        logging.warning(f"Multiprocessing failed: {e}, falling back to sequential processing")
        hash_table = _build_feature_path_hash_table_sequential(feature_tree, prefix)
    
    return hash_table


def _build_feature_path_hash_table_sequential(
    feature_tree: Dict[str, Any], 
    prefix: str = ""
) -> Dict[str, List[str]]:
    """
    Sequential version of hash table building (original implementation).
    Used as fallback and for small trees.
    """
    hash_table = {}
    
    def _build_recursive(node: Union[Dict[str, Any], List[str]], current_path: str):
        if isinstance(node, dict):
            for key, value in node.items():
                new_path = f"{current_path}/{key}" if current_path else key
                
                # Add this feature name to hash table
                if key not in hash_table:
                    hash_table[key] = []
                hash_table[key].append(new_path)
                
                # Recurse into children
                _build_recursive(value, new_path)
                
        elif isinstance(node, list):
            for item in node:
                new_path = f"{current_path}/{item}" if current_path else item
                
                # Add this feature name to hash table
                if item not in hash_table:
                    hash_table[item] = []
                hash_table[item].append(new_path)
    
    _build_recursive(feature_tree, prefix)
    return hash_table