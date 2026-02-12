import os
import fnmatch
import mimetypes
import Stemmer
import pickle
import ast
from multiprocessing.dummy import Pool as ThreadPool  # 线程池版本的 Pool
from multiprocessing import Pool, cpu_count
from functools import partial
import numpy as np
from typing import Dict, Optional, Dict, Union, List
from networkx import MultiDiGraph
from llama_index.core import SimpleDirectoryReader
from llama_index.core import Document
from llama_index.core.node_parser import SimpleFileNodeParser
from llama_index.retrievers.bm25 import BM25Retriever
from zerorepo.utils.repo_index.index.epic_split import EpicSplitter
from ..env import RepoEntitySearcher
from zerorepo.utils.repo import is_test_file
from zerorepo.rpg_gen.base.rpg import NodeType, RPG
from zerorepo.rpg_gen.base.node import RepoSkeleton

import warnings
warnings.simplefilter('ignore', FutureWarning)

# --- 1. 定义顶层辅助函数 (必须在类/主函数之外) ---
def _process_doc_chunk(args):
    """
    Worker function to process a chunk of documents.
    """
    docs_chunk, splitter_config = args
    
    # 在子进程中重新初始化 splitter，避免传递复杂对象带来的序列化问题
    splitter = EpicSplitter(**splitter_config)
    
    # 执行切分
    nodes = splitter.get_nodes_from_documents(docs_chunk, show_progress=False)
    return nodes

def build_code_retriever_from_repo(
    skeleton: RepoSkeleton,
    similarity_top_k: int = 10,
    min_chunk_size: int = 100,
    chunk_size: int = 500,
    max_chunk_size: int = 2000,
    hard_token_limit: int = 2000,
    max_chunks: int = 200,
    persist_path: Optional[str] = None,
    show_progress: bool = False,
):
    """Build a BM25 retriever from the code repository (fully serial version)."""

    test_patterns = [
        '**/test/**',
        '**/tests/**',
        '**/test_*.py',
        '**/*_test.py',
    ]

    def is_test_file_local(file_path: str) -> bool:
        fp = file_path.lstrip("/")
        return any(fnmatch.fnmatch(fp, pattern) for pattern in test_patterns)

    file_map = skeleton.get_file_code_map()

    docs = []
    for file_path, code in file_map.items():
        if is_test_file_local(file_path):
            continue

        metadata = {
            "file_path": file_path,
            "file_name": os.path.basename(file_path),
            "file_type": mimetypes.guess_type(file_path)[0],
            "category": "implementation",
        }

        docs.append(Document(text=code, metadata=metadata, doc_id=file_path))

    splitter = EpicSplitter(
        min_chunk_size=min_chunk_size,
        chunk_size=chunk_size,
        max_chunk_size=max_chunk_size,
        hard_token_limit=hard_token_limit,
        max_chunks=max_chunks,
        repo_path=".",   
    )

    if show_progress:
        print(f"Splitting {len(docs)} documents into nodes (single process)...")

    prepared_nodes = splitter.get_nodes_from_documents(
        docs,
        show_progress=show_progress,
    )
    
    if show_progress:
        print(f"Building BM25 index with {len(prepared_nodes)} nodes...")

    retriever = BM25Retriever.from_defaults(
        nodes=prepared_nodes,
        similarity_top_k=similarity_top_k,
        stemmer=Stemmer.Stemmer("english"),
        language="english",
    )

    if persist_path:
        retriever.persist(persist_path)

    return retriever


def build_retriever_from_persist_dir(path: str):
    return BM25Retriever.from_persist_dir(path)


def build_module_retriever(
    rpg: Optional[RPG] = None,
    entity_searcher: Optional[RepoEntitySearcher] = None,
    search_scope: NodeType = 'all',
    similarity_top_k: int = 10,
):
    """
    Build a BM25 retriever for module/entity search.

    Args:
        rpg: RPG instance (preferred). Will create entity_searcher from it.
        entity_searcher: RepoEntitySearcher instance (alternative to rpg).
        search_scope: Type of nodes to include ('all' or specific NodeType).
        similarity_top_k: Number of top results to return.
    """
    assert search_scope in [NodeType.DIRECTORY, NodeType.FILE, NodeType.CLASS, NodeType.METHOD, NodeType.FUNCTION, 'all']
    assert rpg is not None or isinstance(entity_searcher, RepoEntitySearcher), \
        "Either rpg or entity_searcher must be provided"

    if entity_searcher is None:
        entity_searcher = RepoEntitySearcher(rpg=rpg)

    if entity_searcher.G is None:
        raise ValueError("Entity searcher has no dependency graph. Ensure RPG has dep_graph set.")

    selected_nodes = []
    for nid in entity_searcher.G:
        if is_test_file(nid):
            continue

        ndata_list = entity_searcher.get_node_data([nid])
        if not ndata_list:
            continue
        ndata = ndata_list[0]
        ndata['nid'] = nid

        if search_scope == 'all' or ndata['type'] == search_scope:
            selected_nodes.append(ndata)

    parser = SimpleFileNodeParser()
    docs = [Document(text=node['nid']) for node in selected_nodes]
    nodes = parser.get_nodes_from_documents(docs)

    return BM25Retriever.from_defaults(
        nodes=nodes,
        similarity_top_k=similarity_top_k,
        stemmer=Stemmer.Stemmer("english"),
        language="english",
    )