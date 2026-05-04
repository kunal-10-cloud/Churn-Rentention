import os
import glob
import logging
from typing import List

_logger = logging.getLogger(__name__)

# Fallback memory arrays if libraries fail to load (to prevent breaking agent)
_KNOWLEDGE_BASE: List[str] = []

try:
    import chromadb
    from sentence_transformers import SentenceTransformer
    
    # Initialize components
    # We want a local transient or persistent client. 
    # Use ephemeral client for simplicity as it lives in memory,
    # or PersistentClient to store in disk. We will use PersistentClient 
    # to avoid re-embedding on every run if possible, but ephemeral is also fine.
    # Let's use PersistentClient in memory-like or purely local path.
    DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "chromadb")
    
    _chroma_client = chromadb.PersistentClient(path=DB_PATH)
    try:
        _collection = _chroma_client.get_or_create_collection(name="retention_kb")
    except Exception as e:
        _logger.warning(f"Could not create chromadb collection: {e}")
        _collection = None
        
    _model = SentenceTransformer("all-MiniLM-L6-v2")
    _RAG_AVAILABLE = True

except ImportError:
    _logger.warning("chromadb or sentence_transformers not installed. RAG will fallback gracefully.")
    _RAG_AVAILABLE = False
    _chroma_client = None
    _collection = None
    _model = None


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """A very basic text chunker."""
    words = text.split()
    chunks = []
    
    # Simple chunking by word count
    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
        i += chunk_size - overlap
        
    return chunks

def build_index() -> None:
    """Load documents, generate embeddings, and store in ChromaDB."""
    if not _RAG_AVAILABLE or _collection is None:
        _logger.warning("RAG dependencies unavailable. Skipping index build.")
        return

    # To avoid reinstalling if it's already there, we can check if it has items.
    if _collection.count() > 0:
        _logger.info("ChromaDB already has items. Skipping rebuild.")
        return

    kb_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "knowledge_base")
    md_files = glob.glob(os.path.join(kb_dir, "*.md")) + glob.glob(os.path.join(kb_dir, "*.txt"))

    all_chunks = []
    all_metadatas = []
    all_ids = []

    doc_counter = 0
    for file_path in md_files:
        filename = os.path.basename(file_path)
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                
            chunks = chunk_text(content, chunk_size=100, overlap=20)
            
            for i, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                all_metadatas.append({"source": filename, "chunk": str(i)})
                all_ids.append(f"{filename}_{i}")
                
            doc_counter += 1
        except Exception as e:
            _logger.error(f"Error reading file {file_path}: {e}")

    if not all_chunks:
        _logger.info("No documents found to build index.")
        return
        
    # Generate embeddings and add to chroma
    try:
        embeddings = _model.encode(all_chunks).tolist()
        
        # Add to chroma DB
        _collection.add(
            ids=all_ids,
            embeddings=embeddings,
            documents=all_chunks,
            metadatas=all_metadatas
        )
        _logger.info(f"Built knowledge base index with {len(all_chunks)} chunks from {doc_counter} files.")
    except Exception as e:
        _logger.error(f"Failed to ingest documents into ChromaDB: {e}")

def retrieve(query: str, top_k: int = 3) -> List[str]:
    """Retrieve top-k relevant chunks based on query."""
    if not _RAG_AVAILABLE or _collection is None:
        return []
        
    try:
        # Convert query to embedding
        query_embedding = _model.encode(query).tolist()
        
        # Query chromadb
        results = _collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        
        if results and "documents" in results and results["documents"]:
            # results["documents"] is a list of lists 
            # e.g. [['doc1', 'doc2']]
            return results["documents"][0]
            
    except Exception as e:
        _logger.error(f"Error retrieving from RAG: {e}")
        
    return []

# Optional: expose a method to force initialization
def init_rag():
    build_index()

# Automatically build index upon import if needed
# but better to let the app invoke it or invoke here
try:
    init_rag()
except Exception as e:
    pass
