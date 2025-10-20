"""
RAG Processor Module

Handles document processing, LightRAG initialization, and RAG system management.
"""

import os
import json
import numpy as np
import openai
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

from lightrag import LightRAG
from lightrag.utils import EmbeddingFunc
from lightrag.llm.openai import openai_complete_if_cache, openai_embed


@dataclass
class DocumentChunk:
    """Represents a document chunk with metadata."""
    content: str
    source_file: str
    chunk_id: int


class RAGProcessor:
    """
    RAG Processor for handling document ingestion and LightRAG operations.
    """

    def __init__(
        self,
        llm_config: Dict[str, str],
        embedding_config: Dict[str, str],
        qdrant_config: Dict[str, str],
        neo4j_config: Dict[str, str],
        working_dir: str = "./lightrag_data",
        chunk_size: int = 1200,
        chunk_overlap: int = 100
    ):
        """
        Initialize the RAG processor.
        
        Args:
            llm_config: Configuration for LLM API
            embedding_config: Configuration for embedding API
            qdrant_config: Configuration for Qdrant vector database
            neo4j_config: Configuration for Neo4j graph database
            working_dir: Working directory for LightRAG data
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
        """
        self.llm_config = llm_config
        self.embedding_config = embedding_config
        self.qdrant_config = qdrant_config
        self.neo4j_config = neo4j_config
        self.working_dir = Path(working_dir)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Create working directory if it doesn't exist
        self.working_dir.mkdir(parents=True, exist_ok=True)

        # Initialize LightRAG components
        self.rag: Optional[LightRAG] = None
        self._setup_lightrag()

    def _setup_lightrag(self):
        """Setup LightRAG with the provided configurations using the direct OpenAI functions."""

        # Create LLM function using LightRAG's OpenAI complete function
        def llm_model_func(prompt, system_prompt=None, **kwargs):
            return openai_complete_if_cache(
                model=self.llm_config['model'],
                api_key=self.llm_config['api_key'],
                base_url=self.llm_config['api_base'],
                prompt=prompt,
                system_prompt=system_prompt,
                **kwargs
            )

        # Create custom embedding function that uses float encoding (not base64)
        # This is needed because some OpenAI-compatible APIs don't support base64 encoding
        async def custom_embedding_func(texts: list[str]) -> np.ndarray:
            """Custom embedding function that uses float encoding format."""
            openai_async_client = openai.AsyncOpenAI(
                api_key=self.embedding_config['api_key'],
                base_url=self.embedding_config['api_base']
            )
            
            response = await openai_async_client.embeddings.create(
                model=self.embedding_config['model'],
                input=texts,
                encoding_format="float"  # Use float instead of base64 for compatibility
            )
            
            return np.array([dp.embedding for dp in response.data])

        embedding_func = EmbeddingFunc(
            embedding_dim=768,  # Default for nomic-embed-text (adjust if needed)
            max_token_size=8191,
            func=custom_embedding_func
        )

        # Initialize LightRAG with proper configuration
        self.rag = LightRAG(
            working_dir=str(self.working_dir),
            llm_model_func=llm_model_func,
            llm_model_name=self.llm_config['model'],
            chunk_token_size=self.chunk_size,
            chunk_overlap_token_size=self.chunk_overlap,
            embedding_func=embedding_func,
            kv_storage="JsonKVStorage",
            vector_storage="QdrantVectorDBStorage",
            graph_storage="Neo4JStorage",
            vector_db_storage_cls_kwargs={
                "url": self.qdrant_config['host'],
                "api_key": self.qdrant_config['api_key'],
                "collection_name": self.qdrant_config['collection_name']
            },
            addon_params={
                "graph_db_config": {
                    "url": self.neo4j_config['uri'],
                    "username": self.neo4j_config['user'],
                    "password": self.neo4j_config['password']
                }
            }
        )

        # Note: LightRAG will initialize storages automatically on first use
        # We don't need to explicitly initialize them here

    def scan_documents(self, repo_path: str, file_extensions: List[str]) -> List[str]:
        """
        Scan repository for documents to process.

        Args:
            repo_path: Path to the repository
            file_extensions: List of file extensions to include

        Returns:
            List of file paths to process
        """
        repo_path = Path(repo_path)
        documents = []

        # Common files/directories to ignore
        ignore_patterns = {
            '.git', '__pycache__', 'node_modules', '.venv', 'venv', 'env',
            'build', 'dist', '.next', '.nuxt', 'target', 'bin', 'obj',
            '.DS_Store', 'Thumbs.db'
        }

        for root, dirs, files in os.walk(repo_path):
            # Filter out ignored directories
            dirs[:] = [d for d in dirs if d not in ignore_patterns]

            for file in files:
                file_path = Path(root) / file

                # Check if file extension is allowed
                if file_path.suffix.lower() in file_extensions:
                    # Skip files that are too large (> 10MB)
                    if file_path.stat().st_size > 10 * 1024 * 1024:
                        print(f"Skipping large file: {file_path}")
                        continue

                    documents.append(str(file_path))

        print(f"Found {len(documents)} documents to process")
        return documents

    def read_document(self, file_path: str) -> str:
        """
        Read and extract text content from a document.

        Args:
            file_path: Path to the document file

        Returns:
            Text content of the document
        """
        try:
            file_path_obj = Path(file_path)
            
            # Handle PDF files
            if file_path_obj.suffix.lower() == '.pdf':
                content = self._read_pdf(file_path)
            else:
                # Handle text files
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()

            # Basic cleaning
            content = content.strip()
            if not content:
                return ""

            # Add file path as header for context
            relative_path = os.path.relpath(file_path, self.repo_path)
            header = f"File: {relative_path}\n{'='*50}\n\n"
            return header + content

        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            return ""

    def _read_pdf(self, file_path: str) -> str:
        """
        Extract text from a PDF file.

        Args:
            file_path: Path to the PDF file

        Returns:
            Extracted text content
        """
        try:
            import pymupdf  # PyMuPDF
            
            text_content = []
            with pymupdf.open(file_path) as doc:
                for page_num, page in enumerate(doc, 1):
                    text = page.get_text()
                    if text.strip():
                        text_content.append(f"--- Page {page_num} ---\n{text}")
            
            return "\n\n".join(text_content)
        except ImportError:
            print("Warning: PyMuPDF not installed. Install with: pip install pymupdf")
            return ""
        except Exception as e:
            print(f"Error extracting text from PDF {file_path}: {e}")
            return ""

    def chunk_text(self, text: str, source_file: str) -> List[DocumentChunk]:
        """
        Split text into chunks with overlap.

        Args:
            text: Text to chunk
            source_file: Source file path

        Returns:
            List of document chunks
        """
        if not text:
            return []

        chunks = []
        start = 0
        chunk_id = 0

        while start < len(text):
            end = start + self.chunk_size

            # If we're not at the end, try to find a good break point
            if end < len(text):
                # Look for sentence endings within the last 100 characters
                last_period = text.rfind('.', end - 100, end)
                last_newline = text.rfind('\n', end - 100, end)
                last_space = text.rfind(' ', end - 100, end)

                # Prefer period, then newline, then space
                break_point = max(last_period, last_newline, last_space)
                if break_point > start:
                    end = break_point + 1

            chunk_content = text[start:end].strip()
            if chunk_content:
                chunks.append(DocumentChunk(
                    content=chunk_content,
                    source_file=source_file,
                    chunk_id=chunk_id
                ))
                chunk_id += 1

            # Move start position with overlap
            start = end - self.chunk_overlap

            # Ensure we make progress
            if start >= len(text) - 1:
                break

        return chunks

    def process_documents(
        self,
        repo_path: str,
        file_extensions: List[str],
        rebuild: bool = False
    ):
        """
        Process all documents in the repository and build the RAG system.

        Args:
            repo_path: Path to the repository
            file_extensions: File extensions to process
            rebuild: Whether to rebuild even if data exists
        """
        self.repo_path = repo_path

        # Check if we need to rebuild
        if not rebuild and self._is_processed():
            print("RAG system already exists. Use --rebuild to force rebuild.")
            return

        # Scan for documents
        document_files = self.scan_documents(repo_path, file_extensions)

        if not document_files:
            print("No documents found to process.")
            return

        print(f"Processing {len(document_files)} documents...")

        # Collect all document contents
        all_documents = []
        
        for i, file_path in enumerate(document_files, 1):
            print(f"Reading {i}/{len(document_files)}: {os.path.basename(file_path)}")

            # Read document
            content = self.read_document(file_path)
            if not content:
                continue

            all_documents.append(content)

        if not all_documents:
            print("No valid document content found.")
            return

        # Insert all documents into LightRAG at once
        # LightRAG will handle chunking internally based on chunk_token_size and chunk_overlap_token_size
        print(f"\nInserting {len(all_documents)} documents into LightRAG...")
        print("LightRAG will handle chunking internally...")
        
        try:
            # Use asyncio.run to properly handle the async insert
            import asyncio
            asyncio.run(self._async_insert_documents(all_documents))
            print(f"✓ Successfully inserted {len(all_documents)} documents")
        except Exception as e:
            print(f"✗ Error inserting documents: {e}")
            import traceback
            traceback.print_exc()
            return

        # Save processing metadata
        self._save_metadata(len(document_files), len(all_documents))

    async def _async_insert_documents(self, documents: List[str]):
        """Insert documents asynchronously."""
        # Initialize storages if not already initialized
        await self.rag.initialize_storages()
        
        # Initialize pipeline status (required for document processing)
        from lightrag.kg.shared_storage import initialize_pipeline_status
        await initialize_pipeline_status()
        
        # Now insert the documents
        await self.rag.ainsert(documents)

    async def aquery(
        self, 
        question: str, 
        mode: str = "hybrid",
        only_need_context: bool = False,
        top_k: int = 60,
        max_chunks: Optional[int] = None
    ) -> str:
        """
        Query the RAG system asynchronously.

        Args:
            question: Question to ask
            mode: Query mode - "naive", "local", "global", "hybrid" (default: "hybrid")
            only_need_context: If True, return only the retrieved context without generating an answer
            top_k: Initial retrieval limit - max entities/edges to fetch from vector/graph DBs (default: 60)
            max_chunks: Number of chunks to send to LLM (each chunk = 1024 tokens)
                        Example: max_chunks=10 → 10,240 tokens sent to LLM
                        If None, uses default 30,000 tokens (≈29 chunks)

        Returns:
            Response from the RAG system (or context if only_need_context=True)
        """
        if not self.rag:
            raise RuntimeError("RAG system not initialized")

        try:
            # Ensure storages are initialized
            await self.rag.initialize_storages()
            
            # Initialize pipeline status
            from lightrag.kg.shared_storage import initialize_pipeline_status
            await initialize_pipeline_status()
            
            # Calculate token limits based on max_chunks if specified
            # Each chunk is 1024 tokens
            if max_chunks is not None:
                tokens = max_chunks * 1024
                print(f"[Max Chunks Control] Limiting to {max_chunks} chunks ({tokens} tokens)")
            else:
                # LightRAG defaults
                tokens = 30000
            
            # Query with the specified mode and parameters
            from lightrag.base import QueryParam
            response = await self.rag.aquery(
                question, 
                param=QueryParam(
                    mode=mode,
                    only_need_context=only_need_context,
                    top_k=top_k,
                    max_entity_tokens=tokens,
                    max_relation_tokens=tokens,
                    max_total_tokens=tokens
                )
            )
            return response
        except Exception as e:
            import traceback
            traceback.print_exc()
            return f"Error querying RAG system: {e}"

    def query(
        self, 
        question: str,
        mode: str = "hybrid",
        top_k: int = 60,
        max_chunks: Optional[int] = None
    ) -> str:
        """
        Query the RAG system synchronously.

        Args:
            question: Question to ask
            mode: Query mode (default: "hybrid")
            top_k: Initial retrieval limit (default: 60)
            max_chunks: Optional limit on final chunks sent to LLM

        Returns:
            Response from the RAG system
        """
        import asyncio

        try:
            # Create a new event loop if one doesn't exist
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # If there's already a running loop, we need to handle this differently
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = executor.submit(
                            asyncio.run, 
                            self.aquery(question, mode=mode, top_k=top_k, max_chunks=max_chunks)
                        )
                        return future.result()
                else:
                    return loop.run_until_complete(
                        self.aquery(question, mode=mode, top_k=top_k, max_chunks=max_chunks)
                    )
            except RuntimeError:
                # No event loop exists
                return asyncio.run(
                    self.aquery(question, mode=mode, top_k=top_k, max_chunks=max_chunks)
                )
        except Exception as e:
            return f"Error querying RAG system: {e}"

    def _is_processed(self) -> bool:
        """Check if the repository has already been processed."""
        metadata_file = self.working_dir / "metadata.json"
        return metadata_file.exists()

    def _save_metadata(self, num_files: int, num_documents: int):
        """Save processing metadata."""
        metadata = {
            "repo_path": self.repo_path,
            "num_files": num_files,
            "num_documents": num_documents,
            "chunk_token_size": self.chunk_size,
            "chunk_overlap_token_size": self.chunk_overlap,
            "llm_model": self.llm_config['model'],
            "embedding_model": self.embedding_config['model'],
            "embedding_dim": 768
        }

        metadata_file = self.working_dir / "metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
