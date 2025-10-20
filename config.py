"""
Configuration Module

Handles loading and validation of environment variables for LightRAG.
"""

import os
from pathlib import Path
from typing import Dict, List, Any
from dataclasses import dataclass


@dataclass
class LLMConfig:
    """LLM configuration."""
    api_base: str
    api_key: str
    model: str


@dataclass
class EmbeddingConfig:
    """Embedding configuration."""
    api_base: str
    api_key: str
    model: str


@dataclass
class QdrantConfig:
    """Qdrant configuration."""
    host: str
    api_key: str
    collection_name: str


@dataclass
class Neo4jConfig:
    """Neo4j configuration."""
    uri: str
    user: str
    password: str


@dataclass
class ProcessingConfig:
    """Processing configuration."""
    working_dir: str
    file_extensions: List[str]
    chunk_size: int
    chunk_overlap: int
    repo_path: str


@dataclass
class AppConfig:
    """Complete application configuration."""
    llm: LLMConfig
    embedding: EmbeddingConfig
    qdrant: QdrantConfig
    neo4j: Neo4jConfig
    processing: ProcessingConfig


class ConfigLoader:
    """Loads and validates configuration from environment variables."""

    @staticmethod
    def load_config() -> AppConfig:
        """
        Load configuration from environment variables.

        Returns:
            Complete application configuration

        Raises:
            ValueError: If required configuration is missing or invalid
        """
        # Load environment variables
        config_data = {}

        # Repository path
        repo_path = os.getenv('REPO_PATH')
        if not repo_path:
            raise ValueError("REPO_PATH environment variable is required")
        if not Path(repo_path).exists():
            raise ValueError(f"Repository path does not exist: {repo_path}")
        config_data['repo_path'] = repo_path

        # LLM Configuration
        llm_config = LLMConfig(
            api_base=ConfigLoader._get_required_env('LLM_API_BASE'),
            api_key=ConfigLoader._get_required_env('LLM_API_KEY'),
            model=os.getenv('LLM_MODEL', 'gpt-3.5-turbo')
        )
        ConfigLoader._validate_url(llm_config.api_base, 'LLM_API_BASE')

        # Embedding Configuration
        embedding_config = EmbeddingConfig(
            api_base=ConfigLoader._get_required_env('EMBEDDING_API_BASE'),
            api_key=ConfigLoader._get_required_env('EMBEDDING_API_KEY'),
            model=os.getenv('EMBEDDING_MODEL', 'text-embedding-ada-002')
        )
        ConfigLoader._validate_url(embedding_config.api_base, 'EMBEDDING_API_BASE')

        # Qdrant Configuration
        qdrant_config = QdrantConfig(
            host=ConfigLoader._get_required_env('QDRANT_HOST'),
            api_key=ConfigLoader._get_required_env('QDRANT_API_KEY'),
            collection_name=os.getenv('QDRANT_COLLECTION', 'lightrag_docs')
        )
        ConfigLoader._validate_url(qdrant_config.host, 'QDRANT_HOST')

        # Set Qdrant environment variables that LightRAG expects
        os.environ['QDRANT_URL'] = qdrant_config.host
        os.environ['QDRANT_API_KEY'] = qdrant_config.api_key

        # Neo4j Configuration
        neo4j_config = Neo4jConfig(
            uri=ConfigLoader._get_required_env('NEO4J_URI'),
            user=ConfigLoader._get_required_env('NEO4J_USER'),
            password=ConfigLoader._get_required_env('NEO4J_PASSWORD')
        )
        ConfigLoader._validate_url(neo4j_config.uri, 'NEO4J_URI')

        # Set Neo4j environment variables that LightRAG expects
        os.environ['NEO4J_URI'] = neo4j_config.uri
        os.environ['NEO4J_USERNAME'] = neo4j_config.user
        os.environ['NEO4J_PASSWORD'] = neo4j_config.password

        # Processing Configuration
        file_extensions_str = os.getenv('FILE_EXTENSIONS', '.md,.txt,.rst,.py,.js,.ts,.java,.cpp,.c,.h')
        file_extensions = [ext.strip() for ext in file_extensions_str.split(',') if ext.strip()]

        processing_config = ProcessingConfig(
            working_dir=os.getenv('WORKING_DIR', './lightrag_data'),
            file_extensions=file_extensions,
            chunk_size=int(os.getenv('CHUNK_SIZE', '1200')),
            chunk_overlap=int(os.getenv('CHUNK_OVERLAP', '100')),
            repo_path=repo_path
        )

        return AppConfig(
            llm=llm_config,
            embedding=embedding_config,
            qdrant=qdrant_config,
            neo4j=neo4j_config,
            processing=processing_config
        )

    @staticmethod
    def _get_required_env(var_name: str) -> str:
        """Get required environment variable."""
        value = os.getenv(var_name)
        if not value:
            raise ValueError(f"Required environment variable {var_name} is not set")
        return value

    @staticmethod
    def _validate_url(url: str, var_name: str):
        """Validate URL format."""
        if not url.startswith(('http://', 'https://', 'neo4j://', 'neo4j+ssc://')):
            raise ValueError(f"{var_name} must be a valid URL starting with http://, https://, neo4j://, or neo4j+ssc://")

    @staticmethod
    def create_env_file(env_file_path: str = ".env"):
        """Create a .env file from the example."""
        example_path = Path(".env.example")
        if not example_path.exists():
            print("Error: .env.example not found. Cannot create .env file.")
            return

        env_path = Path(env_file_path)
        if env_path.exists():
            print(f"Warning: {env_file_path} already exists. Not overwriting.")
            return

        # Copy the example file
        import shutil
        shutil.copy(example_path, env_path)
        print(f"Created {env_file_path}. Please edit it with your configuration values.")

    @staticmethod
    def print_config_summary(config: AppConfig):
        """Print a summary of the loaded configuration."""
        print("\n🔧 Configuration Summary:")
        print("=" * 50)
        print(f"Repository: {config.processing.repo_path}")
        print(f"LLM Model: {config.llm.model} ({config.llm.api_base})")
        print(f"Embedding Model: {config.embedding.model} ({config.embedding.api_base})")
        print(f"Qdrant: {config.qdrant.host} (collection: {config.qdrant.collection_name})")
        print(f"Neo4j: {config.neo4j.uri}")
        print(f"Working Directory: {config.processing.working_dir}")
        print(f"File Extensions: {', '.join(config.processing.file_extensions)}")
        print(f"Chunk Size: {config.processing.chunk_size}, Overlap: {config.processing.chunk_overlap}")
        print("=" * 50 + "\n")
