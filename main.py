#!/usr/bin/env python3
"""
LightRAG CLI Application

A command-line interface for creating and querying RAG systems using LightRAG
with configurable LLM, embedding model, vector database (Qdrant), and graph database (Neo4j).
"""

import argparse
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

from config import ConfigLoader, AppConfig
from rag_processor import RAGProcessor
from chat_interface import ChatInterface


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="LightRAG CLI - Create and query RAG systems from documentation repositories",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create embeddings from a repository
  python main.py create

  # Start chat interface with existing embeddings
  python main.py chat

  # Create a new .env file from template
  python main.py init-env
        """
    )

    # Subcommands
    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Init env command
    subparsers.add_parser('init-env', help='Create a .env file from template')

    # Create command
    create_parser = subparsers.add_parser('create', help='Create embeddings from repository')
    create_parser.add_argument(
        '--rebuild',
        action='store_true',
        help='Force rebuild of the RAG system even if data exists'
    )

    # Chat command
    subparsers.add_parser('chat', help='Start chat interface with existing embeddings')

    # Global options
    parser.add_argument(
        '--env-file',
        type=str,
        default='.env',
        help='Path to environment file (default: .env)'
    )

    return parser.parse_args()


def load_configuration(env_file: str) -> AppConfig:
    """Load configuration from environment file."""
    # Load environment variables from file
    if not Path(env_file).exists():
        print(f"Error: Environment file '{env_file}' not found.")
        print("Run 'python main.py init-env' to create a template .env file.")
        sys.exit(1)

    load_dotenv(env_file)

    try:
        config = ConfigLoader.load_config()
        ConfigLoader.print_config_summary(config)
        return config
    except ValueError as e:
        print(f"Configuration error: {e}")
        sys.exit(1)


def create_embeddings_mode(config: AppConfig, rebuild: bool = False):
    """Create embeddings from the repository."""
    print("🚀 Starting embeddings creation mode...")

    try:
        # Initialize RAG processor
        print("Initializing RAG processor...")
        rag_processor = RAGProcessor(
            llm_config={
                'api_base': config.llm.api_base,
                'api_key': config.llm.api_key,
                'model': config.llm.model
            },
            embedding_config={
                'api_base': config.embedding.api_base,
                'api_key': config.embedding.api_key,
                'model': config.embedding.model
            },
            qdrant_config={
                'host': config.qdrant.host,
                'api_key': config.qdrant.api_key,
                'collection_name': config.qdrant.collection_name
            },
            neo4j_config={
                'uri': config.neo4j.uri,
                'user': config.neo4j.user,
                'password': config.neo4j.password
            },
            working_dir=config.processing.working_dir,
            chunk_size=config.processing.chunk_size,
            chunk_overlap=config.processing.chunk_overlap
        )

        # Process documents
        print(f"📚 Processing documents from: {config.processing.repo_path}")
        rag_processor.process_documents(
            repo_path=config.processing.repo_path,
            file_extensions=config.processing.file_extensions,
            rebuild=rebuild
        )

        print("✅ Embeddings creation completed successfully!")

    except KeyboardInterrupt:
        print("\n⏹️  Operation cancelled by user.")
    except Exception as e:
        print(f"❌ Error during embeddings creation: {e}")
        sys.exit(1)


def chat_mode(config: AppConfig):
    """Start chat interface with existing embeddings."""
    print("💬 Starting chat mode...")

    try:
        # Check if embeddings exist
        working_dir = Path(config.processing.working_dir)
        if not working_dir.exists():
            print(f"❌ Error: Working directory '{working_dir}' does not exist.")
            print("Please run 'python main.py create' first to create embeddings.")
            sys.exit(1)

        # Initialize RAG processor
        print("Initializing RAG processor...")
        rag_processor = RAGProcessor(
            llm_config={
                'api_base': config.llm.api_base,
                'api_key': config.llm.api_key,
                'model': config.llm.model
            },
            embedding_config={
                'api_base': config.embedding.api_base,
                'api_key': config.embedding.api_key,
                'model': config.embedding.model
            },
            qdrant_config={
                'host': config.qdrant.host,
                'api_key': config.qdrant.api_key,
                'collection_name': config.qdrant.collection_name
            },
            neo4j_config={
                'uri': config.neo4j.uri,
                'user': config.neo4j.user,
                'password': config.neo4j.password
            },
            working_dir=config.processing.working_dir,
            chunk_size=config.processing.chunk_size,
            chunk_overlap=config.processing.chunk_overlap
        )

        # Start chat interface
        print("🤖 Starting chat interface...")
        chat_interface = ChatInterface(rag_processor)
        chat_interface.start_chat()

    except KeyboardInterrupt:
        print("\n👋 Chat session ended.")
    except Exception as e:
        print(f"❌ Error starting chat: {e}")
        sys.exit(1)


def init_env_mode():
    """Initialize environment file from template."""
    print("📝 Creating .env file from template...")
    ConfigLoader.create_env_file()
    print("\n📋 Next steps:")
    print("1. Edit the .env file with your actual configuration values")
    print("2. Run 'python main.py create' to create embeddings")
    print("3. Run 'python main.py chat' to start chatting with your docs")


def main():
    """Main application entry point."""
    args = parse_arguments()

    # Handle init-env command (doesn't need config)
    if args.command == 'init-env':
        init_env_mode()
        return

    # Load configuration for other commands
    config = load_configuration(args.env_file)

    # Handle different modes
    if args.command == 'create':
        create_embeddings_mode(config, rebuild=args.rebuild)
    elif args.command == 'chat':
        chat_mode(config)
    else:
        print("❌ Error: No command specified. Use 'create' or 'chat'.")
        print("Run 'python main.py --help' for usage information.")
        sys.exit(1)


if __name__ == "__main__":
    main()
