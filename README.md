# LightRAG CLI

A command-line interface for creating and querying RAG (Retrieval-Augmented Generation) systems using LightRAG with configurable LLM, embedding model, vector database (Qdrant), and graph database (Neo4j).

## Features

- 📚 **Document Processing**: Scan and process documentation repositories
- 🤖 **Flexible LLM Support**: OpenAI-compatible API endpoints
- 🔍 **Vector Search**: Qdrant vector database integration
- 🕸️ **Graph Storage**: Neo4j graph database for relationships
- 💬 **Interactive Chat**: Command-line interface for querying documents
- ⚙️ **Environment Configuration**: Clean configuration management via `.env` files

## Quick Start

### 1. Setup Environment

```bash
# Clone or create the project
git clone <repository-url>
cd light_rag_prototype

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
# Create environment file from template
python main.py init-env

# Edit the .env file with your configuration
nano .env  # or your preferred editor
```

### 3. Create Embeddings

```bash
# Process documents and create embeddings
python main.py create
```

### 4. Start Chatting

```bash
# Start the interactive chat interface
python main.py chat
```

## Configuration

The application uses environment variables for configuration. Copy `.env.example` to `.env` and fill in your values:

```bash
# Repository Configuration
REPO_PATH=/path/to/your/docs/repository

# LLM Configuration (OpenAI-compatible API)
LLM_API_BASE=https://api.openai.com/v1
LLM_API_KEY=your_llm_api_key_here
LLM_MODEL=gpt-3.5-turbo

# Embedding Configuration (OpenAI-compatible API)
EMBEDDING_API_BASE=https://api.openai.com/v1
EMBEDDING_API_KEY=your_embedding_api_key_here
EMBEDDING_MODEL=text-embedding-ada-002

# Vector Database Configuration (Qdrant)
QDRANT_HOST=https://your-qdrant-instance.com
QDRANT_API_KEY=your_qdrant_api_key_here
QDRANT_COLLECTION=lightrag_docs

# Graph Database Configuration (Neo4j)
NEO4J_URI=neo4j://your-neo4j-instance.com:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_neo4j_password_here

# Processing Configuration
WORKING_DIR=./lightrag_data
FILE_EXTENSIONS=.md,.txt,.rst,.py,.js,.ts,.java,.cpp,.c,.h
CHUNK_SIZE=1200
CHUNK_OVERLAP=100
```

## Usage

### Commands

```bash
# Initialize environment file
python main.py init-env

# Create embeddings from repository
python main.py create [--rebuild]

# Start chat interface
python main.py chat

# Show help
python main.py --help
```

### Chat Commands

When in chat mode, you can use these commands:

- `/help` - Show available commands
- `/history` - Show conversation history
- `/clear` - Clear conversation history
- `/quit` - Exit the chat

### Options

- `--env-file PATH`: Specify custom environment file path (default: `.env`)
- `--rebuild`: Force rebuild embeddings even if they exist

## Supported File Types

The application processes these file extensions by default:
- `.md` - Markdown files
- `.txt` - Plain text files
- `.rst` - reStructuredText files
- `.py` - Python source files
- `.js`, `.ts` - JavaScript/TypeScript files
- `.java` - Java source files
- `.cpp`, `.c`, `.h` - C/C++ source files

You can modify `FILE_EXTENSIONS` in your `.env` file to customize this list.

## Architecture

### Components

- **`main.py`**: CLI entry point with command parsing
- **`config.py`**: Environment configuration management
- **`rag_processor.py`**: Document processing and LightRAG integration
- **`chat_interface.py`**: Interactive chat interface

### Data Flow

1. **Document Scanning**: Recursively scan repository for supported files
2. **Text Chunking**: Split documents into overlapping chunks
3. **Embedding Generation**: Create vector embeddings using configured model
4. **Vector Storage**: Store embeddings in Qdrant
5. **Graph Construction**: Build knowledge graph in Neo4j
6. **Query Processing**: Retrieve relevant chunks and generate responses

## Requirements

- Python 3.8+
- OpenAI-compatible LLM API (or OpenAI API)
- Qdrant vector database instance
- Neo4j graph database instance
- Repository with documentation files

## Dependencies

- `lightrag`: Core RAG functionality
- `python-dotenv`: Environment variable management
- `openai`: OpenAI API client
- `qdrant-client`: Qdrant vector database client
- `neo4j`: Neo4j graph database driver
- `numpy`: Numerical computations
- `tiktoken`: Token counting
- `dataclasses-json`: JSON serialization for dataclasses

## Development

### Project Structure

```
light_rag_prototype/
├── main.py                 # CLI entry point
├── config.py              # Configuration management
├── rag_processor.py       # RAG processing logic
├── chat_interface.py      # Chat interface
├── requirements.txt       # Python dependencies
├── .env.example          # Environment template
├── README.md             # This file
└── lightrag_data/        # Working directory (created automatically)
```

### Adding New Features

1. Create new command in `main.py`
2. Add configuration options to `config.py`
3. Implement functionality in appropriate module
4. Update `.env.example` if needed
5. Update this README

## Troubleshooting

### Common Issues

1. **"Environment file not found"**
   - Run `python main.py init-env` to create `.env` file

2. **"Repository path does not exist"**
   - Check `REPO_PATH` in your `.env` file
   - Ensure the path is absolute or relative to the project directory

3. **Connection errors**
   - Verify API endpoints and keys in `.env`
   - Check network connectivity to external services

4. **Memory issues with large repositories**
   - Increase `CHUNK_SIZE` to reduce number of chunks
   - Process repository in smaller batches if needed

### Logs and Debugging

The application provides detailed output during processing. For additional debugging:

1. Check LightRAG data directory for intermediate files
2. Verify Qdrant collections exist
3. Check Neo4j for graph data

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## Support

For issues and questions:
- Check the troubleshooting section above
- Create an issue on GitHub
- Review LightRAG documentation: https://github.com/HKUDS/LightRAG
