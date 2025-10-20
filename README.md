# LightRAG CLI

A command-line interface for creating and querying RAG (Retrieval-Augmented Generation) systems using LightRAG with configurable LLM, embedding model, vector database (Qdrant), and graph database (Neo4j).

## Features

- 📚 **Document Processing**: Scan and process documentation repositories (including PDFs)
- 🤖 **Flexible LLM Support**: OpenAI-compatible API endpoints
- 🔍 **Vector Search**: Qdrant vector database integration
- 🕸️ **Graph Storage**: Neo4j graph database for relationships
- 💬 **Interactive Chat**: Command-line interface for querying documents
- 🎛️ **Query Control**: Adjust retrieval parameters and chunk limits
- ⚙️ **Environment Configuration**: Clean configuration management via `.env` files

## Quick Start

### Step 1: Prerequisites

Before starting, ensure you have:
- ✅ Python 3.8 or higher installed
- ✅ Access to an OpenAI-compatible LLM API (OpenAI, DeepSeek, etc.)
- ✅ Access to an OpenAI-compatible embedding API
- ✅ A Qdrant instance (cloud or local)
- ✅ A Neo4j instance (cloud or local)
- ✅ Documents/PDFs you want to query

### Step 2: Clone and Setup Virtual Environment

```bash
# Navigate to your projects directory
cd ~/Desktop/Projects

# Clone or navigate to the project
cd light_rag_prototype

# Create a virtual environment
python3 -m venv .venv

# Activate the virtual environment
source .venv/bin/activate  # On macOS/Linux
# OR on Windows: .venv\Scripts\activate

# You should see (.venv) in your terminal prompt
```

### Step 3: Install Dependencies

```bash
# Make sure you're in the project root and venv is activated
pip install --upgrade pip
pip install -r requirements.txt

# This will install:
# - LightRAG (from GitHub)
# - OpenAI client
# - Qdrant client
# - Neo4j driver
# - PyMuPDF (for PDF processing)
# - Other dependencies
```

### Step 4: Setup Your Document Repository

```bash
# Create a directory for your documents
mkdir -p data

# Copy your documents (PDFs, text files, etc.) into the data/ directory
# Example:
cp /path/to/your/documents/*.pdf data/
```

### Step 5: Configure Environment Variables

```bash
# Create the .env file from the example
cp .env.example .env

# Edit the .env file with your actual credentials
nano .env  # or use: vim .env, code .env, etc.
```

**Fill in these required values in `.env`:**

```bash
# Path to your documents (the data/ folder you created)
REPO_PATH=data

# Your LLM API configuration
LLM_API_BASE=https://api.openai.com/v1  # Or your provider's endpoint
LLM_API_KEY=sk-your-api-key-here
LLM_MODEL=gpt-4  # Or gpt-3.5-turbo, etc.

# Your embedding API configuration
EMBEDDING_API_BASE=https://api.openai.com/v1
EMBEDDING_API_KEY=sk-your-api-key-here
EMBEDDING_MODEL=text-embedding-ada-002  # Or nomic-embed-text, etc.

# Qdrant configuration (get from Qdrant Cloud or local instance)
QDRANT_HOST=https://your-cluster.qdrant.io:6333
QDRANT_API_KEY=your-qdrant-api-key
QDRANT_COLLECTION=lightrag_docs

# Neo4j configuration (get from Neo4j Aura or local instance)
NEO4J_URI=neo4j+ssc://your-instance.databases.neo4j.io:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your-neo4j-password

# Processing settings (optional, defaults shown)
WORKING_DIR=./lightrag_data
FILE_EXTENSIONS=.pdf,.txt,.md
CHUNK_SIZE=1200
CHUNK_OVERLAP=100
```

### Step 6: Verify Configuration

```bash
# Test that your environment is set up correctly
python main.py --help

# You should see the help menu with available commands
```

### Step 7: Process Your Documents

```bash
# This will:
# - Scan your REPO_PATH directory
# - Extract text from documents (including PDFs)
# - Create embeddings
# - Store in Qdrant and Neo4j
python main.py create

# You should see output like:
# "📄 Processing documents..."
# "Processing file: data/document.pdf"
# "✅ Successfully processed and stored documents"
```

**Note:** This step may take several minutes depending on:
- Number and size of documents
- API rate limits
- Network speed

### Step 8: Start Querying!

```bash
# Start the interactive chat interface
python main.py chat

# You'll see:
# 🤖 LightRAG Chat Interface
# Type your questions...
```

**Example queries:**
```
You: What is this document about?
You: Summarize the main topics covered
You: What does section 7 say about benefits?
```

### Step 9: (Optional) Clear Cache and Rebuild

```bash
# If you want to start fresh or update documents:
rm -f lightrag_data/kv_store_llm_response_cache.json

# Rebuild everything from scratch:
python main.py create --rebuild
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

### Advanced Query Control

You can customize query behavior programmatically in `rag_processor.py`:

```python
# Control how many chunks are sent to the LLM
response = rag_processor.query(
    question="Your question",
    mode="hybrid",      # Query mode: naive, local, global, or hybrid
    top_k=60,          # Number of entities/relations to retrieve
    max_chunks=10      # Limit to 10 chunks (10,240 tokens)
)
```

**Parameters:**
- `mode`: Query strategy
  - `naive`: Simple vector search
  - `local`: Focus on specific entities and relationships
  - `global`: Use community summaries for broader context
  - `hybrid`: Combine all approaches (recommended)
- `top_k`: Number of candidates to retrieve from vector/graph databases (default: 60)
- `max_chunks`: Maximum chunks sent to LLM (each chunk = 1024 tokens)
  - Example: `max_chunks=10` → 10,240 tokens
  - If not specified, uses default of 30,000 tokens (≈29 chunks)

See `USAGE_EXAMPLES.md` for detailed examples and use cases.

## How It Works

### What Happens During `python main.py create`

1. **Document Scanning**
   - Scans `REPO_PATH` directory recursively
   - Filters files by `FILE_EXTENSIONS` (.pdf, .txt, .md, etc.)
   - Reads and extracts text (PDFs processed with PyMuPDF)

2. **Text Processing**
   - Chunks documents into ~1024 token pieces
   - Applies overlap between chunks for context continuity
   - Adds metadata (source file, page numbers for PDFs)

3. **Embedding Generation**
   - Sends text chunks to embedding API
   - Creates 768-dimensional vectors (for nomic-embed-text) or 1536-dim (for OpenAI)
   - Stores embeddings in Qdrant vector database

4. **Knowledge Graph Construction**
   - LLM extracts entities (people, places, concepts) from chunks
   - Identifies relationships between entities
   - Builds graph structure in Neo4j

5. **Storage**
   - Vector embeddings → Qdrant (for similarity search)
   - Knowledge graph → Neo4j (for relationship queries)
   - Metadata → Local JSON files in `lightrag_data/`

### What Happens During `python main.py chat`

1. **Query Processing**
   - Your question is embedded using the same embedding model
   - Similarity search finds relevant chunks in Qdrant

2. **Context Retrieval** (based on mode)
   - **Naive**: Simple vector similarity search
   - **Local**: Finds related entities and their connections in Neo4j
   - **Global**: Uses community summaries for broader context
   - **Hybrid** (default): Combines all three approaches

3. **Context Assembly**
   - Retrieves top_k candidates (default: 60 entities/relations)
   - Deduplicates and ranks by relevance
   - Limits to max_chunks (if specified) or token limits

4. **LLM Generation**
   - Sends assembled context + your question to LLM
   - LLM generates answer based on retrieved context
   - Returns response to you

5. **Caching**
   - Responses cached in `kv_store_llm_response_cache.json`
   - Future identical queries return instantly from cache

### Options

- `--env-file PATH`: Specify custom environment file path (default: `.env`)
- `--rebuild`: Force rebuild embeddings even if they exist

## Supported File Types

The application processes these file extensions:
- **`.pdf`** - PDF documents (with text extraction via PyMuPDF)
- **`.txt`** - Plain text files
- **`.md`** - Markdown files
- **`.rst`** - reStructuredText files
- **`.py`** - Python source files
- **`.js`, `.ts`** - JavaScript/TypeScript files
- **`.java`** - Java source files
- **`.cpp`, `.c`, `.h`** - C/C++ source files

**Note:** You can modify `FILE_EXTENSIONS` in your `.env` file to customize this list.

**PDF Processing:** The application automatically extracts text from PDFs and preserves page numbers for better citation and reference.

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

Key dependencies installed via `requirements.txt`:

- **`lightrag`**: Core RAG functionality (installed from GitHub)
- **`python-dotenv`**: Environment variable management
- **`openai`**: OpenAI API client (works with OpenAI-compatible APIs)
- **`qdrant-client`**: Qdrant vector database client
- **`neo4j`**: Neo4j graph database driver
- **`numpy`**: Numerical computations for embeddings
- **`tiktoken`**: Token counting and text processing
- **`dataclasses-json`**: JSON serialization
- **`nest-asyncio`**: Async event loop management
- **`pymupdf`**: PDF text extraction

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

#### 1. **"command not found: python"**
**Solution:** Activate your virtual environment first:
```bash
source .venv/bin/activate  # macOS/Linux
# OR
.venv\Scripts\activate  # Windows
```

#### 2. **"Environment file not found"**
**Solution:** Create the `.env` file:
```bash
cp .env.example .env
# Then edit .env with your credentials
```

#### 3. **"Repository path does not exist"**
**Solution:** 
- Check `REPO_PATH` in your `.env` file
- Make sure the `data/` directory exists and contains documents
```bash
ls -la data/  # Verify files are present
```

#### 4. **API Connection Errors**
**Solution:**
- Verify API endpoints in `.env` (should include `/v1` for OpenAI-compatible)
- Check API keys are valid and have proper permissions
- Test connectivity:
```bash
curl -H "Authorization: Bearer YOUR_API_KEY" YOUR_API_BASE/models
```

#### 5. **Qdrant Connection Issues**
**Solution:**
- Verify `QDRANT_HOST` includes the protocol and port: `https://your-cluster.qdrant.io:6333`
- Check API key is valid in Qdrant dashboard
- Test if collections exist in Qdrant UI

#### 6. **Neo4j Connection Issues**
**Solution:**
- Use the correct protocol: `neo4j+ssc://` (for Aura) or `neo4j://` (for local)
- Verify credentials in Neo4j dashboard
- Check if database is running

#### 7. **"TypeError: QueryParam.__init__() got an unexpected keyword argument"**
**Solution:** This was fixed in the latest version. Make sure you have the latest code:
```bash
git pull  # If using git
# Or re-download the latest version
```

#### 8. **PDF Text Extraction Fails**
**Solution:** Make sure PyMuPDF is installed:
```bash
pip install pymupdf
```

#### 9. **Memory Issues with Large Documents**
**Solution:**
- Reduce `CHUNK_SIZE` in `.env` (e.g., from 1200 to 800)
- Process fewer documents at a time
- Use `max_chunks` parameter to limit context size

#### 10. **Cached Responses (Stale Answers)**
**Solution:** Clear the LLM response cache:
```bash
rm -f lightrag_data/kv_store_llm_response_cache.json
```

### Logs and Debugging

**Enable Verbose Logging:**
The application automatically shows INFO logs. Look for:
- `INFO: [_] Process XXX KV load...` - Loading data
- `INFO: [base] Connected to Neo4j...` - Database connections
- `ERROR:` lines indicate problems

**Check Data Integrity:**
```bash
# Verify Qdrant collections
ls -la lightrag_data/

# Check metadata
cat lightrag_data/metadata.json

# View Neo4j data in Neo4j Browser
# Navigate to your Neo4j instance and run:
# MATCH (n) RETURN count(n)
```

**Common Debugging Steps:**
1. Check `.env` file has correct credentials
2. Verify all services (Qdrant, Neo4j) are accessible
3. Ensure documents exist in `REPO_PATH`
4. Try with `--rebuild` flag to recreate embeddings
5. Check available disk space for `lightrag_data/`

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## Quick Reference Card

### Essential Commands
```bash
# Setup
source .venv/bin/activate        # Activate environment
pip install -r requirements.txt  # Install dependencies

# Usage
python main.py create            # Process documents (first time)
python main.py create --rebuild  # Rebuild from scratch
python main.py chat             # Start chatting

# Maintenance
rm -f lightrag_data/kv_store_llm_response_cache.json  # Clear cache
```

### Key Files
- **`.env`** - Your configuration (API keys, endpoints)
- **`data/`** - Your documents (PDFs, text files)
- **`lightrag_data/`** - Generated embeddings and metadata
- **`requirements.txt`** - Python dependencies

### Important Parameters
- **`CHUNK_SIZE`**: Token size per chunk (default: 1200)
- **`top_k`**: Candidates to retrieve (default: 60)
- **`max_chunks`**: Limit chunks sent to LLM (each = 1024 tokens)
- **`mode`**: Query strategy (naive/local/global/hybrid)

### File Support
- ✅ PDF documents (with text extraction)
- ✅ Text files (.txt, .md, .rst)
- ✅ Source code (.py, .js, .ts, .java, .cpp, .c, .h)

## Support

For issues and questions:
- Check the troubleshooting section above
- Review `USAGE_EXAMPLES.md` for practical examples
- Create an issue on GitHub
- Review LightRAG documentation: https://github.com/HKUDS/LightRAG
