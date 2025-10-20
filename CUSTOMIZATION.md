# LightRAG Customization Guide

This guide explains how to customize your LightRAG system for better results.

## System Prompt Customization

You can customize the system prompt to change how the AI responds. Add this to your `.env` file:

```bash
# Custom system prompt (optional)
SYSTEM_PROMPT="You are a helpful AI assistant that answers questions based on the provided context. Always cite sources and page numbers."
```

Or modify it directly in your code when initializing `RAGProcessor`:

```python
rag_processor = RAGProcessor(
    llm_config=config.llm,
    embedding_config=config.embedding,
    qdrant_config=config.qdrant,
    neo4j_config=config.neo4j,
    system_prompt="Your custom prompt here with instructions to cite sources..."
)
```

### Default System Prompt

The default prompt instructs the AI to:
- Cite specific sources and page numbers
- Be precise and factual  
- Clearly state when information isn't available
- Structure answers with citations in format: `[Source: filename, Page: X]`

## Query Parameters

### Control Chunk Retrieval

When querying, you can control how many chunks are retrieved and processed:

```python
# In rag_processor.py aquery method
response = await rag_processor.aquery(
    question="Your question",
    mode="hybrid",          # naive, local, global, or hybrid
    top_k=60,              # Max results to retrieve (default: 60)
    max_token_for_text_unit=4000,
    max_token_for_global_context=4000,
    max_token_for_local_context=4000
)
```

### Query Modes

- **naive**: Simple vector search
- **local**: Focus on local entities and relationships
- **global**: Use global community summaries
- **hybrid**: Combine all approaches (recommended)

### Get Only Context (No Generation)

To see what context is being retrieved without generating an answer:

```python
context = await rag_processor.aquery(
    question="Your question",
    only_need_context=True  # Returns only retrieved context
)
```

## Source Citations

### In Document Processing

LightRAG automatically tracks document chunks. To add better source tracking:

1. **File Names**: The system uses relative file paths from `REPO_PATH`
2. **Page Numbers**: For PDFs, page information is added in the `_read_pdf` method

The current implementation adds page markers like:
```
--- Page 1 ---
content...
--- Page 2 ---
content...
```

### Improving Citations

The default system prompt asks the AI to cite sources. LightRAG's chunk metadata includes:
- File path
- Chunk ID
- Position in document

## Advanced Configuration

### Deduplication

LightRAG handles deduplication internally during:
- Entity merging
- Relationship consolidation
- Chunk selection

You can see dedup logs like:
```
INFO: Round-robin merged chunks: 19 -> 19 (deduplicated 0)
```

### Understanding `top_k`

**`top_k` controls the INITIAL retrieval, NOT the final chunks sent to LLM.**

#### The Retrieval Pipeline:

1. **Stage 1 - Initial Retrieval** (`top_k` parameter):
   ```python
   top_k=60  # Fetch top 60 entities and top 60 edges from vector/graph DBs
   ```
   
2. **Stage 2 - Merging & Deduplication** (automatic):
   - Combines local + global query results
   - Removes duplicate entities/relations
   - Example: `83 entities, 67 relations` after merge
   
3. **Stage 3 - Chunk Selection** (automatic):
   - Selects text chunks relevant to the entities/relations
   - Example: `19 entity-related chunks` selected
   - After dedup: `16 final chunks`
   
4. **Stage 4 - Token Truncation** (`max_token_for_*` parameters):
   - Truncates chunks to fit within token limits
   - Example: `max_token_for_text_unit=4000`

#### Example from Query Logs:

```
INFO: Query nodes: (top_k:40, cosine:0.2)        # Stage 1: Retrieve top 40
INFO: Local query: 40 entities, 35 relations      # Initial results
INFO: Global query: 52 entities, 40 relations
INFO: Raw search results: 83 entities, 67 relations  # Stage 2: Merged
INFO: Selecting 19 from 19 entity-related chunks   # Stage 3: Chunk selection
INFO: Round-robin merged chunks: 19 -> 19 (deduplicated 0)
INFO: Final context: 83 entities, 67 relations, 16 chunks  # Stage 4: Final
```

**Key Insight**: `top_k=40` retrieved 40 candidates, but only **16 chunks** were sent to the LLM!

#### When to Adjust `top_k`:

```python
# High precision (focused answers)
top_k=20  # Fewer initial candidates → more focused results

# Balanced (recommended)
top_k=60  # Default, good for most cases

# High recall (comprehensive answers)  
top_k=100  # More candidates → better chance of finding relevant info
```

**Trade-off**: Higher `top_k` = slower retrieval but potentially better coverage

### Directly Control Final Chunks Sent to LLM

**New Feature**: You can now directly limit the number of chunks sent to the LLM using the `max_chunks` parameter:

```python
# Simple way: Limit to approximately 10 final chunks
response = await rag_processor.aquery(
    question="Your question",
    max_chunks=10  # Will send approximately 10 chunks to LLM
)

# In sync mode (from chat interface):
response = rag_processor.query(
    question="Your question",
    max_chunks=10
)
```

#### How `max_chunks` Works:

1. **Calculates token limits** based on average chunk size (~350 tokens)
2. **Sets all context limits** accordingly
3. **Example**: `max_chunks=10` → `~3,500 tokens per context`

#### Comparison:

```python
# Method 1: Direct chunk control (simpler)
max_chunks=10  # Target: 10 chunks

# Method 2: Token control (more precise)
max_token_for_text_unit=3500
max_token_for_global_context=3500
max_token_for_local_context=3500
```

**Note**: `max_chunks` is approximate because:
- Chunk sizes vary (200-500 tokens typically)
- Deduplication may reduce the actual count
- Token limits are the final constraint

### Token Limits (Advanced Control)

For precise control over context size, use token limits directly:

```python
response = await rag_processor.aquery(
    question="Your question",
    max_token_for_text_unit=2000,      # Reduce for shorter context
    max_token_for_global_context=2000,
    max_token_for_local_context=2000
)
```

## Example: Complete Customization

```python
# Custom system prompt with strict citation requirements
custom_prompt = """You are a legal document assistant.
IMPORTANT: You MUST cite every piece of information with [Source: filename, Page: number].
If information is not in the provided context, respond: "I cannot find this information in the provided documents."
Format your answers as:
1. Direct answer
2. Citations
3. Relevant context"""

# Initialize with custom settings
rag_processor = RAGProcessor(
    llm_config=config.llm,
    embedding_config=config.embedding,
    qdrant_config=config.qdrant,
    neo4j_config=config.neo4j,
    system_prompt=custom_prompt,
    chunk_size=1200,
    chunk_overlap=100
)

# Query with controlled parameters
response = await rag_processor.aquery(
    question="What are the employee benefits under §7?",
    mode="hybrid",
    top_k=40,  # Focus on top 40 most relevant chunks
    max_token_for_text_unit=3000
)
```

## Tips for Better Results

1. **System Prompt**: Be explicit about citation format and requirements
2. **top_k**: Lower values (20-40) for focused answers, higher (60-100) for comprehensive coverage
3. **Mode**: Use "hybrid" for best results, "local" for specific entity queries
4. **Chunk Size**: Larger chunks (1200-2000) preserve more context
5. **Token Limits**: Adjust based on your LLM's context window

## Environment Variables Reference

```bash
# Query Configuration (optional, set in code)
DEFAULT_QUERY_MODE=hybrid
DEFAULT_TOP_K=60
DEFAULT_MAX_TOKENS=4000

# System Behavior
SYSTEM_PROMPT="Your custom prompt..."
```

For more information, see the [LightRAG documentation](https://github.com/HKUDS/LightRAG).

