# LightRAG Usage Examples

## Quick Reference

### Basic Query
```python
response = rag_processor.query("What is this document about?")
```

### Control Number of Chunks Sent to LLM
```python
# Limit to approximately 10 chunks
response = rag_processor.query(
    question="What are the employee benefits?",
    max_chunks=10
)
```

### Control Initial Retrieval
```python
# Retrieve fewer candidates for faster, focused results
response = rag_processor.query(
    question="What is §7 Nr. 4?",
    top_k=30,  # Retrieve top 30 instead of default 60
    max_chunks=8  # Limit final chunks to ~8
)
```

### Different Query Modes
```python
# Hybrid mode (default - recommended)
response = rag_processor.query(
    question="Your question",
    mode="hybrid"  # Combines local + global + naive
)

# Local mode (for specific entities)
response = rag_processor.query(
    question="Your question",
    mode="local"  # Focus on local entities and relationships
)

# Global mode (for broader context)
response = rag_processor.query(
    question="Your question",
    mode="global"  # Use community summaries
)

# Naive mode (simple vector search)
response = rag_processor.query(
    question="Your question",
    mode="naive"  # Just vector similarity
)
```

## Complete Examples

### Example 1: Precise, Fast Query
```python
# When you need quick, focused answers
response = rag_processor.query(
    question="What is the wage for work with respiratory equipment?",
    mode="hybrid",
    top_k=20,      # Fetch fewer candidates
    max_chunks=5   # Send ~5 chunks to LLM
)
```
**Use case**: Specific factual questions, faster responses

### Example 2: Comprehensive Query
```python
# When you need detailed, comprehensive answers
response = rag_processor.query(
    question="Summarize all employee benefits mentioned",
    mode="hybrid",
    top_k=100,     # Fetch more candidates
    max_chunks=20  # Send ~20 chunks to LLM
)
```
**Use case**: Broad questions, need full coverage

### Example 3: Cost-Optimized Query
```python
# Minimize token usage for cost savings
response = rag_processor.query(
    question="What is §7?",
    top_k=15,      # Very focused retrieval
    max_chunks=3   # Minimal context (~1000 tokens)
)
```
**Use case**: High-volume queries, cost-sensitive applications

### Example 4: Get Context Without LLM Generation
```python
# Just see what context would be retrieved
import asyncio

context = asyncio.run(rag_processor.aquery(
    question="What are the benefits?",
    only_need_context=True,  # Skip LLM generation
    max_chunks=10
))
print("Retrieved context:", context)
```
**Use case**: Debug retrieval, check what's being found

## Parameter Cheat Sheet

### `top_k` (Initial Retrieval)
- **Default**: 60
- **Low (20-30)**: Fast, focused, precision-oriented
- **Medium (40-60)**: Balanced (recommended)
- **High (80-100)**: Comprehensive, recall-oriented

### `max_chunks` (Final Context)
- **Default**: None (uses 4000 tokens ≈ 11-12 chunks)
- **Low (3-5)**: Minimal context, fast, cheap
- **Medium (8-12)**: Balanced context
- **High (15-20)**: Rich context, comprehensive

### `mode` (Query Strategy)
- **hybrid**: Best overall (default)
- **local**: Entity-focused queries
- **global**: Broad, summary queries
- **naive**: Simple similarity search

## Real-World Scenarios

### Scenario 1: Legal Document Q&A
```python
# Precise legal questions requiring citations
response = rag_processor.query(
    question="Welche Leistungen stehen ihm laut § 7 Nr. 4 zu?",
    mode="hybrid",
    top_k=40,
    max_chunks=10  # Focused but comprehensive
)
```

### Scenario 2: Document Exploration
```python
# Understanding document structure and topics
response = rag_processor.query(
    question="What are all the main sections in this document?",
    mode="global",  # Use global summaries
    top_k=100,
    max_chunks=15
)
```

### Scenario 3: Specific Fact Lookup
```python
# Quick factual answer
response = rag_processor.query(
    question="What is the price for respiratory masks?",
    mode="local",  # Focus on specific entities
    top_k=25,
    max_chunks=5
)
```

### Scenario 4: Comparison Questions
```python
# Comparing multiple items
response = rag_processor.query(
    question="Compare the benefits under §7 Nr. 4 vs §7 Nr. 5",
    mode="hybrid",
    top_k=60,
    max_chunks=15  # Need more context for comparison
)
```

## Async Usage (Advanced)

```python
import asyncio

async def main():
    response = await rag_processor.aquery(
        question="Your question",
        mode="hybrid",
        top_k=60,
        max_chunks=10
    )
    print(response)

asyncio.run(main())
```

## Performance Tips

1. **Start with defaults** then adjust based on results
2. **Lower `top_k` and `max_chunks`** for faster responses
3. **Higher `max_chunks`** when answers seem incomplete
4. **Use `mode="local"`** for entity-specific questions
5. **Use `mode="global"`** for document-level questions
6. **Monitor logs** to see actual chunks retrieved vs sent to LLM

## Monitoring Your Queries

Watch for these log messages:
```
INFO: Query nodes: (top_k:40, cosine:0.2)        # Initial retrieval
INFO: Local query: 40 entities, 35 relations      # What was found
INFO: Final context: 83 entities, 67 relations, 16 chunks  # What's sent to LLM
[Max Chunks Control] Limiting to ~10 chunks (≈3500 tokens per context)  # Your limit
```

Compare "16 chunks" (actual) vs your `max_chunks` setting to see if limit was reached.

