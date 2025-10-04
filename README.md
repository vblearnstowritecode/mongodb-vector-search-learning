# MongoDB Vector Search Learning Scratchpad

This is a rough scratch pad repository created while learning MongoDB and vector search concepts.

## What I'm Learning

- MongoDB Atlas vector search fundamentals
- LLM-powered filter extraction from natural language
- Building complete RAG (Retrieval-Augmented Generation) pipelines
- Semantic similarity search with embeddings
- Pydantic data validation and modeling

## Inspiration

Inspired by the course: [Prompt Compression and Query Optimization](https://learn.deeplearning.ai/courses/prompt-compression-and-query-optimization/)

## Experiments

### Experiment 1: Basic Vector Search
**File**: `vector_search_complete_ordered.ipynb`

**What it does**: Fundamental vector search with Airbnb listings

**Learning goals**:
- Load and validate data with Pydantic
- Create vector search indexes in MongoDB Atlas
- Perform semantic similarity searches
- Generate recommendations using GPT

**Example**:
```
Input:  "beach house with pool"
Output: Returns listings semantically similar to beach houses with pools,
        even if they don't contain those exact words
        → "Ocean View Waikiki Marina" (Oahu, $115)
        → "Copacabana Apartment" (Rio, $119)
```

### Experiment 2: Intelligent Search with Filter Extraction
**File**: `filter_extraction_vector_search.ipynb`

**What it does**: Natural language query → structured filters → vector search → GPT answer

**Learning goals**:
- Extract structured MongoDB filters from natural language using LLM
- Combine filters with vector search for precise results
- Build a complete RAG pipeline (parse → retrieve → generate)
- Handle real-world data constraints

**Example**:
```
Input:  "Find me an affordable cozy place in New York for 4 people"

Filter extraction:
  → Semantic: "cozy place"
  → Filters: {price: <$150, accommodates: ≥4, market: "New York"}

Output:
  → 1 matching listing with GPT-generated recommendation
  → "I recommend the [listing name] because it's cozy,
     affordable at $X, and perfectly sized for 4 guests..."
```

## Note

This is a learning repository - code is intentionally simple and exploratory. Not production-ready!
