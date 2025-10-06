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

### Experiment 3: Projections for Efficient Data Transfer
**File**: `learning_projections.ipynb`

**What it does**: Demonstrates MongoDB projections to reduce data transfer by 65%

**Learning goals**:
- Understand what projections are and why they matter
- Compare searches WITH vs WITHOUT projections
- Use the `$project` stage in aggregation pipelines
- Access metadata with `$meta` operator (similarity scores)
- Optimize data transfer for LLM applications

**Example**:
```
Same query, two approaches:

WITHOUT projection:
  → MongoDB returns: 40 fields (full documents)
  → Includes: reviews, images, host details, amenities, etc.

WITH projection:
  → MongoDB returns: 14 fields (only what we need)
  → Includes: name, address, summary, score
  → Result: 65% less data transferred!
```

**Key insight**: When building RAG systems, projections keep your LLM context focused and reduce bandwidth costs.

### Experiment 4: Boosting with Review Scores
**File**: `lesson4_boosting.ipynb` (private - not in repo)

**What it does**: Re-ranks vector search results by combining semantic similarity with quality metrics

**Learning goals**:
- Understand when pure vector search isn't enough (similar ≠ best)
- Calculate aggregate scores from multiple review dimensions
- Use MongoDB aggregation pipelines to boost high-quality results
- Learn when to use dynamic boosting vs pre-computed scores

**Example**:
```
Query: "warm friendly place near restaurants"

Without boosting:
  → Listing A ranks #1 (similarity 0.95, reviews 6.2/10, 3 reviews)

With boosting (90% quality + 10% popularity):
  → Listing B ranks #1 (similarity 0.82, reviews 9.7/10, 143 reviews)
  → combinedScore = (9.7 × 0.9) + (143 × 0.1) = 23.03
```

**When dynamic boosting makes sense**:
- User-specific personalization (each user has different preferences)
- Time-sensitive factors (availability windows, seasonal pricing)
- A/B testing different ranking formulas
- Distance from user's search location
- Real-time inventory or demand signals

**Key insight**: Post-search boosting lets you combine semantic relevance with business logic that can't be pre-computed.

## Note

This is a learning repository - code is intentionally simple and exploratory. Not production-ready!
