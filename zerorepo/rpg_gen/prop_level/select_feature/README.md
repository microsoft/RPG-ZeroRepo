# ZeroRepo-Compatible Faiss-based Property Level Feature Selection

This directory contains a sophisticated Faiss-based feature selection system for property-level agent development. The system is **fully compatible** with the existing ZeroRepo FaissDocDB implementation while adding enhanced features for repository feature tree generation.

## ZeroRepo Compatibility

✅ **100% Compatible** with `/mnt/jianwen/temp/ZeroRepo/select_features/faiss_db.py`

- Same metadata storage format: `{"key": text, "doc": document}`
- Same embedding model support: `infly/inf-retriever-v1` (with fallback)
- Same L2 normalization and IndexIDMap usage
- Same search result format and GPU acceleration
- Same save/load file formats (`index.faiss`, `id2doc.json`)

## Overview

The feature selection system operates in two modes:

1. **Simple Mode**: Direct feature path generation using a single agent
2. **Feature Mode**: Complex multi-agent approach with exploitation, exploration, and validation phases

Both modes leverage Faiss vector similarity search to find relevant features based on semantic similarity.

## Architecture Overview

The system uses a **clean separation of concerns** architecture:

```
PropLevelAgent
├── LLMClient.call_with_structured_output()  # All LLM interactions
├── ActionHandler                            # Business logic processing  
├── PropLevelEnv                            # State management
└── Memory (specialized per agent role)     # Context management
```

**Key Architectural Principles:**
- **No Tool System**: Replaced with structured Pydantic output schemas
- **Type Safety**: All LLM outputs validated with Pydantic models
- **Action Separation**: Business logic completely separated from LLM calls
- **Memory Specialization**: Different memory contexts for different agent roles
- **Retry Handling**: Built-in resilience for LLM call failures

## Key Components

### 1. PropLevelAgent (`prop_level_agent.py`)

**New Architecture**: Uses `LLMClient.call_with_structured_output()` instead of tools for clean separation of concerns.

The main agent class that orchestrates feature tree generation:

- **Simple Mode**: Direct feature generation with self-validation
- **Feature Mode**: Multi-agent system with specialized roles:
  - Exploitation Agent: Extends existing feature branches
  - Exploration Agent: Adds new feature categories  
  - Self-Check Agent: Validates selected features
  - Missing Feature Agent: Identifies gaps in the feature tree

**Key Features:**
- **Structured Output**: All LLM interactions use Pydantic models for type safety
- **Action Handler**: Separate `ActionHandler` class handles all business logic independent from LLM calls
- **Memory Management**: Specialized memory contexts for different agent roles
- **Retry Logic**: Built-in retry mechanism for robust LLM interactions

### 2. FaissDocDB (`faiss_db.py`)

ZeroRepo-compatible vector database implementation for efficient feature similarity search:

- **Compatible with ZeroRepo**: Same interface and data formats as original
- Uses sentence transformers for embedding generation (`infly/inf-retriever-v1` default)
- Supports both CPU and GPU execution with multi-GPU support
- L2 normalized embeddings with IndexIDMap for custom ID mapping  
- Metadata format: `{"key": query_text, "doc": document_content}`
- Handles index persistence and loading (`.faiss` and `.json` files)

### 3. Prompt Templates (`../prompts/feature_tree.py`)

Comprehensive prompt templates for different agent roles:

- Feature generation prompts
- Exploitation and exploration selection prompts
- Validation and ranking prompts
- Missing feature identification prompts

## Installation

### Dependencies

```bash
# Core dependencies (ZeroRepo compatible)
pip install faiss-cpu  # or faiss-gpu for GPU support
pip install sentence-transformers
pip install torch
pip install numpy

# Optional: for GPU support
pip install faiss-gpu

# ZeroRepo default model (optional, fallback available)
# Model: infly/inf-retriever-v1 
```

### Quick Setup

```python
from rpg_building.prop_level.select_feature import PropLevelAgent, FaissDocDB, ActionHandler
from rpg_building.base.llm_client import LLMConfig

# Initialize vector database (ZeroRepo compatible)
vec_db = FaissDocDB(
    model_name="infly/inf-retriever-v1",  # ZeroRepo default
    use_gpu=True  # ZeroRepo default
)

# Configure LLM with structured output support
llm_config = LLMConfig(
    provider="openai",
    model="gpt-4", 
    api_key="your-api-key",
    temperature=0.7
)

# Create agent (no tools needed - uses structured output)
agent = PropLevelAgent(
    feature_tree=your_feature_tree,
    frequency=your_frequency_data,
    vec_db=vec_db,
    llm_cfg=llm_config,
    mode="feature"  # or "simple"
)

# Action handler is automatically created and accessible
action_handler = agent.action_handler
```

## Usage Examples

### Simple Mode

```python
# Run simple feature generation
result = agent.run_simple_mode(
    repo_data=repository_info,
    iterations=10,
    batch_size=3
)

print("Generated features:", result['Feature_tree'])
```

### Feature Mode

```python
# Run complex multi-agent generation
result = agent.run_feature_mode(
    repo_data=repository_info,
    iterations=5
)

print("Current tree:", result['Feature_tree'])
print("Validated tree:", result['Validated_Feature_tree'])
```

### Vector Search

```python
# Direct similarity search
results = vec_db.search(
    queries=["sorting algorithm", "memory management"],
    top_k=5
)

for query_results in results:
    for result in query_results:
        print(f"Feature: {result['doc']}, Similarity: {result['similarity']}")
```

## Configuration

### Agent Parameters

- `explore_conditions`: Sampling conditions for exploration [5, 8, 5, 4, 5]
- `exploit_conditions`: Sampling conditions for exploitation [5, 8, 5, 4, 5] 
- `temperature`: Sampling temperature (default: 10)
- `overlap_pct`: Maximum overlap with previous samples (default: 0.3)
- `context_window`: LLM memory context size (default: 1)

### Vector Database Parameters

- `model_name`: Sentence transformer model ("all-MiniLM-L6-v2")
- `use_gpu`: Enable GPU acceleration (requires faiss-gpu)
- `normalize_embeddings`: Use cosine similarity (recommended: True)

## File Structure

```
select_feature/
├── prop_level_agent.py    # Main agent implementation
├── faiss_db.py           # Vector database implementation  
├── example_usage.py      # Usage examples and demos
├── feature_gen.py        # Legacy implementation (reference)
└── README.md            # This documentation
```

## Advanced Features

### Feature Tree Generation

The system uses two complementary strategies for generating candidate feature trees:

**Exploitation Trees (Vector Retrieval)**:
- Built from vector database retrieval using missing features as semantic queries
- Leverages semantic similarity to find relevant features
- Provides focused, contextually relevant feature candidates
- No randomness - deterministic based on similarity scores

**Exploration Trees (Sampling)**:
- **Frequency-based sampling**: More common features have higher selection probability
- **Temperature control**: Controls randomness in feature selection
- **Overlap prevention**: Avoids selecting too many similar features
- Provides diversity and discovery of new feature combinations

### Multi-Agent Coordination

Feature mode employs specialized agents:

1. **Missing Feature Phase**: Identifies gaps in current feature coverage
2. **Tree Generation Phase**: 
   - **Exploitation Tree**: Built from vector database retrieval using missing features as queries
   - **Exploration Tree**: Generated through sampling from the feature tree
3. **Selection Phase**: Exploitation and exploration agents select features from their respective trees
4. **Validation Phase**: Self-check agent validates selections

### Vector Similarity Matching

- Semantic similarity search using sentence transformers
- Efficient FAISS indexing for large feature sets  
- Support for both exact and approximate nearest neighbor search
- Configurable similarity thresholds and result ranking

## Performance Considerations

### Memory Usage

- Vector database size scales with feature tree size
- Embedding dimension: 384 (for all-MiniLM-L6-v2)
- Memory usage: ~1.5KB per indexed feature

### Speed Optimization

- Use GPU acceleration for large feature sets (>10K features)
- Batch processing for multiple queries
- Index persistence to avoid rebuilding
- Parallel agent execution where possible

### Scaling Guidelines

- **Small projects** (<1K features): CPU-only, simple mode
- **Medium projects** (1K-10K features): GPU optional, feature mode
- **Large projects** (>10K features): GPU recommended, batch processing

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure faiss-cpu/faiss-gpu is installed
2. **Memory Issues**: Use CPU mode for large trees, enable GPU for speed
3. **Poor Results**: Check feature tree structure and frequency data quality
4. **API Timeouts**: Adjust LLM timeout settings and retry logic

### Debug Mode

Enable detailed logging:

```python
import logging
logging.basicConfig(level=logging.INFO)

# Run with verbose logging
agent.logger.setLevel(logging.DEBUG)
```

## Contributing

When extending this system:

1. Add new prompt templates to `../prompts/feature_tree.py`
2. Implement new sampling strategies in utility functions
3. Add agent specializations by extending the base agent class
4. Optimize vector search with custom similarity metrics

## License

This implementation follows the project's overall licensing terms.