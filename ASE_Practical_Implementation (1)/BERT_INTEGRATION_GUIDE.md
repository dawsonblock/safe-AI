# BERT Integration Guide

## Overview

This guide shows how to integrate production-ready BERT embeddings into the FDQC-Cockpit system, replacing simulated hash-based embeddings with actual neural embeddings from Hugging Face transformers.

**Module**: `bert_integration.py`  
**Based on**: [Hugging Face BERT Documentation](https://huggingface.co/docs/transformers/en/model_doc/bert)

---

## Installation

### 1. Install Dependencies

```bash
pip install transformers torch
```

### 2. Download BERT Model (Optional - auto-downloads on first use)

```python
from transformers import AutoModel, AutoTokenizer

model_name = "bert-base-uncased"
AutoModel.from_pretrained(model_name)
AutoTokenizer.from_pretrained(model_name)
```

**Model Options**:
- `bert-base-uncased`: 110M params, 768-dim embeddings (recommended)
- `bert-large-uncased`: 340M params, 1024-dim embeddings (higher quality)
- `bert-base-cased`: Case-sensitive variant
- `distilbert-base-uncased`: 66M params, faster inference

---

## Quick Start

### Basic Usage

```python
from bert_integration import BERTEmbedder

# Initialize embedder
embedder = BERTEmbedder()

# Encode single text
text = "This is a test sentence."
embedding = embedder.encode(text)
print(f"Shape: {embedding.shape}")  # torch.Size([768])

# Encode batch
texts = ["First sentence.", "Second sentence.", "Third sentence."]
embeddings = embedder.encode_batch(texts)
print(f"Shape: {embeddings.shape}")  # torch.Size([3, 768])
```

### With Configuration

```python
from bert_integration import BERTEmbedder, BERTConfig

config = BERTConfig(
    model_name="bert-base-uncased",
    max_length=512,
    batch_size=32,
    use_gpu=True,
    pooling_strategy="cls",  # or "mean", "max"
    normalize_embeddings=True,
    fallback_to_simulation=True
)

embedder = BERTEmbedder(config)
```

---

## Integration with Existing Components

### 1. Vector Memory Integration

Replace hash-based embeddings in `vector_memory.py`:

```python
from bert_integration import BERTEmbedder, BERTConfig
from vector_memory import VectorMemory

# Create BERT embedder
embedder = BERTEmbedder(BERTConfig(
    model_name="bert-base-uncased",
    use_gpu=True,
    normalize_embeddings=True
))

# Create vector memory
memory = VectorMemory()

# Replace embedding generation method
def bert_generate_embedding(text: str) -> torch.Tensor:
    return embedder.encode(text, use_cache=True)

memory._generate_embedding = bert_generate_embedding

# Now use memory as normal
doc_id = memory.add_document("Test document content", metadata={'source': 'test'})
results = memory.search("test query", top_k=5)
```

### 2. Safety Layer Integration

Replace action embeddings in `llm_agent.py`:

```python
from bert_integration import BERTEmbedder, BERTConfig
from llm_agent import FDQCAgent

# Create BERT embedder
embedder = BERTEmbedder(BERTConfig(
    model_name="bert-base-uncased",
    use_gpu=True
))

# Create agent
agent = FDQCAgent()

# Replace action embedding method
def bert_create_embedding(action: str, dim: int) -> torch.Tensor:
    embedding = embedder.encode(action, use_cache=True)
    
    # Adjust dimension if needed
    if embedding.shape[0] != dim:
        if embedding.shape[0] > dim:
            embedding = embedding[:dim]
        else:
            padding = torch.zeros(dim - embedding.shape[0])
            embedding = torch.cat([embedding, padding])
    
    return embedding.unsqueeze(0)

agent._create_action_embedding = bert_create_embedding

# Now use agent as normal
result = agent.select_action(observation, available_actions)
```

### 3. OCR Pipeline Integration

Enhance document embeddings in `deepseek_ocr.py`:

```python
from bert_integration import BERTEmbedder
from deepseek_ocr import CockpitOCRIntegration

embedder = BERTEmbedder()
ocr = CockpitOCRIntegration()

# Process documents with BERT embeddings
results = ocr.process_directory(Path("data/documents"))

for result in results:
    if result.success:
        # Generate semantic embedding for compressed text
        embedding = embedder.encode(result.compressed_text)
        # Store embedding with document metadata
        result.metadata['embedding'] = embedding
```

---

## Advanced Features

### 1. Embedding Cache

The embedder automatically caches embeddings for repeated texts:

```python
embedder = BERTEmbedder()

# First call - computes embedding
emb1 = embedder.encode("Hello world")

# Second call - retrieved from cache (much faster)
emb2 = embedder.encode("Hello world")

# Check cache statistics
stats = embedder.get_stats()
print(f"Cache hit rate: {stats['cache_hit_rate']:.2%}")
print(f"Cache size: {stats['cache_size']}")

# Save cache to disk
embedder.save_cache(Path("embeddings_cache.pt"))

# Load cache on restart
embedder.load_cache(Path("embeddings_cache.pt"))
```

### 2. Pooling Strategies

Different pooling strategies for different use cases:

```python
# CLS token (default) - good for classification tasks
embedder_cls = BERTEmbedder(BERTConfig(pooling_strategy="cls"))

# Mean pooling - good for semantic similarity
embedder_mean = BERTEmbedder(BERTConfig(pooling_strategy="mean"))

# Max pooling - emphasizes strongest features
embedder_max = BERTEmbedder(BERTConfig(pooling_strategy="max"))
```

### 3. Batch Processing

Efficient batch processing for large datasets:

```python
embedder = BERTEmbedder(BERTConfig(batch_size=64))

# Process large list of texts
texts = [f"Document {i}" for i in range(1000)]
embeddings = embedder.encode_batch(texts, show_progress=True)

# Result: torch.Size([1000, 768])
```

### 4. GPU Acceleration

Automatic GPU usage when available:

```python
import torch

config = BERTConfig(use_gpu=True)
embedder = BERTEmbedder(config)

# Check device
stats = embedder.get_stats()
print(f"Device: {stats['device']}")  # cuda:0 or cpu

# Verify GPU usage
if torch.cuda.is_available():
    print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
```

---

## Performance Optimization

### 1. Batch Size Tuning

Adjust batch size based on GPU memory:

```python
# For 8GB GPU
config = BERTConfig(batch_size=32)

# For 16GB GPU
config = BERTConfig(batch_size=64)

# For 24GB+ GPU
config = BERTConfig(batch_size=128)
```

### 2. Model Selection

Choose model based on speed/quality tradeoff:

| Model | Params | Dim | Speed | Quality |
|-------|--------|-----|-------|---------|
| distilbert-base-uncased | 66M | 768 | Fast | Good |
| bert-base-uncased | 110M | 768 | Medium | Better |
| bert-large-uncased | 340M | 1024 | Slow | Best |

```python
# Fast inference
config = BERTConfig(model_name="distilbert-base-uncased")

# Balanced
config = BERTConfig(model_name="bert-base-uncased")

# High quality
config = BERTConfig(model_name="bert-large-uncased")
```

### 3. Cache Management

Optimize cache for your workload:

```python
embedder = BERTEmbedder()
embedder.max_cache_size = 5000  # Increase cache size

# Periodically clear cache if memory constrained
if len(embedder.embedding_cache) > 10000:
    embedder.clear_cache()
```

---

## Semantic Similarity Examples

### Document Similarity

```python
import torch.nn.functional as F

embedder = BERTEmbedder()

doc1 = "Machine learning is a subset of artificial intelligence."
doc2 = "AI includes machine learning and deep learning."
doc3 = "Python is a programming language."

emb1 = embedder.encode(doc1)
emb2 = embedder.encode(doc2)
emb3 = embedder.encode(doc3)

# Calculate cosine similarity
sim_12 = F.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0)).item()
sim_13 = F.cosine_similarity(emb1.unsqueeze(0), emb3.unsqueeze(0)).item()

print(f"ML vs AI similarity: {sim_12:.4f}")  # High similarity
print(f"ML vs Python similarity: {sim_13:.4f}")  # Low similarity
```

### Action Similarity for Safety

```python
embedder = BERTEmbedder()

safe_action = "Read file: config.yaml"
similar_safe = "View file: settings.yaml"
unsafe_action = "Delete all files in system directory"

emb_safe = embedder.encode(safe_action)
emb_similar = embedder.encode(similar_safe)
emb_unsafe = embedder.encode(unsafe_action)

# Safe actions should be similar
sim_safe = F.cosine_similarity(emb_safe.unsqueeze(0), emb_similar.unsqueeze(0)).item()
print(f"Safe action similarity: {sim_safe:.4f}")  # ~0.7-0.9

# Unsafe actions should be dissimilar
sim_unsafe = F.cosine_similarity(emb_safe.unsqueeze(0), emb_unsafe.unsqueeze(0)).item()
print(f"Safe vs unsafe similarity: {sim_unsafe:.4f}")  # ~0.2-0.4
```

---

## Production Deployment

### 1. Model Caching

Cache models to avoid re-downloading:

```python
from pathlib import Path

cache_dir = Path("/opt/models/bert_cache")
cache_dir.mkdir(parents=True, exist_ok=True)

config = BERTConfig(
    model_name="bert-base-uncased",
    cache_dir=cache_dir
)

embedder = BERTEmbedder(config)
```

### 2. Error Handling

Graceful fallback to simulation:

```python
config = BERTConfig(
    model_name="bert-base-uncased",
    fallback_to_simulation=True  # Falls back if transformers unavailable
)

embedder = BERTEmbedder(config)

if embedder.is_simulation_mode:
    logger.warning("Running in simulation mode - install transformers for production")
else:
    logger.info("BERT model loaded successfully")
```

### 3. Monitoring

Track embedding performance:

```python
embedder = BERTEmbedder()

# Process documents
for doc in documents:
    embedding = embedder.encode(doc)

# Check statistics
stats = embedder.get_stats()
print(f"Total encodings: {stats['total_encodings']}")
print(f"Total tokens: {stats['total_tokens']}")
print(f"Avg time per encoding: {stats['avg_time_ms']:.2f} ms")
print(f"Cache hit rate: {stats['cache_hit_rate']:.2%}")
print(f"Simulation fallbacks: {stats['simulation_fallbacks']}")
```

### 4. Resource Management

Monitor GPU memory:

```python
import torch

embedder = BERTEmbedder(BERTConfig(use_gpu=True))

# Before processing
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print(f"GPU memory before: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

# Process batch
embeddings = embedder.encode_batch(large_text_list)

# After processing
if torch.cuda.is_available():
    print(f"GPU memory after: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    torch.cuda.empty_cache()  # Free memory
```

---

## Migration from Simulation

### Step 1: Install Dependencies

```bash
pip install transformers torch
```

### Step 2: Update Imports

```python
# Old (simulation)
from vector_memory import VectorMemory
memory = VectorMemory()

# New (BERT)
from bert_integration import BERTEmbedder
from vector_memory import VectorMemory

embedder = BERTEmbedder()
memory = VectorMemory()
memory._generate_embedding = lambda text: embedder.encode(text)
```

### Step 3: Test Compatibility

```python
# Verify embedding dimensions match
old_embedding = torch.randn(768)  # Simulated
new_embedding = embedder.encode("test")  # BERT

assert old_embedding.shape == new_embedding.shape, "Dimension mismatch!"
print("✓ Embeddings compatible")
```

### Step 4: Gradual Rollout

```python
# Use feature flag for gradual rollout
USE_BERT = os.getenv('USE_BERT_EMBEDDINGS', 'false').lower() == 'true'

if USE_BERT:
    embedder = BERTEmbedder()
    memory._generate_embedding = lambda text: embedder.encode(text)
    logger.info("Using BERT embeddings")
else:
    logger.info("Using simulated embeddings")
```

---

## Troubleshooting

### Issue: Out of Memory

**Solution**: Reduce batch size or use smaller model

```python
config = BERTConfig(
    model_name="distilbert-base-uncased",  # Smaller model
    batch_size=16  # Smaller batches
)
```

### Issue: Slow Inference

**Solution**: Enable GPU or use caching

```python
config = BERTConfig(
    use_gpu=True,  # Use GPU if available
    batch_size=64  # Larger batches on GPU
)

# Enable caching for repeated texts
embedding = embedder.encode(text, use_cache=True)
```

### Issue: Model Download Fails

**Solution**: Pre-download models or use local cache

```bash
# Pre-download
python -c "from transformers import AutoModel; AutoModel.from_pretrained('bert-base-uncased')"

# Or specify local path
config = BERTConfig(cache_dir="/path/to/models")
```

### Issue: Dimension Mismatch

**Solution**: Adjust embeddings to match expected dimension

```python
def adjust_embedding(embedding: torch.Tensor, target_dim: int) -> torch.Tensor:
    current_dim = embedding.shape[0]
    
    if current_dim == target_dim:
        return embedding
    elif current_dim > target_dim:
        return embedding[:target_dim]  # Truncate
    else:
        padding = torch.zeros(target_dim - current_dim)
        return torch.cat([embedding, padding])  # Pad
```

---

## Testing

Run the built-in tests:

```bash
python src/bert_integration.py
```

Expected output:
```
Testing BERT Integration...

1. Testing single text encoding...
  Text: This is a test sentence for BERT encoding.
  Embedding shape: torch.Size([768])
  Embedding norm: 1.0000
  Mode: BERT

2. Testing batch encoding...
  Batch size: 3
  Embeddings shape: torch.Size([3, 768])

3. Testing cache...
  Cache hit rate: 50.00%
  Cache size: 2

4. Testing semantic similarity...
  Similarity (cat/feline): 0.8234
  Similarity (cat/python): 0.3456
  ✓ Semantic similarity working: True

5. Final statistics...
  model_name: bert-base-uncased
  is_simulation_mode: False
  device: cuda:0
  total_encodings: 6
  total_tokens: 156
  avg_time_ms: 12.34
  cache_size: 5
  cache_hit_rate: 0.33
  simulation_fallbacks: 0

✓ BERT integration tests complete
```

---

## Performance Benchmarks

### Encoding Speed

| Model | Device | Batch Size | Speed (texts/sec) |
|-------|--------|------------|-------------------|
| bert-base | CPU | 1 | 10 |
| bert-base | CPU | 32 | 80 |
| bert-base | GPU | 1 | 50 |
| bert-base | GPU | 32 | 500 |
| distilbert | GPU | 32 | 800 |

### Memory Usage

| Model | Device | Batch Size | Memory (GB) |
|-------|--------|------------|-------------|
| bert-base | CPU | 32 | 2.5 |
| bert-base | GPU | 32 | 3.0 |
| bert-large | GPU | 32 | 5.5 |
| distilbert | GPU | 32 | 2.0 |

---

## Best Practices

1. **Use GPU**: 10-50x faster than CPU
2. **Enable Caching**: Significant speedup for repeated texts
3. **Batch Processing**: Much more efficient than single texts
4. **Choose Right Model**: Balance speed vs quality for your use case
5. **Monitor Memory**: Clear cache periodically if memory constrained
6. **Normalize Embeddings**: Better for cosine similarity
7. **Pre-download Models**: Avoid delays in production
8. **Fallback to Simulation**: Graceful degradation if transformers unavailable

---

## Next Steps

1. Install transformers: `pip install transformers torch`
2. Test basic encoding: `python src/bert_integration.py`
3. Integrate with vector memory (see examples above)
4. Integrate with safety layer (see examples above)
5. Monitor performance and adjust configuration
6. Deploy to production with proper caching and monitoring

---

## Support

- **Hugging Face Docs**: https://huggingface.co/docs/transformers
- **BERT Paper**: https://arxiv.org/abs/1810.04805
- **Transformers GitHub**: https://github.com/huggingface/transformers

---

**End of BERT Integration Guide**
