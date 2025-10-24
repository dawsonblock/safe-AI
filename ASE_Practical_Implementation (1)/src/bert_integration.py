#!/usr/bin/env python3
"""
Production BERT Integration for FDQC-Cockpit

Provides real BERT embeddings using Hugging Face transformers library.
Replaces simulated hash-based embeddings with actual neural embeddings.

Based on: https://huggingface.co/docs/transformers/en/model_doc/bert
"""

import torch
import torch.nn.functional as F
from typing import List, Optional, Union, Dict, Any
from dataclasses import dataclass
import logging
from pathlib import Path
import hashlib
import time

logger = logging.getLogger(__name__)


@dataclass
class BERTConfig:
    """Configuration for BERT embedding generation"""
    model_name: str = "bert-base-uncased"
    max_length: int = 512
    batch_size: int = 32
    use_gpu: bool = True
    cache_dir: Optional[Path] = None
    pooling_strategy: str = "cls"  # 'cls', 'mean', 'max'
    normalize_embeddings: bool = True
    fallback_to_simulation: bool = True


class BERTEmbedder:
    """
    Production BERT embedding generator
    
    Uses Hugging Face transformers library for actual BERT embeddings.
    Falls back to simulation if transformers not available.
    
    Example:
        embedder = BERTEmbedder()
        embedding = embedder.encode("Hello world")
        embeddings = embedder.encode_batch(["Text 1", "Text 2"])
    """
    
    def __init__(self, config: Optional[BERTConfig] = None):
        self.config = config or BERTConfig()
        self.model = None
        self.tokenizer = None
        self.device = None
        self.is_simulation_mode = False
        
        # Try to load actual BERT model
        self._initialize_model()
        
        # Statistics
        self.stats = {
            'total_encodings': 0,
            'total_tokens': 0,
            'total_time_ms': 0.0,
            'cache_hits': 0,
            'simulation_fallbacks': 0
        }
        
        # Embedding cache for repeated texts
        self.embedding_cache: Dict[str, torch.Tensor] = {}
        self.max_cache_size = 1000
    
    def _initialize_model(self):
        """Initialize BERT model and tokenizer"""
        try:
            from transformers import AutoModel, AutoTokenizer
            
            logger.info(f"Loading BERT model: {self.config.model_name}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_name,
                cache_dir=str(self.config.cache_dir) if self.config.cache_dir else None
            )
            
            # Load model
            self.model = AutoModel.from_pretrained(
                self.config.model_name,
                cache_dir=str(self.config.cache_dir) if self.config.cache_dir else None
            )
            
            # Set device
            if self.config.use_gpu and torch.cuda.is_available():
                self.device = torch.device('cuda')
                self.model = self.model.to(self.device)
                logger.info("Using GPU for BERT inference")
            else:
                self.device = torch.device('cpu')
                logger.info("Using CPU for BERT inference")
            
            # Set to evaluation mode
            self.model.eval()
            
            logger.info(f"✓ BERT model loaded successfully: {self.config.model_name}")
            self.is_simulation_mode = False
            
        except ImportError as e:
            logger.warning(f"Transformers library not available: {e}")
            if self.config.fallback_to_simulation:
                logger.warning("Falling back to simulation mode")
                self.is_simulation_mode = True
            else:
                raise RuntimeError("BERT model required but transformers not available")
        
        except Exception as e:
            logger.error(f"Error loading BERT model: {e}")
            if self.config.fallback_to_simulation:
                logger.warning("Falling back to simulation mode")
                self.is_simulation_mode = True
            else:
                raise
    
    def encode(
        self,
        text: str,
        use_cache: bool = True
    ) -> torch.Tensor:
        """
        Generate embedding for a single text
        
        Args:
            text: Input text to encode
            use_cache: Whether to use embedding cache
            
        Returns:
            embedding: 768-dim tensor (for bert-base)
        """
        # Check cache
        if use_cache:
            cache_key = self._get_cache_key(text)
            if cache_key in self.embedding_cache:
                self.stats['cache_hits'] += 1
                return self.embedding_cache[cache_key].clone()
        
        start_time = time.time()
        
        if self.is_simulation_mode:
            embedding = self._simulate_embedding(text)
            self.stats['simulation_fallbacks'] += 1
        else:
            embedding = self._encode_with_bert(text)
        
        # Update stats
        self.stats['total_encodings'] += 1
        self.stats['total_time_ms'] += (time.time() - start_time) * 1000
        
        # Cache result
        if use_cache:
            self._update_cache(cache_key, embedding)
        
        return embedding
    
    def encode_batch(
        self,
        texts: List[str],
        show_progress: bool = False
    ) -> torch.Tensor:
        """
        Generate embeddings for multiple texts (batched for efficiency)
        
        Args:
            texts: List of input texts
            show_progress: Whether to show progress bar
            
        Returns:
            embeddings: Tensor of shape [len(texts), 768]
        """
        if len(texts) == 0:
            return torch.zeros(0, 768)
        
        # Check cache for all texts
        cached_embeddings = []
        uncached_texts = []
        uncached_indices = []
        
        for i, text in enumerate(texts):
            cache_key = self._get_cache_key(text)
            if cache_key in self.embedding_cache:
                cached_embeddings.append((i, self.embedding_cache[cache_key]))
                self.stats['cache_hits'] += 1
            else:
                uncached_texts.append(text)
                uncached_indices.append(i)
        
        # Process uncached texts
        if uncached_texts:
            if self.is_simulation_mode:
                new_embeddings = torch.stack([
                    self._simulate_embedding(text) for text in uncached_texts
                ])
                self.stats['simulation_fallbacks'] += len(uncached_texts)
            else:
                new_embeddings = self._encode_batch_with_bert(
                    uncached_texts, show_progress
                )
            
            # Cache new embeddings
            for text, embedding in zip(uncached_texts, new_embeddings):
                cache_key = self._get_cache_key(text)
                self._update_cache(cache_key, embedding)
        else:
            new_embeddings = torch.zeros(0, 768)
        
        # Combine cached and new embeddings in correct order
        all_embeddings = torch.zeros(len(texts), 768)
        
        for i, embedding in cached_embeddings:
            all_embeddings[i] = embedding
        
        for idx, embedding in zip(uncached_indices, new_embeddings):
            all_embeddings[idx] = embedding
        
        self.stats['total_encodings'] += len(texts)
        
        return all_embeddings
    
    def _encode_with_bert(self, text: str) -> torch.Tensor:
        """Encode single text with actual BERT model"""
        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_length,
            padding=True
        )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get embeddings
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Apply pooling strategy
        embedding = self._pool_embeddings(
            outputs.last_hidden_state,
            inputs['attention_mask']
        )
        
        # Normalize if requested
        if self.config.normalize_embeddings:
            embedding = F.normalize(embedding, p=2, dim=-1)
        
        # Update token count
        self.stats['total_tokens'] += inputs['input_ids'].shape[1]
        
        return embedding.squeeze(0).cpu()
    
    def _encode_batch_with_bert(
        self,
        texts: List[str],
        show_progress: bool = False
    ) -> torch.Tensor:
        """Encode multiple texts with actual BERT model (batched)"""
        all_embeddings = []
        
        # Process in batches
        for i in range(0, len(texts), self.config.batch_size):
            batch_texts = texts[i:i + self.config.batch_size]
            
            # Tokenize batch
            inputs = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                truncation=True,
                max_length=self.config.max_length,
                padding=True
            )
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Apply pooling strategy
            batch_embeddings = self._pool_embeddings(
                outputs.last_hidden_state,
                inputs['attention_mask']
            )
            
            # Normalize if requested
            if self.config.normalize_embeddings:
                batch_embeddings = F.normalize(batch_embeddings, p=2, dim=-1)
            
            all_embeddings.append(batch_embeddings.cpu())
            
            # Update token count
            self.stats['total_tokens'] += inputs['input_ids'].shape[1]
            
            if show_progress:
                print(f"Processed {min(i + self.config.batch_size, len(texts))}/{len(texts)} texts")
        
        return torch.cat(all_embeddings, dim=0)
    
    def _pool_embeddings(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply pooling strategy to hidden states
        
        Args:
            hidden_states: [batch_size, seq_len, hidden_dim]
            attention_mask: [batch_size, seq_len]
            
        Returns:
            pooled: [batch_size, hidden_dim]
        """
        if self.config.pooling_strategy == "cls":
            # Use [CLS] token (first token)
            return hidden_states[:, 0, :]
        
        elif self.config.pooling_strategy == "mean":
            # Mean pooling (attention-weighted)
            mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
            sum_embeddings = torch.sum(hidden_states * mask_expanded, dim=1)
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            return sum_embeddings / sum_mask
        
        elif self.config.pooling_strategy == "max":
            # Max pooling
            mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
            hidden_states[mask_expanded == 0] = -1e9  # Mask padding tokens
            return torch.max(hidden_states, dim=1)[0]
        
        else:
            raise ValueError(f"Unknown pooling strategy: {self.config.pooling_strategy}")
    
    def _simulate_embedding(self, text: str) -> torch.Tensor:
        """Fallback simulation using hash-based embedding"""
        # Normalize text
        text_lower = text.lower()
        words = text_lower.split()
        
        # Create 768-dim embedding (BERT-base dimension)
        embedding = torch.zeros(768, dtype=torch.float32)
        
        # Hash each word and add to embedding
        for word in words:
            word_hash = hashlib.sha256(word.encode()).digest()
            for i in range(768):
                byte_idx = i % len(word_hash)
                embedding[i] += float(word_hash[byte_idx]) / 255.0
        
        # Normalize to unit length
        norm = torch.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        return embedding
    
    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text"""
        return hashlib.sha256(text.encode()).hexdigest()[:16]
    
    def _update_cache(self, key: str, embedding: torch.Tensor):
        """Update embedding cache with LRU eviction"""
        if len(self.embedding_cache) >= self.max_cache_size:
            # Remove oldest entry (simple FIFO)
            oldest_key = next(iter(self.embedding_cache))
            del self.embedding_cache[oldest_key]
        
        self.embedding_cache[key] = embedding.clone()
    
    def clear_cache(self):
        """Clear embedding cache"""
        self.embedding_cache.clear()
        logger.info("Embedding cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get encoding statistics"""
        avg_time = (
            self.stats['total_time_ms'] / self.stats['total_encodings']
            if self.stats['total_encodings'] > 0 else 0
        )
        
        cache_hit_rate = (
            self.stats['cache_hits'] / 
            (self.stats['total_encodings'] + self.stats['cache_hits'])
            if (self.stats['total_encodings'] + self.stats['cache_hits']) > 0 else 0
        )
        
        return {
            'model_name': self.config.model_name,
            'is_simulation_mode': self.is_simulation_mode,
            'device': str(self.device) if self.device else 'none',
            'total_encodings': self.stats['total_encodings'],
            'total_tokens': self.stats['total_tokens'],
            'avg_time_ms': avg_time,
            'cache_size': len(self.embedding_cache),
            'cache_hit_rate': cache_hit_rate,
            'simulation_fallbacks': self.stats['simulation_fallbacks']
        }
    
    def save_cache(self, path: Path):
        """Save embedding cache to disk"""
        torch.save(self.embedding_cache, path)
        logger.info(f"Saved embedding cache to {path}")
    
    def load_cache(self, path: Path):
        """Load embedding cache from disk"""
        if path.exists():
            self.embedding_cache = torch.load(path)
            logger.info(f"Loaded embedding cache from {path} ({len(self.embedding_cache)} entries)")
        else:
            logger.warning(f"Cache file not found: {path}")


def integrate_with_vector_memory():
    """
    Example: Integrate BERT embedder with VectorMemory
    
    Replaces simulated embeddings in vector_memory.py
    """
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
    original_generate_embedding = memory._generate_embedding
    
    def bert_generate_embedding(text: str) -> torch.Tensor:
        """Use BERT instead of hash-based embedding"""
        return embedder.encode(text, use_cache=True)
    
    memory._generate_embedding = bert_generate_embedding
    
    logger.info("Integrated BERT embedder with VectorMemory")
    return memory, embedder


def integrate_with_safety_layer():
    """
    Example: Integrate BERT embedder with safety validation
    
    Replaces action embedding generation in llm_agent.py
    """
    from llm_agent import FDQCAgent
    
    # Create BERT embedder
    embedder = BERTEmbedder(BERTConfig(
        model_name="bert-base-uncased",
        use_gpu=True,
        normalize_embeddings=True
    ))
    
    # Create agent
    agent = FDQCAgent()
    
    # Replace action embedding method
    original_create_embedding = agent._create_action_embedding
    
    def bert_create_embedding(action: str, dim: int) -> torch.Tensor:
        """Use BERT for action embeddings"""
        embedding = embedder.encode(action, use_cache=True)
        
        # Adjust dimension if needed
        if embedding.shape[0] != dim:
            # Simple projection
            if embedding.shape[0] > dim:
                embedding = embedding[:dim]
            else:
                # Pad with zeros
                padding = torch.zeros(dim - embedding.shape[0])
                embedding = torch.cat([embedding, padding])
        
        return embedding.unsqueeze(0)
    
    agent._create_action_embedding = bert_create_embedding
    
    logger.info("Integrated BERT embedder with FDQCAgent")
    return agent, embedder


if __name__ == "__main__":
    print("Testing BERT Integration...")
    
    # Test basic encoding
    print("\n1. Testing single text encoding...")
    embedder = BERTEmbedder()
    
    text = "This is a test sentence for BERT encoding."
    embedding = embedder.encode(text)
    print(f"  Text: {text}")
    print(f"  Embedding shape: {embedding.shape}")
    print(f"  Embedding norm: {torch.norm(embedding):.4f}")
    print(f"  Mode: {'Simulation' if embedder.is_simulation_mode else 'BERT'}")
    
    # Test batch encoding
    print("\n2. Testing batch encoding...")
    texts = [
        "First test sentence.",
        "Second test sentence with different content.",
        "Third sentence for batch processing."
    ]
    embeddings = embedder.encode_batch(texts)
    print(f"  Batch size: {len(texts)}")
    print(f"  Embeddings shape: {embeddings.shape}")
    
    # Test cache
    print("\n3. Testing cache...")
    _ = embedder.encode(text)  # Should hit cache
    stats = embedder.get_stats()
    print(f"  Cache hit rate: {stats['cache_hit_rate']:.2%}")
    print(f"  Cache size: {stats['cache_size']}")
    
    # Test similarity
    print("\n4. Testing semantic similarity...")
    text1 = "The cat sits on the mat."
    text2 = "A feline rests on the rug."
    text3 = "Python is a programming language."
    
    emb1 = embedder.encode(text1)
    emb2 = embedder.encode(text2)
    emb3 = embedder.encode(text3)
    
    sim_12 = F.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0)).item()
    sim_13 = F.cosine_similarity(emb1.unsqueeze(0), emb3.unsqueeze(0)).item()
    
    print(f"  Similarity (cat/feline): {sim_12:.4f}")
    print(f"  Similarity (cat/python): {sim_13:.4f}")
    print(f"  ✓ Semantic similarity working: {sim_12 > sim_13}")
    
    # Print stats
    print("\n5. Final statistics...")
    stats = embedder.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\n✓ BERT integration tests complete")
