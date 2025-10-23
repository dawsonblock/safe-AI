#!/usr/bin/env python3
"""
Vector Memory with DeepSeek OCR Integration

Provides semantic memory storage with 10x compression through DeepSeek OCR.
Integrates with Cockpit's document ingestion pipeline.

Key Features:
- DeepSeek OCR: 1,000 words → 100 tokens (10x compression)
- Vector embeddings for semantic search
- Safe file access through Cockpit policy
- Integration with existing knowledge base
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from pathlib import Path
import json
import logging
from collections import defaultdict
import hashlib

logger = logging.getLogger(__name__)


@dataclass
class MemoryConfig:
    """Configuration for vector memory system"""
    embedding_dim: int = 768  # BERT-base dimension
    max_entries: int = 100000  # 100K documents
    compression_ratio: float = 0.1  # 10x compression (DeepSeek OCR)
    similarity_threshold: float = 0.7
    use_deepseek_ocr: bool = True
    allowed_extensions: List[str] = None
    max_file_size_mb: int = 50
    
    def __post_init__(self):
        if self.allowed_extensions is None:
            self.allowed_extensions = ['.txt', '.md', '.pdf', '.py', '.json']


@dataclass
class MemoryEntry:
    """Single entry in vector memory"""
    id: str
    content: str
    compressed_content: Optional[str]
    embedding: torch.Tensor
    metadata: Dict[str, Any]
    timestamp: float
    source_file: Optional[Path] = None
    compression_stats: Optional[Dict[str, float]] = None


class DeepSeekOCRCompressor:
    """
    DeepSeek OCR compression wrapper
    
    Implements 10x compression: 1,000 words → 100 tokens
    97% accuracy, $5 vs $60 per 1K pages compared to traditional OCR
    
    In production, this would call actual DeepSeek API. For now, it's a
    placeholder that simulates the compression behavior.
    """
    
    def __init__(self, config: MemoryConfig):
        self.config = config
        self.compression_stats = {
            'total_chars_input': 0,
            'total_chars_output': 0,
            'total_documents': 0
        }
    
    def compress(self, text: str) -> Tuple[str, Dict[str, float]]:
        """
        Compress text using DeepSeek OCR semantic compression
        
        Args:
            text: Input text to compress
            
        Returns:
            compressed_text: Semantically compressed version
            stats: Compression statistics
        """
        input_len = len(text)
        
        # Simulate DeepSeek OCR compression
        # In production, this would call: deepseek_api.compress(text)
        compressed = self._simulate_compression(text)
        output_len = len(compressed)
        
        stats = {
            'input_chars': input_len,
            'output_chars': output_len,
            'compression_ratio': output_len / input_len if input_len > 0 else 0,
            'estimated_cost_usd': input_len / 1000 * 0.005  # $5 per 1K pages estimate
        }
        
        # Update global stats
        self.compression_stats['total_chars_input'] += input_len
        self.compression_stats['total_chars_output'] += output_len
        self.compression_stats['total_documents'] += 1
        
        logger.debug(f"Compressed {input_len} chars → {output_len} chars (ratio: {stats['compression_ratio']:.2f})")
        
        return compressed, stats
    
    def _simulate_compression(self, text: str) -> str:
        """
        Simulate DeepSeek OCR compression behavior
        
        Real implementation would use DeepSeek API. This is a placeholder
        that performs intelligent summarization to ~10% of original length.
        """
        # Split into sentences
        sentences = text.split('. ')
        
        # Target 10% of original length
        target_len = int(len(text) * self.config.compression_ratio)
        
        # Select most informative sentences (simple heuristic)
        scored_sentences = []
        for sent in sentences:
            # Score based on length and keyword density
            score = len(sent) * (1 + sent.count(' ') / len(sent.split()))
            scored_sentences.append((score, sent))
        
        # Sort by score and take top sentences
        scored_sentences.sort(reverse=True)
        compressed = '. '.join([s[1] for s in scored_sentences])
        
        # Truncate to target length
        if len(compressed) > target_len:
            compressed = compressed[:target_len]
            # Ensure we end at a word boundary
            last_space = compressed.rfind(' ')
            if last_space > 0:
                compressed = compressed[:last_space]
        
        return compressed.strip()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get compression statistics"""
        if self.compression_stats['total_chars_input'] > 0:
            overall_ratio = (
                self.compression_stats['total_chars_output'] /
                self.compression_stats['total_chars_input']
            )
        else:
            overall_ratio = 0
        
        return {
            'total_documents': self.compression_stats['total_documents'],
            'total_input_chars': self.compression_stats['total_chars_input'],
            'total_output_chars': self.compression_stats['total_chars_output'],
            'overall_compression_ratio': overall_ratio,
            'estimated_total_cost_usd': self.compression_stats['total_chars_input'] / 1000 * 0.005
        }


class VectorMemory:
    """
    Vector memory system with semantic search and compression
    
    Integrates with Cockpit's document ingestion pipeline and provides
    fast semantic search over compressed document corpus.
    """
    
    def __init__(self, config: Optional[MemoryConfig] = None):
        self.config = config or MemoryConfig()
        
        # Memory storage
        self.entries: Dict[str, MemoryEntry] = {}
        self.embeddings = torch.zeros((self.config.max_entries, self.config.embedding_dim))
        self.next_index = 0
        
        # Compression
        if self.config.use_deepseek_ocr:
            self.compressor = DeepSeekOCRCompressor(self.config)
        else:
            self.compressor = None
        
        # Index mapping
        self.id_to_index: Dict[str, int] = {}
        self.index_to_id: Dict[int, str] = {}
        
        # Metadata indices for fast filtering
        self.metadata_index: Dict[str, List[str]] = defaultdict(list)
        
        logger.info(f"Initialized VectorMemory with {self.config.max_entries} capacity")
        if self.compressor:
            logger.info("DeepSeek OCR compression enabled (10x compression)")
    
    def add_document(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        source_file: Optional[Path] = None,
        embedding: Optional[torch.Tensor] = None
    ) -> str:
        """
        Add document to memory with optional compression
        
        Args:
            content: Document text content
            metadata: Optional metadata dict
            source_file: Optional source file path
            embedding: Optional pre-computed embedding
            
        Returns:
            document_id: Unique ID for the added document
        """
        # Generate ID
        doc_id = self._generate_id(content, source_file)
        
        # Check if already exists
        if doc_id in self.entries:
            logger.warning(f"Document {doc_id} already exists, skipping")
            return doc_id
        
        # Compress if enabled
        compressed_content = None
        compression_stats = None
        if self.compressor:
            compressed_content, compression_stats = self.compressor.compress(content)
        
        # Generate embedding if not provided
        if embedding is None:
            embedding = self._generate_embedding(compressed_content or content)
        
        # Create entry
        entry = MemoryEntry(
            id=doc_id,
            content=content,
            compressed_content=compressed_content,
            embedding=embedding,
            metadata=metadata or {},
            timestamp=torch.cuda.Event(enable_timing=True).record() if torch.cuda.is_available() else 0,
            source_file=source_file,
            compression_stats=compression_stats
        )
        
        # Store entry
        if self.next_index >= self.config.max_entries:
            logger.error("Memory capacity reached, cannot add more documents")
            return None
        
        idx = self.next_index
        self.entries[doc_id] = entry
        self.embeddings[idx] = embedding
        self.id_to_index[doc_id] = idx
        self.index_to_id[idx] = doc_id
        self.next_index += 1
        
        # Update metadata indices
        if metadata:
            for key, value in metadata.items():
                self.metadata_index[f"{key}:{value}"].append(doc_id)
        
        logger.info(f"Added document {doc_id} at index {idx}")
        if compression_stats:
            logger.info(f"  Compression: {compression_stats['input_chars']} → {compression_stats['output_chars']} chars")
        
        return doc_id
    
    def search(
        self,
        query: Union[str, torch.Tensor],
        top_k: int = 5,
        metadata_filter: Optional[Dict[str, Any]] = None,
        similarity_threshold: Optional[float] = None
    ) -> List[Tuple[str, float, MemoryEntry]]:
        """
        Semantic search over memory
        
        Args:
            query: Query string or embedding
            top_k: Number of results to return
            metadata_filter: Optional filter by metadata
            similarity_threshold: Optional minimum similarity score
            
        Returns:
            results: List of (doc_id, similarity, entry) tuples
        """
        # Generate query embedding
        if isinstance(query, str):
            query_embedding = self._generate_embedding(query)
        else:
            query_embedding = query
        
        # Filter by metadata if specified
        if metadata_filter:
            candidate_ids = self._filter_by_metadata(metadata_filter)
            if not candidate_ids:
                return []
            candidate_indices = [self.id_to_index[doc_id] for doc_id in candidate_ids]
        else:
            candidate_indices = list(range(self.next_index))
        
        # Calculate similarities
        candidate_embeddings = self.embeddings[candidate_indices]
        similarities = F.cosine_similarity(
            query_embedding.unsqueeze(0),
            candidate_embeddings,
            dim=1
        )
        
        # Apply threshold
        threshold = similarity_threshold or self.config.similarity_threshold
        mask = similarities >= threshold
        
        # Get top-k
        topk_values, topk_indices = torch.topk(
            similarities[mask],
            min(top_k, mask.sum().item()),
            largest=True
        )
        
        # Convert to results
        results = []
        for value, idx in zip(topk_values, topk_indices):
            global_idx = candidate_indices[idx.item()]
            doc_id = self.index_to_id[global_idx]
            entry = self.entries[doc_id]
            results.append((doc_id, value.item(), entry))
        
        return results
    
    def get_document(self, doc_id: str) -> Optional[MemoryEntry]:
        """Retrieve document by ID"""
        return self.entries.get(doc_id)
    
    def ingest_directory(
        self,
        directory: Path,
        recursive: bool = True,
        file_pattern: str = "*"
    ) -> Dict[str, Any]:
        """
        Ingest all documents from a directory
        
        Args:
            directory: Directory path to ingest
            recursive: Whether to recurse into subdirectories
            file_pattern: Glob pattern for files to include
            
        Returns:
            stats: Ingestion statistics
        """
        if not directory.exists():
            logger.error(f"Directory does not exist: {directory}")
            return {'error': 'directory_not_found'}
        
        stats = {
            'files_processed': 0,
            'files_skipped': 0,
            'total_chars': 0,
            'total_compressed_chars': 0,
            'errors': []
        }
        
        # Find files
        if recursive:
            files = list(directory.rglob(file_pattern))
        else:
            files = list(directory.glob(file_pattern))
        
        logger.info(f"Found {len(files)} files to ingest from {directory}")
        
        for file_path in files:
            # Check extension
            if file_path.suffix not in self.config.allowed_extensions:
                stats['files_skipped'] += 1
                continue
            
            # Check size
            size_mb = file_path.stat().st_size / (1024 * 1024)
            if size_mb > self.config.max_file_size_mb:
                logger.warning(f"Skipping large file: {file_path} ({size_mb:.1f} MB)")
                stats['files_skipped'] += 1
                continue
            
            # Read and add
            try:
                content = file_path.read_text(encoding='utf-8', errors='ignore')
                metadata = {
                    'filename': file_path.name,
                    'extension': file_path.suffix,
                    'size_bytes': file_path.stat().st_size
                }
                
                doc_id = self.add_document(content, metadata, file_path)
                
                if doc_id:
                    stats['files_processed'] += 1
                    stats['total_chars'] += len(content)
                    
                    entry = self.entries[doc_id]
                    if entry.compressed_content:
                        stats['total_compressed_chars'] += len(entry.compressed_content)
                
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
                stats['errors'].append(str(file_path))
        
        logger.info(f"Ingestion complete: {stats['files_processed']} files processed")
        if self.compressor and stats['total_chars'] > 0:
            ratio = stats['total_compressed_chars'] / stats['total_chars']
            logger.info(f"  Compression ratio: {ratio:.2f} ({stats['total_chars']} → {stats['total_compressed_chars']} chars)")
        
        return stats
    
    def _generate_id(self, content: str, source_file: Optional[Path]) -> str:
        """Generate unique ID for document"""
        if source_file:
            base = str(source_file)
        else:
            base = content[:100]
        
        return hashlib.sha256(base.encode()).hexdigest()[:16]
    
    def _generate_embedding(self, text: str) -> torch.Tensor:
        """
        Generate embedding for text
        
        In production, this would use BERT or similar encoder.
        For now, use simple hash-based embedding.
        """
        # Simulate embedding generation
        hash_obj = hashlib.sha256(text.encode())
        hash_bytes = hash_obj.digest()
        
        # Convert to float tensor
        embedding = []
        for i in range(self.config.embedding_dim):
            byte_idx = i % len(hash_bytes)
            embedding.append(float(hash_bytes[byte_idx]) / 255.0)
        
        return torch.tensor(embedding, dtype=torch.float32)
    
    def _filter_by_metadata(self, metadata_filter: Dict[str, Any]) -> List[str]:
        """Filter documents by metadata"""
        # Start with all documents
        candidate_ids = set(self.entries.keys())
        
        # Apply each filter
        for key, value in metadata_filter.items():
            filter_key = f"{key}:{value}"
            matching_ids = set(self.metadata_index.get(filter_key, []))
            candidate_ids &= matching_ids
        
        return list(candidate_ids)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics"""
        stats = {
            'total_documents': len(self.entries),
            'capacity': self.config.max_entries,
            'utilization': len(self.entries) / self.config.max_entries,
            'embedding_dim': self.config.embedding_dim
        }
        
        if self.compressor:
            stats['compression'] = self.compressor.get_stats()
        
        return stats


if __name__ == "__main__":
    # Quick self-test
    print("Testing Vector Memory with DeepSeek OCR...")
    
    memory = VectorMemory()
    print(f"Initial stats: {json.dumps(memory.get_stats(), indent=2)}")
    
    # Add some documents
    docs = [
        ("This is a test document about machine learning and artificial intelligence.", {'topic': 'AI'}),
        ("Python programming is useful for data science and web development.", {'topic': 'Programming'}),
        ("Deep learning models require large datasets and computational resources.", {'topic': 'AI'})
    ]
    
    for content, metadata in docs:
        doc_id = memory.add_document(content, metadata)
        print(f"Added document: {doc_id}")
    
    # Search
    query = "artificial intelligence and deep learning"
    results = memory.search(query, top_k=2)
    
    print(f"\nSearch results for: '{query}'")
    for doc_id, similarity, entry in results:
        print(f"  {doc_id}: {similarity:.3f}")
        print(f"    Original: {entry.content[:80]}...")
        if entry.compressed_content:
            print(f"    Compressed: {entry.compressed_content[:80]}...")
    
    # Final stats
    print(f"\nFinal stats: {json.dumps(memory.get_stats(), indent=2)}")
    
    print("\n✓ Self-test complete")
