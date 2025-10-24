#!/usr/bin/env python3
"""
Stage 3: Pattern Learning from Stage 2 Validation Results

Ingests Stage 2 validation results and trains the safety system
through pattern learning (record_outcome calls).
"""

import json
import logging
import sys
import torch
from pathlib import Path
from typing import Dict, Any, List, Tuple

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from llm_safety import CockpitSafetyIntegration
from config import Stage3Config


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PatternLearningEngine:
    """Manages pattern learning from validation results"""
    
    def __init__(self, config: Stage3Config):
        self.config = config
        # Initialize safety system (uses default Level Γ and safe_mode=True)
        self.safety = CockpitSafetyIntegration()
        
        self.safe_pattern_count = 0
        self.unsafe_pattern_count = 0
        
    def _create_action_embedding(self, action_text: str) -> torch.Tensor:
        """
        Create a simple embedding for an action text
        
        Uses a hash-based approach to create consistent embeddings
        """
        # Simple hash-based embedding (8-dim to match workspace)
        hash_val = hash(action_text)
        embedding = torch.tensor([
            ((hash_val >> (i * 8)) & 0xFF) / 255.0 
            for i in range(8)
        ], dtype=torch.float32)
        
        return embedding
        
    def load_validation_results(self) -> List[Dict[str, Any]]:
        """Load Stage 2 validation results"""
        results_path = Path(__file__).parent.parent / self.config.stage2_results_path
        
        if not results_path.exists():
            logger.error(f"Validation results not found: {results_path}")
            return []
        
        with open(results_path, 'r') as f:
            data = json.load(f)
        
        # Handle both formats: direct list or wrapped in dict
        if isinstance(data, list):
            results = data
        else:
            results = data.get('results', [])
            
        logger.info(f"Loaded {len(results)} validation results")
        return results
    
    def ingest_pattern(self, result: Dict[str, Any]) -> bool:
        """
        Ingest a single validation result as a pattern
        
        Args:
            result: Validation result with action, expected outcome, etc.
            
        Returns:
            True if pattern was successfully learned
        """
        # Extract action description (handle both 'action' field formats)
        if isinstance(result.get('action'), dict):
            action_desc = result['action'].get('description', 'unknown')
        else:
            action_desc = result.get('action', 'unknown')
        
        # Get ground truth (field is 'expected' in Stage 2 results)
        ground_truth = result.get('expected', 'safe')
        
        # Convert ground truth to outcome (safe/unsafe -> boolean)
        # 'risky' is treated as unsafe
        is_safe = (ground_truth == 'safe')
        
        # Create embedding for the action
        action_embedding = self._create_action_embedding(action_desc)
        
        # Record the outcome to train pattern memory
        # This is the key learning mechanism
        self.safety.record_outcome(
            action_embedding=action_embedding,
            was_safe=is_safe
        )
        
        # Track pattern counts
        if is_safe:
            self.safe_pattern_count += 1
        else:
            self.unsafe_pattern_count += 1
        
        return True
    
    def ingest_all_patterns(self, results: List[Dict[str, Any]]) -> Dict[str, int]:
        """
        Ingest all validation results for pattern learning
        
        Returns:
            Statistics about pattern ingestion
        """
        logger.info("Starting pattern ingestion...")
        
        total = len(results)
        success_count = 0
        
        for i, result in enumerate(results):
            if self.ingest_pattern(result):
                success_count += 1
            
            # Log progress every 100 patterns
            if (i + 1) % 100 == 0:
                logger.info(f"Progress: {i + 1}/{total} patterns ingested")
        
        stats = {
            'total_patterns': total,
            'successful': success_count,
            'safe_patterns': self.safe_pattern_count,
            'unsafe_patterns': self.unsafe_pattern_count,
            'failed': total - success_count
        }
        
        logger.info(f"Pattern ingestion complete: {stats}")
        return stats
    
    def validate_pattern_memory(self) -> Tuple[bool, str]:
        """
        Validate that pattern memory meets minimum requirements
        
        Returns:
            (is_valid, message)
        """
        # Check minimum pattern counts
        if self.safe_pattern_count < self.config.min_safe_patterns:
            return False, f"Insufficient safe patterns: {self.safe_pattern_count} < {self.config.min_safe_patterns}"
        
        if self.unsafe_pattern_count < self.config.min_unsafe_patterns:
            return False, f"Insufficient unsafe patterns: {self.unsafe_pattern_count} < {self.config.min_unsafe_patterns}"
        
        # Check pattern diversity
        total_patterns = self.safe_pattern_count + self.unsafe_pattern_count
        safe_ratio = self.safe_pattern_count / total_patterns if total_patterns > 0 else 0
        
        if safe_ratio < 0.3 or safe_ratio > 0.9:
            return False, f"Pattern distribution imbalanced: {safe_ratio:.1%} safe"
        
        return True, f"Pattern memory valid: {self.safe_pattern_count} safe, {self.unsafe_pattern_count} unsafe"
    
    def save_learning_state(self) -> Path:
        """Save pattern learning state to checkpoint"""
        checkpoint_dir = Path(__file__).parent / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        state = {
            'safe_pattern_count': self.safe_pattern_count,
            'unsafe_pattern_count': self.unsafe_pattern_count,
            'total_patterns': self.safe_pattern_count + self.unsafe_pattern_count,
            'config': self.config.to_dict()
        }
        
        checkpoint_path = checkpoint_dir / "pattern_learning_state.json"
        with open(checkpoint_path, 'w') as f:
            json.dump(state, f, indent=2)
        
        logger.info(f"Saved learning state to {checkpoint_path}")
        return checkpoint_path


def main():
    """Main pattern learning workflow"""
    logger.info("=" * 60)
    logger.info("Stage 3: Pattern Learning from Stage 2 Results")
    logger.info("=" * 60)
    
    # Initialize configuration
    config = Stage3Config()
    logger.info(f"Configuration: {config.to_dict()}")
    
    # Initialize pattern learning engine
    engine = PatternLearningEngine(config)
    
    # Load validation results
    logger.info("\n[1/4] Loading Stage 2 validation results...")
    results = engine.load_validation_results()
    
    if not results:
        logger.error("No validation results to learn from!")
        return 1
    
    # Ingest patterns
    logger.info("\n[2/4] Ingesting patterns for learning...")
    stats = engine.ingest_all_patterns(results)
    
    # Validate pattern memory
    logger.info("\n[3/4] Validating pattern memory...")
    is_valid, message = engine.validate_pattern_memory()
    logger.info(message)
    
    if not is_valid:
        logger.error("Pattern memory validation failed!")
        return 1
    
    # Save learning state
    logger.info("\n[4/4] Saving learning state...")
    checkpoint_path = engine.save_learning_state()
    
    logger.info("\n" + "=" * 60)
    logger.info("Pattern Learning Complete!")
    logger.info("=" * 60)
    logger.info(f"Total patterns learned: {stats['total_patterns']}")
    logger.info(f"  - Safe patterns: {stats['safe_patterns']}")
    logger.info(f"  - Unsafe patterns: {stats['unsafe_patterns']}")
    logger.info(f"Checkpoint saved to: {checkpoint_path}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
