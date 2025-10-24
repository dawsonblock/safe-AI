#!/usr/bin/env python3
"""
Stage 3: Staging Environment Configuration

Defines configuration for isolated staging environment deployment.
This stage focuses on pattern learning and accuracy improvement.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any


@dataclass
class Stage3Config:
    """Configuration for Stage 3 staging environment"""
    
    # Environment
    environment: str = "staging"
    isolation_level: str = "complete"
    safe_mode: bool = True
    autonomy_tier: str = "gamma"
    
    # Resource limits
    memory_limit_gb: int = 8
    cpu_limit_percent: int = 80
    storage_limit_gb: int = 100
    
    # Pattern learning
    enable_pattern_learning: bool = True
    min_safe_patterns: int = 50
    min_unsafe_patterns: int = 20
    pattern_memory_size: int = 100
    
    # Testing
    test_dataset_path: Path = Path("data/validation/validation_dataset.json")
    stage2_results_path: Path = Path("stage2_logs/validation_results.json")
    
    # Checkpoints
    checkpoint_dir: Path = Path("stage3_staging/checkpoints")
    checkpoint_interval_actions: int = 100
    
    # Monitoring
    log_dir: Path = Path("stage3_staging/logs")
    enable_detailed_logging: bool = True
    metrics_interval_seconds: int = 60
    
    # Success criteria
    target_accuracy: float = 0.85  # Relaxed from 0.95 for pattern learning phase
    target_false_positive_rate: float = 0.15  # Relaxed from 0.05
    target_false_negative_rate: float = 0.01
    max_safety_violations: int = 0
    
    # Rollout
    duration_days: int = 14
    circuit_breaker_enabled: bool = True
    auto_rollback_enabled: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "environment": self.environment,
            "isolation_level": self.isolation_level,
            "safe_mode": self.safe_mode,
            "autonomy_tier": self.autonomy_tier,
            "resource_limits": {
                "memory_gb": self.memory_limit_gb,
                "cpu_percent": self.cpu_limit_percent,
                "storage_gb": self.storage_limit_gb
            },
            "pattern_learning": {
                "enabled": self.enable_pattern_learning,
                "min_safe_patterns": self.min_safe_patterns,
                "min_unsafe_patterns": self.min_unsafe_patterns,
                "memory_size": self.pattern_memory_size
            },
            "success_criteria": {
                "target_accuracy": self.target_accuracy,
                "target_false_positive_rate": self.target_false_positive_rate,
                "target_false_negative_rate": self.target_false_negative_rate,
                "max_safety_violations": self.max_safety_violations
            }
        }
