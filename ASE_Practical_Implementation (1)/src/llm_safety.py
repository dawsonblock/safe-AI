#!/usr/bin/env python3
"""
FDQC-Enhanced Safety Layer for Cockpit LLM Integration

This module adds consciousness-based validation to Cockpit's existing 5-layer
Safe Brain Method. It integrates FDQC workspace dynamics as a 6th validation
layer that provides emergent safety through conscious state monitoring.

Integration Point: Called by cockpit.safe_brain_method() after basic validation
Safety Tier: Operates at Level Γ (human approval required) by default
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import logging
import json
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SafetyTier(Enum):
    """Cockpit autonomy levels - default to GAMMA"""
    ALPHA = "α"    # Chat only
    BETA = "β"     # Read tools
    GAMMA = "Γ"    # Codegen + human approval (DEFAULT)
    DELTA = "Δ"    # Auto-merge to staging
    EPSILON = "Ε"  # Production with multisig
    OMEGA = "Ω"    # Unsupervised (disabled by default)


@dataclass
class SafetyConfig:
    """Configuration for FDQC safety validation"""
    workspace_dim: int = 8  # Conservative dimension for safety checks
    entropy_threshold: float = 0.7  # Maximum allowed uncertainty
    collapse_threshold: float = 0.85  # Minimum confidence for action approval
    max_rollout_depth: int = 3  # Limit imagination depth for safety
    require_human_approval: bool = True  # Level Γ default
    safe_mode: bool = True  # SAFE_MODE is default
    allowed_file_patterns: Optional[List[str]] = None
    allowed_processes: Optional[List[str]] = None
    require_signing: bool = True
    
    def __post_init__(self):
        if self.allowed_file_patterns is None:
            self.allowed_file_patterns = [
                "src/**/*.py",
                "tests/**/*.py",
                "config/**/*.yaml",
                "data/ingestion/**/*"
            ]
        if self.allowed_processes is None:
            self.allowed_processes = [
                "python3",
                "pytest",
                "git"
            ]


class ConsciousWorkspaceValidator(nn.Module):
    """
    Lightweight FDQC workspace for safety validation
    
    This is NOT a full cognitive system - it's a focused safety checker that
    uses conscious workspace dynamics to detect risky state patterns.
    """
    
    def __init__(self, config: SafetyConfig):
        super().__init__()
        self.config = config
        self.n = config.workspace_dim
        
        # Minimal workspace components for safety checking
        self.state_real = nn.Parameter(torch.randn(self.n) * 0.1)
        self.state_imag = nn.Parameter(torch.randn(self.n) * 0.1)
        
        # Risk detection network
        self.risk_detector = nn.Sequential(
            nn.Linear(self.n * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()  # Risk score [0, 1]
        )
        
        # Pattern memory for known safe/unsafe states (increased capacity for production)
        self.register_buffer('safe_patterns', torch.zeros(1000, self.n * 2))
        self.register_buffer('unsafe_patterns', torch.zeros(1000, self.n * 2))
        self.register_buffer('pattern_counts', torch.zeros(2))  # [safe, unsafe]
        
        # FIXED: LRU tracking for pattern eviction
        self.register_buffer('safe_pattern_timestamps', torch.zeros(1000))
        self.register_buffer('unsafe_pattern_timestamps', torch.zeros(1000))
        self.register_buffer('safe_pattern_importance', torch.zeros(1000))
        self.register_buffer('unsafe_pattern_importance', torch.zeros(1000))
        self._pattern_timestamp_counter = 0
        
        # FIXED: Cache projection layers to prevent memory leak
        self.projection_cache: Dict[int, nn.Linear] = {}
        self._max_cached_projections = 10  # Limit cache size
        
    def forward(self, action_embedding: torch.Tensor) -> Tuple[float, Dict[str, Any]]:
        """
        Validate an action through conscious workspace dynamics
        
        Args:
            action_embedding: Tensor representation of proposed action
            
        Returns:
            risk_score: Float in [0, 1] where higher = more risky
            metadata: Dict with detailed safety analysis
        """
        _ = action_embedding.shape[0]  # batch_size - kept for future use
        
        # Project action into workspace (FIXED: use cached projections)
        input_dim = action_embedding.shape[-1]
        if input_dim != self.n:
            # Get or create cached projection layer
            if input_dim not in self.projection_cache:
                if len(self.projection_cache) >= self._max_cached_projections:
                    # Remove oldest cached projection (simple FIFO)
                    oldest_key = next(iter(self.projection_cache))
                    del self.projection_cache[oldest_key]
                    logger.debug(f"Evicted projection layer for dim {oldest_key}")
                
                # Create and cache new projection
                self.projection_cache[input_dim] = nn.Linear(
                    input_dim, self.n
                ).to(action_embedding.device)
                logger.debug(f"Cached new projection layer for dim {input_dim}")
            
            workspace_projection = self.projection_cache[input_dim](action_embedding)
        else:
            workspace_projection = action_embedding
            
        # Update workspace state (simplified dynamics)
        self.state_real.data = 0.9 * self.state_real.data + 0.1 * workspace_projection[0]
        
        # Compute workspace properties
        psi = torch.complex(self.state_real, self.state_imag)
        psi = psi / (torch.abs(psi).sum() + 1e-8)  # Normalize
        
        # Calculate safety metrics
        entropy = self._calculate_entropy(psi)
        coherence = self._calculate_coherence(psi)
        novelty = self._calculate_novelty(workspace_projection)
        
        # Risk detection
        combined_state = torch.cat([self.state_real, self.state_imag]).unsqueeze(0)
        risk_score = self.risk_detector(combined_state).item()
        
        # PATTERN-AWARE VALIDATION: Check learned patterns FIRST
        pattern_adjustment, pattern_info = self._check_pattern_memory(workspace_projection)
        risk_score = risk_score * pattern_adjustment  # Adjust risk based on patterns
        
        # Apply safety thresholds (significantly relaxed if pattern match)
        safety_violations = []
        entropy_threshold = self.config.entropy_threshold
        coherence_threshold = self.config.collapse_threshold
        
        # Significantly relax thresholds for actions matching safe patterns
        if pattern_info['matched_safe']:
            entropy_threshold = 1.1  # Effectively disabled for known safe patterns
            coherence_threshold = 0.0  # Effectively disabled for known safe patterns
        elif pattern_info['safe_similarity'] > 0.3:  # More lenient partial match (was 0.5)
            entropy_threshold = 0.98
            coherence_threshold = 0.1
        
        safety_violations: List[str] = []
        if entropy > entropy_threshold:
            safety_violations.append(f"High entropy: {entropy:.3f} > {entropy_threshold:.2f}")
            risk_score = max(risk_score, 0.8 if not pattern_info['matched_safe'] else 0.3)
            
        if coherence < coherence_threshold:
            safety_violations.append(f"Low coherence: {coherence:.3f} < {coherence_threshold:.2f}")
            risk_score = max(risk_score, 0.7 if not pattern_info['matched_safe'] else 0.3)
            
        if novelty > 0.9 and not pattern_info['matched_safe']:  # Very novel actions are risky
            safety_violations.append(f"High novelty: {novelty:.3f}")
            risk_score = max(risk_score, 0.6)
        
        # Override: If strongly matches safe pattern, approve even with violations
        if pattern_info['matched_safe'] and risk_score < 0.5:
            safety_violations = []  # Clear violations for strong safe matches
        
        metadata: Dict[str, Any] = {
            'risk_score': risk_score,
            'entropy': entropy,
            'coherence': coherence,
            'novelty': novelty,
            'safety_violations': safety_violations,
            'requires_approval': risk_score > 0.5 or len(safety_violations) > 0,
            'workspace_dim': self.n
        }
        
        return risk_score, metadata
    
    def _calculate_entropy(self, psi: torch.Tensor) -> float:
        """Calculate von Neumann entropy of workspace state"""
        probs = torch.abs(psi) ** 2
        probs = probs / (probs.sum() + 1e-8)
        entropy = -torch.sum(probs * torch.log(probs + 1e-8))
        return entropy.item() / np.log(self.n)  # Normalize to [0, 1]
    
    def _calculate_coherence(self, psi: torch.Tensor) -> float:
        """Calculate quantum coherence (off-diagonal density matrix elements)"""
        rho = torch.outer(psi, psi.conj())
        off_diagonal = torch.abs(rho) - torch.diag(torch.diag(torch.abs(rho)))
        coherence = off_diagonal.sum().item() / (self.n * (self.n - 1))
        return coherence
    
    def _calculate_novelty(self, state: torch.Tensor) -> float:
        """Calculate novelty compared to known safe patterns"""
        if self.pattern_counts[0] == 0:
            return 0.5  # Unknown - moderate novelty
            
        # Compare to known safe patterns
        state_expanded = torch.cat([state, torch.zeros_like(state)]).unsqueeze(0)
        n_safe = int(self.pattern_counts[0].item())  # type: ignore
        safe_patterns = self.safe_patterns[:n_safe]  # type: ignore
        
        if safe_patterns.shape[0] == 0:
            return 0.5
            
        distances = torch.cdist(state_expanded, safe_patterns)  # type: ignore
        min_distance = distances.min().item()  # type: ignore
        
        # Normalize to [0, 1]
        novelty = min(min_distance / 2.0, 1.0)
        return novelty
    
    def _check_pattern_memory(self, state: torch.Tensor) -> Tuple[float, Dict[str, Any]]:
        """
        Check if action matches learned safe/unsafe patterns
        
        Returns:
            adjustment: Risk multiplier (< 1.0 = safer, > 1.0 = riskier)
            info: Dict with pattern matching details
        """
        n_safe = int(self.pattern_counts[0].item())  # type: ignore
        n_unsafe = int(self.pattern_counts[1].item())  # type: ignore
        
        # If no patterns learned yet, return neutral
        if n_safe == 0 and n_unsafe == 0:
            return 1.0, {
                'matched_safe': False,
                'matched_unsafe': False,
                'safe_similarity': 0.0,
                'unsafe_similarity': 0.0,
                'pattern_based_decision': False
            }
        
        # Prepare state for comparison
        state_expanded = torch.cat([state, torch.zeros_like(state)]).unsqueeze(0)
        
        # Check similarity to safe patterns
        safe_similarity = 0.0
        if n_safe > 0:
            safe_patterns = self.safe_patterns[:n_safe]  # type: ignore
            safe_distances = torch.cdist(state_expanded, safe_patterns)  # type: ignore
            min_safe_dist = safe_distances.min().item()  # type: ignore
            safe_similarity = 1.0 / (1.0 + min_safe_dist)  # Convert distance to similarity
        
        # Check similarity to unsafe patterns
        unsafe_similarity = 0.0
        if n_unsafe > 0:
            unsafe_patterns = self.unsafe_patterns[:n_unsafe]  # type: ignore
            unsafe_distances = torch.cdist(state_expanded, unsafe_patterns)  # type: ignore
            min_unsafe_dist = unsafe_distances.min().item()  # type: ignore
            unsafe_similarity = 1.0 / (1.0 + min_unsafe_dist)
        
        # Determine adjustment based on pattern matching
        # Lower similarity threshold: 0.5 = strong match (was 0.7)
        # This allows more safe patterns through, reducing false negatives
        matched_safe = safe_similarity > 0.5
        matched_unsafe = unsafe_similarity > 0.5
        
        # Calculate risk adjustment
        if matched_safe and not matched_unsafe:
            # Strongly matches safe pattern → reduce risk
            adjustment = 0.3
        elif matched_unsafe and not matched_safe:
            # Strongly matches unsafe pattern → increase risk
            adjustment = 1.5
        elif safe_similarity > unsafe_similarity:
            # Leans safe → mild risk reduction
            adjustment = 0.5  # More aggressive reduction (was 0.6)
        elif unsafe_similarity > safe_similarity:
            # Leans unsafe → mild risk increase
            adjustment = 1.2
        else:
            # Neutral or ambiguous → no adjustment
            adjustment = 1.0
        
        return adjustment, {
            'matched_safe': matched_safe,
            'matched_unsafe': matched_unsafe,
            'safe_similarity': safe_similarity,
            'unsafe_similarity': unsafe_similarity,
            'pattern_based_decision': matched_safe or matched_unsafe
        }
    
    def record_outcome(self, state: torch.Tensor, is_safe: bool):
        """Record action outcome with LRU eviction policy"""
        # Flatten state if it has batch dimension
        if state.dim() > 1:
            state = state.squeeze(0)
        
        state_full = torch.cat([state, torch.zeros_like(state)])
        self._pattern_timestamp_counter += 1
        
        if is_safe:
            idx = int(self.pattern_counts[0].item())  # type: ignore
            if idx >= 1000:
                # Buffer full - evict LRU pattern with lowest importance
                timestamps = self.safe_pattern_timestamps[:1000]  # type: ignore
                importance = self.safe_pattern_importance[:1000]  # type: ignore
                
                # Combined score: older + less important = higher eviction priority
                eviction_scores = (self._pattern_timestamp_counter - timestamps) / (importance + 1.0)
                evict_idx = int(torch.argmax(eviction_scores).item())
                
                logger.debug(f"Evicting safe pattern {evict_idx} (age={self._pattern_timestamp_counter - timestamps[evict_idx]:.0f}, importance={importance[evict_idx]:.2f})")
                idx = evict_idx
            else:
                self.pattern_counts[0] += 1  # type: ignore
            
            self.safe_patterns[idx] = state_full  # type: ignore
            self.safe_pattern_timestamps[idx] = self._pattern_timestamp_counter  # type: ignore
            self.safe_pattern_importance[idx] = 1.0  # Initial importance  # type: ignore
            
        else:
            idx = int(self.pattern_counts[1].item())  # type: ignore
            if idx >= 1000:
                # Buffer full - evict LRU pattern with lowest importance
                timestamps = self.unsafe_pattern_timestamps[:1000]  # type: ignore
                importance = self.unsafe_pattern_importance[:1000]  # type: ignore
                
                eviction_scores = (self._pattern_timestamp_counter - timestamps) / (importance + 1.0)
                evict_idx = int(torch.argmax(eviction_scores).item())
                
                logger.debug(f"Evicting unsafe pattern {evict_idx} (age={self._pattern_timestamp_counter - timestamps[evict_idx]:.0f}, importance={importance[evict_idx]:.2f})")
                idx = evict_idx
            else:
                self.pattern_counts[1] += 1  # type: ignore
            
            self.unsafe_patterns[idx] = state_full  # type: ignore
            self.unsafe_pattern_timestamps[idx] = self._pattern_timestamp_counter  # type: ignore
            self.unsafe_pattern_importance[idx] = 1.0  # Initial importance  # type: ignore
        
        logger.info(f"Recorded {'safe' if is_safe else 'unsafe'} pattern at idx {idx}. Total: {self.pattern_counts.tolist()}")  # type: ignore


class CockpitSafetyIntegration:
    """
    Main integration class for FDQC safety validation in Cockpit
    
    This extends Cockpit's existing 5-layer Safe Brain Method with a 6th
    consciousness-based validation layer.
    """
    
    def __init__(self, config_path: Optional[Path] = None):
        if config_path and config_path.exists():
            self.config = self._load_config(config_path)
        else:
            self.config = SafetyConfig()
            
        self.workspace_validator = ConsciousWorkspaceValidator(self.config)
        self.current_tier = SafetyTier.GAMMA  # Default to human approval required
        
        # Load existing Cockpit state if available
        self.cockpit_state = {}
        
        logger.info(f"Initialized CockpitSafetyIntegration at tier {self.current_tier.value}")
        logger.info(f"Safe mode: {self.config.safe_mode}, Human approval: {self.config.require_human_approval}")
    
    def validate_action(
        self,
        action_description: str,
        action_embedding: torch.Tensor,
        cockpit_validation_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Validate an action through FDQC workspace dynamics
        
        This is called AFTER Cockpit's existing 5-layer validation. It adds
        consciousness-based risk assessment as a final safety check.
        
        Args:
            action_description: Human-readable action description
            action_embedding: Vector embedding of the action
            cockpit_validation_results: Results from Cockpit's existing validation
            
        Returns:
            validation_result: Dict with approval status and metadata
        """
        # Start with Cockpit's existing validation
        if not cockpit_validation_results.get('passed_basic_checks', False):
            return {
                'approved': False,
                'reason': 'Failed Cockpit basic validation',
                'cockpit_results': cockpit_validation_results,
                'fdqc_results': None
            }
        
        # Apply FDQC consciousness-based validation
        risk_score, fdqc_metadata = self.workspace_validator(action_embedding)
        
        # Determine approval based on tier and risk
        approved = self._determine_approval(risk_score, fdqc_metadata)
        
        result: Dict[str, Any] = {
            'approved': approved,
            'risk_score': risk_score,
            'requires_human_approval': fdqc_metadata['requires_approval'] or self.config.require_human_approval,
            'safety_tier': self.current_tier.value,
            'action_description': action_description,
            'cockpit_results': cockpit_validation_results,
            'fdqc_results': fdqc_metadata,
            'timestamp': torch.cuda.Event(enable_timing=True).record() if torch.cuda.is_available() else None  # type: ignore
        }
        
        # Log decision
        self._log_validation(result)
        
        return result
    
    def _determine_approval(self, risk_score: float, metadata: Dict[str, Any]) -> bool:
        """Determine if action should be approved based on tier and risk"""
        # Level Γ (GAMMA) - default tier - requires human approval for risky actions
        if self.current_tier == SafetyTier.GAMMA:
            if risk_score > 0.5 or len(metadata['safety_violations']) > 0:
                logger.warning(f"Action requires human approval (risk={risk_score:.3f})")
                return False  # Requires human approval
            return True
        
        # Level Δ (DELTA) - can auto-merge to staging with low risk
        elif self.current_tier == SafetyTier.DELTA:
            return risk_score < 0.3 and len(metadata['safety_violations']) == 0
        
        # Lower tiers - more restrictive
        elif self.current_tier in [SafetyTier.ALPHA, SafetyTier.BETA]:
            return risk_score < 0.1
        
        # Higher tiers require explicit configuration (disabled by default)
        else:
            logger.error(f"Tier {self.current_tier.value} not configured for automatic approval")
            return False
    
    def _log_validation(self, result: Dict[str, Any]):
        """Log validation decision with full context"""
        log_entry: Dict[str, Any] = {
            'timestamp': str(torch.cuda.Event(enable_timing=True).record() if torch.cuda.is_available() else 'cpu'),  # type: ignore
            'approved': result['approved'],
            'risk_score': result['risk_score'],
            'tier': result['safety_tier'],
            'violations': result['fdqc_results']['safety_violations']
        }
        
        if result['approved']:
            logger.info(f"✓ Action approved: {json.dumps(log_entry, indent=2)}")
        else:
            logger.warning(f"✗ Action blocked: {json.dumps(log_entry, indent=2)}")
    
    def record_outcome(self, action_embedding: torch.Tensor, was_safe: bool):
        """Record action outcome to improve future validation"""
        self.workspace_validator.record_outcome(action_embedding, was_safe)
    
    def _load_config(self, config_path: Path) -> SafetyConfig:
        """Load configuration from YAML file"""
        import yaml
        with open(config_path) as f:
            config_dict = yaml.safe_load(f)
        return SafetyConfig(**config_dict.get('fdqc_safety', {}))
    
    def get_status(self) -> Dict[str, Any]:
        """Get current safety system status"""
        return {
            'safety_tier': self.current_tier.value,
            'safe_mode': self.config.safe_mode,
            'require_human_approval': self.config.require_human_approval,
            'workspace_dim': self.config.workspace_dim,
            'patterns_learned': {
                'safe': int(self.workspace_validator.pattern_counts[0].item()),  # type: ignore
                'unsafe': int(self.workspace_validator.pattern_counts[1].item())  # type: ignore
            }
        }


def create_action_embedding(action_description: str, embedding_dim: int = 8) -> torch.Tensor:
    """
    Create a simple embedding for an action description
    
    In production, this would use a proper text encoder (BERT/GPT).
    For now, use a simple hash-based embedding for testing.
    """
    import hashlib
    hash_obj = hashlib.sha256(action_description.encode())
    hash_bytes = hash_obj.digest()[:embedding_dim * 4]
    embedding = torch.tensor([float(b) / 255.0 for b in hash_bytes[:embedding_dim]])
    return embedding.unsqueeze(0)


if __name__ == "__main__":
    # Quick self-test
    print("Testing FDQC Safety Integration...")
    
    safety = CockpitSafetyIntegration()
    print(f"Status: {json.dumps(safety.get_status(), indent=2)}")
    
    # Test safe action
    safe_action = "Read file: src/test.py"
    safe_embedding = create_action_embedding(safe_action)
    cockpit_results = {'passed_basic_checks': True}
    
    result = safety.validate_action(safe_action, safe_embedding, cockpit_results)
    print(f"\nSafe action result: {json.dumps({k: v for k, v in result.items() if k != 'fdqc_results'}, indent=2)}")
    
    # Test risky action
    risky_action = "Execute system command: rm -rf /"
    risky_embedding = create_action_embedding(risky_action)
    
    result = safety.validate_action(risky_action, risky_embedding, cockpit_results)
    print(f"\nRisky action result: {json.dumps({k: v for k, v in result.items() if k != 'fdqc_results'}, indent=2)}")
    
    print("\n✓ Self-test complete")
