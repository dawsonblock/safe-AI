#!/usr/bin/env python3
"""
FDQC-Cockpit Quick Start Script

Demonstrates production deployment with:
- DeepSeek API integration
- BERT embeddings
- Pattern memory persistence
- Monitoring and alerting
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from config_loader import get_config, validate_config
from bert_integration import BERTEmbedder, BERTConfig
from production_hardening import (
    PatternMemoryPersistence,
    MonitoringSystem,
    AdversarialDetector,
    DeepSeekAPIClient
)
from llm_agent import FDQCAgent, AgentConfig
from llm_safety import CockpitSafetyIntegration, SafetyConfig
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main quick start demonstration"""
    
    print("=" * 70)
    print("FDQC-Cockpit Production Quick Start")
    print("=" * 70)
    
    # Step 1: Load Configuration
    print("\n[1/6] Loading configuration...")
    try:
        config = get_config()
        validate_config(config)
        print(f"✓ Configuration loaded")
        print(f"  - DeepSeek API: Configured")
        print(f"  - Safety Tier: {config.safety_tier}")
        print(f"  - Safe Mode: {config.safe_mode}")
    except Exception as e:
        print(f"✗ Configuration error: {e}")
        print("\nPlease ensure .env file exists with DEEPSEEK_API_KEY set")
        return 1
    
    # Step 2: Initialize BERT Embedder
    print("\n[2/6] Initializing BERT embedder...")
    try:
        bert_config = BERTConfig(
            model_name=config.bert_model_name,
            cache_dir=config.bert_cache_dir,
            use_gpu=config.bert_use_gpu,
            batch_size=config.bert_batch_size,
            normalize_embeddings=True
        )
        embedder = BERTEmbedder(bert_config)
        
        if embedder.is_simulation_mode:
            print("⚠ Running in simulation mode (transformers not installed)")
            print("  Install with: pip install transformers torch")
        else:
            print(f"✓ BERT model loaded: {config.bert_model_name}")
            stats = embedder.get_stats()
            print(f"  - Device: {stats['device']}")
    except Exception as e:
        print(f"✗ BERT initialization error: {e}")
        return 1
    
    # Step 3: Initialize DeepSeek API Client
    print("\n[3/6] Initializing DeepSeek API client...")
    try:
        deepseek = DeepSeekAPIClient(
            api_key=config.deepseek_api_key,
            max_retries=3
        )
        print("✓ DeepSeek API client initialized")
        print(f"  - API Key: {'*' * 20}{config.deepseek_api_key[-8:]}")
    except Exception as e:
        print(f"✗ DeepSeek initialization error: {e}")
        return 1
    
    # Step 4: Initialize Safety System
    print("\n[4/6] Initializing safety system...")
    try:
        safety_config = SafetyConfig(
            workspace_dim=config.workspace_dim,
            entropy_threshold=config.entropy_threshold,
            collapse_threshold=config.collapse_threshold,
            require_human_approval=config.require_human_approval,
            safe_mode=config.safe_mode
        )
        safety = CockpitSafetyIntegration()
        safety.config = safety_config
        
        print("✓ Safety system initialized")
        print(f"  - Tier: {config.safety_tier}")
        print(f"  - Human Approval: {config.require_human_approval}")
    except Exception as e:
        print(f"✗ Safety initialization error: {e}")
        return 1
    
    # Step 5: Initialize Monitoring
    print("\n[5/6] Initializing monitoring...")
    try:
        monitor = MonitoringSystem(alert_threshold={
            'risk_score': config.alert_risk_threshold,
            'error_rate': config.alert_error_rate_threshold,
            'memory_utilization': config.alert_memory_utilization_threshold
        })
        
        adversarial_detector = AdversarialDetector(sensitivity=0.95)
        
        persistence = PatternMemoryPersistence()
        
        print("✓ Monitoring system initialized")
        print(f"  - Checkpoint dir: {config.checkpoint_dir}")
        print(f"  - Auto-save interval: {config.auto_save_interval_seconds}s")
    except Exception as e:
        print(f"✗ Monitoring initialization error: {e}")
        return 1
    
    # Step 6: Initialize Agent
    print("\n[6/6] Initializing FDQC agent...")
    try:
        agent_config = AgentConfig(
            workspace_dim_range=(4, 12),
            imagination_depth=3,
            require_approval=config.require_human_approval
        )
        agent = FDQCAgent(agent_config=agent_config)
        
        # Integrate BERT embeddings
        def bert_create_embedding(action: str, dim: int):
            embedding = embedder.encode(action, use_cache=True)
            if embedding.shape[0] != dim:
                if embedding.shape[0] > dim:
                    embedding = embedding[:dim]
                else:
                    import torch
                    padding = torch.zeros(dim - embedding.shape[0])
                    embedding = torch.cat([embedding, padding])
            return embedding.unsqueeze(0)
        
        agent._create_action_embedding = bert_create_embedding
        
        print("✓ FDQC agent initialized")
        print(f"  - Workspace range: {agent_config.workspace_dim_range}")
        print(f"  - Imagination depth: {agent_config.imagination_depth}")
    except Exception as e:
        print(f"✗ Agent initialization error: {e}")
        return 1
    
    # Demonstration
    print("\n" + "=" * 70)
    print("System Ready - Running Demonstration")
    print("=" * 70)
    
    # Test action selection
    print("\n[Demo] Testing action selection...")
    observation = {
        'current_file': 'src/test.py',
        'task': 'code_review',
        'complexity': 0.5
    }
    available_actions = [
        "Read file: src/test.py",
        "Write file: src/output.py",
        "Execute: pytest tests/"
    ]
    
    try:
        result = agent.select_action(observation, available_actions)
        
        print(f"\nSelected Action: {result['action']}")
        print(f"Workspace Dimension: {result['workspace_dim']}")
        print(f"Risk Score: {result['safety_validation']['risk_score']:.3f}")
        print(f"Approved: {result['approved']}")
        print(f"Requires Approval: {result['requires_approval']}")
        
        # Record in monitoring
        monitor.record_validation(
            approved=result['approved'],
            risk_score=result['safety_validation']['risk_score'],
            requires_approval=result['requires_approval']
        )
        
        # Check for adversarial patterns
        is_outlier, anomaly_score, reason = adversarial_detector.detect_outlier(
            result['safety_validation']['fdqc_results']['workspace_dim'],
            is_safe=result['approved']
        )
        
        if is_outlier:
            print(f"\n⚠ Adversarial detection: {reason}")
        
    except Exception as e:
        print(f"✗ Action selection error: {e}")
        monitor.record_error('action_selection', str(e))
    
    # Show statistics
    print("\n" + "=" * 70)
    print("System Statistics")
    print("=" * 70)
    
    print("\nBERT Embedder:")
    bert_stats = embedder.get_stats()
    for key, value in bert_stats.items():
        print(f"  {key}: {value}")
    
    print("\nMonitoring:")
    monitor_stats = monitor.get_metrics()
    for key, value in monitor_stats.items():
        print(f"  {key}: {value}")
    
    print("\nSafety System:")
    safety_stats = safety.get_status()
    for key, value in safety_stats.items():
        print(f"  {key}: {value}")
    
    print("\n" + "=" * 70)
    print("✓ Quick Start Complete")
    print("=" * 70)
    print("\nNext Steps:")
    print("1. Review PRODUCTION_DEPLOYMENT_GUIDE.md for full deployment")
    print("2. Review BERT_INTEGRATION_GUIDE.md for embedding optimization")
    print("3. Configure monitoring alerts (Slack, etc.)")
    print("4. Set up automated checkpointing")
    print("5. Deploy to production environment")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
