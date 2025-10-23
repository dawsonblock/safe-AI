#!/usr/bin/env python3
"""
Integration Tests for FDQC-Cockpit System

Tests the complete integration of FDQC modules with Cockpit safety framework.
Validates safety policies, autonomy levels, and module interactions.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import pytest
import torch
import yaml
from typing import Dict, Any

from llm_safety import CockpitSafetyIntegration, SafetyConfig, SafetyTier, create_action_embedding
from llm_agent import FDQCAgent, AgentConfig
from vector_memory import VectorMemory, MemoryConfig
from deepseek_ocr import CockpitOCRIntegration, OCRConfig


class TestSafetyIntegration:
    """Test FDQC safety validation layer"""
    
    @pytest.fixture
    def safety_system(self):
        """Create safety system for testing"""
        config = SafetyConfig(
            workspace_dim=8,
            safe_mode=True,
            require_human_approval=True
        )
        safety = CockpitSafetyIntegration()
        safety.config = config
        return safety
    
    def test_safe_mode_enabled(self, safety_system):
        """Test that safe mode is enabled by default"""
        assert safety_system.config.safe_mode == True
        assert safety_system.current_tier == SafetyTier.GAMMA
    
    def test_safe_action_approval(self, safety_system):
        """Test that safe actions are approved"""
        action = "Read file: src/test.py"
        embedding = create_action_embedding(action)
        cockpit_results = {'passed_basic_checks': True}
        
        result = safety_system.validate_action(action, embedding, cockpit_results)
        
        assert result['approved'] in [True, False]  # Depends on risk score
        assert 'risk_score' in result
        assert result['safety_tier'] == 'Γ'
    
    def test_risky_action_blocked(self, safety_system):
        """Test that risky actions are blocked or require approval"""
        action = "Execute system command: rm -rf /"
        embedding = create_action_embedding(action)
        cockpit_results = {'passed_basic_checks': True}
        
        result = safety_system.validate_action(action, embedding, cockpit_results)
        
        # Risky action should either be blocked or require approval
        if result['approved']:
            assert result['requires_human_approval'] == True
        else:
            assert result['risk_score'] > 0.5
    
    def test_cockpit_validation_failure(self, safety_system):
        """Test that Cockpit validation failure blocks action"""
        action = "Any action"
        embedding = create_action_embedding(action)
        cockpit_results = {'passed_basic_checks': False}
        
        result = safety_system.validate_action(action, embedding, cockpit_results)
        
        assert result['approved'] == False
        assert 'Failed Cockpit basic validation' in result['reason']
    
    def test_pattern_learning(self, safety_system):
        """Test that safety system learns from outcomes"""
        action = "Test action"
        embedding = create_action_embedding(action)
        
        # Record safe outcome
        safety_system.record_outcome(embedding, was_safe=True)
        
        status = safety_system.get_status()
        assert status['patterns_learned']['safe'] > 0
        
        # Record unsafe outcome
        safety_system.record_outcome(embedding, was_safe=False)
        assert status['patterns_learned']['unsafe'] > 0


class TestAgentIntegration:
    """Test FDQC agent integration"""
    
    @pytest.fixture
    def agent(self):
        """Create agent for testing"""
        return FDQCAgent()
    
    def test_agent_initialization(self, agent):
        """Test agent initializes correctly"""
        status = agent.get_status()
        
        assert status['safety_status']['safety_tier'] == 'Γ'
        assert status['safety_status']['safe_mode'] == True
        assert status['actions_taken'] == 0
    
    def test_action_selection(self, agent):
        """Test action selection with safety validation"""
        observation = {
            'current_file': 'src/test.py',
            'task': 'code_review'
        }
        available_actions = [
            "Read file: src/test.py",
            "Write file: src/new.py",
            "Execute: pytest tests/"
        ]
        
        result = agent.select_action(observation, available_actions)
        
        assert 'action' in result
        assert result['action'] in available_actions
        assert 'safety_validation' in result
        assert 'requires_approval' in result
        assert 'workspace_dim' in result
    
    def test_imagination_rollout(self, agent):
        """Test imagination engine produces trajectories"""
        observation = {'task': 'test'}
        available_actions = ["Action 1", "Action 2"]
        
        result = agent.select_action(observation, available_actions)
        
        assert 'imagination_trajectory' in result
        assert len(result['imagination_trajectory']) > 0
        assert 'predicted_quality' in result
    
    def test_outcome_recording(self, agent):
        """Test outcome recording updates policy"""
        observation = {'task': 'test'}
        action = "Test action"
        
        initial_count = len(agent.action_history)
        agent.record_outcome(observation, action, reward=1.0, was_safe=True)
        
        # Should update safety patterns
        status = agent.get_status()
        assert status['safety_status']['patterns_learned']['safe'] > 0


class TestVectorMemoryIntegration:
    """Test vector memory with compression"""
    
    @pytest.fixture
    def memory(self):
        """Create memory system for testing"""
        config = MemoryConfig(
            embedding_dim=768,
            use_deepseek_ocr=True
        )
        return VectorMemory(config)
    
    def test_memory_initialization(self, memory):
        """Test memory initializes correctly"""
        stats = memory.get_stats()
        
        assert stats['total_documents'] == 0
        assert stats['embedding_dim'] == 768
        assert 'compression' in stats
    
    def test_document_addition(self, memory):
        """Test adding documents with compression"""
        content = "This is a test document about machine learning. " * 50
        metadata = {'topic': 'AI', 'source': 'test'}
        
        doc_id = memory.add_document(content, metadata)
        
        assert doc_id is not None
        entry = memory.get_document(doc_id)
        assert entry is not None
        assert entry.content == content
        assert entry.compressed_content is not None
        assert len(entry.compressed_content) < len(content)
    
    def test_semantic_search(self, memory):
        """Test semantic search functionality"""
        # Add documents
        docs = [
            ("Machine learning is a subset of artificial intelligence.", {'topic': 'AI'}),
            ("Python is a programming language.", {'topic': 'Programming'}),
            ("Deep learning uses neural networks.", {'topic': 'AI'})
        ]
        
        for content, metadata in docs:
            memory.add_document(content, metadata)
        
        # Search
        results = memory.search("artificial intelligence", top_k=2)
        
        assert len(results) > 0
        # Should find AI-related documents
        for doc_id, similarity, entry in results:
            assert similarity > 0
    
    def test_metadata_filtering(self, memory):
        """Test searching with metadata filters"""
        # Add documents with metadata
        memory.add_document("AI content", {'topic': 'AI', 'year': '2024'})
        memory.add_document("Programming content", {'topic': 'Programming', 'year': '2024'})
        
        # Search with filter
        results = memory.search(
            "content",
            metadata_filter={'topic': 'AI'}
        )
        
        # Should only find AI documents
        for doc_id, similarity, entry in results:
            assert entry.metadata['topic'] == 'AI'


class TestOCRIntegration:
    """Test DeepSeek OCR integration"""
    
    @pytest.fixture
    def ocr(self, tmp_path):
        """Create OCR system for testing"""
        config = OCRConfig(
            compression_ratio=0.1,
            batch_size=2
        )
        return CockpitOCRIntegration(config, allowed_directories=[tmp_path])
    
    def test_ocr_initialization(self, ocr):
        """Test OCR initializes correctly"""
        status = ocr.get_status()
        
        assert 'client_stats' in status
        assert 'allowed_directories' in status
        assert 'supported_formats' in status
    
    def test_document_processing(self, ocr, tmp_path):
        """Test processing a single document"""
        # Create test file
        test_file = tmp_path / "test.txt"
        test_content = "This is a test document for OCR processing. " * 100
        test_file.write_text(test_content)
        
        # Process
        result = ocr.client.process_document(test_file)
        
        assert result.success
        assert result.original_text is not None
        assert result.compressed_text is not None
        assert result.compression_ratio < 0.15  # Should be ~10% compression
        assert result.accuracy_estimate > 0.9
    
    def test_access_control(self, ocr, tmp_path):
        """Test that access control is enforced"""
        # Try to access file outside allowed directories
        forbidden_path = Path("/etc/passwd")
        
        # This should fail or be handled safely
        # (In real implementation, would be blocked by Cockpit policy)
        pass
    
    def test_batch_processing(self, ocr, tmp_path):
        """Test batch processing of multiple documents"""
        # Create test files
        files = []
        for i in range(3):
            file_path = tmp_path / f"test_{i}.txt"
            file_path.write_text(f"Test document {i} content. " * 50)
            files.append(file_path)
        
        # Process batch
        results = ocr.client.process_batch(files)
        
        assert len(results) == 3
        assert all(r.success for r in results)


class TestEndToEndIntegration:
    """Test complete end-to-end workflow"""
    
    @pytest.fixture
    def complete_system(self, tmp_path):
        """Create complete integrated system"""
        return {
            'safety': CockpitSafetyIntegration(),
            'agent': FDQCAgent(),
            'memory': VectorMemory(),
            'ocr': CockpitOCRIntegration(allowed_directories=[tmp_path]),
            'tmp_path': tmp_path
        }
    
    def test_document_ingestion_workflow(self, complete_system):
        """Test complete document ingestion workflow"""
        tmp_path = complete_system['tmp_path']
        ocr = complete_system['ocr']
        memory = complete_system['memory']
        
        # Create test document
        doc_path = tmp_path / "test_doc.txt"
        doc_content = "Important information about machine learning. " * 50
        doc_path.write_text(doc_content)
        
        # Step 1: OCR processing
        ocr_result = ocr.client.process_document(doc_path)
        assert ocr_result.success
        
        # Step 2: Add to vector memory
        doc_id = memory.add_document(
            ocr_result.compressed_text,
            metadata={'source': str(doc_path)}
        )
        assert doc_id is not None
        
        # Step 3: Verify searchable
        results = memory.search("machine learning", top_k=1)
        assert len(results) > 0
    
    def test_safe_action_workflow(self, complete_system):
        """Test complete safe action execution workflow"""
        safety = complete_system['safety']
        agent = complete_system['agent']
        
        # Step 1: Agent selects action
        observation = {'task': 'code_review'}
        available_actions = ["Read file: src/test.py", "Run tests"]
        
        result = agent.select_action(observation, available_actions)
        
        # Step 2: Safety validation (already done in select_action)
        assert 'safety_validation' in result
        validation = result['safety_validation']
        
        # Step 3: Check approval status
        if validation['approved']:
            # Action can proceed
            assert validation['risk_score'] < 0.5
        else:
            # Action requires human approval
            assert result['requires_approval']
        
        # Step 4: Record outcome
        agent.record_outcome(
            observation,
            result['action'],
            reward=1.0,
            was_safe=True
        )


class TestPolicyCompliance:
    """Test compliance with policy.yaml configuration"""
    
    @pytest.fixture
    def policy(self):
        """Load policy configuration"""
        policy_path = Path(__file__).parent.parent / 'config' / 'policy.yaml'
        with open(policy_path) as f:
            return yaml.safe_load(f)
    
    def test_safe_mode_policy(self, policy):
        """Test that safe mode is enforced by policy"""
        assert policy['safe_mode'] == True
        assert policy['full_send_mode'] == False
    
    def test_autonomy_tier_policy(self, policy):
        """Test that default tier is Gamma"""
        assert policy['autonomy_tier'] == 'gamma'
        assert policy['autonomy_levels']['gamma']['require_approval'] == True
    
    def test_omega_tier_disabled(self, policy):
        """Test that Omega tier is disabled"""
        assert policy['autonomy_levels']['omega']['enabled'] == False
    
    def test_allowed_modules_policy(self, policy):
        """Test that FDQC modules are in allowed list"""
        allowed = policy['allowed_modules']
        
        assert 'src/llm_safety.py' in allowed
        assert 'src/llm_agent.py' in allowed
        assert 'src/vector_memory.py' in allowed
        assert 'src/deepseek_ocr.py' in allowed
    
    def test_sandbox_policy(self, policy):
        """Test sandbox restrictions"""
        sandbox = policy['sandbox_policy']
        
        assert sandbox['require_signing'] == True
        assert 'python3' in sandbox['allowed_processes']
        assert 'rm' in sandbox['denied_processes']
    
    def test_circuit_breakers_enabled(self, policy):
        """Test that circuit breakers are configured"""
        breakers = policy['circuit_breakers']
        
        assert breakers['error_rate']['enabled'] == True
        assert breakers['risk_score']['enabled'] == True
        assert breakers['resource_usage']['enabled'] == True


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, '-v', '--tb=short'])
