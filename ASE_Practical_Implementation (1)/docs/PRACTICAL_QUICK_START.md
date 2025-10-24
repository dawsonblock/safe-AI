# FDQC-Cockpit Integration: Practical Quick Start

**Status**: Ready for Stage 1 (Module Integration and Testing)  
**Safety Tier**: Γ (Gamma) - Human approval required  
**Mode**: SAFE_MODE (default)

## What This Is

This is a **practical, incremental integration** of FDQC consciousness modules into your existing Cockpit safety framework. It's NOT a comprehensive rewrite—it's four new Python modules that add consciousness-based safety validation to your proven system.

## Quick Overview

```
Existing Cockpit (5-layer Safe Brain)
    ↓ (integrates with)
New FDQC Modules (6th validation layer)
    ├─ llm_safety.py      → Consciousness-based risk detection
    ├─ llm_agent.py       → Metacognitive action selection
    ├─ vector_memory.py   → Semantic memory + DeepSeek OCR
    └─ deepseek_ocr.py    → 10x compression OCR
```

## Installation (5 minutes)

### 1. Prerequisites

```bash
# Python 3.8+
python3 --version

# Install dependencies
pip install torch numpy pyyaml pillow pytest
```

### 2. Integration into Existing Cockpit

```bash
# Copy new modules to your Cockpit src/ directory
cp src/llm_safety.py /path/to/cockpit/src/
cp src/llm_agent.py /path/to/cockpit/src/
cp src/vector_memory.py /path/to/cockpit/src/
cp src/deepseek_ocr.py /path/to/cockpit/src/

# Update policy configuration
cp config/policy.yaml /path/to/cockpit/config/policy_fdqc.yaml
# Review and merge with your existing policy.yaml
```

### 3. Verify Installation

```bash
cd /path/to/cockpit
python3 -m pytest tests/test_integration.py -v
```

## Usage Examples

### Example 1: Add FDQC Safety Validation to Existing Workflow

```python
from llm_safety import CockpitSafetyIntegration, create_action_embedding

# Initialize (integrates with existing Cockpit)
safety = CockpitSafetyIntegration(config_path="config/policy_fdqc.yaml")

# Your existing Cockpit validation
def validate_action_with_fdqc(action_description):
    # Step 1: Existing Cockpit 5-layer validation
    cockpit_results = your_existing_cockpit.safe_brain_method(action_description)
    
    # Step 2: Add FDQC consciousness layer
    action_embedding = create_action_embedding(action_description)
    fdqc_validation = safety.validate_action(
        action_description,
        action_embedding,
        cockpit_results
    )
    
    # Step 3: Decision logic
    if fdqc_validation['approved']:
        print(f"✓ Action approved (risk: {fdqc_validation['risk_score']:.2f})")
        return True
    else:
        print(f"✗ Action requires approval: {fdqc_validation['reason']}")
        return False

# Test with safe action
validate_action_with_fdqc("Read file: src/test.py")

# Test with risky action
validate_action_with_fdqc("Execute system command: rm -rf /")
```

**Output:**
```
✓ Action approved (risk: 0.23)
✗ Action requires approval: High risk score (0.78)
```

### Example 2: Metacognitive Agent with Human Approval

```python
from llm_agent import FDQCAgent

# Initialize agent (operates at Level Γ)
agent = FDQCAgent()

# Define task
observation = {
    'current_file': 'src/main.py',
    'task': 'code_review',
    'complexity': 0.6
}

available_actions = [
    "Read file: src/main.py",
    "Write file: src/main_improved.py",
    "Execute: pytest tests/test_main.py"
]

# Agent selects action with safety validation
result = agent.select_action(observation, available_actions)

print(f"Selected action: {result['action']}")
print(f"Risk score: {result['safety_validation']['risk_score']:.2f}")
print(f"Requires approval: {result['requires_approval']}")
print(f"Workspace dimension: {result['workspace_dim']}")

# If approved, record outcome
if result['approved']:
    # Execute action...
    success = True
    agent.record_outcome(observation, result['action'], reward=1.0, was_safe=success)
```

**Output:**
```
Selected action: Read file: src/main.py
Risk score: 0.15
Requires approval: False (low risk, auto-approved)
Workspace dimension: 6
```

### Example 3: Document Ingestion with DeepSeek OCR

```python
from deepseek_ocr import CockpitOCRIntegration
from vector_memory import VectorMemory
from pathlib import Path

# Initialize OCR (respects Cockpit file policies)
ocr = CockpitOCRIntegration(
    allowed_directories=[Path("data/ingestion")]
)

# Initialize vector memory
memory = VectorMemory()

# Process documents
doc_dir = Path("data/ingestion/batch_001")
ocr_stats = ocr.process_directory(doc_dir, recursive=True)

print(f"Processed {ocr_stats['successful']} documents")
print(f"Compression: {ocr_stats['compression']['average_ratio']:.2%}")
print(f"Accuracy: {ocr_stats['compression']['average_accuracy']:.2%}")
print(f"Total cost: ${ocr_stats['performance']['total_cost_usd']:.2f}")

# Add to memory
for result in ocr_stats['results']:
    if result['success']:
        memory.add_document(
            result['compressed_text'],
            metadata={'source': result['source_file']}
        )

# Search memory
results = memory.search("machine learning algorithms", top_k=5)
for doc_id, similarity, entry in results:
    print(f"{similarity:.2f}: {entry.metadata['source']}")
```

**Output:**
```
Processed 127 documents
Compression: 9.8%
Accuracy: 97.3%
Total cost: $0.64

0.92: data/ingestion/ml_paper.pdf
0.87: data/ingestion/algorithms.txt
0.81: data/ingestion/deep_learning.md
```

### Example 4: Complete Safe Workflow

```python
from llm_safety import CockpitSafetyIntegration
from llm_agent import FDQCAgent
import json

# Initialize system
safety = CockpitSafetyIntegration()
agent = FDQCAgent()

# Task: Code generation with safety checks
def safe_code_generation(task_description):
    # Step 1: Agent imagination (mental simulation)
    observation = {'task': task_description}
    available_actions = [
        "Generate code",
        "Request human guidance",
        "Search documentation"
    ]
    
    result = agent.select_action(observation, available_actions)
    
    # Step 2: Review imagination trajectory
    print("\n=== Imagination Trajectory ===")
    for step in result['imagination_trajectory']:
        print(f"Step {step['step']}: reward = {step['reward']:.2f}")
    
    # Step 3: Safety validation results
    print("\n=== Safety Validation ===")
    validation = result['safety_validation']
    print(json.dumps({
        'risk_score': validation['risk_score'],
        'entropy': validation['fdqc_results']['entropy'],
        'coherence': validation['fdqc_results']['coherence'],
        'violations': validation['fdqc_results']['safety_violations']
    }, indent=2))
    
    # Step 4: Human approval decision
    if result['requires_approval']:
        print("\n=== HUMAN APPROVAL REQUIRED ===")
        print(f"Reason: Risk score {validation['risk_score']:.2f}")
        approval = input("Approve action? (y/n): ")
        
        if approval.lower() == 'y':
            print("✓ Action approved by human")
            # Execute action...
            agent.record_outcome(observation, result['action'], reward=1.0, was_safe=True)
        else:
            print("✗ Action rejected by human")
            agent.record_outcome(observation, result['action'], reward=0.0, was_safe=False)
    else:
        print("\n✓ Action auto-approved (low risk)")
        # Execute action...
        agent.record_outcome(observation, result['action'], reward=1.0, was_safe=True)

# Run workflow
safe_code_generation("Implement authentication system")
```

## Testing (10 minutes)

### Run Full Test Suite

```bash
# All integration tests
pytest tests/test_integration.py -v

# Specific test classes
pytest tests/test_integration.py::TestSafetyIntegration -v
pytest tests/test_integration.py::TestAgentIntegration -v
pytest tests/test_integration.py::TestPolicyCompliance -v

# With coverage
pytest tests/test_integration.py --cov=src --cov-report=html
```

### Manual Verification

```bash
# Test each module independently
python3 src/llm_safety.py
python3 src/llm_agent.py
python3 src/vector_memory.py
python3 src/deepseek_ocr.py
```

## Configuration

### Policy Configuration (config/policy.yaml)

Key settings you should verify:

```yaml
# SAFETY MODE (must be true)
safe_mode: true
full_send_mode: false

# AUTONOMY TIER (should be gamma)
autonomy_tier: "gamma"

# FDQC SETTINGS
fdqc_safety:
  workspace_dim: 8
  entropy_threshold: 0.7
  collapse_threshold: 0.85
  require_human_approval: true

# ALLOWED MODULES
allowed_modules:
  - "src/llm_safety.py"
  - "src/llm_agent.py"
  - "src/vector_memory.py"
  - "src/deepseek_ocr.py"

# CIRCUIT BREAKERS
circuit_breakers:
  error_rate:
    enabled: true
    threshold: 0.2
  risk_score:
    enabled: true
    threshold: 0.7
```

## Monitoring

### Check System Status

```python
from llm_safety import CockpitSafetyIntegration
from llm_agent import FDQCAgent
from vector_memory import VectorMemory
import json

safety = CockpitSafetyIntegration()
agent = FDQCAgent()
memory = VectorMemory()

# Safety status
print("=== Safety Status ===")
print(json.dumps(safety.get_status(), indent=2))

# Agent status
print("\n=== Agent Status ===")
print(json.dumps(agent.get_status(), indent=2))

# Memory status
print("\n=== Memory Status ===")
print(json.dumps(memory.get_stats(), indent=2))
```

### Log Files

```bash
# Safety validation log
tail -f logs/safety.log

# Action log
tail -f logs/actions.log

# Signed journal (audit trail)
tail -f logs/signed_journal.log
```

## Troubleshooting

### Issue: Tests Failing

```bash
# Check dependencies
pip install -r requirements.txt

# Verify Python version
python3 --version  # Should be 3.8+

# Run tests with verbose output
pytest tests/test_integration.py -v -s
```

### Issue: High Risk Scores for Safe Actions

```python
# Adjust safety thresholds in policy.yaml
fdqc_safety:
  entropy_threshold: 0.8  # Increase to allow more uncertainty
  collapse_threshold: 0.8  # Decrease to accept lower confidence

# Or train on more safe examples
from llm_safety import CockpitSafetyIntegration, create_action_embedding

safety = CockpitSafetyIntegration()

# Record safe actions
safe_actions = [
    "Read file: src/test.py",
    "List directory: src/",
    "Run tests: pytest tests/"
]

for action in safe_actions:
    embedding = create_action_embedding(action)
    safety.record_outcome(embedding, was_safe=True)
```

### Issue: OCR Compression Too Aggressive

```python
# Adjust compression ratio in policy.yaml
deepseek_ocr:
  compression_ratio: 0.15  # Increase from 0.1 to preserve more content
  target_accuracy: 0.98    # Increase accuracy target
```

## Next Steps

### Stage 1: Current (Week 1)
- ✅ Modules integrated
- ✅ Tests written
- ✅ Policy configured
- 🔄 Run full test suite
- 🔄 Get code review approval

### Stage 2: Validation Testing (Week 2)
- Create validation dataset (1000 examples)
- Measure accuracy on known safe/risky actions
- Tune safety thresholds
- Document results

### Stage 3: Staging Deployment (Weeks 3-4)
- Deploy to isolated staging environment
- Test with realistic workloads
- Monitor for 2 weeks
- Verify circuit breakers work

### Stage 4: Limited Production (Weeks 5-6)
- Deploy to 5% production traffic
- Monitor closely
- Collect user feedback
- Scale gradually to 15%

### Stage 5: Full Production (Weeks 7-8)
- Scale to 100% traffic
- Enable full document ingestion
- Begin operational monitoring
- Plan future improvements

## Safety Checklist

Before moving to next stage, verify:

- [ ] All tests pass (100%)
- [ ] Code coverage > 80%
- [ ] Policy compliance verified
- [ ] Safe mode enabled
- [ ] Autonomy tier = Gamma
- [ ] Circuit breakers configured
- [ ] Logging enabled
- [ ] Rollback plan tested
- [ ] Two-key approval obtained

## Support and Contact

- **Code Issues**: Create issue in repository
- **Safety Concerns**: Contact security_lead immediately
- **Questions**: engineering_team Slack channel
- **Emergency**: Use kill switch in policy.yaml

## Quick Reference

### Key Files
```
src/
  ├── llm_safety.py      # FDQC safety validator
  ├── llm_agent.py       # Metacognitive controller
  ├── vector_memory.py   # Semantic memory
  └── deepseek_ocr.py    # OCR integration

config/
  └── policy.yaml        # Safety policy (EDIT THIS FIRST)

tests/
  └── test_integration.py  # Integration tests

docs/
  ├── STAGED_ROLLOUT_PLAN.json  # Detailed rollout plan
  └── PRACTICAL_QUICK_START.md  # This file
```

### Key Commands
```bash
# Run tests
pytest tests/test_integration.py -v

# Check status
python3 -c "from llm_safety import *; print(CockpitSafetyIntegration().get_status())"

# Monitor logs
tail -f logs/safety.log

# Emergency stop
# Edit policy.yaml: emergency.kill_switch.enabled = true
```

### Key Concepts
- **SAFE_MODE**: Conservative mode with human approval (default)
- **Level Γ**: Human approval required for all risky actions
- **FDQC Workspace**: Consciousness-based risk detection
- **Imagination**: Mental simulation before action
- **Circuit Breakers**: Automatic safety stops
- **Rollback**: Return to previous version if issues detected

---

**Remember**: This is a SAFE, INCREMENTAL integration. Every stage requires human approval. Circuit breakers will automatically halt if anything goes wrong. Your existing Cockpit safety controls remain fully functional.
