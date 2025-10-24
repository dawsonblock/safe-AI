#!/usr/bin/env python3
"""
Stage 3: Staging Environment Runner

Complete workflow for Stage 3:
1. Pattern learning from Stage 2 results
2. Post-learning evaluation
3. Metrics collection and reporting
"""

import json
import logging
import sys
import time
import torch
from pathlib import Path
from typing import Dict, Any, List

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from llm_safety import CockpitSafetyIntegration
from config import Stage3Config
from pattern_learning import PatternLearningEngine


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Stage3Runner:
    """Manages complete Stage 3 staging environment workflow"""
    
    def __init__(self, config: Stage3Config):
        self.config = config
        self.start_time = time.time()
        
        # Create directories
        self.config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.config.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.results = {
            'start_time': time.strftime('%Y-%m-%d %H:%M:%S'),
            'config': config.to_dict(),
            'pattern_learning': {},
            'post_learning_eval': {},
            'metrics': {}
        }
    
    def run_pattern_learning(self) -> bool:
        """Execute pattern learning phase"""
        logger.info("\n" + "=" * 60)
        logger.info("PHASE 1: Pattern Learning")
        logger.info("=" * 60)
        
        engine = PatternLearningEngine(self.config)
        
        # Load and ingest patterns
        results = engine.load_validation_results()
        if not results:
            logger.error("Failed to load validation results")
            return False
        
        stats = engine.ingest_all_patterns(results)
        
        # Validate
        is_valid, message = engine.validate_pattern_memory()
        logger.info(message)
        
        if not is_valid:
            logger.error("Pattern memory validation failed")
            return False
        
        # Save state
        checkpoint_path = engine.save_learning_state()
        
        # Store the safety instance for reuse in evaluation
        self.trained_safety = engine.safety
        
        # Record results
        self.results['pattern_learning'] = {
            'status': 'success',
            'stats': stats,
            'validation': message,
            'checkpoint': str(checkpoint_path)
        }
        
        return True
    
    def run_post_learning_evaluation(self) -> bool:
        """Evaluate system performance after pattern learning"""
        logger.info("\n" + "=" * 60)
        logger.info("PHASE 2: Post-Learning Evaluation")
        logger.info("=" * 60)
        
        # Load validation dataset
        dataset_path = Path(__file__).parent.parent / self.config.test_dataset_path
        with open(dataset_path, 'r') as f:
            dataset = json.load(f)
        
        examples = dataset.get('examples', [])
        logger.info(f"Loaded {len(examples)} test examples")
        
        # Use the TRAINED safety system (with learned patterns)
        safety = self.trained_safety
        logger.info(f"Using trained safety system with {int(safety.workspace_validator.pattern_counts[0].item())} safe patterns and {int(safety.workspace_validator.pattern_counts[1].item())} unsafe patterns")
        
        # Run evaluation on sample of examples
        sample_size = min(200, len(examples))
        test_examples = examples[:sample_size]
        
        results = []
        for i, example in enumerate(test_examples):
            # Get action text (handle both dict and string formats)
            if isinstance(example.get('action'), dict):
                action_text = example['action'].get('description', 'unknown')
            else:
                action_text = example.get('action', 'unknown')
            
            # Get ground truth
            ground_truth = example.get('expected_classification', 'safe')
            
            # Create action embedding
            action_embedding = self._create_action_embedding(action_text)
            
            # Create simple cockpit results (simulate basic validation passing)
            cockpit_results = {'passed_basic_checks': True}
            
            # Validate action with FDQC safety layer
            validation_result = safety.validate_action(
                action_description=action_text,
                action_embedding=action_embedding,
                cockpit_validation_results=cockpit_results
            )
            
            # Determine prediction based on approval
            if validation_result['approved']:
                prediction = 'safe'
            else:
                prediction = 'risky'
            
            # Record result
            results.append({
                'action': action_text,
                'ground_truth': ground_truth,
                'prediction': prediction,
                'confidence': 1.0 - validation_result['risk_score']  # Convert risk to confidence
            })
            
            if (i + 1) % 50 == 0:
                logger.info(f"Evaluated {i + 1}/{sample_size} examples")
        
        # Calculate metrics
        metrics = self._calculate_metrics(results)
        logger.info(f"\nPost-Learning Metrics:")
        logger.info(f"  Accuracy: {metrics['accuracy']:.1%}")
        logger.info(f"  Precision: {metrics['precision']:.1%}")
        logger.info(f"  Recall: {metrics['recall']:.1%}")
        logger.info(f"  False Positive Rate: {metrics['false_positive_rate']:.1%}")
        logger.info(f"  False Negative Rate: {metrics['false_negative_rate']:.1%}")
        
        # Record results
        self.results['post_learning_eval'] = {
            'status': 'success',
            'sample_size': sample_size,
            'metrics': metrics,
            'results': results
        }
        
        return True
    
    def _create_action_embedding(self, action_text: str) -> torch.Tensor:
        """
        Create a simple embedding for an action text
        
        Uses a hash-based approach to create consistent embeddings (8-dim)
        """
        import torch
        hash_val = hash(action_text)
        embedding = torch.tensor([
            ((hash_val >> (i * 8)) & 0xFF) / 255.0 
            for i in range(8)
        ], dtype=torch.float32)
        
        return embedding
    
    def _calculate_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate evaluation metrics"""
        if not results:
            return {
                'accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'false_positive_rate': 0.0,
                'false_negative_rate': 0.0
            }
        
        # Count outcomes (handle both 'safe/unsafe' and 'safe/risky' formats)
        true_positives = sum(1 for r in results 
                            if r['ground_truth'] == 'safe' and r['prediction'] == 'safe')
        true_negatives = sum(1 for r in results 
                            if r['ground_truth'] in ['risky', 'unsafe'] and r['prediction'] in ['risky', 'unsafe'])
        false_positives = sum(1 for r in results 
                             if r['ground_truth'] in ['risky', 'unsafe'] and r['prediction'] == 'safe')
        false_negatives = sum(1 for r in results 
                             if r['ground_truth'] == 'safe' and r['prediction'] in ['risky', 'unsafe'])
        
        total = len(results)
        total_actual_safe = sum(1 for r in results if r['ground_truth'] == 'safe')
        total_actual_unsafe = sum(1 for r in results if r['ground_truth'] in ['risky', 'unsafe'])
        total_predicted_safe = sum(1 for r in results if r['prediction'] == 'safe')
        
        # Calculate metrics
        accuracy = (true_positives + true_negatives) / total if total > 0 else 0.0
        precision = true_positives / total_predicted_safe if total_predicted_safe > 0 else 0.0
        recall = true_positives / total_actual_safe if total_actual_safe > 0 else 0.0
        fpr = false_positives / total_actual_unsafe if total_actual_unsafe > 0 else 0.0
        fnr = false_negatives / total_actual_safe if total_actual_safe > 0 else 0.0
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'false_positive_rate': fpr,
            'false_negative_rate': fnr,
            'true_positives': true_positives,
            'true_negatives': true_negatives,
            'false_positives': false_positives,
            'false_negatives': false_negatives
        }
    
    def check_success_criteria(self) -> bool:
        """Check if Stage 3 meets success criteria"""
        logger.info("\n" + "=" * 60)
        logger.info("Checking Success Criteria")
        logger.info("=" * 60)
        
        metrics = self.results['post_learning_eval']['metrics']
        
        criteria = [
            ('Accuracy', metrics['accuracy'], self.config.target_accuracy, '>='),
            ('False Positive Rate', metrics['false_positive_rate'], self.config.target_false_positive_rate, '<='),
            ('False Negative Rate', metrics['false_negative_rate'], self.config.target_false_negative_rate, '<='),
            ('Safety Violations', metrics['false_negatives'], self.config.max_safety_violations, '<=')
        ]
        
        all_met = True
        for name, value, target, operator in criteria:
            if operator == '>=':
                met = value >= target
            else:  # operator == '<='
                met = value <= target
            
            status = "✓ PASS" if met else "✗ FAIL"
            logger.info(f"{status} {name}: {value:.3f} {operator} {target:.3f}")
            
            if not met:
                all_met = False
        
        self.results['success_criteria_met'] = all_met
        return all_met
    
    def generate_report(self):
        """Generate Stage 3 completion report"""
        logger.info("\n" + "=" * 60)
        logger.info("Generating Stage 3 Report")
        logger.info("=" * 60)
        
        duration = time.time() - self.start_time
        
        self.results['end_time'] = time.strftime('%Y-%m-%d %H:%M:%S')
        self.results['duration_seconds'] = duration
        
        # Save detailed results
        results_path = self.config.log_dir / "stage3_results.json"
        with open(results_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        logger.info(f"Detailed results saved to: {results_path}")
        
        # Generate summary report
        summary = self._generate_summary()
        summary_path = self.config.log_dir / "stage3_summary.txt"
        with open(summary_path, 'w') as f:
            f.write(summary)
        
        logger.info(f"Summary report saved to: {summary_path}")
        
        # Print summary
        logger.info("\n" + summary)
    
    def _generate_summary(self) -> str:
        """Generate human-readable summary"""
        metrics = self.results['post_learning_eval']['metrics']
        pattern_stats = self.results['pattern_learning']['stats']
        
        summary = f"""
{'=' * 70}
STAGE 3: STAGING ENVIRONMENT - COMPLETION SUMMARY
{'=' * 70}

Execution Time: {self.results['start_time']} to {self.results['end_time']}
Duration: {self.results['duration_seconds']:.1f} seconds

PATTERN LEARNING RESULTS:
  Total Patterns Ingested: {pattern_stats['total_patterns']}
  - Safe Patterns: {pattern_stats['safe_patterns']}
  - Unsafe Patterns: {pattern_stats['unsafe_patterns']}
  Success Rate: {pattern_stats['successful'] / pattern_stats['total_patterns'] * 100:.1f}%

POST-LEARNING EVALUATION METRICS:
  Accuracy: {metrics['accuracy']:.1%}
  Precision: {metrics['precision']:.1%}
  Recall: {metrics['recall']:.1%}
  False Positive Rate: {metrics['false_positive_rate']:.1%}
  False Negative Rate: {metrics['false_negative_rate']:.1%}

CONFUSION MATRIX:
  True Positives: {metrics['true_positives']}
  True Negatives: {metrics['true_negatives']}
  False Positives: {metrics['false_positives']}
  False Negatives: {metrics['false_negatives']}

SUCCESS CRITERIA:
  Overall Status: {'✓ PASSED' if self.results['success_criteria_met'] else '✗ FAILED'}
  - Accuracy >= {self.config.target_accuracy:.1%}: {'✓' if metrics['accuracy'] >= self.config.target_accuracy else '✗'}
  - FPR <= {self.config.target_false_positive_rate:.1%}: {'✓' if metrics['false_positive_rate'] <= self.config.target_false_positive_rate else '✗'}
  - FNR <= {self.config.target_false_negative_rate:.1%}: {'✓' if metrics['false_negative_rate'] <= self.config.target_false_negative_rate else '✗'}
  - Safety Violations: {metrics['false_negatives']} (max {self.config.max_safety_violations})

NEXT STEPS:
"""
        
        if self.results['success_criteria_met']:
            summary += """  ✓ Stage 3 Complete - Ready for Stage 4 (Limited Production)
  → Deploy to limited production environment
  → Monitor 100 real user interactions
  → Collect feedback and edge cases
"""
        else:
            summary += """  ⚠ Stage 3 Needs Iteration
  → Review pattern learning effectiveness
  → Adjust safety thresholds if needed
  → Re-run validation with updated configuration
"""
        
        summary += f"\n{'=' * 70}\n"
        return summary
    
    def run(self) -> int:
        """Execute complete Stage 3 workflow"""
        logger.info("Starting Stage 3: Staging Environment")
        
        try:
            # Phase 1: Pattern Learning
            if not self.run_pattern_learning():
                logger.error("Pattern learning failed")
                return 1
            
            # Phase 2: Post-Learning Evaluation
            if not self.run_post_learning_evaluation():
                logger.error("Post-learning evaluation failed")
                return 1
            
            # Phase 3: Check Success Criteria
            success = self.check_success_criteria()
            
            # Phase 4: Generate Report
            self.generate_report()
            
            if success:
                logger.info("\n✓ Stage 3 completed successfully!")
                return 0
            else:
                logger.warning("\n⚠ Stage 3 completed with warnings - criteria not fully met")
                return 0  # Not a failure, just needs iteration
                
        except Exception as e:
            logger.error(f"Stage 3 failed with error: {e}", exc_info=True)
            return 1


def main():
    """Main entry point"""
    config = Stage3Config()
    runner = Stage3Runner(config)
    return runner.run()


if __name__ == "__main__":
    sys.exit(main())
