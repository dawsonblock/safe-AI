#!/usr/bin/env python3
"""
FDQC-Cockpit Production GUI

Modern web-based interface for monitoring and controlling the FDQC-Cockpit system.
Built with Flask for backend and real-time updates via Server-Sent Events.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from flask import Flask, render_template, jsonify, request, Response
from flask_cors import CORS
import json
import time
import threading
from datetime import datetime
from queue import Queue
import logging

from config_loader import get_config, validate_config
from bert_integration import BERTEmbedder, BERTConfig
from production_hardening import (
    MonitoringSystem,
    AdversarialDetector,
    PatternMemoryPersistence
)
from llm_agent import FDQCAgent, AgentConfig
from llm_safety import CockpitSafetyIntegration, SafetyConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Global state
system_state = {
    'initialized': False,
    'config': None,
    'agent': None,
    'embedder': None,
    'monitor': None,
    'adversarial_detector': None,
    'persistence': None,
    'safety': None,
    'status': 'initializing',
    'error': None
}

# Event queue for real-time updates
event_queue = Queue()


def initialize_system():
    """Initialize all system components"""
    try:
        logger.info("Initializing FDQC-Cockpit system...")
        
        # Load configuration
        config = get_config()
        validate_config(config)
        system_state['config'] = config
        
        # Initialize BERT embedder
        bert_config = BERTConfig(
            model_name=config.bert_model_name,
            cache_dir=config.bert_cache_dir,
            use_gpu=config.bert_use_gpu,
            batch_size=config.bert_batch_size,
            normalize_embeddings=True
        )
        embedder = BERTEmbedder(bert_config)
        system_state['embedder'] = embedder
        
        # Initialize monitoring
        monitor = MonitoringSystem(alert_threshold={
            'risk_score': config.alert_risk_threshold,
            'error_rate': config.alert_error_rate_threshold,
            'memory_utilization': config.alert_memory_utilization_threshold
        })
        system_state['monitor'] = monitor
        
        # Initialize adversarial detector
        adversarial_detector = AdversarialDetector(sensitivity=0.95)
        system_state['adversarial_detector'] = adversarial_detector
        
        # Initialize persistence
        persistence = PatternMemoryPersistence()
        system_state['persistence'] = persistence
        
        # Initialize safety system
        safety_config = SafetyConfig(
            workspace_dim=config.workspace_dim,
            entropy_threshold=config.entropy_threshold,
            collapse_threshold=config.collapse_threshold,
            require_human_approval=config.require_human_approval,
            safe_mode=config.safe_mode
        )
        safety = CockpitSafetyIntegration()
        safety.config = safety_config
        system_state['safety'] = safety
        
        # Initialize agent
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
        system_state['agent'] = agent
        
        system_state['initialized'] = True
        system_state['status'] = 'ready'
        
        logger.info("✓ System initialized successfully")
        send_event('system_status', {'status': 'ready', 'message': 'System ready'})
        
    except Exception as e:
        logger.error(f"System initialization failed: {e}")
        system_state['status'] = 'error'
        system_state['error'] = str(e)
        send_event('system_status', {'status': 'error', 'message': str(e)})


def send_event(event_type: str, data: dict):
    """Send event to all connected clients"""
    event = {
        'type': event_type,
        'data': data,
        'timestamp': time.time()
    }
    event_queue.put(event)


# Routes

@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('dashboard.html')


@app.route('/api/status')
def get_status():
    """Get system status"""
    if not system_state['initialized']:
        return jsonify({
            'status': system_state['status'],
            'error': system_state['error']
        })
    
    config = system_state['config']
    monitor = system_state['monitor']
    embedder = system_state['embedder']
    safety = system_state['safety']
    
    return jsonify({
        'status': system_state['status'],
        'config': {
            'safety_tier': config.safety_tier,
            'safe_mode': config.safe_mode,
            'bert_model': config.bert_model_name,
            'require_approval': config.require_human_approval
        },
        'monitoring': monitor.get_metrics(),
        'bert': embedder.get_stats(),
        'safety': safety.get_status()
    })


@app.route('/api/action/select', methods=['POST'])
def select_action():
    """Select action with safety validation"""
    if not system_state['initialized']:
        return jsonify({'error': 'System not initialized'}), 503
    
    data = request.json
    observation = data.get('observation', {})
    available_actions = data.get('available_actions', [])
    
    if not available_actions:
        return jsonify({'error': 'No actions provided'}), 400
    
    try:
        agent = system_state['agent']
        monitor = system_state['monitor']
        adversarial_detector = system_state['adversarial_detector']
        
        # Select action
        result = agent.select_action(observation, available_actions)
        
        # Record in monitoring
        monitor.record_validation(
            approved=result['approved'],
            risk_score=result['safety_validation']['risk_score'],
            requires_approval=result['requires_approval']
        )
        
        # Check for adversarial patterns
        action_embedding = result['safety_validation']['fdqc_results']
        is_outlier, anomaly_score, reason = adversarial_detector.detect_outlier(
            agent._create_action_embedding(result['action'], 8),
            is_safe=result['approved']
        )
        
        result['adversarial_detection'] = {
            'is_outlier': is_outlier,
            'anomaly_score': anomaly_score,
            'reason': reason
        }
        
        # Send real-time update
        send_event('action_selected', {
            'action': result['action'],
            'risk_score': result['safety_validation']['risk_score'],
            'approved': result['approved'],
            'workspace_dim': result['workspace_dim']
        })
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Action selection error: {e}")
        monitor.record_error('action_selection', str(e))
        return jsonify({'error': str(e)}), 500


@app.route('/api/action/record', methods=['POST'])
def record_outcome():
    """Record action outcome for learning"""
    if not system_state['initialized']:
        return jsonify({'error': 'System not initialized'}), 503
    
    data = request.json
    observation = data.get('observation', {})
    action = data.get('action', '')
    reward = data.get('reward', 0.0)
    was_safe = data.get('was_safe', True)
    
    try:
        agent = system_state['agent']
        agent.record_outcome(observation, action, reward, was_safe)
        
        send_event('outcome_recorded', {
            'action': action,
            'reward': reward,
            'was_safe': was_safe
        })
        
        return jsonify({'success': True})
        
    except Exception as e:
        logger.error(f"Outcome recording error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/monitoring/metrics')
def get_metrics():
    """Get detailed monitoring metrics"""
    if not system_state['initialized']:
        return jsonify({'error': 'System not initialized'}), 503
    
    monitor = system_state['monitor']
    return jsonify(monitor.get_metrics())


@app.route('/api/monitoring/alerts')
def get_alerts():
    """Get recent alerts"""
    if not system_state['initialized']:
        return jsonify({'error': 'System not initialized'}), 503
    
    severity = request.args.get('severity')
    limit = int(request.args.get('limit', 50))
    
    monitor = system_state['monitor']
    alerts = monitor.get_recent_alerts(severity=severity, limit=limit)
    
    return jsonify([{
        'severity': a.severity,
        'component': a.component,
        'message': a.message,
        'timestamp': a.timestamp,
        'metadata': a.metadata
    } for a in alerts])


@app.route('/api/persistence/save', methods=['POST'])
def save_checkpoint():
    """Save system checkpoint"""
    if not system_state['initialized']:
        return jsonify({'error': 'System not initialized'}), 503
    
    try:
        persistence = system_state['persistence']
        safety = system_state['safety']
        
        checkpoint_path = persistence.save_patterns(
            safety.workspace_validator.safe_patterns,
            safety.workspace_validator.unsafe_patterns,
            safety.workspace_validator.pattern_counts,
            safety.workspace_validator.safe_pattern_timestamps,
            safety.workspace_validator.unsafe_pattern_timestamps,
            safety.workspace_validator.safe_pattern_importance,
            safety.workspace_validator.unsafe_pattern_importance
        )
        
        send_event('checkpoint_saved', {
            'path': str(checkpoint_path),
            'timestamp': time.time()
        })
        
        return jsonify({
            'success': True,
            'checkpoint_path': str(checkpoint_path)
        })
        
    except Exception as e:
        logger.error(f"Checkpoint save error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/events')
def event_stream():
    """Server-Sent Events stream for real-time updates"""
    def generate():
        while True:
            try:
                event = event_queue.get(timeout=30)
                yield f"data: {json.dumps(event)}\n\n"
            except:
                # Send keepalive
                yield f"data: {json.dumps({'type': 'keepalive'})}\n\n"
    
    return Response(generate(), mimetype='text/event-stream')


@app.route('/api/config', methods=['GET', 'POST'])
def manage_config():
    """Get or update configuration"""
    if request.method == 'GET':
        if not system_state['initialized']:
            return jsonify({'error': 'System not initialized'}), 503
        
        config = system_state['config']
        return jsonify({
            'safety_tier': config.safety_tier,
            'safe_mode': config.safe_mode,
            'require_human_approval': config.require_human_approval,
            'workspace_dim': config.workspace_dim,
            'entropy_threshold': config.entropy_threshold,
            'collapse_threshold': config.collapse_threshold,
            'bert_model_name': config.bert_model_name,
            'bert_use_gpu': config.bert_use_gpu
        })
    
    elif request.method == 'POST':
        # Update configuration (requires restart)
        return jsonify({
            'message': 'Configuration update requires system restart',
            'success': False
        }), 501


if __name__ == '__main__':
    # Initialize system in background thread
    init_thread = threading.Thread(target=initialize_system, daemon=True)
    init_thread.start()
    
    # Start Flask server
    logger.info("Starting FDQC-Cockpit GUI on http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
