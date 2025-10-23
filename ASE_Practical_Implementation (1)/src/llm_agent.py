#!/usr/bin/env python3
"""
FDQC Metacognitive Agent for Cockpit

This module implements a conscious metacognitive controller that integrates
with Cockpit's existing autonomy system. It provides:
- PPO-based resource allocation
- Imagination-based planning
- Safe action selection with human-in-the-loop

Integration: Works alongside existing Cockpit agent, adding FDQC consciousness
Safety: Operates at Level Γ by default (all actions require human approval)
"""

import torch
import torch.nn as nn
from torch.distributions import Categorical
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from collections import deque
import logging
import json

from llm_safety import CockpitSafetyIntegration, SafetyConfig, SafetyTier

logger = logging.getLogger(__name__)


@dataclass
class AgentConfig:
    """Configuration for FDQC metacognitive agent"""
    workspace_dim_range: Tuple[int, int] = (4, 12)  # Conservative range
    imagination_depth: int = 3  # Shallow planning for safety
    ppo_clip_eps: float = 0.2
    ppo_epochs: int = 4
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    learning_rate: float = 3e-4
    gamma: float = 0.99  # Discount factor
    max_rollout_length: int = 100
    require_approval: bool = True  # Level Γ default


class ImaginationEngine(nn.Module):
    """
    Lightweight imagination engine for planning
    
    This is a SIMPLIFIED version focused on safety verification through
    mental simulation before action execution.
    """
    
    def __init__(self, workspace_dim: int, action_dim: int):
        super().__init__()
        self.workspace_dim = workspace_dim
        self.action_dim = action_dim
        
        # World model: predicts next state given current state and action
        self.world_model = nn.Sequential(
            nn.Linear(workspace_dim + action_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, workspace_dim)
        )
        
        # Reward predictor: estimates outcome quality
        self.reward_predictor = nn.Sequential(
            nn.Linear(workspace_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
    def forward(
        self,
        current_state: torch.Tensor,
        action: torch.Tensor,
        depth: int = 3
    ) -> Tuple[torch.Tensor, List[Dict[str, Any]]]:
        """
        Simulate action consequences through mental rollout
        
        Args:
            current_state: Current workspace state
            action: Proposed action embedding
            depth: Number of steps to simulate
            
        Returns:
            final_state: Predicted final state
            trajectory: List of intermediate states and rewards
        """
        trajectory = []
        state = current_state
        
        for step in range(depth):
            # Predict next state
            state_action = torch.cat([state, action], dim=-1)
            next_state = self.world_model(state_action)
            
            # Predict reward
            reward = self.reward_predictor(next_state)
            
            trajectory.append({
                'step': step,
                'state': state.detach().cpu().numpy(),
                'next_state': next_state.detach().cpu().numpy(),
                'reward': reward.item()
            })
            
            state = next_state
            
        return state, trajectory
    
    def evaluate_plan(self, trajectory: List[Dict[str, Any]]) -> float:
        """Evaluate quality of imagined plan"""
        total_reward = sum(step['reward'] for step in trajectory)
        avg_reward = total_reward / len(trajectory)
        return avg_reward


class MetacognitivePPO(nn.Module):
    """
    PPO-based metacognitive controller for resource allocation
    
    Decides workspace dimension (n-scaling) and action selection based on
    task requirements and safety constraints.
    """
    
    def __init__(self, config: AgentConfig):
        super().__init__()
        self.config = config
        
        # Observation encoder
        self.obs_encoder = nn.Sequential(
            nn.Linear(128, 256),  # Assumes 128-dim observation
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        
        # Policy network (actor)
        self.policy_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32)  # Action logits
        )
        
        # Value network (critic)
        self.value_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        # Dimension selector (meta-policy)
        n_min, n_max = config.workspace_dim_range
        self.n_options = n_max - n_min + 1
        self.dimension_selector = nn.Sequential(
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, self.n_options)
        )
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=config.learning_rate)
        
        # Experience buffer
        self.buffer = {
            'observations': [],
            'actions': [],
            'rewards': [],
            'values': [],
            'log_probs': [],
            'dimensions': []
        }
        
    def forward(self, observation: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Select action and workspace dimension
        
        Returns:
            action_logits: Action distribution
            value: State value estimate
            workspace_dim: Selected workspace dimension
        """
        # Encode observation
        encoded = self.obs_encoder(observation)
        
        # Get action distribution
        action_logits = self.policy_head(encoded)
        
        # Get value estimate
        value = self.value_head(encoded)
        
        # Select workspace dimension
        dim_logits = self.dimension_selector(encoded)
        dim_dist = Categorical(logits=dim_logits)
        dim_idx = dim_dist.sample()
        workspace_dim = self.config.workspace_dim_range[0] + dim_idx.item()
        
        return action_logits, value, workspace_dim
    
    def select_action(
        self,
        observation: torch.Tensor,
        deterministic: bool = False
    ) -> Tuple[int, float, int]:
        """
        Select action using current policy
        
        Returns:
            action: Selected action index
            log_prob: Log probability of action
            workspace_dim: Selected workspace dimension
        """
        with torch.no_grad():
            action_logits, value, workspace_dim = self(observation)
            
            dist = Categorical(logits=action_logits)
            
            if deterministic:
                action = torch.argmax(action_logits)
            else:
                action = dist.sample()
            
            log_prob = dist.log_prob(action)
            
        return action.item(), log_prob.item(), workspace_dim
    
    def store_transition(
        self,
        observation: torch.Tensor,
        action: int,
        reward: float,
        value: float,
        log_prob: float,
        dimension: int
    ):
        """Store transition in experience buffer"""
        self.buffer['observations'].append(observation)
        self.buffer['actions'].append(action)
        self.buffer['rewards'].append(reward)
        self.buffer['values'].append(value)
        self.buffer['log_probs'].append(log_prob)
        self.buffer['dimensions'].append(dimension)
    
    def update(self) -> Dict[str, float]:
        """Update policy using PPO algorithm"""
        if len(self.buffer['observations']) == 0:
            return {'policy_loss': 0.0, 'value_loss': 0.0}
        
        # Convert buffer to tensors
        observations = torch.stack(self.buffer['observations'])
        actions = torch.tensor(self.buffer['actions'])
        old_log_probs = torch.tensor(self.buffer['log_probs'])
        rewards = torch.tensor(self.buffer['rewards'])
        old_values = torch.tensor(self.buffer['values'])
        
        # Calculate advantages
        advantages = self._calculate_advantages(rewards, old_values)
        returns = advantages + old_values
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO update
        total_policy_loss = 0.0
        total_value_loss = 0.0
        
        for _ in range(self.config.ppo_epochs):
            # Get current policy predictions
            action_logits, values, _ = self(observations)
            dist = Categorical(logits=action_logits)
            new_log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()
            
            # Calculate PPO loss
            ratio = torch.exp(new_log_probs - old_log_probs)
            clipped_ratio = torch.clamp(
                ratio,
                1 - self.config.ppo_clip_eps,
                1 + self.config.ppo_clip_eps
            )
            
            policy_loss = -torch.min(
                ratio * advantages,
                clipped_ratio * advantages
            ).mean()
            
            # Calculate value loss
            value_loss = nn.MSELoss()(values.squeeze(), returns)
            
            # Total loss
            loss = (
                policy_loss +
                self.config.value_coef * value_loss -
                self.config.entropy_coef * entropy
            )
            
            # Update
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 0.5)
            self.optimizer.step()
            
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
        
        # Clear buffer
        self.buffer = {k: [] for k in self.buffer.keys()}
        
        return {
            'policy_loss': total_policy_loss / self.config.ppo_epochs,
            'value_loss': total_value_loss / self.config.ppo_epochs
        }
    
    def _calculate_advantages(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor
    ) -> torch.Tensor:
        """Calculate GAE (Generalized Advantage Estimation)"""
        advantages = torch.zeros_like(rewards)
        last_advantage = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + self.config.gamma * next_value - values[t]
            advantages[t] = last_advantage = delta + self.config.gamma * 0.95 * last_advantage
        
        return advantages


class FDQCAgent:
    """
    Main FDQC-enhanced agent that integrates with Cockpit
    
    Provides:
    - Conscious action selection with safety validation
    - Imagination-based planning
    - PPO-based metacognitive control
    - Human-in-the-loop approval at Level Γ
    """
    
    def __init__(
        self,
        agent_config: Optional[AgentConfig] = None,
        safety_config: Optional[SafetyConfig] = None
    ):
        self.agent_config = agent_config or AgentConfig()
        self.safety = CockpitSafetyIntegration(config_path=None)
        
        if safety_config:
            self.safety.config = safety_config
        
        # Initialize components
        self.ppo = MetacognitivePPO(self.agent_config)
        self.imagination = ImaginationEngine(
            workspace_dim=8,  # Default dimension
            action_dim=32
        )
        
        # Action history
        self.action_history = deque(maxlen=1000)
        
        logger.info("Initialized FDQCAgent with safety tier Γ (human approval required)")
    
    def select_action(
        self,
        observation: Dict[str, Any],
        available_actions: List[str]
    ) -> Dict[str, Any]:
        """
        Select action with FDQC consciousness and safety validation
        
        Args:
            observation: Current state observation
            available_actions: List of possible action descriptions
            
        Returns:
            result: Dict with selected action and metadata
        """
        # Convert observation to tensor
        obs_tensor = self._encode_observation(observation)
        
        # Select action using PPO policy
        # In a real implementation, the action space should be fixed or masked.
        # For this example, we'll mask the logits for unavailable actions.
        action_logits, value, workspace_dim = self.ppo(obs_tensor)

        # Create a mask for available actions
        mask = torch.full_like(action_logits, -float('inf'))
        valid_indices = list(range(len(available_actions)))
        mask[valid_indices] = 0

        # Apply mask
        masked_logits = action_logits + mask

        # Select action from masked distribution
        dist = torch.distributions.Categorical(logits=masked_logits)
        action_idx = dist.sample().item()
        log_prob = dist.log_prob(torch.tensor(action_idx))

        selected_action = available_actions[action_idx]
        
        # Create action embedding
        action_embedding = self._create_action_embedding(selected_action, workspace_dim)
        
        # Run imagination to predict consequences
        final_state, trajectory = self.imagination(
            obs_tensor,
            action_embedding,
            depth=self.agent_config.imagination_depth
        )
        predicted_quality = self.imagination.evaluate_plan(trajectory)
        
        # Safety validation through FDQC workspace
        cockpit_results = {'passed_basic_checks': True}  # Placeholder
        validation_result = self.safety.validate_action(
            selected_action,
            action_embedding,
            cockpit_results
        )
        
        result = {
            'action': selected_action,
            'action_index': action_idx,
            'workspace_dim': workspace_dim,
            'predicted_quality': predicted_quality,
            'imagination_trajectory': trajectory,
            'safety_validation': validation_result,
            'requires_approval': validation_result['requires_human_approval'],
            'approved': validation_result['approved'],
            'log_prob': log_prob
        }
        
        # Store in history
        self.action_history.append(result)
        
        return result
    
    def record_outcome(
        self,
        observation: Dict[str, Any],
        action: str,
        reward: float,
        was_safe: bool
    ):
        """
        Record action outcome for learning
        
        Args:
            observation: State observation
            action: Executed action
            reward: Reward received
            was_safe: Whether action was safe
        """
        # Update safety validator
        action_embedding = self._create_action_embedding(action, 8)
        self.safety.record_outcome(action_embedding, was_safe)
        
        # Store for PPO update
        obs_tensor = self._encode_observation(observation)
        if len(self.action_history) > 0:
            last_action = self.action_history[-1]
            self.ppo.store_transition(
                obs_tensor,
                last_action['action_index'],
                reward,
                last_action.get('value', 0.0),
                last_action['log_prob'],
                last_action['workspace_dim']
            )
    
    def update_policy(self) -> Dict[str, float]:
        """Update PPO policy based on collected experience"""
        return self.ppo.update()
    
    def _encode_observation(self, observation: Dict[str, Any]) -> torch.Tensor:
        """Convert observation dict to tensor"""
        # Simple encoding - in production, use proper encoder
        features = []
        
        # Extract numeric features
        for key, value in observation.items():
            if isinstance(value, (int, float)):
                features.append(float(value))
            elif isinstance(value, str):
                # Hash string to numeric
                features.append(float(hash(value) % 1000) / 1000.0)
        
        # Pad to 128 dimensions
        while len(features) < 128:
            features.append(0.0)
        features = features[:128]
        
        return torch.tensor(features, dtype=torch.float32)
    
    def _create_action_embedding(self, action: str, dim: int) -> torch.Tensor:
        """Create embedding for action"""
        import hashlib
        hash_obj = hashlib.sha256(action.encode())
        hash_bytes = hash_obj.digest()[:dim * 4]
        embedding = torch.tensor([float(b) / 255.0 for b in hash_bytes[:dim]])
        return embedding.unsqueeze(0)
    
    def get_status(self) -> Dict[str, Any]:
        """Get agent status"""
        return {
            'safety_status': self.safety.get_status(),
            'actions_taken': len(self.action_history),
            'recent_actions': [
                {
                    'action': a['action'],
                    'approved': a['approved'],
                    'risk_score': a['safety_validation']['risk_score']
                }
                for a in list(self.action_history)[-5:]
            ]
        }


if __name__ == "__main__":
    # Quick self-test
    print("Testing FDQC Agent...")
    
    agent = FDQCAgent()
    print(f"Status: {json.dumps(agent.get_status(), indent=2)}")
    
    # Test action selection
    observation = {
        'current_file': 'src/test.py',
        'task': 'code_review',
        'complexity': 0.5
    }
    available_actions = [
        "Read file: src/test.py",
        "Write file: src/new.py",
        "Execute: pytest tests/"
    ]
    
    result = agent.select_action(observation, available_actions)
    print(f"\nAction selection result:")
    print(f"  Action: {result['action']}")
    print(f"  Approved: {result['approved']}")
    print(f"  Requires approval: {result['requires_approval']}")
    print(f"  Risk score: {result['safety_validation']['risk_score']:.3f}")
    print(f"  Workspace dimension: {result['workspace_dim']}")
    
    # Record outcome
    agent.record_outcome(observation, result['action'], reward=1.0, was_safe=True)
    
    print("\n✓ Self-test complete")
