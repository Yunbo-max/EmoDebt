"""
Reinforcement Learning Agents for Emotional Debt Negotiation
Includes DQN, Q-Learning, and Policy Gradient approaches
"""

import numpy as np
import random
from typing import Dict, List, Any, Optional, Tuple
from models.base_model import BaseEmotionModel
import json
from datetime import datetime
import os
from utils.statistical_analysis import enhance_results_with_statistics, analyze_negotiation_results, format_ci_results
from llm.negotiator import DebtNegotiator

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("⚠️ PyTorch not available - using basic Q-learning only")

class DQNAgent(nn.Module):
    """Deep Q-Network for emotion selection"""
    
    def __init__(self, state_size: int, action_size: int, hidden_size: int = 128):
        super(DQNAgent, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class DeepQEmotionModel(BaseEmotionModel):
    """DQN-based emotion model for debt negotiation"""
    
    def __init__(self, exploration_rate: float = 0.3, learning_rate: float = 0.001):
        super().__init__()
        
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for DQN agent")
        
        # Emotion mapping
        self.emotions = ['happy', 'surprising', 'angry', 'sad', 'disgust', 'fear', 'neutral']
        self.emotion_to_idx = {e: i for i, e in enumerate(self.emotions)}
        
        # State representation: [round, debtor_emotion, gap_size, debt_amount_normalized, success_history]
        self.state_size = 5
        self.action_size = len(self.emotions)
        
        # DQN Networks
        self.q_network = DQNAgent(self.state_size, self.action_size)
        self.target_network = DQNAgent(self.state_size, self.action_size)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Training parameters
        self.exploration_rate = exploration_rate
        self.min_exploration = 0.1
        self.exploration_decay = 0.995
        self.batch_size = 32
        self.memory_size = 10000
        self.target_update_freq = 100
        
        # Experience replay memory
        self.memory = []
        self.memory_idx = 0
        
        # Training statistics
        self.training_step = 0
        self.total_reward = 0
        self.episode_rewards = []
        self.success_history = []
        
    def _normalize_state(self, model_state: Dict[str, Any]) -> np.ndarray:
        """Convert model state to normalized feature vector"""
        
        # Extract features
        round_num = min(model_state.get('round', 1), 20) / 20.0  # Normalize to [0,1]
        
        # Debtor emotion encoding
        debtor_emotion = model_state.get('debtor_emotion', 'neutral')
        if isinstance(debtor_emotion, str):
            debtor_emotion_idx = self.emotion_to_idx.get(debtor_emotion, 6) / len(self.emotions)
        else:
            # Handle encoded emotions (A, S, D, etc.)
            emotion_map = {'A': 'angry', 'S': 'sad', 'D': 'disgust', 'F': 'fear', 
                          'J': 'happy', 'Su': 'surprising', 'N': 'neutral'}
            emotion_name = emotion_map.get(debtor_emotion, 'neutral')
            debtor_emotion_idx = self.emotion_to_idx.get(emotion_name, 6) / len(self.emotions)
        
        # Gap size (normalized)
        gap_size = min(model_state.get('gap_size', 0), 100) / 100.0
        
        # Debt amount (log-normalized)
        debt_amount = model_state.get('debt_amount', 1000)
        debt_normalized = min(np.log(debt_amount + 1) / 15.0, 1.0)  # Log scale
        
        # Success history (rolling average)
        success_rate = np.mean(self.success_history[-10:]) if self.success_history else 0.5
        
        return np.array([round_num, debtor_emotion_idx, gap_size, debt_normalized, success_rate])
    
    def select_emotion(self, model_state: Dict[str, Any]) -> Dict[str, Any]:
        """Select emotion using DQN with epsilon-greedy exploration"""
        
        state = self._normalize_state(model_state)
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        # Epsilon-greedy action selection
        if random.random() < self.exploration_rate:
            # Exploration: random action
            action_idx = random.randint(0, self.action_size - 1)
            exploration = True
        else:
            # Exploitation: best action from Q-network
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
                action_idx = q_values.argmax().item()
            exploration = False
        
        selected_emotion = self.emotions[action_idx]
        
        # Store state for learning
        self.current_state = state
        self.current_action = action_idx
        
        # Decay exploration rate
        if self.exploration_rate > self.min_exploration:
            self.exploration_rate *= self.exploration_decay
        
        return {
            'emotion': selected_emotion,
            'confidence': 1.0 - self.exploration_rate if not exploration else self.exploration_rate,
            'exploration': exploration,
            'q_values': q_values.squeeze().tolist() if not exploration else None,
            'emotion_text': self._get_emotion_prompt(selected_emotion),
            'temperature': 0.7,
            'reasoning': f"DQN {'exploration' if exploration else 'exploitation'} (ε={self.exploration_rate:.3f})"
        }
    
    def update_model(self, success: bool, final_days: int = None, target_days: int = None):
        """Update DQN based on negotiation outcome"""
        
        if not hasattr(self, 'current_state') or not hasattr(self, 'current_action'):
            return
        
        # Calculate reward
        if success and final_days and target_days:
            # Reward based on how close to target
            closeness = 1.0 - min(abs(final_days - target_days) / target_days, 1.0)
            reward = 1.0 + closeness  # Base reward + closeness bonus
        elif success:
            reward = 1.0
        else:
            reward = -0.5
        
        # Store experience
        self._store_experience(self.current_state, self.current_action, reward, None, success)
        
        # Train network if enough experience
        if len(self.memory) > self.batch_size:
            self._train_network()
        
        # Update target network periodically
        if self.training_step % self.target_update_freq == 0:
            self._update_target_network()
        
        # Update statistics
        self.total_reward += reward
        self.success_history.append(success)
        self.training_step += 1
    
    def _store_experience(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        experience = (state, action, reward, next_state, done)
        
        if len(self.memory) < self.memory_size:
            self.memory.append(experience)
        else:
            self.memory[self.memory_idx] = experience
            self.memory_idx = (self.memory_idx + 1) % self.memory_size
    
    def _train_network(self):
        """Train DQN using experience replay"""
        batch = random.sample(self.memory, self.batch_size)
        
        states = torch.FloatTensor([e[0] for e in batch])
        actions = torch.LongTensor([e[1] for e in batch])
        rewards = torch.FloatTensor([e[2] for e in batch])
        next_states = torch.FloatTensor([e[3] if e[3] is not None else e[0] for e in batch])
        dones = torch.BoolTensor([e[4] for e in batch])
        
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0].detach()
            target_q_values = rewards + (0.99 * next_q_values * ~dones)
        
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    def _update_target_network(self):
        """Update target network with current network weights"""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def _get_emotion_prompt(self, emotion: str) -> str:
        """Get emotion-specific prompt text"""
        prompts = {
            'happy': "Maintain an optimistic and cooperative tone. Show enthusiasm for reaching an agreement.",
            'surprising': "Express amazement at the situation and seek clarification with curiosity.",
            'angry': "Show controlled frustration about the situation while remaining professional.",
            'sad': "Express concern and disappointment about the circumstances.",
            'disgust': "Show dissatisfaction with the current terms while seeking better options.",
            'fear': "Express anxiety about potential consequences and seek reassurance.",
            'neutral': "Maintain a professional, matter-of-fact approach focused on facts."
        }
        return prompts.get(emotion, prompts['neutral'])
    
    def get_stats(self) -> Dict[str, Any]:
        """Get model statistics"""
        return {
            'model_type': 'deep_q_network',
            'exploration_rate': self.exploration_rate,
            'training_steps': self.training_step,
            'total_reward': self.total_reward,
            'success_history': self.success_history[-10:],  # Last 10 results
            'memory_size': len(self.memory),
            'q_network_parameters': sum(p.numel() for p in self.q_network.parameters()),
            'episode_rewards': self.episode_rewards[-5:] if self.episode_rewards else []  # Last 5 episodes
        }
    
    def reset(self) -> None:
        """Reset model for new scenario"""
        # Keep learning history but reset current state tracking
        if hasattr(self, 'current_state'):
            delattr(self, 'current_state')
        if hasattr(self, 'current_action'):
            delattr(self, 'current_action')

class QLearningEmotionModel(BaseEmotionModel):
    """Q-Learning based emotion model (no PyTorch required)"""
    
    def __init__(self, exploration_rate: float = 0.3, learning_rate: float = 0.1):
        super().__init__()
        
        self.emotions = ['happy', 'surprising', 'angry', 'sad', 'disgust', 'fear', 'neutral']
        self.exploration_rate = exploration_rate
        self.learning_rate = learning_rate
        self.discount_factor = 0.95
        
        # Q-table: state -> emotion -> Q-value
        self.q_table = {}
        
        # Training statistics
        self.total_episodes = 0
        self.success_history = []
        self.exploration_history = []
    
    def _get_state_key(self, model_state: Dict[str, Any]) -> str:
        """Convert model state to discrete state key"""
        round_bin = min(model_state.get('round', 1) // 5, 5)  # Bins: 0-4, 5-9, 10-14, 15-19, 20+
        
        # Debtor emotion
        debtor_emotion = model_state.get('debtor_emotion', 'neutral')
        if isinstance(debtor_emotion, str):
            debtor_key = debtor_emotion
        else:
            # Handle encoded emotions
            emotion_map = {'A': 'angry', 'S': 'sad', 'D': 'disgust', 'F': 'fear', 
                          'J': 'happy', 'Su': 'surprising', 'N': 'neutral'}
            debtor_key = emotion_map.get(debtor_emotion, 'neutral')
        
        # Gap size bins
        gap_size = model_state.get('gap_size', 0)
        if gap_size <= 5:
            gap_bin = 'small'
        elif gap_size <= 20:
            gap_bin = 'medium'
        else:
            gap_bin = 'large'
        
        # Success history
        success_rate = np.mean(self.success_history[-5:]) if self.success_history else 0.5
        if success_rate >= 0.7:
            success_bin = 'high'
        elif success_rate >= 0.4:
            success_bin = 'medium'
        else:
            success_bin = 'low'
        
        return f"{round_bin}_{debtor_key}_{gap_bin}_{success_bin}"
    
    def select_emotion(self, model_state: Dict[str, Any]) -> Dict[str, Any]:
        """Select emotion using Q-learning with epsilon-greedy exploration"""
        
        state_key = self._get_state_key(model_state)
        
        # Initialize Q-values for new states
        if state_key not in self.q_table:
            self.q_table[state_key] = {emotion: 0.0 for emotion in self.emotions}
        
        # Epsilon-greedy action selection
        if random.random() < self.exploration_rate:
            selected_emotion = random.choice(self.emotions)
            exploration = True
        else:
            q_values = self.q_table[state_key]
            selected_emotion = max(q_values, key=q_values.get)
            exploration = False
        
        # Store current state and action for learning
        self.current_state = state_key
        self.current_action = selected_emotion
        
        self.exploration_history.append(exploration)
        
        return {
            'emotion': selected_emotion,
            'confidence': max(self.q_table[state_key].values()) if not exploration else 0.5,
            'exploration': exploration,
            'q_values': self.q_table[state_key],
            'state_key': state_key,
            'emotion_text': self._get_emotion_prompt(selected_emotion),
            'temperature': 0.7,
            'reasoning': f"Q-Learning {'exploration' if exploration else 'exploitation'} (ε={self.exploration_rate:.3f})"
        }
    
    def update_model(self, success: bool, final_days: int = None, target_days: int = None):
        """Update Q-values based on negotiation outcome"""
        
        if not hasattr(self, 'current_state') or not hasattr(self, 'current_action'):
            return
        
        # Calculate reward
        if success and final_days and target_days:
            closeness = 1.0 - min(abs(final_days - target_days) / target_days, 1.0)
            reward = 1.0 + closeness
        elif success:
            reward = 1.0
        else:
            reward = -0.5
        
        # Q-learning update
        current_q = self.q_table[self.current_state][self.current_action]
        
        # For terminal states, there's no future reward
        max_future_q = 0
        
        # Q-learning formula: Q(s,a) = Q(s,a) + α[r + γ*max(Q(s',a')) - Q(s,a)]
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_future_q - current_q)
        self.q_table[self.current_state][self.current_action] = new_q
        
        # Update statistics
        self.success_history.append(success)
        self.total_episodes += 1
        
        # Decay exploration rate
        if self.exploration_rate > 0.1:
            self.exploration_rate *= 0.995
    
    def get_stats(self) -> Dict[str, Any]:
        """Get model statistics"""
        return {
            'model_type': 'q_learning',
            'exploration_rate': self.exploration_rate,
            'total_episodes': self.total_episodes,
            'success_history': self.success_history[-10:],  # Last 10 results
            'q_table_size': len(self.q_table),
            'states_explored': list(self.q_table.keys()),
            'exploration_history': self.exploration_history[-10:] if self.exploration_history else [],
            'learning_rate': self.learning_rate,
            'discount_factor': self.discount_factor
        }
    
    def reset(self) -> None:
        """Reset model for new scenario"""
        # Keep Q-table and learning history but reset current state tracking
        if hasattr(self, 'current_state'):
            delattr(self, 'current_state')
        if hasattr(self, 'current_action'):
            delattr(self, 'current_action')
    
    def _get_emotion_prompt(self, emotion: str) -> str:
        """Get emotion-specific prompt text"""
        prompts = {
            'happy': "Maintain an optimistic and cooperative tone. Show enthusiasm for reaching an agreement.",
            'surprising': "Express amazement at the situation and seek clarification with curiosity.",
            'angry': "Show controlled frustration about the situation while remaining professional.",
            'sad': "Express concern and disappointment about the circumstances.",
            'disgust': "Show dissatisfaction with the current terms while seeking better options.",
            'fear': "Express anxiety about potential consequences and seek reassurance.",
            'neutral': "Maintain a professional, matter-of-fact approach focused on facts."
        }
        return prompts.get(emotion, prompts['neutral'])

class PolicyGradientEmotionModel(BaseEmotionModel):
    """Policy Gradient based emotion model"""
    
    def __init__(self, exploration_rate: float = 0.2, learning_rate: float = 0.01):
        super().__init__()
        
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for Policy Gradient agent")
        
        self.emotions = ['happy', 'surprising', 'angry', 'sad', 'disgust', 'fear', 'neutral']
        self.emotion_to_idx = {e: i for i, e in enumerate(self.emotions)}
        
        # State size: same as DQN
        self.state_size = 5
        self.action_size = len(self.emotions)
        
        # Policy network
        self.policy_network = nn.Sequential(
            nn.Linear(self.state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, self.action_size),
            nn.Softmax(dim=-1)
        )
        
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=learning_rate)
        
        # Episode memory
        self.episode_states = []
        self.episode_actions = []
        self.episode_rewards = []
        
        # Statistics
        self.success_history = []
    
    def _normalize_state(self, model_state: Dict[str, Any]) -> np.ndarray:
        """Same state normalization as DQN"""
        round_num = min(model_state.get('round', 1), 20) / 20.0
        
        debtor_emotion = model_state.get('debtor_emotion', 'neutral')
        if isinstance(debtor_emotion, str):
            debtor_emotion_idx = self.emotion_to_idx.get(debtor_emotion, 6) / len(self.emotions)
        else:
            emotion_map = {'A': 'angry', 'S': 'sad', 'D': 'disgust', 'F': 'fear', 
                          'J': 'happy', 'Su': 'surprising', 'N': 'neutral'}
            emotion_name = emotion_map.get(debtor_emotion, 'neutral')
            debtor_emotion_idx = self.emotion_to_idx.get(emotion_name, 6) / len(self.emotions)
        
        gap_size = min(model_state.get('gap_size', 0), 100) / 100.0
        debt_amount = model_state.get('debt_amount', 1000)
        debt_normalized = min(np.log(debt_amount + 1) / 15.0, 1.0)
        success_rate = np.mean(self.success_history[-10:]) if self.success_history else 0.5
        
        return np.array([round_num, debtor_emotion_idx, gap_size, debt_normalized, success_rate])
    
    def select_emotion(self, model_state: Dict[str, Any]) -> Dict[str, Any]:
        """Select emotion using policy network"""
        
        state = self._normalize_state(model_state)
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        # Get action probabilities from policy network
        with torch.no_grad():
            action_probs = self.policy_network(state_tensor)
        
        # Sample action from probability distribution
        action_dist = torch.distributions.Categorical(action_probs)
        action_idx = action_dist.sample().item()
        
        selected_emotion = self.emotions[action_idx]
        confidence = action_probs[0][action_idx].item()
        
        # Store for training
        self.episode_states.append(state)
        self.episode_actions.append(action_idx)
        
        return {
            'emotion': selected_emotion,
            'confidence': confidence,
            'action_probs': action_probs.squeeze().tolist(),
            'emotion_text': self._get_emotion_prompt(selected_emotion),
            'temperature': 0.7,
            'reasoning': f"Policy Gradient (confidence: {confidence:.3f})"
        }
    
    def update_model(self, success: bool, final_days: int = None, target_days: int = None):
        """Update policy using REINFORCE algorithm"""
        
        # Calculate reward
        if success and final_days and target_days:
            closeness = 1.0 - min(abs(final_days - target_days) / target_days, 1.0)
            reward = 1.0 + closeness
        elif success:
            reward = 1.0
        else:
            reward = -0.5
        
        self.episode_rewards.append(reward)
        self.success_history.append(success)
        
        # Train at end of episode (when we have complete trajectory)
        if len(self.episode_rewards) >= 5:  # Mini-batch training
            self._train_policy()
    
    def _train_policy(self):
        """Train policy network using REINFORCE"""
        if not self.episode_states:
            return
        
        # Convert to tensors
        states = torch.FloatTensor(self.episode_states)
        actions = torch.LongTensor(self.episode_actions)
        rewards = torch.FloatTensor(self.episode_rewards)
        
        # Calculate discounted rewards
        discounted_rewards = []
        R = 0
        for r in reversed(rewards):
            R = r + 0.99 * R
            discounted_rewards.insert(0, R)
        
        discounted_rewards = torch.FloatTensor(discounted_rewards)
        
        # Normalize rewards
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-8)
        
        # Calculate policy gradient
        action_probs = self.policy_network(states)
        action_log_probs = torch.log(action_probs.gather(1, actions.unsqueeze(1)).squeeze())
        
        policy_loss = -(action_log_probs * discounted_rewards).mean()
        
        # Update policy
        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()
        
        # Clear episode memory
        self.episode_states = []
        self.episode_actions = []
        self.episode_rewards = []
    
    def get_stats(self) -> Dict[str, Any]:
        """Get model statistics"""
        return {
            'model_type': 'policy_gradient',
            'success_history': self.success_history[-10:],  # Last 10 results
            'policy_network_parameters': sum(p.numel() for p in self.policy_network.parameters()),
            'episode_states_count': len(self.episode_states),
            'episode_actions_count': len(self.episode_actions),
            'episode_rewards_count': len(self.episode_rewards),
            'current_episode_reward': sum(self.episode_rewards) if self.episode_rewards else 0
        }
    
    def reset(self) -> None:
        """Reset model for new scenario"""
        # Reset episode memory for new scenario
        self.episode_states = []
        self.episode_actions = []
        self.episode_rewards = []
    
    def _get_emotion_prompt(self, emotion: str) -> str:
        """Get emotion-specific prompt text"""
        prompts = {
            'happy': "Maintain an optimistic and cooperative tone. Show enthusiasm for reaching an agreement.",
            'surprising': "Express amazement at the situation and seek clarification with curiosity.",
            'angry': "Show controlled frustration about the situation while remaining professional.",
            'sad': "Express concern and disappointment about the circumstances.",
            'disgust': "Show dissatisfaction with the current terms while seeking better options.",
            'fear': "Express anxiety about potential consequences and seek reassurance.",
            'neutral': "Maintain a professional, matter-of-fact approach focused on facts."
        }
        return prompts.get(emotion, prompts['neutral'])


def run_dqn_experiment(
    scenarios: List[Dict[str, Any]],
    iterations: int = 5,
    model_creditor: str = "gpt-4o-mini",
    model_debtor: str = "gpt-4o-mini",
    debtor_emotion: str = "neutral",
    debtor_model_type: str = "vanilla",
    max_dialog_len: int = 30,
    out_dir: str = "results/dqn"
) -> Dict[str, Any]:
    """Run DQN-based emotion optimization experiment"""
    
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    # Create DQN model
    model = DeepQEmotionModel(exploration_rate=0.3, learning_rate=0.001)
    
    results = {
        'experiment_type': 'dqn_emotion_optimization',
        'model_type': 'deep_q_network',
        'iterations': iterations,
        'scenarios_used': [s['id'] if 'id' in s else f"scenario_{i}" for i, s in enumerate(scenarios)],
        'config': {
            'model_creditor': model_creditor,
            'model_debtor': model_debtor,
            'debtor_emotion': debtor_emotion,
            'debtor_model_type': debtor_model_type,
            'max_dialog_len': max_dialog_len,
            'exploration_rate': model.exploration_rate
        },
        'detailed_results': []
    }
    
    all_negotiations = []
    
    for iteration in range(iterations):
        print(f"\n🎯 DQN Iteration {iteration + 1}/{iterations}")
        
        for scenario_idx, scenario in enumerate(scenarios):
            print(f"  📋 Scenario {scenario_idx + 1}/{len(scenarios)}: ", end="")
            
            # Create negotiator with DQN model
            negotiator = DebtNegotiator(
                config=scenario,
                emotion_model=model,
                model_creditor=model_creditor,
                model_debtor=model_debtor,
                debtor_emotion=debtor_emotion,
                debtor_model_type=debtor_model_type
            )
            
            # Run negotiation
            result = negotiator.run_negotiation(max_dialog_len=max_dialog_len)
            all_negotiations.append(result)
            
            # Update DQN based on result
            success = result.get('final_state') == 'accept'
            final_days = result.get('collection_days')
            target_days = int(scenario['seller']['target_price'])
            
            model.update_model(success, final_days, target_days)
            
            # Print result
            if success:
                print(f"✅ Success - {final_days} days (target: {target_days})")
            else:
                print(f"❌ Failed after {result.get('negotiation_rounds', 0)} rounds")
    
    # Calculate summary statistics - MATCH VANILLA METHOD
    successful_negotiations = [n for n in all_negotiations if n.get('final_state') == 'accept']
    failed_negotiations = [n for n in all_negotiations if n.get('final_state') != 'accept']
    success_rate = len(successful_negotiations) / len(all_negotiations) if all_negotiations else 0
    
    # Calculate collection rates for successful negotiations only - MATCH VANILLA METHOD
    collection_rates = []
    for r in successful_negotiations:
        target_days = r.get('creditor_target_days', 30)
        actual_days = r.get('collection_days', target_days)
        if target_days > 0 and actual_days > 0:
            collection_rate = min(1.0, target_days / actual_days)  # Higher is better, capped at 1.0
            collection_rates.append(collection_rate)
    
    avg_collection_rate = np.mean(collection_rates) if collection_rates else 0.0
    avg_rounds = np.mean([len(r.get('dialog', [])) for r in all_negotiations])
    
    # Enhance results with comprehensive statistical analysis
    results = enhance_results_with_statistics(
        results, 
        all_negotiations, 
        scenarios, 
        method="bootstrap"
    )
    
    # Add performance metrics like vanilla
    results['performance'] = {
        'success_rate': success_rate,
        'avg_collection_rate': float(avg_collection_rate),
        'avg_negotiation_rounds': float(avg_rounds),
        'total_negotiations': len(all_negotiations),
        'successful_negotiations': len(successful_negotiations),
        'failed_negotiations': len(failed_negotiations)
    }
    
    # Add detailed breakdown like vanilla
    failure_reasons = {}
    for result in failed_negotiations:
        reason = result.get('final_state', 'unknown')
        failure_reasons[reason] = failure_reasons.get(reason, 0) + 1
    
    results['analysis'] = {
        'failure_breakdown': failure_reasons,
        'success_patterns': {
            'avg_rounds_successful': float(np.mean([len(r.get('dialog', [])) for r in successful_negotiations])) if successful_negotiations else 0,
            'avg_rounds_failed': float(np.mean([len(r.get('dialog', [])) for r in failed_negotiations])) if failed_negotiations else 0
        }
    }
    
    results.update({
        'summary_statistics': {
            'success_rate': success_rate,
            'total_negotiations': len(all_negotiations),
            'successful_negotiations': len(successful_negotiations),
            'failed_negotiations': len(failed_negotiations),
            'collection_rate': {
                'mean': float(avg_collection_rate),
                'std': float(np.std(collection_rates)) if collection_rates else 0,
                'min': float(min(collection_rates)) if collection_rates else 0,
                'max': float(max(collection_rates)) if collection_rates else 0
            },
            'negotiation_rounds': {
                'mean': float(avg_rounds),
                'std': float(np.std([n.get('negotiation_rounds', 0) for n in all_negotiations])),
                'min': min([n.get('negotiation_rounds', 0) for n in all_negotiations]),
                'max': max([n.get('negotiation_rounds', 0) for n in all_negotiations])
            }
        },
        'dqn_statistics': {
            'final_exploration_rate': model.exploration_rate,
            'total_training_steps': model.training_step,
            'episode_rewards': model.episode_rewards,
            'success_history': model.success_history,
            'q_network_size': sum(p.numel() for p in model.q_network.parameters()),
            'memory_size': len(model.memory)
        },
        'detailed_results': all_negotiations
    })
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = f"{out_dir}/dqn_results_{timestamp}.json"
    
    with open(result_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # ===== PRINT RESULTS WITH CONFIDENCE INTERVALS (SINGLE TIME LIKE VANILLA) =====
    print("\n" + "="*80)
    print("📊 DQN RESULTS WITH 95% CONFIDENCE INTERVALS")
    print("="*80)
    
    # Print statistical analysis with CIs
    if 'statistical_analysis' in results:
        print(format_ci_results(results['statistical_analysis']))
        print("="*80)
    
    # Save human-readable summary like vanilla
    summary_file = f"{out_dir}/summary_{timestamp}.txt"
    with open(summary_file, "w") as f:
        f.write("DQN EMOTION OPTIMIZATION EXPERIMENT SUMMARY\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Experiment Type: DQN Emotion Optimization\n")
        f.write(f"Total Negotiations: {len(all_negotiations)}\n")
        
        # Add CI information to summary
        if 'statistical_analysis' in results:
            stat_analysis = results['statistical_analysis']
            sr_ci = stat_analysis['success_rate']['ci_95']
            cr_ci = stat_analysis['collection_rate']['ci_95']
            nr_ci = stat_analysis['negotiation_rounds']['ci_95']
            
            f.write(f"Success Rate: {stat_analysis['success_rate']['mean']:.1%} ")
            f.write(f"(95% CI: [{sr_ci[0]:.1%}, {sr_ci[1]:.1%}])\n")
            f.write(f"Collection Rate: {stat_analysis['collection_rate']['mean']:.3f} ")
            f.write(f"(95% CI: [{cr_ci[0]:.3f}, {cr_ci[1]:.3f}])\n")
            f.write(f"Negotiation Rounds: {stat_analysis['negotiation_rounds']['mean']:.1f} ")
            f.write(f"(95% CI: [{nr_ci[0]:.1f}, {nr_ci[1]:.1f}])\n\n")
        else:
            f.write(f"Success Rate: {success_rate:.1%}\n")
            f.write(f"Average Collection Rate: {avg_collection_rate:.3f}\n")
            f.write(f"Average Rounds: {avg_rounds:.1f}\n\n")
        
        f.write("FAILURE BREAKDOWN:\n")
      
        
        f.write(f"\nModel Configuration:\n")
        f.write(f"  Creditor Model: {model_creditor}\n")
        f.write(f"  Debtor Model: {model_debtor}\n")
        f.write(f"  Emotion Strategy: Deep Q-Network\n")
        f.write(f"  Max Dialog Length: {max_dialog_len}\n")
        f.write(f"  Final Exploration Rate: {model.exploration_rate:.3f}\n")
    
    print(f"📧 DQN Training Steps: {model.training_step}")
    print(f"🎯 Final Exploration Rate: {model.exploration_rate:.3f}")
    print(f"💾 Results saved to: {result_file}")
    
    return results


def run_qlearning_experiment(
    scenarios: List[Dict[str, Any]],
    iterations: int = 5,
    model_creditor: str = "gpt-4o-mini",
    model_debtor: str = "gpt-4o-mini",
    debtor_emotion: str = "neutral",
    debtor_model_type: str = "vanilla",
    max_dialog_len: int = 30,
    out_dir: str = "results/qlearning"
) -> Dict[str, Any]:
    """Run Q-Learning based emotion optimization experiment"""
    
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    # Create Q-learning model
    model = QLearningEmotionModel(exploration_rate=0.3, learning_rate=0.1)
    
    results = {
        'experiment_type': 'qlearning_emotion_optimization',
        'model_type': 'q_learning',
        'iterations': iterations,
        'scenarios_used': [s['id'] if 'id' in s else f"scenario_{i}" for i, s in enumerate(scenarios)],
        'config': {
            'model_creditor': model_creditor,
            'model_debtor': model_debtor,
            'debtor_emotion': debtor_emotion,
            'debtor_model_type': debtor_model_type,
            'max_dialog_len': max_dialog_len,
            'exploration_rate': model.exploration_rate,
            'learning_rate': model.learning_rate
        },
        'detailed_results': []
    }
    
    all_negotiations = []
    
    for iteration in range(iterations):
        print(f"\n🎯 Q-Learning Iteration {iteration + 1}/{iterations}")
        
        for scenario_idx, scenario in enumerate(scenarios):
            print(f"  📋 Scenario {scenario_idx + 1}/{len(scenarios)}: ", end="")
            
            # Create negotiator with Q-learning model
            negotiator = DebtNegotiator(
                config=scenario,
                emotion_model=model,
                model_creditor=model_creditor,
                model_debtor=model_debtor,
                debtor_emotion=debtor_emotion,
                debtor_model_type=debtor_model_type
            )
            
            # Run negotiation
            result = negotiator.run_negotiation(max_dialog_len=max_dialog_len)
            all_negotiations.append(result)
            
            # Update Q-learning based on result
            success = result.get('final_state') == 'accept'
            final_days = result.get('collection_days')
            target_days = int(scenario['seller']['target_price'])
            
            model.update_model(success, final_days, target_days)
            
            # Print result
            if success:
                print(f"✅ Success - {final_days} days (target: {target_days})")
            else:
                print(f"❌ Failed after {result.get('negotiation_rounds', 0)} rounds")
    
    # Calculate summary statistics - MATCH VANILLA METHOD
    successful_negotiations = [n for n in all_negotiations if n.get('final_state') == 'accept']
    failed_negotiations = [n for n in all_negotiations if n.get('final_state') != 'accept']
    success_rate = len(successful_negotiations) / len(all_negotiations) if all_negotiations else 0
    
    # Calculate collection rates for successful negotiations only - MATCH VANILLA METHOD
    collection_rates = []
    for r in successful_negotiations:
        target_days = r.get('creditor_target_days', 30)
        actual_days = r.get('collection_days', target_days)
        if target_days > 0 and actual_days > 0:
            collection_rate = min(1.0, target_days / actual_days)  # Higher is better, capped at 1.0
            collection_rates.append(collection_rate)
    
    avg_collection_rate = np.mean(collection_rates) if collection_rates else 0.0
    avg_rounds = np.mean([n.get('negotiation_rounds', 0) for n in all_negotiations])
    
    # Enhance results with comprehensive statistical analysis
    results = enhance_results_with_statistics(
        results, 
        all_negotiations, 
        scenarios, 
        method="bootstrap"
    )
    
    # Add performance metrics like vanilla
    results['performance'] = {
        'success_rate': success_rate,
        'avg_collection_rate': float(avg_collection_rate),
        'avg_negotiation_rounds': float(avg_rounds),
        'total_negotiations': len(all_negotiations),
        'successful_negotiations': len(successful_negotiations),
        'failed_negotiations': len(failed_negotiations)
    }
    
    # Add detailed breakdown like vanilla
    failure_reasons = {}
    for result in failed_negotiations:
        reason = result.get('final_state', 'unknown')
        failure_reasons[reason] = failure_reasons.get(reason, 0) + 1
    
    results['analysis'] = {
        'failure_breakdown': failure_reasons,
        'success_patterns': {
            'avg_rounds_successful': float(np.mean([len(r.get('dialog', [])) for r in successful_negotiations])) if successful_negotiations else 0,
            'avg_rounds_failed': float(np.mean([len(r.get('dialog', [])) for r in failed_negotiations])) if failed_negotiations else 0
        }
    }
    
    results.update({
        'summary_statistics': {
            'success_rate': success_rate,
            'total_negotiations': len(all_negotiations),
            'successful_negotiations': len(successful_negotiations),
            'failed_negotiations': len(failed_negotiations),
            'collection_rate': {
                'mean': float(avg_collection_rate),
                'std': float(np.std(collection_rates)) if collection_rates else 0,
                'min': float(min(collection_rates)) if collection_rates else 0,
                'max': float(max(collection_rates)) if collection_rates else 0
            },
            'negotiation_rounds': {
                'mean': float(avg_rounds),
                'std': float(np.std([n.get('negotiation_rounds', 0) for n in all_negotiations])),
                'min': min([n.get('negotiation_rounds', 0) for n in all_negotiations]),
                'max': max([n.get('negotiation_rounds', 0) for n in all_negotiations])
            }
        },
        'qlearning_statistics': {
            'final_exploration_rate': model.exploration_rate,
            'total_episodes': model.total_episodes,
            'q_table_size': len(model.q_table),
            'states_explored': list(model.q_table.keys()),
            'success_history': model.success_history,
            'exploration_history': model.exploration_history
        },
        'detailed_results': all_negotiations
    })
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = f"{out_dir}/qlearning_results_{timestamp}.json"
    
    with open(result_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # ===== PRINT RESULTS WITH CONFIDENCE INTERVALS (SINGLE TIME LIKE VANILLA) =====
    print("\\n" + "="*80)
    print("📊 Q-LEARNING RESULTS WITH 95% CONFIDENCE INTERVALS")
    print("="*80)
    
    # Print statistical analysis with CIs
    if 'statistical_analysis' in results:
        print(format_ci_results(results['statistical_analysis']))
        print("="*80)
    
    # Save human-readable summary like vanilla
    summary_file = f"{out_dir}/summary_{timestamp}.txt"
    with open(summary_file, "w") as f:
        f.write("Q-LEARNING EMOTION OPTIMIZATION EXPERIMENT SUMMARY\\n")
        f.write("=" * 50 + "\\n\\n")
        f.write(f"Experiment Type: Q-Learning Emotion Optimization\\n")
        f.write(f"Total Negotiations: {len(all_negotiations)}\\n")
        
        # Add CI information to summary
        if 'statistical_analysis' in results:
            stat_analysis = results['statistical_analysis']
            sr_ci = stat_analysis['success_rate']['ci_95']
            cr_ci = stat_analysis['collection_rate']['ci_95']
            nr_ci = stat_analysis['negotiation_rounds']['ci_95']
            
            f.write(f"Success Rate: {stat_analysis['success_rate']['mean']:.1%} ")
            f.write(f"(95% CI: [{sr_ci[0]:.1%}, {sr_ci[1]:.1%}])\\n")
            f.write(f"Collection Rate: {stat_analysis['collection_rate']['mean']:.3f} ")
            f.write(f"(95% CI: [{cr_ci[0]:.3f}, {cr_ci[1]:.3f}])\\n")
            f.write(f"Negotiation Rounds: {stat_analysis['negotiation_rounds']['mean']:.1f} ")
            f.write(f"(95% CI: [{nr_ci[0]:.1f}, {nr_ci[1]:.1f}])\\n\\n")
        else:
            f.write(f"Success Rate: {success_rate:.1%}\n")
            f.write(f"Average Collection Rate: {avg_collection_rate:.3f}\n")
            f.write(f"Average Rounds: {avg_rounds:.1f}\n\n")
        
        f.write("FAILURE BREAKDOWN:\n")
        for reason, count in failure_reasons.items():
            if failed:
                f.write(f"  {reason}: {count} ({count/len(failed)*100:.1f}%)\n")
            else:
                f.write(f"  {reason}: {count} (0.0%)\n")
        
        f.write(f"\nModel Configuration:\n")
        f.write(f"  Creditor Model: {model_creditor}\n")
        f.write(f"  Debtor Model: {model_debtor}\n")
        f.write(f"  Emotion Strategy: Q-Learning\n")
        f.write(f"  Max Dialog Length: {max_dialog_len}\n")
        f.write(f"  Final Exploration Rate: {model.exploration_rate:.3f}\n")
        f.write(f"  Q-Table Size: {len(model.q_table)}\n")
    
    print(f"\n📊 Q-LEARNING RESULTS")
    print("=" * 50)
    print(f"✅ Success Rate: {success_rate:.1%}")
    print(f"📈 Avg Collection Rate: {avg_collection_rate:.3f}")
    print(f"💬 Avg Negotiation Rounds: {avg_rounds:.1f}")
    print(f"🎯 Final Exploration Rate: {model.exploration_rate:.3f}")
    print(f"🗺 Q-Table States: {len(model.q_table)}")
    print(f"📁 Results saved to: {result_file}")
    print(f"📄 Summary saved to: {summary_file}")
    
    return results


def run_rl_agents_experiment(
    scenarios: List[Dict[str, Any]],
    iterations: int = 5,
    model_creditor: str = "gpt-4o-mini",
    model_debtor: str = "gpt-4o-mini",
    debtor_emotion: str = "neutral",
    debtor_model_type: str = "vanilla",
    max_dialog_len: int = 30,
    out_dir: str = "results/rl_agents"
) -> Dict[str, Any]:
    """Run RL agents experiment - defaults to Q-Learning if PyTorch not available"""
    
    # Create output directory
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    
    print(f"🎮 Running RL Agents Experiment")
    
    # Use Q-Learning by default (no PyTorch dependency)
    if TORCH_AVAILABLE:
        print(f"🔥 PyTorch available - using DQN agent")
        return run_dqn_experiment(
            scenarios=scenarios,
            iterations=iterations,
            model_creditor=model_creditor,
            model_debtor=model_debtor,
            debtor_emotion=debtor_emotion,
            debtor_model_type=debtor_model_type,
            max_dialog_len=max_dialog_len,
            out_dir=out_dir
        )
    else:
        print(f"📊 PyTorch not available - using Q-Learning agent")
        return run_qlearning_experiment(
            scenarios=scenarios,
            iterations=iterations,
            model_creditor=model_creditor,
            model_debtor=model_debtor,
            debtor_emotion=debtor_emotion,
            debtor_model_type=debtor_model_type,
            max_dialog_len=max_dialog_len,
            out_dir=out_dir
        )