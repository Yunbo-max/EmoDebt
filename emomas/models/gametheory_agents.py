"""
Game Theory Agents for Emotional Debt Negotiation
Includes Nash Equilibrium, Minimax, and Strategic Dominance approaches
"""

import numpy as np
import itertools
from typing import Dict, List, Any, Optional, Tuple
from models.base_model import BaseEmotionModel
import json
from datetime import datetime
import os
from scipy.optimize import minimize
from scipy.spatial.distance import euclidean
from utils.statistical_analysis import enhance_results_with_statistics, analyze_negotiation_results, format_ci_results

class NashEquilibriumEmotionModel(BaseEmotionModel):
    """Nash Equilibrium based emotion selection for debt negotiation"""
    
    def __init__(self, exploration_rate: float = 0.1):
        super().__init__()
        
        # Emotion strategies
        self.emotions = ['happy', 'surprising', 'angry', 'sad', 'disgust', 'fear', 'neutral']
        self.emotion_to_idx = {e: i for i, e in enumerate(self.emotions)}
        
        # Payoff matrix: [creditor_emotion][debtor_emotion] -> (creditor_payoff, debtor_payoff)
        self.payoff_matrix = self._initialize_payoff_matrix()
        
        # Nash equilibrium strategies
        self.creditor_strategy = np.ones(len(self.emotions)) / len(self.emotions)  # Uniform initially
        self.debtor_strategy = np.ones(len(self.emotions)) / len(self.emotions)
        
        # Learning parameters
        self.exploration_rate = exploration_rate
        self.learning_rate = 0.05
        self.experience_history = []
        
        # Statistics
        self.strategy_history = []
        self.payoff_history = []
        self.equilibrium_updates = 0
    
    def _initialize_payoff_matrix(self):
        """Initialize empirical payoff matrix based on emotion psychology"""
        emotions = self.emotions
        n = len(emotions)
        
        # Initialize with domain knowledge
        payoffs = {}
        
        for i, creditor_emotion in enumerate(emotions):
            payoffs[creditor_emotion] = {}
            for j, debtor_emotion in enumerate(emotions):
                # Base payoffs on psychological compatibility
                creditor_payoff, debtor_payoff = self._get_base_payoff(creditor_emotion, debtor_emotion)
                payoffs[creditor_emotion][debtor_emotion] = (creditor_payoff, debtor_payoff)
        
        return payoffs
    
    def _get_base_payoff(self, creditor_emotion: str, debtor_emotion: str) -> Tuple[float, float]:
        """Get base payoff for emotion combination based on psychological theory"""
        
        # Compatibility matrix based on negotiation psychology
        compatibility = {
            ('happy', 'happy'): (0.8, 0.8),      # Both cooperative
            ('happy', 'sad'): (0.7, 0.6),        # Creditor sympathetic to distress
            ('happy', 'fear'): (0.6, 0.5),       # Creditor patient with anxious debtor
            ('angry', 'angry'): (0.2, 0.2),      # Escalation, poor outcomes
            ('angry', 'fear'): (0.5, 0.3),       # Intimidation may work short-term
            ('angry', 'sad'): (0.4, 0.4),        # Anger vs sadness - moderate tension
            ('neutral', 'neutral'): (0.6, 0.6),  # Professional baseline
            ('sad', 'sad'): (0.5, 0.7),          # Mutual sympathy, debtor favored
            ('disgust', 'disgust'): (0.3, 0.3),  # Mutual dissatisfaction
            ('fear', 'angry'): (0.3, 0.5),       # Debtor dominance
            ('surprising', 'surprising'): (0.4, 0.4), # Confusion, uncertain outcomes
        }
        
        if (creditor_emotion, debtor_emotion) in compatibility:
            return compatibility[(creditor_emotion, debtor_emotion)]
        
        # Default calculation for unlisted combinations
        creditor_score = self._emotion_assertiveness(creditor_emotion)
        debtor_score = self._emotion_assertiveness(debtor_emotion)
        
        # Balance: higher assertiveness generally better for creditor
        creditor_payoff = 0.5 + 0.3 * (creditor_score - debtor_score)
        debtor_payoff = 0.5 + 0.3 * (debtor_score - creditor_score)
        
        # Normalize to [0, 1]
        return (max(0, min(1, creditor_payoff)), max(0, min(1, debtor_payoff)))
    
    def _emotion_assertiveness(self, emotion: str) -> float:
        """Get assertiveness score for emotion [-1 to 1]"""
        scores = {
            'angry': 0.8,
            'disgust': 0.6,
            'neutral': 0.0,
            'surprising': 0.2,
            'happy': -0.2,
            'sad': -0.6,
            'fear': -0.8
        }
        return scores.get(emotion, 0.0)
    
    def select_emotion(self, model_state: Dict[str, Any]) -> Dict[str, Any]:
        """Select emotion using mixed Nash equilibrium strategy"""
        
        # Update debtor strategy estimate based on recent observations
        self._update_debtor_strategy_estimate(model_state)
        
        # Calculate best response to current debtor strategy
        creditor_payoffs = []
        for i, creditor_emotion in enumerate(self.emotions):
            expected_payoff = 0
            for j, debtor_emotion in enumerate(self.emotions):
                prob = self.debtor_strategy[j]
                payoff = self.payoff_matrix[creditor_emotion][debtor_emotion][0]  # Creditor payoff
                expected_payoff += prob * payoff
            creditor_payoffs.append(expected_payoff)
        
        # Epsilon-greedy: mostly best response, sometimes explore
        if np.random.random() < self.exploration_rate:
            # Exploration: sample from current mixed strategy
            emotion_idx = np.random.choice(len(self.emotions), p=self.creditor_strategy)
            exploration = True
        else:
            # Exploitation: best response
            emotion_idx = np.argmax(creditor_payoffs)
            exploration = False
        
        selected_emotion = self.emotions[emotion_idx]
        
        # Update strategy history
        self.strategy_history.append({
            'creditor_strategy': self.creditor_strategy.copy(),
            'debtor_strategy': self.debtor_strategy.copy(),
            'selected_emotion': selected_emotion,
            'expected_payoffs': creditor_payoffs
        })
        
        return {
            'emotion': selected_emotion,
            'confidence': creditor_payoffs[emotion_idx],
            'exploration': exploration,
            'creditor_strategy': self.creditor_strategy.tolist(),
            'debtor_strategy': self.debtor_strategy.tolist(),
            'expected_payoffs': creditor_payoffs,
            'emotion_text': self._get_emotion_prompt(selected_emotion),
            'temperature': 0.7,
            'reasoning': f"Nash {'exploration' if exploration else 'best response'} (equilibrium {self.equilibrium_updates})"
        }
    
    def _update_debtor_strategy_estimate(self, model_state: Dict[str, Any]):
        """Update estimate of debtor's mixed strategy"""
        
        # Get debtor emotion from state
        debtor_emotion = model_state.get('debtor_emotion', 'neutral')
        
        # Handle encoded emotions
        if isinstance(debtor_emotion, str) and len(debtor_emotion) <= 2:
            emotion_map = {'A': 'angry', 'S': 'sad', 'D': 'disgust', 'F': 'fear', 
                          'J': 'happy', 'Su': 'surprising', 'N': 'neutral'}
            debtor_emotion = emotion_map.get(debtor_emotion, 'neutral')
        
        if debtor_emotion in self.emotion_to_idx:
            debtor_idx = self.emotion_to_idx[debtor_emotion]
            
            # Update debtor strategy using exponential smoothing
            decay = 0.9
            self.debtor_strategy *= decay
            self.debtor_strategy[debtor_idx] += (1 - decay)
            
            # Renormalize
            self.debtor_strategy /= self.debtor_strategy.sum()
    
    def update_model(self, success: bool, final_days: int = None, target_days: int = None):
        """Update payoff matrix and recompute Nash equilibrium"""
        
        if not hasattr(self, 'last_creditor_emotion') or not hasattr(self, 'last_debtor_emotion'):
            return
        
        # Calculate actual payoff
        if success and final_days and target_days:
            # Payoff based on how close to target
            closeness = 1.0 - min(abs(final_days - target_days) / target_days, 1.0)
            creditor_actual_payoff = 0.5 + 0.5 * closeness  # [0.5, 1.0]
            debtor_actual_payoff = 0.5 + 0.3 * (final_days - target_days) / target_days  # More days = better for debtor
            debtor_actual_payoff = max(0, min(1, debtor_actual_payoff))
        elif success:
            creditor_actual_payoff = 0.7
            debtor_actual_payoff = 0.7
        else:
            creditor_actual_payoff = 0.1
            debtor_actual_payoff = 0.1
        
        # Update payoff matrix using learning
        creditor_emotion = self.last_creditor_emotion
        debtor_emotion = self.last_debtor_emotion
        
        if creditor_emotion in self.payoff_matrix and debtor_emotion in self.payoff_matrix[creditor_emotion]:
            current_payoffs = self.payoff_matrix[creditor_emotion][debtor_emotion]
            
            # Exponential moving average update
            alpha = self.learning_rate
            new_creditor_payoff = (1 - alpha) * current_payoffs[0] + alpha * creditor_actual_payoff
            new_debtor_payoff = (1 - alpha) * current_payoffs[1] + alpha * debtor_actual_payoff
            
            self.payoff_matrix[creditor_emotion][debtor_emotion] = (new_creditor_payoff, new_debtor_payoff)
            
            # Store experience
            self.experience_history.append({
                'creditor_emotion': creditor_emotion,
                'debtor_emotion': debtor_emotion,
                'success': success,
                'creditor_payoff': creditor_actual_payoff,
                'debtor_payoff': debtor_actual_payoff,
                'final_days': final_days,
                'target_days': target_days
            })
            
            # Recompute Nash equilibrium periodically
            if len(self.experience_history) % 5 == 0:
                self._compute_nash_equilibrium()
    
    def _compute_nash_equilibrium(self):
        """Compute mixed strategy Nash equilibrium using iterative best response"""
        
        max_iterations = 100
        tolerance = 1e-6
        
        for iteration in range(max_iterations):
            old_creditor_strategy = self.creditor_strategy.copy()
            old_debtor_strategy = self.debtor_strategy.copy()
            
            # Update creditor strategy (best response to current debtor strategy)
            creditor_payoffs = np.zeros(len(self.emotions))
            for i, creditor_emotion in enumerate(self.emotions):
                for j, debtor_emotion in enumerate(self.emotions):
                    payoff = self.payoff_matrix[creditor_emotion][debtor_emotion][0]
                    creditor_payoffs[i] += self.debtor_strategy[j] * payoff
            
            # Convert to probability distribution (softmax with temperature)
            temperature = 0.1
            exp_payoffs = np.exp(creditor_payoffs / temperature)
            self.creditor_strategy = exp_payoffs / exp_payoffs.sum()
            
            # Update debtor strategy (best response to current creditor strategy)
            debtor_payoffs = np.zeros(len(self.emotions))
            for j, debtor_emotion in enumerate(self.emotions):
                for i, creditor_emotion in enumerate(self.emotions):
                    payoff = self.payoff_matrix[creditor_emotion][debtor_emotion][1]
                    debtor_payoffs[j] += self.creditor_strategy[i] * payoff
            
            exp_payoffs = np.exp(debtor_payoffs / temperature)
            self.debtor_strategy = exp_payoffs / exp_payoffs.sum()
            
            # Check convergence
            creditor_change = np.linalg.norm(self.creditor_strategy - old_creditor_strategy)
            debtor_change = np.linalg.norm(self.debtor_strategy - old_debtor_strategy)
            
            if creditor_change < tolerance and debtor_change < tolerance:
                break
        
        self.equilibrium_updates += 1
    
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
            'model_type': 'nash_equilibrium',
            'exploration_rate': self.exploration_rate,
            'equilibrium_updates': self.equilibrium_updates,
            'creditor_strategy': self.creditor_strategy.tolist(),
            'debtor_strategy': self.debtor_strategy.tolist(),
            'experience_count': len(self.experience_history),
            'strategy_history_length': len(self.strategy_history),
            'payoff_history_length': len(self.payoff_history)
        }
    
    def reset(self) -> None:
        """Reset model for new scenario"""
        # Keep learned strategies and payoff matrix but reset current tracking
        if hasattr(self, 'last_creditor_emotion'):
            delattr(self, 'last_creditor_emotion')
        if hasattr(self, 'last_debtor_emotion'):
            delattr(self, 'last_debtor_emotion')

class MinimaxEmotionModel(BaseEmotionModel):
    """Minimax decision making for emotion selection"""
    
    def __init__(self, depth: int = 2):
        super().__init__()
        
        self.emotions = ['happy', 'surprising', 'angry', 'sad', 'disgust', 'fear', 'neutral']
        self.depth = depth  # Look-ahead depth
        
        # Outcome prediction model
        self.outcome_probabilities = self._initialize_outcome_model()
        
        # Game tree statistics
        self.evaluations_made = 0
        self.decision_history = []
    
    def _initialize_outcome_model(self):
        """Initialize model for predicting negotiation outcomes"""
        
        # P(success | creditor_emotion, debtor_emotion)
        model = {}
        
        for creditor_emotion in self.emotions:
            model[creditor_emotion] = {}
            for debtor_emotion in self.emotions:
                # Base success probability on emotion compatibility
                if creditor_emotion == 'happy' and debtor_emotion in ['happy', 'sad', 'neutral']:
                    prob = 0.8
                elif creditor_emotion == 'angry' and debtor_emotion == 'fear':
                    prob = 0.6
                elif creditor_emotion == 'angry' and debtor_emotion == 'angry':
                    prob = 0.2
                elif creditor_emotion == 'neutral':
                    prob = 0.6
                elif creditor_emotion == debtor_emotion:
                    prob = 0.5  # Mirroring baseline
                else:
                    # Default based on emotion assertiveness difference
                    cred_assert = self._emotion_assertiveness(creditor_emotion)
                    debt_assert = self._emotion_assertiveness(debtor_emotion)
                    prob = 0.5 + 0.2 * (cred_assert - debt_assert)
                    prob = max(0.1, min(0.9, prob))
                
                model[creditor_emotion][debtor_emotion] = prob
        
        return model
    
    def _emotion_assertiveness(self, emotion: str) -> float:
        """Get assertiveness score for emotion"""
        scores = {
            'angry': 0.8,
            'disgust': 0.6,
            'neutral': 0.0,
            'surprising': 0.2,
            'happy': -0.2,
            'sad': -0.6,
            'fear': -0.8
        }
        return scores.get(emotion, 0.0)
    
    def select_emotion(self, model_state: Dict[str, Any]) -> Dict[str, Any]:
        """Select emotion using minimax algorithm"""
        
        # Get current debtor emotion
        debtor_emotion = model_state.get('debtor_emotion', 'neutral')
        
        # Handle encoded emotions
        if isinstance(debtor_emotion, str) and len(debtor_emotion) <= 2:
            emotion_map = {'A': 'angry', 'S': 'sad', 'D': 'disgust', 'F': 'fear', 
                          'J': 'happy', 'Su': 'surprising', 'N': 'neutral'}
            debtor_emotion = emotion_map.get(debtor_emotion, 'neutral')
        
        # Minimax decision
        best_emotion, best_value, decision_tree = self._minimax_decision(debtor_emotion, self.depth)
        
        self.evaluations_made += 1
        
        # Store decision information
        decision_info = {
            'debtor_emotion': debtor_emotion,
            'selected_emotion': best_emotion,
            'minimax_value': best_value,
            'depth_searched': self.depth,
            'decision_tree': decision_tree
        }
        self.decision_history.append(decision_info)
        
        return {
            'emotion': best_emotion,
            'confidence': abs(best_value),
            'minimax_value': best_value,
            'decision_tree': decision_tree,
            'debtor_emotion': debtor_emotion,
            'emotion_text': self._get_emotion_prompt(best_emotion),
            'temperature': 0.7,
            'reasoning': f"Minimax depth-{self.depth} search (value: {best_value:.3f})"
        }
    
    def _minimax_decision(self, debtor_emotion: str, depth: int) -> Tuple[str, float, Dict]:
        """Minimax algorithm implementation"""
        
        if depth == 0:
            return None, 0, {}
        
        best_emotion = None
        best_value = float('-inf')
        decision_tree = {}
        
        for creditor_emotion in self.emotions:
            # Evaluate this emotion choice
            value = self._evaluate_emotion_choice(creditor_emotion, debtor_emotion, depth)
            decision_tree[creditor_emotion] = value
            
            if value > best_value:
                best_value = value
                best_emotion = creditor_emotion
        
        return best_emotion, best_value, decision_tree
    
    def _evaluate_emotion_choice(self, creditor_emotion: str, debtor_emotion: str, depth: int) -> float:
        """Evaluate the value of choosing a specific creditor emotion"""
        
        # Base case: immediate outcome
        if depth == 1:
            success_prob = self.outcome_probabilities[creditor_emotion][debtor_emotion]
            # Value function: probability of success weighted by expected payoff
            return 2 * success_prob - 1  # Scale to [-1, 1]
        
        # Recursive case: consider debtor's best response
        min_value = float('inf')
        
        for next_debtor_emotion in self.emotions:
            # Assume debtor will choose emotion that minimizes creditor's advantage
            next_value = 0
            for next_creditor_emotion in self.emotions:
                outcome_value = self._evaluate_emotion_choice(next_creditor_emotion, next_debtor_emotion, depth - 1)
                next_value += outcome_value / len(self.emotions)  # Average over creditor choices
            
            if next_value < min_value:
                min_value = next_value
        
        # Current immediate value plus discounted future value
        immediate_value = 2 * self.outcome_probabilities[creditor_emotion][debtor_emotion] - 1
        future_value = 0.8 * min_value  # Discount factor
        
        return immediate_value + future_value
    
    def update_model(self, success: bool, final_days: int = None, target_days: int = None):
        """Update outcome prediction model based on results"""
        
        if not self.decision_history:
            return
        
        last_decision = self.decision_history[-1]
        creditor_emotion = last_decision['selected_emotion']
        debtor_emotion = last_decision['debtor_emotion']
        
        if creditor_emotion in self.outcome_probabilities and debtor_emotion in self.outcome_probabilities[creditor_emotion]:
            # Update success probability using exponential moving average
            current_prob = self.outcome_probabilities[creditor_emotion][debtor_emotion]
            actual_success = 1.0 if success else 0.0
            
            # Learning rate
            alpha = 0.1
            new_prob = (1 - alpha) * current_prob + alpha * actual_success
            
            self.outcome_probabilities[creditor_emotion][debtor_emotion] = new_prob
    
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
            'model_type': 'minimax',
            'search_depth': self.depth,
            'evaluations_made': self.evaluations_made,
            'decision_history_length': len(self.decision_history),
            'outcome_probabilities': self.outcome_probabilities
        }
    
    def reset(self) -> None:
        """Reset model for new scenario"""
        # Keep outcome model but reset decision tracking
        self.decision_history = []

class StrategicDominanceModel(BaseEmotionModel):
    """Strategic dominance based emotion selection"""
    
    def __init__(self):
        super().__init__()
        
        self.emotions = ['happy', 'surprising', 'angry', 'sad', 'disgust', 'fear', 'neutral']
        
        # Dominance relationships
        self.dominance_matrix = self._build_dominance_matrix()
        
        # Strategy statistics
        self.domination_history = []
        self.strategy_performance = {emotion: {'wins': 0, 'total': 0} for emotion in self.emotions}
    
    def _build_dominance_matrix(self):
        """Build matrix of strategic dominance relationships"""
        
        # dominance[A][B] = 1 if A dominates B, -1 if B dominates A, 0 if equal
        dominance = {}
        
        for emotion_a in self.emotions:
            dominance[emotion_a] = {}
            for emotion_b in self.emotions:
                dominance[emotion_a][emotion_b] = self._calculate_dominance(emotion_a, emotion_b)
        
        return dominance
    
    def _calculate_dominance(self, emotion_a: str, emotion_b: str) -> float:
        """Calculate dominance relationship between two emotions"""
        
        # Domain-specific dominance rules for debt negotiation
        rules = {
            # Angry emotions dominate fearful ones in assertiveness
            ('angry', 'fear'): 0.8,
            ('angry', 'sad'): 0.6,
            
            # Cooperative emotions work well together
            ('happy', 'neutral'): 0.3,
            ('neutral', 'happy'): 0.3,
            
            # Extreme emotions conflict
            ('angry', 'happy'): -0.4,
            ('disgust', 'happy'): -0.3,
            
            # Neutral is generally stable
            ('neutral', 'angry'): 0.2,
            ('neutral', 'disgust'): 0.2,
            
            # Fear can be exploited but may backfire
            ('fear', 'angry'): -0.7,
            ('fear', 'neutral'): -0.3,
            
            # Sad may evoke sympathy
            ('sad', 'angry'): 0.4,
            ('sad', 'neutral'): 0.1
        }
        
        # Check direct rules
        if (emotion_a, emotion_b) in rules:
            return rules[(emotion_a, emotion_b)]
        elif (emotion_b, emotion_a) in rules:
            return -rules[(emotion_b, emotion_a)]
        
        # Default: compare assertiveness
        assert_a = self._emotion_assertiveness(emotion_a)
        assert_b = self._emotion_assertiveness(emotion_b)
        
        return 0.3 * np.tanh(assert_a - assert_b)  # Bounded dominance
    
    def _emotion_assertiveness(self, emotion: str) -> float:
        """Get assertiveness score for emotion"""
        scores = {
            'angry': 0.8,
            'disgust': 0.6,
            'neutral': 0.0,
            'surprising': 0.2,
            'happy': -0.2,
            'sad': -0.6,
            'fear': -0.8
        }
        return scores.get(emotion, 0.0)
    
    def select_emotion(self, model_state: Dict[str, Any]) -> Dict[str, Any]:
        """Select emotion using strategic dominance analysis"""
        
        # Get debtor emotion
        debtor_emotion = model_state.get('debtor_emotion', 'neutral')
        
        # Handle encoded emotions
        if isinstance(debtor_emotion, str) and len(debtor_emotion) <= 2:
            emotion_map = {'A': 'angry', 'S': 'sad', 'D': 'disgust', 'F': 'fear', 
                          'J': 'happy', 'Su': 'surprising', 'N': 'neutral'}
            debtor_emotion = emotion_map.get(debtor_emotion, 'neutral')
        
        # Find emotions that dominate the debtor's current emotion
        dominance_scores = {}
        for creditor_emotion in self.emotions:
            dominance_scores[creditor_emotion] = self.dominance_matrix[creditor_emotion][debtor_emotion]
        
        # Select emotion with highest dominance, with some randomization
        emotions_by_dominance = sorted(dominance_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Top-k selection with weighted random choice
        top_k = 3
        top_emotions = emotions_by_dominance[:top_k]
        
        # Weighted selection (higher dominance = higher probability)
        weights = [max(0.1, score + 1) for _, score in top_emotions]  # Ensure positive weights
        total_weight = sum(weights)
        probabilities = [w / total_weight for w in weights]
        
        selected_idx = np.random.choice(len(top_emotions), p=probabilities)
        selected_emotion = top_emotions[selected_idx][0]
        dominance_value = top_emotions[selected_idx][1]
        
        # Record selection
        self.domination_history.append({
            'debtor_emotion': debtor_emotion,
            'selected_emotion': selected_emotion,
            'dominance_value': dominance_value,
            'all_dominance_scores': dominance_scores,
            'top_emotions': top_emotions
        })
        
        return {
            'emotion': selected_emotion,
            'confidence': (dominance_value + 1) / 2,  # Scale to [0, 1]
            'dominance_value': dominance_value,
            'dominance_scores': dominance_scores,
            'debtor_emotion': debtor_emotion,
            'top_alternatives': top_emotions,
            'emotion_text': self._get_emotion_prompt(selected_emotion),
            'temperature': 0.7,
            'reasoning': f"Strategic dominance (dominates {debtor_emotion} by {dominance_value:.2f})"
        }
    
    def update_model(self, success: bool, final_days: int = None, target_days: int = None):
        """Update dominance relationships based on outcomes"""
        
        if not self.domination_history:
            return
        
        last_decision = self.domination_history[-1]
        selected_emotion = last_decision['selected_emotion']
        debtor_emotion = last_decision['debtor_emotion']
        
        # Update strategy performance
        self.strategy_performance[selected_emotion]['total'] += 1
        if success:
            self.strategy_performance[selected_emotion]['wins'] += 1
        
        # Update dominance matrix based on outcome
        current_dominance = self.dominance_matrix[selected_emotion][debtor_emotion]
        
        # Learning rate
        alpha = 0.05
        
        if success:
            # Successful outcome reinforces dominance
            new_dominance = current_dominance + alpha * (1.0 - current_dominance)
        else:
            # Failed outcome reduces dominance
            new_dominance = current_dominance + alpha * (-1.0 - current_dominance)
        
        # Update with bounds
        self.dominance_matrix[selected_emotion][debtor_emotion] = max(-1, min(1, new_dominance))
    
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
            'model_type': 'strategic_dominance',
            'domination_history_length': len(self.domination_history),
            'strategy_performance': self.strategy_performance,
            'dominance_matrix': self.dominance_matrix
        }
    
    def reset(self) -> None:
        """Reset model for new scenario"""
        # Keep dominance matrix and strategy performance but reset current tracking
        pass  # Strategic dominance doesn't need per-scenario reset


def run_gametheory_experiment(
    scenarios: List[Dict[str, Any]],
    iterations: int = 5,
    model_creditor: str = "gpt-4o-mini",
    model_debtor: str = "gpt-4o-mini",
    debtor_emotion: str = "neutral",
    debtor_model_type: str = "vanilla",
    max_dialog_len: int = 30,
    out_dir: str = "results/gametheory"
) -> Dict[str, Any]:
    """Run Game Theory experiment - defaults to Nash Equilibrium"""
    
    # Create output directory
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    
    print(f"🎯 Running Game Theory Experiment - Nash Equilibrium")
    
    # Use Nash Equilibrium by default
    return run_nash_equilibrium_experiment(
        scenarios=scenarios,
        iterations=iterations,
        model_creditor=model_creditor,
        model_debtor=model_debtor,
        debtor_emotion=debtor_emotion,
        debtor_model_type=debtor_model_type,
        max_dialog_len=max_dialog_len,
        out_dir=out_dir
    )


def run_gametheory_experiment(
    scenarios: List[Dict[str, Any]],
    iterations: int = 5,
    model_creditor: str = "gpt-4o-mini",
    model_debtor: str = "gpt-4o-mini",
    debtor_emotion: str = "neutral",
    debtor_model_type: str = "vanilla",
    max_dialog_len: int = 30,
    out_dir: str = "results/nash_equilibrium"
) -> Dict[str, Any]:
    """Run Nash Equilibrium based emotion optimization experiment"""
    
    from llm.negotiator import DebtNegotiator
    
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    # Create Nash equilibrium model
    model = NashEquilibriumEmotionModel(exploration_rate=0.1)
    
    results = {
        'experiment_type': 'nash_equilibrium_emotion_optimization',
        'model_type': 'nash_equilibrium',
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
        print(f"\n🎯 Nash Equilibrium Iteration {iteration + 1}/{iterations}")
        
        for scenario_idx, scenario in enumerate(scenarios):
            print(f"  📋 Scenario {scenario_idx + 1}/{len(scenarios)}: ", end="")
            
            # Create negotiator with Nash equilibrium model
            negotiator = DebtNegotiator(
                config=scenario,
                emotion_model=model,
                model_creditor=model_creditor,
                model_debtor=model_debtor,
                debtor_emotion=debtor_emotion,
                debtor_model_type=debtor_model_type
            )
            
            # Store emotion choices for learning
            model.last_creditor_emotion = None
            model.last_debtor_emotion = debtor_emotion
            
            # Run negotiation
            result = negotiator.run_negotiation(max_dialog_len=max_dialog_len)
            all_negotiations.append(result)
            
            # Extract emotions from result
            if hasattr(model, 'strategy_history') and model.strategy_history:
                model.last_creditor_emotion = model.strategy_history[-1]['selected_emotion']
            
            # Update model based on result
            success = result.get('final_state') == 'accept'
            final_days = result.get('collection_days')
            target_days = int(scenario['seller']['target_price'])
            
            model.update_model(success, final_days, target_days)
            
            # Print result
            if success:
                print(f"✅ Success - {final_days} days (target: {target_days})")
            else:
                print(f"❌ Failed after {result.get('negotiation_rounds', 0)} rounds")
    
    # Calculate summary statistics
    successful_negotiations = [n for n in all_negotiations if n.get('final_state') == 'accept']
    success_rate = len(successful_negotiations) / len(all_negotiations) if all_negotiations else 0
    
    collection_rates = []
    target_days_list = [int(scenarios[i % len(scenarios)]['seller']['target_price']) for i in range(len(all_negotiations))]
    
    for i, negotiation in enumerate(all_negotiations):
        if negotiation.get('final_state') == 'accept' and negotiation.get('collection_days'):
            target = target_days_list[i]
            actual = negotiation['collection_days']
            rate = target / actual if actual > 0 else 0
            collection_rates.append(rate)
    
    # Enhance results with comprehensive statistical analysis
    results = enhance_results_with_statistics(
        {
            'summary_statistics': {
                'success_rate': success_rate,
                'total_negotiations': len(all_negotiations),
                'successful_negotiations': len(successful_negotiations),
                'failed_negotiations': len(all_negotiations) - len(successful_negotiations),
                'collection_rate': {
                    'mean': np.mean(collection_rates) if collection_rates else 0,
                    'std': np.std(collection_rates) if collection_rates else 0,
                    'min': min(collection_rates) if collection_rates else 0,
                    'max': max(collection_rates) if collection_rates else 0
                },
                'negotiation_rounds': {
                    'mean': np.mean([n.get('negotiation_rounds', 0) for n in all_negotiations]),
                    'std': np.std([n.get('negotiation_rounds', 0) for n in all_negotiations]),
                    'min': min([n.get('negotiation_rounds', 0) for n in all_negotiations]),
                    'max': max([n.get('negotiation_rounds', 0) for n in all_negotiations])
                }
            },
            'nash_statistics': {
                'equilibrium_updates': model.equilibrium_updates,
                'final_creditor_strategy': model.creditor_strategy.tolist(),
                'final_debtor_strategy': model.debtor_strategy.tolist(),
                'strategy_history': model.strategy_history,
                'payoff_matrix': {k: dict(v) for k, v in model.payoff_matrix.items()},
                'experience_count': len(model.experience_history)
            },
            'detailed_results': all_negotiations
        },
        all_negotiations,
        scenarios,
        method="bootstrap"
    )
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = f"{out_dir}/nash_results_{timestamp}.json"
    
    with open(result_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Print enhanced statistics with confidence intervals
    if 'statistical_analysis' in results:
        print(format_ci_results(results['statistical_analysis']))
    
    print(f"💾 Results saved to: {result_file}")
    
    return results


def run_minimax_experiment(
    scenarios: List[Dict[str, Any]],
    iterations: int = 5,
    model_creditor: str = "gpt-4o-mini",
    model_debtor: str = "gpt-4o-mini",
    debtor_emotion: str = "neutral",
    debtor_model_type: str = "vanilla",
    max_dialog_len: int = 30,
    depth: int = 2,
    out_dir: str = "results/minimax"
) -> Dict[str, Any]:
    """Run Minimax based emotion optimization experiment"""
    
    from llm.negotiator import DebtNegotiator
    
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    # Create Minimax model
    model = MinimaxEmotionModel(depth=depth)
    
    results = {
        'experiment_type': 'minimax_emotion_optimization',
        'model_type': 'minimax',
        'iterations': iterations,
        'scenarios_used': [s['id'] if 'id' in s else f"scenario_{i}" for i, s in enumerate(scenarios)],
        'config': {
            'model_creditor': model_creditor,
            'model_debtor': model_debtor,
            'debtor_emotion': debtor_emotion,
            'debtor_model_type': debtor_model_type,
            'max_dialog_len': max_dialog_len,
            'search_depth': depth
        },
        'detailed_results': []
    }
    
    all_negotiations = []
    
    for iteration in range(iterations):
        print(f"\n🎯 Minimax Iteration {iteration + 1}/{iterations}")
        
        for scenario_idx, scenario in enumerate(scenarios):
            print(f"  📋 Scenario {scenario_idx + 1}/{len(scenarios)}: ", end="")
            
            # Create negotiator with Minimax model
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
            
            # Update model based on result
            success = result.get('final_state') == 'accept'
            final_days = result.get('collection_days')
            target_days = int(scenario['seller']['target_price'])
            
            model.update_model(success, final_days, target_days)
            
            # Print result
            if success:
                print(f"✅ Success - {final_days} days (target: {target_days})")
            else:
                print(f"❌ Failed after {result.get('negotiation_rounds', 0)} rounds")
    
    # Calculate summary statistics (same as other models)
    successful_negotiations = [n for n in all_negotiations if n.get('final_state') == 'accept']
    success_rate = len(successful_negotiations) / len(all_negotiations) if all_negotiations else 0
    
    collection_rates = []
    target_days_list = [int(scenarios[i % len(scenarios)]['seller']['target_price']) for i in range(len(all_negotiations))]
    
    for i, negotiation in enumerate(all_negotiations):
        if negotiation.get('final_state') == 'accept' and negotiation.get('collection_days'):
            target = target_days_list[i]
            actual = negotiation['collection_days']
            rate = target / actual if actual > 0 else 0
            collection_rates.append(rate)
    
    # Enhance results with comprehensive statistical analysis
    results = enhance_results_with_statistics(
        {
            'summary_statistics': {
                'success_rate': success_rate,
                'total_negotiations': len(all_negotiations),
                'successful_negotiations': len(successful_negotiations),
                'failed_negotiations': len(all_negotiations) - len(successful_negotiations),
                'collection_rate': {
                    'mean': np.mean(collection_rates) if collection_rates else 0,
                    'std': np.std(collection_rates) if collection_rates else 0,
                    'min': min(collection_rates) if collection_rates else 0,
                    'max': max(collection_rates) if collection_rates else 0
                },
                'negotiation_rounds': {
                    'mean': np.mean([n.get('negotiation_rounds', 0) for n in all_negotiations]),
                    'std': np.std([n.get('negotiation_rounds', 0) for n in all_negotiations]),
                    'min': min([n.get('negotiation_rounds', 0) for n in all_negotiations]),
                    'max': max([n.get('negotiation_rounds', 0) for n in all_negotiations])
                }
            },
            'minimax_statistics': {
                'search_depth': model.depth,
                'evaluations_made': model.evaluations_made,
                'decision_history': model.decision_history,
                'outcome_probabilities': model.outcome_probabilities
            },
            'detailed_results': all_negotiations
        },
        all_negotiations,
        scenarios,
        method="bootstrap"
    )
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = f"{out_dir}/minimax_results_{timestamp}.json"
    
    with open(result_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n📊 Minimax Results: {success_rate:.1%} success rate, {len(all_negotiations)} negotiations")
    print(f"� Results saved to: {result_file}")
    
    return results


def run_strategic_dominance_experiment(
    scenarios: List[Dict[str, Any]],
    iterations: int = 5,
    model_creditor: str = "gpt-4o-mini",
    model_debtor: str = "gpt-4o-mini",
    debtor_emotion: str = "neutral",
    debtor_model_type: str = "vanilla",
    max_dialog_len: int = 30,
    out_dir: str = "results/strategic_dominance"
) -> Dict[str, Any]:
    """Run Strategic Dominance based emotion optimization experiment"""
    
    from llm.negotiator import DebtNegotiator
    
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    # Create Strategic Dominance model
    model = StrategicDominanceModel()
    
    results = {
        'experiment_type': 'strategic_dominance_emotion_optimization',
        'model_type': 'strategic_dominance',
        'iterations': iterations,
        'scenarios_used': [s['id'] if 'id' in s else f"scenario_{i}" for i, s in enumerate(scenarios)],
        'config': {
            'model_creditor': model_creditor,
            'model_debtor': model_debtor,
            'debtor_emotion': debtor_emotion,
            'debtor_model_type': debtor_model_type,
            'max_dialog_len': max_dialog_len
        },
        'detailed_results': []
    }
    
    all_negotiations = []
    
    for iteration in range(iterations):
        print(f"\n🎯 Strategic Dominance Iteration {iteration + 1}/{iterations}")
        
        for scenario_idx, scenario in enumerate(scenarios):
            print(f"  📋 Scenario {scenario_idx + 1}/{len(scenarios)}: ", end="")
            
            # Create negotiator with Strategic Dominance model
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
            
            # Update model based on result
            success = result.get('final_state') == 'accept'
            final_days = result.get('collection_days')
            target_days = int(scenario['seller']['target_price'])
            
            model.update_model(success, final_days, target_days)
            
            # Print result
            if success:
                print(f"✅ Success - {final_days} days (target: {target_days})")
            else:
                print(f"❌ Failed after {result.get('negotiation_rounds', 0)} rounds")
    
    # Calculate summary statistics
    successful_negotiations = [n for n in all_negotiations if n.get('final_state') == 'accept']
    success_rate = len(successful_negotiations) / len(all_negotiations) if all_negotiations else 0
    
    collection_rates = []
    target_days_list = [int(scenarios[i % len(scenarios)]['seller']['target_price']) for i in range(len(all_negotiations))]
    
    for i, negotiation in enumerate(all_negotiations):
        if negotiation.get('final_state') == 'accept' and negotiation.get('collection_days'):
            target = target_days_list[i]
            actual = negotiation['collection_days']
            rate = target / actual if actual > 0 else 0
            collection_rates.append(rate)
    
    # Enhance results with comprehensive statistical analysis
    results = enhance_results_with_statistics(
        {
            'summary_statistics': {
                'success_rate': success_rate,
                'total_negotiations': len(all_negotiations),
                'successful_negotiations': len(successful_negotiations),
                'failed_negotiations': len(all_negotiations) - len(successful_negotiations),
                'collection_rate': {
                    'mean': np.mean(collection_rates) if collection_rates else 0,
                    'std': np.std(collection_rates) if collection_rates else 0,
                    'min': min(collection_rates) if collection_rates else 0,
                    'max': max(collection_rates) if collection_rates else 0
                },
                'negotiation_rounds': {
                    'mean': np.mean([n.get('negotiation_rounds', 0) for n in all_negotiations]),
                    'std': np.std([n.get('negotiation_rounds', 0) for n in all_negotiations]),
                    'min': min([n.get('negotiation_rounds', 0) for n in all_negotiations]),
                    'max': max([n.get('negotiation_rounds', 0) for n in all_negotiations])
                }
            },
            'dominance_statistics': {
                'strategy_performance': model.strategy_performance,
                'dominance_matrix': model.dominance_matrix,
                'domination_history': model.domination_history
            },
            'detailed_results': all_negotiations
        },
        all_negotiations,
        scenarios,
        method="bootstrap"
    )
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = f"{out_dir}/dominance_results_{timestamp}.json"
    
    with open(result_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n📊 Strategic Dominance Results: {success_rate:.1%} success rate, {len(all_negotiations)} negotiations")
    print(f"� Results saved to: {result_file}")
    
    return results