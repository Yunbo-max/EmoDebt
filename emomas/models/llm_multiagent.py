"""
Bayesian Multi-Agent System for Optimizing Emotional Transitions
Combines Game Theory, Online RL, and Emotional Coherence agents
using Bayesian probability to optimize emotional state transitions in negotiations
"""

import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
from models.base_model import BaseEmotionModel
import json
from datetime import datetime
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')
from utils.statistical_analysis import enhance_results_with_statistics, analyze_negotiation_results, format_ci_results

# Emotion definitions matching your paper
EMOTIONS = ['J', 'S', 'A', 'F', 'Su', 'D', 'N']  # Joy, Sadness, Anger, Fear, Surprise, Disgust, Neutral
EMOTION_NAMES = {
    'J': 'Joy', 'S': 'Sadness', 'A': 'Anger', 'F': 'Fear', 
    'Su': 'Surprise', 'D': 'Disgust', 'N': 'Neutral'
}
N_EMOTIONS = len(EMOTIONS)
EMOTION_TO_IDX = {e: i for i, e in enumerate(EMOTIONS)}
IDX_TO_EMOTION = {i: e for i, e in enumerate(EMOTIONS)}

# Payoff matrix from your paper Table 2
PAYOFF_MATRIX = np.array([
    # J    S    A    F    Su   D    N
    [(4,4), (2,3), (1,2), (2,1), (3,3), (2,2), (3,3)],  # J
    [(3,2), (3,3), (1,2), (2,1), (2,2), (1,1), (2,3)],  # S
    [(2,1), (2,1), (1,1), (1,0), (1,2), (0,1), (1,2)],  # A
    [(1,2), (1,2), (0,1), (2,2), (1,2), (0,1), (2,3)],  # F
    [(3,3), (2,2), (2,1), (2,1), (4,4), (1,2), (3,3)],  # Su
    [(2,2), (1,1), (1,0), (1,0), (2,1), (2,2), (2,2)],  # D
    [(3,3), (2,3), (2,1), (3,2), (3,3), (2,2), (3,3)],  # N
])

# # HMM parameters from your paper Table 1 (commented out as not used)
# TRANSITION_PROBS = np.array([
#     # J    S    A    F    Su   D    N
#     [0.50, 0.10, 0.05, 0.05, 0.20, 0.05, 0.05],  # J
#     [0.20, 0.40, 0.10, 0.10, 0.05, 0.10, 0.05],  # S
#     [0.10, 0.20, 0.40, 0.10, 0.05, 0.10, 0.05],  # A
#     [0.10, 0.20, 0.10, 0.40, 0.05, 0.10, 0.05],  # F
#     [0.30, 0.05, 0.05, 0.05, 0.50, 0.05, 0.05],  # Su
#     [0.10, 0.20, 0.10, 0.10, 0.05, 0.40, 0.05],  # D
#     [0.20, 0.10, 0.05, 0.05, 0.20, 0.05, 0.35],  # N
# ])

# # Emission probabilities: P(client_emotion | agent_emotion)
# EMISSION_PROBS = np.array([
#     # J    S    A    F    Su   D    N
#     [0.60, 0.05, 0.05, 0.05, 0.10, 0.05, 0.10],  # J
#     [0.05, 0.50, 0.20, 0.10, 0.05, 0.05, 0.05],  # S
#     [0.05, 0.20, 0.50, 0.10, 0.05, 0.05, 0.05],  # A
#     [0.05, 0.20, 0.10, 0.50, 0.05, 0.05, 0.05],  # F
#     [0.10, 0.05, 0.05, 0.05, 0.60, 0.05, 0.10],  # Su
#     [0.05, 0.10, 0.20, 0.10, 0.05, 0.50, 0.05],  # D
#     [0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.40],  # N
# ])

# # Hidden states for HMM (commented out as not used)
# HIDDEN_STATES = ['Cooperative', 'Confrontational', 'Distressed', 'Strategic']
# N_HIDDEN_STATES = len(HIDDEN_STATES)
# STATE_TO_IDX = {s: i for i, s in enumerate(HIDDEN_STATES)}

@dataclass
class AgentPrediction:
    """Prediction from a single agent for emotional transition"""
    agent_name: str
    target_emotion: str  # Emotion to transition TO (in EMOTIONS format)
    confidence: float
    reasoning: str
    transition_value: float  # Value of this specific transition
    context_features: Dict[str, Any]

@dataclass
class TransitionContext:
    """Context for emotional transition optimization"""
    current_emotion: str  # Current agent emotion
    debtor_emotion: str   # Current debtor emotion
    negotiation_phase: str  # 'early', 'middle', 'late', 'crisis'
    round_number: int
    emotional_history: List[str]  # List of emotion symbols
    debt_amount: float
    recent_success_rate: float
    gap_size: float  # Days difference in negotiation
    
    def to_features(self) -> Dict[str, Any]:
        """Convert context to feature vector for learning"""
        return {
            'current_emotion': self.current_emotion,
            'debtor_emotion': self.debtor_emotion,
            'phase': self.negotiation_phase,
            'round': self.round_number,
            'history_len': len(self.emotional_history),
            'debt_large': self.debt_amount > 10000,
            'recent_success': self.recent_success_rate,
            'gap_large': self.gap_size > 20,
        }


class TransitionGameTheoryAgent:
    """Game Theory agent implementing pure WSLS strategy with payoff maximization"""
    
    def __init__(self, exploration_rate: float = 0.1):
        self.name = "GameTheory"
        self.exploration_rate = exploration_rate
        self.last_debtor_emotion = None
        self.last_agent_emotion = None
        self.last_successful = True  # Win-Stay, Lose-Shift tracking
        
        # Positive and negative emotion sets for WSLS
        self.positive_emotions = {'J', 'N', 'Su'}  # Joy, Neutral, Surprise
        self.negative_emotions = {'A', 'D', 'F'}   # Anger, Disgust, Fear
        
    def get_payoff(self, debtor_emotion: str, agent_emotion: str) -> float:
        """Get creditor payoff from payoff matrix"""
        d_idx = EMOTION_TO_IDX[debtor_emotion]
        a_idx = EMOTION_TO_IDX[agent_emotion]
        return PAYOFF_MATRIX[d_idx, a_idx][1]  # Second element is creditor payoff
    
    def wsls_strategy(self, debtor_emotion: str, context: TransitionContext) -> str:
        """Pure Win-Stay, Lose-Shift strategy"""
        if self.last_debtor_emotion is None or self.last_agent_emotion is None:
            # First move: start with neutral
            return 'N'
        
        # Check if last move was successful (positive payoff)
        last_payoff = self.get_payoff(self.last_debtor_emotion, self.last_agent_emotion)
        was_successful = last_payoff >= 2.0  # Threshold for "win"
        
        if was_successful:
            # WIN: Stay with similar strategy
            if debtor_emotion in self.positive_emotions:
                # Positive debtor emotion: maintain
                return self.last_agent_emotion
            else:
                # Negative debtor emotion: cautious neutral
                return 'N'
        else:
            # LOSE: Shift strategy
            if debtor_emotion in self.negative_emotions:
                # Facing negative emotion: neutral de-escalation
                return 'N'
            elif debtor_emotion in self.positive_emotions:
                # Facing positive emotion: reciprocate
                return debtor_emotion
            else:
                # Sadness or other: neutral response
                return 'N'
    
    def predict(self, context: TransitionContext) -> AgentPrediction:
        """Predict optimal emotional transition using pure game theory"""
        debtor_emotion = context.debtor_emotion
        current_emotion = context.current_emotion

        # Exploration: With small probability, explore random emotion
        if np.random.random() < self.exploration_rate:
            random_emotion = np.random.choice(EMOTIONS)
            reasoning = f"EXPLORATION: Random choice {random_emotion}"
            return AgentPrediction(
                agent_name=self.name,
                target_emotion=random_emotion,
                confidence=0.3,
                reasoning=reasoning,
                transition_value=0.5,
                context_features=context.to_features()
            )
        
        # Get pure WSLS strategy
        wsls_emotion = self.wsls_strategy(debtor_emotion, context)
        
        # Calculate payoffs for all possible emotions
        payoffs = {}
        for emotion in EMOTIONS:
            payoff = self.get_payoff(debtor_emotion, emotion)
            
            # PURE WSLS: Only favor the WSLS-recommended emotion
            if emotion == wsls_emotion:
                payoff *= 1.3  # Favor WSLS-recommended emotion
            
            payoffs[emotion] = payoff
        
        # Select emotion with maximum payoff
        best_emotion = max(payoffs, key=payoffs.get)
        max_payoff = payoffs[best_emotion]
        
        # Normalize confidence
        total_payoff = sum(payoffs.values())
        confidence = max_payoff / total_payoff if total_payoff > 0 else 0.5
        
        # Store for next round's WSLS
        self.last_debtor_emotion = debtor_emotion
        self.last_agent_emotion = best_emotion
        
        reasoning = f"Pure WSLS: {current_emotion} → {best_emotion} (payoff: {max_payoff:.2f})"
        reasoning += f" | Facing {debtor_emotion}, {'Win-Stay' if self.last_successful else 'Lose-Shift'}"
        
        return AgentPrediction(
            agent_name=self.name,
            target_emotion=best_emotion,
            confidence=confidence,
            reasoning=reasoning,
            transition_value=float(max_payoff),
            context_features=context.to_features()
        )
    
    def update_success(self, success: bool):
        """Update WSLS success tracking"""
        self.last_successful = success




class OnlineRLTransitionModel:
    """
    Online Reinforcement Learning agent for emotion transitions
    Learns directly from negotiation trajectories
    """
    
    def __init__(self, exploration_rate: float = 0.2, learning_rate: float = 0.1):
        self.name = "OnlineRL"
        self.exploration_rate = exploration_rate
        self.learning_rate = learning_rate
        
        # Q-table: (current_emotion, debtor_emotion, phase) -> emotion values
        self.q_table = defaultdict(lambda: np.zeros(N_EMOTIONS))
        
        # Context features for generalization
        # self.feature_weights = defaultdict(lambda: np.random.randn(10) * 0.1)
        # In __init__ method of OnlineRLTransitionModel:
        # Context features for generalization - KEEP at 10
        self.feature_weights = defaultdict(lambda: np.random.randn(10) * 0.1)
                
        # Success tracking
        self.success_counts = defaultdict(int)
        self.total_counts = defaultdict(int)
        
        # Recent experience replay buffer
        self.experience_buffer = []
        self.buffer_size = 100
        
        # Adaptive exploration
        self.min_exploration = 0.05
        self.max_exploration = 0.3
        
    def get_state_key(self, context: TransitionContext) -> str:
        """Create state key from context"""
        key_parts = [
            context.current_emotion,
            context.debtor_emotion,
            context.negotiation_phase,
            'large_gap' if context.gap_size > 20 else 'small_gap'
        ]
        return "_".join(key_parts)
    
    def get_state_features(self, context: TransitionContext) -> np.ndarray:
        """Extract features from context for generalization"""
        features = np.zeros(10)  # Keep it at 10, more reasonable
        
        # 1. Emotion pair encoding (3 features)
        # Instead of 14 one-hot, encode relationship between emotions
        curr_idx = EMOTION_TO_IDX[context.current_emotion]
        debtor_idx = EMOTION_TO_IDX[context.debtor_emotion]
        
        # Feature 0: Are emotions the same?
        features[0] = 1.0 if curr_idx == debtor_idx else 0.0
        
        # Feature 1: Is current emotion positive? (J, Su, N)
        positive_emotions = {'J', 'Su', 'N'}
        features[1] = 1.0 if context.current_emotion in positive_emotions else 0.0
        
        # Feature 2: Is debtor emotion positive?
        features[2] = 1.0 if context.debtor_emotion in positive_emotions else 0.0
        
        # 2. Negotiation context (4 features)
        # Feature 3: Phase (early=0.0, middle=0.5, late/crisis=1.0)
        phase_encoding = {'early': 0.0, 'middle': 0.5, 'late': 1.0, 'crisis': 1.0}
        features[3] = phase_encoding.get(context.negotiation_phase, 0.5)
        
        # Feature 4: Round progress (0.0 to 1.0)
        features[4] = min(context.round_number / 20.0, 1.0)
        
        # Feature 5: Gap size normalized
        features[5] = min(context.gap_size / 100.0, 1.0)
        
        # Feature 6: Recent success rate
        features[6] = context.recent_success_rate
        
        # 3. Strategic context (3 features)
        # Feature 7: Is gap large?
        features[7] = 1.0 if context.gap_size > 20 else 0.0
        
        # Feature 8: Is debt large?
        features[8] = 1.0 if context.debt_amount > 10000 else 0.0
        
        # Feature 9: History length normalized
        features[9] = min(len(context.emotional_history) / 10.0, 1.0)
        
        return features
        
      
    def predict(self, context: TransitionContext) -> AgentPrediction:
        """Predict next emotion using online RL"""
        
        # 1. EXPLORATION: Random choice
        if np.random.random() < self.exploration_rate:
            random_emotion = np.random.choice(EMOTIONS)
            reasoning = f"RL EXPLORATION: Random choice {random_emotion}"
            return AgentPrediction(
                agent_name=self.name,
                target_emotion=random_emotion,
                confidence=0.3,
                reasoning=reasoning,
                transition_value=0.5,
                context_features=context.to_features()
            )
        
        # 2. Get Q-values for current state
        state_key = self.get_state_key(context)
        q_values = self.q_table[state_key].copy()
        
        # 3. Add feature-based generalization
        features = self.get_state_features(context)
        for e_idx in range(N_EMOTIONS):
            if state_key in self.feature_weights:
                feature_bonus = np.dot(features, self.feature_weights[state_key])
                q_values[e_idx] += self.learning_rate * feature_bonus
        
        # 4. Softmax selection with temperature
        temperature = 0.1  # Greedy but stochastic
        exp_q = np.exp(q_values / temperature)
        probs = exp_q / exp_q.sum()
        
        selected_idx = np.random.choice(N_EMOTIONS, p=probs)
        selected_emotion = EMOTIONS[selected_idx]
        selected_q = q_values[selected_idx]
        
        # 5. IMPROVED Confidence calculation
        # Use softmax probability of selected action
        confidence = probs[selected_idx]
        
        # Alternative: Gap between top 2 Q-values
        # sorted_q = np.sort(q_values)[::-1]
        # if len(sorted_q) > 1 and sorted_q[0] > 0:
        #     confidence = (sorted_q[0] - sorted_q[1]) / (sorted_q[0] + 1e-10)
        # else:
        #     confidence = 0.5  # Default confidence
        
        # 6. Store for learning
        self.experience_buffer.append({
            'context': context,
            'selected_emotion': selected_emotion,
            'selected_idx': selected_idx,
            'q_values': q_values.copy(),
            'probs': probs.copy(),  # Store probabilities too
            'state_key': state_key,
            'features': features.copy()
        })
        
        if len(self.experience_buffer) > self.buffer_size:
            self.experience_buffer.pop(0)
        
        reasoning = f"RL: {context.current_emotion}→{selected_emotion} facing {context.debtor_emotion}"
        reasoning += f" | Q-value: {selected_q:.3f}, Prob: {probs[selected_idx]:.3f}"
        reasoning += f" | Phase: {context.negotiation_phase}"
        
        return AgentPrediction(
            agent_name=self.name,
            target_emotion=selected_emotion,
            confidence=confidence,
            reasoning=reasoning,
            transition_value=float(selected_q),
            context_features=context.to_features()
        )
    
    def learn_from_trajectory(self, emotion_trajectory: List[str],
                            context_history: List[Dict[str, Any]],
                            success: bool, macro_reward: float):
        """
        Online learning from entire negotiation trajectory
        Uses experience replay and TD learning
        """
        if len(emotion_trajectory) < 2 or len(context_history) < 2:
            return
        
        print(f"  🎯 OnlineRL: Learning from {len(emotion_trajectory)}-round trajectory")
        print(f"    Success: {success}, Macro reward: {macro_reward:.3f}")
        
        # Calculate discounted rewards for each transition
        rewards = self._calculate_discounted_rewards(
            emotion_trajectory, context_history, success, macro_reward
        )
        
        # Q-learning updates for each transition
        for t in range(len(emotion_trajectory) - 1):
            current_emotion = emotion_trajectory[t]
            next_emotion = emotion_trajectory[t + 1]
            
            if t < len(context_history):
                context_data = context_history[t]
                
                # Reconstruct context
                context = TransitionContext(
                    current_emotion=current_emotion,
                    debtor_emotion=context_data.get('debtor_emotion', 'N'),
                    negotiation_phase=context_data.get('phase', 'middle'),
                    round_number=context_data.get('round', t + 1),
                    emotional_history=emotion_trajectory[:t+1] if t > 0 else [],
                    debt_amount=context_data.get('debt_amount', 0),
                    recent_success_rate=context_data.get('recent_success', 0.5),
                    gap_size=context_data.get('gap_size', 0)
                )
                
                state_key = self.get_state_key(context)
                next_idx = EMOTION_TO_IDX[next_emotion]
                reward = rewards[t]
                
                # Q-learning update: Q(s,a) ← Q(s,a) + α[r + γ·max_a'Q(s',a') - Q(s,a)]
                old_q = self.q_table[state_key][next_idx]
                
                # Estimate next state value (simplified)
                next_state_value = 0.0
                if t + 1 < len(context_history):
                    next_context_data = context_history[t + 1]
                    next_context = TransitionContext(
                        current_emotion=next_emotion,
                        debtor_emotion=next_context_data.get('debtor_emotion', 'N'),
                        negotiation_phase=next_context_data.get('phase', 'middle'),
                        round_number=next_context_data.get('round', t + 2),
                        emotional_history=emotion_trajectory[:t+2],
                        debt_amount=next_context_data.get('debt_amount', 0),
                        recent_success_rate=next_context_data.get('recent_success', 0.5),
                        gap_size=next_context_data.get('gap_size', 0)
                    )
                    next_state_key = self.get_state_key(next_context)
                    next_state_value = self.q_table[next_state_key].max()
                
                # TD target
                gamma = 0.9  # Discount factor
                td_target = reward + gamma * next_state_value
                td_error = td_target - old_q
                
                # Update Q-value
                self.q_table[state_key][next_idx] += self.learning_rate * td_error
                
                # Update success statistics
                self.total_counts[state_key] += 1
                if success:
                    self.success_counts[state_key] += 1
        
        # Experience replay (learn from recent experiences)
        self._experience_replay(success, macro_reward)
        
        # Adjust exploration rate
        self._adjust_exploration(success)
        
        # Print learning statistics
        self._print_learning_stats()
    
    def _calculate_discounted_rewards(self, emotion_trajectory: List[str],
                                    context_history: List[Dict[str, Any]],
                                    success: bool, macro_reward: float) -> List[float]:
        """Calculate discounted rewards for each transition"""
        rewards = []
        n_transitions = len(emotion_trajectory) - 1
        
        if success:
            # Successful negotiation: distribute reward
            base_reward = macro_reward / n_transitions
            for i in range(n_transitions):
                # Early transitions get more weight (they set the tone)
                discount = 0.9 ** i
                rewards.append(base_reward * discount)
        else:
            # Failed negotiation: small negative rewards
            for i in range(n_transitions):
                rewards.append(-0.1)
        
        return rewards
    
    def _experience_replay(self, success: bool, macro_reward: float):
        """Learn from experience replay buffer"""
        if not self.experience_buffer:
            return
        
        # Sample recent experiences
        replay_size = min(10, len(self.experience_buffer))
        replay_samples = np.random.choice(
            len(self.experience_buffer), 
            size=replay_size, 
            replace=False
        )
        
        for idx in replay_samples:
            experience = self.experience_buffer[idx]
            state_key = experience['state_key']
            selected_idx = experience['selected_idx']
            
            # Update based on outcome
            if success:
                self.q_table[state_key][selected_idx] += self.learning_rate * macro_reward * 0.1
            else:
                self.q_table[state_key][selected_idx] -= self.learning_rate * 0.05
    
    def _adjust_exploration(self, success: bool):
        """Adjust exploration rate adaptively"""
        if success:
            # Successful: reduce exploration
            self.exploration_rate *= 0.95
        else:
            # Unsuccessful: increase exploration
            self.exploration_rate *= 1.05
        
        # Clip to bounds
        self.exploration_rate = np.clip(
            self.exploration_rate, 
            self.min_exploration, 
            self.max_exploration
        )
        
        print(f"    Updated exploration rate: {self.exploration_rate:.3f}")
    
    def _print_learning_stats(self):
        """Print learning statistics"""
        if self.total_counts:
            total_states = len(self.total_counts)
            avg_samples = np.mean(list(self.total_counts.values()))
            success_rate = np.mean([
                self.success_counts.get(k, 0) / v 
                for k, v in self.total_counts.items() 
                if v > 0
            ])
            
            print(f"    RL Stats: {total_states} states, avg {avg_samples:.1f} samples")
            print(f"    Success rate: {success_rate:.1%}")
    
    def get_top_policies(self, top_n: int = 5) -> List[Dict]:
        """Get top learned policies"""
        policies = []
        
        for state_key, q_values in self.q_table.items():
            if self.total_counts.get(state_key, 0) > 3:  # Enough samples
                best_idx = np.argmax(q_values)
                best_emotion = EMOTIONS[best_idx]
                best_q = q_values[best_idx]
                
                # Calculate success rate
                success_rate = (
                    self.success_counts.get(state_key, 0) / 
                    self.total_counts.get(state_key, 1)
                )
                
                policies.append({
                    'state': state_key,
                    'recommended': best_emotion,
                    'q_value': float(best_q),
                    'success_rate': float(success_rate),
                    'samples': self.total_counts.get(state_key, 0)
                })
        
        # Sort by Q-value * success_rate
        policies.sort(key=lambda x: x['q_value'] * (1 + x['success_rate']), reverse=True)
        return policies[:top_n]




class EmotionalCoherenceAgent:
    """LLM-based agent for psychological plausibility and credible emotional transitions"""
    
    def __init__(self, llm_client=None):
        self.name = "EmotionalCoherence"
        self.llm_client = llm_client
        
        # No hard-coded rules - relying entirely on LLM reasoning
        self.coherence_threshold = 0.6  # Minimum plausibility score
        self.min_transition_confidence = 0.1  # Minimum confidence for recommendations
        
        # Track historical transitions for learning
        self.transition_history = []
        self.successful_transitions = []
    
    def _create_coherence_prompt(self, context: TransitionContext) -> str:
        """Create LLM prompt for assessing emotional transition plausibility"""
        
        prompt = f"""You are an emotional intelligence expert analyzing emotional transitions in debt collection.

Current Situation:
- Your current emotion: {context.current_emotion}
- Debtor's emotion: {context.debtor_emotion}
- Negotiation phase: {context.negotiation_phase}
- Round: {context.round_number}
- Gap: {context.gap_size} days
- Debt: ${context.debt_amount:,.0f}

Analyze each possible next emotion (J, S, A, F, Su, D, N) from a psychological perspective.

For each emotion, provide:
1. Plausibility (0-1): How natural does this transition feel?
2. Appropriateness (0-1): How suitable for this negotiation phase?
3. Strategic value (0-1): How likely to help collect the debt?
4. Brief reasoning: Psychological rationale

Return JSON array with 7 items - one for each emotion."""        
        return prompt
    
    def _parse_llm_response(self, response: str) -> List[Dict[str, Any]]:
        """Parse LLM response into structured assessments"""
        try:
            # Extract JSON from response
            import json
            import re
            
            # Try to find JSON array in response
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                assessments = json.loads(json_match.group(0))
                
                # Validate structure and length
                valid_assessments = []
                for item in assessments:
                    if isinstance(item, dict) and 'emotion' in item:
                        valid_item = {
                            'emotion': str(item['emotion']),
                            'plausibility': float(item.get('plausibility', 0.5)),
                            'appropriateness': float(item.get('appropriateness', 0.5)),
                            'strategic_value': float(item.get('strategic_value', 0.5)),
                            'reasoning': str(item.get('reasoning', 'No reasoning provided'))
                        }
                        valid_assessments.append(valid_item)
                
                # Verify we have assessments for all 7 emotions
                if len(valid_assessments) == 7:
                    return valid_assessments
                else:
                    print(f"⚠️ LLM returned {len(valid_assessments)} assessments, expected 7")
        except Exception as e:
            print(f"⚠️ Error parsing LLM response: {e}")
        
        # Fallback: Generate default assessments
        return self._generate_fallback_assessments()
    
    def _generate_fallback_assessments(self) -> List[Dict[str, Any]]:
        """Generate fallback assessments when LLM fails"""
        assessments = []
        for emotion in EMOTIONS:
            assessments.append({
                'emotion': emotion,
                'plausibility': 0.5,
                'appropriateness': 0.5,
                'strategic_value': 0.5,
                'reasoning': f"Fallback assessment for {emotion}"
            })
        return assessments
    
    def _calculate_coherence_score(self, assessment: Dict[str, Any], context: TransitionContext) -> float:
        """Calculate overall coherence score from assessment components"""
        # Simple weighted average - let LLM do the reasoning
        plausibility_weight = 0.4
        appropriateness_weight = 0.3
        strategic_weight = 0.3
        
        score = (
            assessment['plausibility'] * plausibility_weight +
            assessment['appropriateness'] * appropriateness_weight +
            assessment['strategic_value'] * strategic_weight
        )
        
        # Apply coherence threshold
        if score < self.coherence_threshold:
            score *= 0.7  # Penalize low-coherence transitions
        
        return min(max(score, 0.0), 1.0)
    
    def predict(self, context: TransitionContext) -> AgentPrediction:
        """Predict next emotion based on psychological coherence using LLM reasoning"""
        
        emotion_scores = {}
        
        # Method 1: Use LLM if available (PRIMARY METHOD)
        if self.llm_client is not None:
            try:
                # Create prompt
                prompt = self._create_coherence_prompt(context)
                
                # Get LLM response
                response = self.llm_client.generate(
                    prompt=prompt,
                    temperature=0.7,  # Medium temperature for creative reasoning
                    max_tokens=800  # More tokens for detailed reasoning
                )
                
                # Parse assessments
                assessments = self._parse_llm_response(response)
                
                # Calculate scores for each emotion based on LLM reasoning
                for assessment in assessments:
                    emotion = assessment['emotion']
                    if emotion in EMOTIONS:
                        score = self._calculate_coherence_score(assessment, context)
                        emotion_scores[emotion] = {
                            'score': score,
                            'reasoning': assessment['reasoning'],
                            'plausibility': assessment['plausibility'],
                            'appropriateness': assessment['appropriateness'],
                            'strategic_value': assessment['strategic_value']
                        }
                
                print(f"        🤔 EmotionalCoherence LLM analysis complete")
                
            except Exception as e:
                print(f"⚠️ LLM coherence check failed: {e}")
                emotion_scores = self._fallback_prediction(context)
        
        # Method 2: Fallback if LLM unavailable
        else:
            emotion_scores = self._fallback_prediction(context)
        
        # **CRITICAL FIX: Enable exploration instead of fixed selection**
        # Convert scores to probabilities for exploration
        emotions = list(emotion_scores.keys())
        scores = [emotion_scores[e]['score'] for e in emotions]
        print(scores)

        
        
        # Apply softmax for exploration
        import math
        
        # Temperature for exploration (higher = more exploration)
        temperature = 1.0  # Adjustable parameter
        
        # Calculate softmax probabilities
        exp_scores = [math.exp(score / temperature) for score in scores]
        sum_exp = sum(exp_scores)
        probabilities = [exp / sum_exp for exp in exp_scores]
        
        # **EXPLORATION: Sample from distribution instead of taking max**
        import random
        
        # With 80% probability, choose top emotion (exploitation)
        # With 20% probability, sample from distribution (exploration)
        if random.random() < 0.8:
            # Exploitation: Choose best emotion
            best_idx = scores.index(max(scores))
            selected_emotion = emotions[best_idx]
            selection_type = "exploitation"
        else:
            # Exploration: Sample from probability distribution
            selected_emotion = random.choices(emotions, weights=probabilities, k=1)[0]
            selection_type = "exploration"
        
        # Get reasoning and score for selected emotion
        selected_score = emotion_scores[selected_emotion]['score']
        reasoning = emotion_scores[selected_emotion]['reasoning']
        
        # Calculate confidence based on score distribution
        if len(scores) > 1:
            sorted_scores = sorted(scores, reverse=True)
            if sorted_scores[0] > 0:
                # Confidence = how much better than second best
                confidence = (sorted_scores[0] - sorted_scores[1]) / sorted_scores[0]
                confidence = max(confidence, self.min_transition_confidence)
            else:
                confidence = self.min_transition_confidence
        else:
            confidence = selected_score
        
        # Add exploration indicator to reasoning
        full_reasoning = f"{reasoning} [{selection_type.upper()}] (Score: {selected_score:.2f})"
        
        return AgentPrediction(
            agent_name=self.name,
            target_emotion=selected_emotion,
            confidence=confidence,
            reasoning=full_reasoning,
            transition_value=float(selected_score),
            context_features=context.to_features()
        )
    
    def _fallback_prediction(self, context: TransitionContext) -> Dict[str, Dict[str, Any]]:
        """Fallback prediction method without LLM - MINIMAL RULES"""
        
        emotion_scores = {}
        
        # Minimal baseline - let LLM handle reasoning in normal operation
        for emotion in EMOTIONS:
            # Start with neutral baseline
            score = 0.5
            reasoning = "Fallback: Neutral assessment"
            
            emotion_scores[emotion] = {
                'score': score,
                'reasoning': reasoning
            }
        
        return emotion_scores
    
    def learn_from_transition(self, transition: Tuple[str, str], success: bool, context: TransitionContext):
        """Learn from transition outcomes to improve coherence understanding"""
        self.transition_history.append({
            'transition': transition,
            'success': success,
            'context': {
                'phase': context.negotiation_phase,
                'debtor_emotion': context.debtor_emotion,
                'gap_size': context.gap_size
            }
        })
        
        if success:
            self.successful_transitions.append(transition)
        
        # Simple learning: Adjust coherence threshold based on history
        if len(self.transition_history) > 10:
            success_rate = len(self.successful_transitions) / len(self.transition_history)
            
            # Adjust coherence threshold
            if success_rate > 0.7:
                self.coherence_threshold = min(0.7, self.coherence_threshold + 0.05)
            elif success_rate < 0.3:
                self.coherence_threshold = max(0.3, self.coherence_threshold - 0.05)



class GPTOrchestrator:
    """
    Simplified GPT-4o-mini orchestrator that ONLY uses LLM reasoning 
    to select between agent recommendations
    """
    
    def __init__(self, model_name: str = "gpt-4o-mini", temperature: float = 0.3):
        self.name = "GPTOrchestrator"
        self.model_name = model_name
        self.temperature = temperature
        self.client = None
        
        # Simple cache to avoid duplicate calls
        self.cache = {}
        
        # Basic statistics
        self.total_calls = 0
        self.cache_hits = 0

    def _init_client(self):
        """Initialize OpenAI client only when needed"""
        if self.client is None:
            try:
                from openai import OpenAI
                self.client = OpenAI()
            except ImportError:
                raise ImportError("OpenAI package not installed. Run: pip install openai")

    def combine_transition_predictions(self, predictions: List[AgentPrediction],
                                     context: TransitionContext) -> Dict[str, Any]:
        """
        Use GPT to simply choose between agent recommendations with reasoning
        """
        # Create simple cache key
        cache_key = f"{context.current_emotion}:{context.debtor_emotion}:{context.negotiation_phase}:"
        cache_key += ":".join(f"{p.agent_name}:{p.target_emotion}" for p in predictions)
        
        # Check cache
        if cache_key in self.cache:
            self.cache_hits += 1
            result = self.cache[cache_key].copy()
            result['selection_mechanism'] = 'gpt-cached'
            return result
        
        self._init_client()
        
        # Create simple prompt
        prompt = self._create_simple_prompt(predictions, context)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a negotiation expert. Choose the best emotional response."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=150
            )
            
            gpt_response = response.choices[0].message.content
            result = self._parse_gpt_response(gpt_response, predictions)
            
            # Cache the result
            self.cache[cache_key] = result
            self.total_calls += 1
            
            print(f"  🤖 GPT: Selected {result['selected_emotion']} ({result['confidence']:.2f})")
            print(f"     Reasoning: {result['gpt_reasoning'][:80]}...")
            
            return result
            
        except Exception as e:
            print(f"  ❌ GPT Error: {e}")
            return self._simple_fallback(predictions)

    def _create_simple_prompt(self, predictions: List[AgentPrediction], 
                             context: TransitionContext) -> str:
        """Create a simple, focused prompt for emotion selection"""
        
        # Current situation
        situation = f"""
Current situation:
- Creditor emotion: {EMOTION_NAMES[context.current_emotion]} ({context.current_emotion})
- Debtor emotion: {EMOTION_NAMES[context.debtor_emotion]} ({context.debtor_emotion})
- Negotiation phase: {context.negotiation_phase}
- Round: {context.round_number}
- Gap size: {context.gap_size} days
"""
        
        # Agent recommendations
        agents_text = "\nAgent recommendations:\n"
        for pred in predictions:
            agents_text += f"- {pred.agent_name}: {EMOTION_NAMES[pred.target_emotion]} ({pred.target_emotion}) "
            agents_text += f"| Reason: {pred.reasoning}\n"
        
        # Task
        task = f"""
Task: Choose ONE emotional response from the agent recommendations above.

Consider:
1. What's most effective for debt collection negotiation?
2. What fits the current negotiation phase?
3. How to respond to the debtor's current emotion?

Output format (must follow exactly):
SELECTED_EMOTION: [J/S/A/F/Su/D/N]
CONFIDENCE: [0.0-1.0]
REASONING: [1-2 sentences explaining your choice]

Available emotions: {[p.target_emotion for p in predictions]}
"""
        
        return situation + agents_text + task

    def _parse_gpt_response(self, response: str, 
                          predictions: List[AgentPrediction]) -> Dict[str, Any]:
        """Parse GPT response to extract selection"""
        import re
        
        # Find selected emotion
        selected_emotion = None
        agent_emotions = [p.target_emotion for p in predictions]
        
        # Look for emotion code
        for emotion in agent_emotions:
            if f"SELECTED_EMOTION: {emotion}" in response:
                selected_emotion = emotion
                break
        
        # Fallback: search for emotion code anywhere
        if not selected_emotion:
            for emotion in agent_emotions:
                if re.search(rf"\b{emotion}\b", response):
                    selected_emotion = emotion
                    break
        
        # Final fallback: pick highest confidence agent
        if not selected_emotion:
            selected_emotion = max(predictions, key=lambda p: p.confidence).target_emotion
        
        # Extract confidence
        confidence = 0.7
        conf_match = re.search(r"CONFIDENCE:\s*([0-9.]+)", response)
        if conf_match:
            try:
                confidence = float(conf_match.group(1))
                confidence = max(0.1, min(1.0, confidence))
            except:
                pass
        
        # Extract reasoning
        reasoning = "No reasoning provided"
        reason_match = re.search(r"REASONING:\s*(.+?)(?=\n\n|$)", response, re.DOTALL)
        if reason_match:
            reasoning = reason_match.group(1).strip()
        
        # Find agreeing agents
        agreeing_agents = [p.agent_name for p in predictions if p.target_emotion == selected_emotion]
        
        return {
            'selected_emotion': selected_emotion,
            'selected_probability': float(confidence),
            'confidence': float(confidence),
            'exploration': False,
            'agent_agreement': {
                'agreement_score': len(agreeing_agents) / len(predictions),
                'num_agreeing': len(agreeing_agents),
                'total_agents': len(predictions),
                'agreeing_agents': agreeing_agents
            },
            'gpt_reasoning': reasoning,
            'selection_mechanism': 'gpt-4o-mini_reasoning'
        }

    def _simple_fallback(self, predictions: List[AgentPrediction]) -> Dict[str, Any]:
        """Simple fallback: choose most confident agent"""
        best_pred = max(predictions, key=lambda p: p.confidence)
        
        agreeing_agents = [p.agent_name for p in predictions if p.target_emotion == best_pred.target_emotion]
        
        return {
            'selected_emotion': best_pred.target_emotion,
            'selected_probability': float(best_pred.confidence),
            'confidence': float(best_pred.confidence),
            'exploration': False,
            'agent_agreement': {
                'agreement_score': len(agreeing_agents) / len(predictions),
                'num_agreeing': len(agreeing_agents),
                'total_agents': len(predictions),
                'agreeing_agents': agreeing_agents
            },
            'gpt_reasoning': f"Fallback: Using {best_pred.agent_name}'s recommendation",
            'selection_mechanism': 'fallback_highest_confidence'
        }

    # Minimal required methods for compatibility
    def get_context_success_rates(self, context: TransitionContext) -> Dict[str, float]:
        """Simple success rates - always return 0.5 (no optimization)"""
        return {emotion: 0.5 for emotion in EMOTIONS}

    def learn_from_trajectory(self, agent_performance: Dict[str, Dict], 
                            success: bool, negotiation_rounds: int):
        """No learning - just record"""
        pass  # No optimization, no learning

    def get_learning_history_summary(self) -> Dict[str, Any]:
        """Simple learning summary"""
        return {
            'total_events': 0,
            'success_rate': 0.5,
            'agent_accuracy': {},
            'learning_disabled': True
        }

    def update_success(self, context: TransitionContext, 
                      selected_emotion: str, 
                      success: bool):
        """No success tracking"""
        pass  # No optimization, no tracking

    def get_stats(self) -> Dict[str, Any]:
        """Simple statistics"""
        return {
            'name': self.name,
            'model': self.model_name,
            'total_calls': self.total_calls,
            'cache_hits': self.cache_hits,
            'cache_rate': self.cache_hits / self.total_calls if self.total_calls > 0 else 0,
            'optimization': 'disabled',
            'mode': 'llm_reasoning_only'
        }
    


class GPTEnhancedBayesianTransitionModel:
    """
    Simplified model: Uses GPT-4o-mini ONLY for reasoning between agent predictions
    No Bayesian optimization, no reliability tracking
    """
    
    def __init__(self, exploration_rate: float = 0.2, agent_exploration_rate: float = 0.1,
                 gpt_model: str = "gpt-4o-mini"):
        # Initialize specialized agents
        self.game_theory_agent = TransitionGameTheoryAgent(exploration_rate=agent_exploration_rate)
        self.online_rl_agent = OnlineRLTransitionModel(exploration_rate=0.2)
        self.coherence_agent = EmotionalCoherenceAgent()
        
        # Use simplified GPT orchestrator
        self.orchestrator = GPTOrchestrator(model_name=gpt_model, temperature=0.3)
        
        # For compatibility with existing code
        self.use_gpt = True  # Always true for this simplified model
        
        # State tracking
        self.current_emotion = 'N'
        self.emotion_history = []
        self.negotiation_round = 0
        self.recent_successes = []
        
        # Basic performance tracking
        self.performance = {
            'total_transitions': 0,
            'successful_transitions': 0,
            'agent_selections': defaultdict(int)
        }
        
        print(f"🧠 Using SIMPLIFIED GPT Orchestrator ({gpt_model})")
        print(f"   Mode: LLM reasoning only, no optimization")
    
    # ... [rest of the class remains the same] ...
    
    def select_emotion(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Select next emotion using GPT to reason between agents"""
        self.negotiation_round += 1
        
        # Create transition context
        context = self._create_transition_context(state)
        
        # Get predictions from all agents
        predictions = [
            self.game_theory_agent.predict(context),
            self.online_rl_agent.predict(context),
            self.coherence_agent.predict(context)
        ]
        
        # Use GPT to choose between agents
        orchestration_result = self.orchestrator.combine_transition_predictions(predictions, context)
        
        selected_emotion = orchestration_result['selected_emotion']
        
        # Update state
        previous_emotion = self.current_emotion
        self.current_emotion = selected_emotion
        self.emotion_history.append(selected_emotion)
        
        # Track which agent was selected
        for pred in predictions:
            if pred.target_emotion == selected_emotion:
                self.performance['agent_selections'][pred.agent_name] += 1
        self.performance['total_transitions'] += 1
        
        # Build explanation
        explanation = self._build_simple_explanation(predictions, orchestration_result, 
                                                   context, previous_emotion)
        
        # Determine temperature
        temperature = 0.7  # Fixed for simplicity
        
        # Prepare result
        result = {
            "emotion": selected_emotion,
            "emotion_name": EMOTION_NAMES[selected_emotion],
            "emotion_text": self._get_emotion_prompt(selected_emotion),
            "temperature": temperature,
            "confidence": float(orchestration_result['confidence']),
            "explanation": explanation,
            "transition": f"{previous_emotion} → {selected_emotion}",
            "agent_predictions": [
                {
                    'agent': p.agent_name,
                    'target': p.target_emotion,
                    'confidence': float(p.confidence),
                    'reasoning': p.reasoning[:80] if p.reasoning else ""
                } for p in predictions
            ],
            "policy_mode": "gpt_reasoning_only",
            "orchestrator_type": "gpt-4o-mini"
        }
        
        # Add GPT reasoning if available
        if 'gpt_reasoning' in orchestration_result:
            result['gpt_analysis'] = {
                'reasoning': orchestration_result['gpt_reasoning'],
                'selection_mechanism': orchestration_result.get('selection_mechanism', 'unknown')
            }
        
        return result
    
    def _build_simple_explanation(self, predictions: List[AgentPrediction],
                                result: Dict[str, Any],
                                context: TransitionContext,
                                previous_emotion: str) -> str:
        """Build simple explanation"""
        lines = []
        
        lines.append(f"🤖 GPT REASONING BETWEEN AGENTS")
        lines.append(f"Emotion: {EMOTION_NAMES[previous_emotion]} → {EMOTION_NAMES[result['selected_emotion']]}")
        lines.append(f"Round {context.round_number}, {context.negotiation_phase} phase")
        lines.append(f"Debtor: {EMOTION_NAMES[context.debtor_emotion]}")
        lines.append("")
        
        lines.append("AGENT RECOMMENDATIONS:")
        for pred in predictions:
            mark = "✓" if pred.target_emotion == result['selected_emotion'] else "○"
            lines.append(f"  {mark} {pred.agent_name}: {EMOTION_NAMES[pred.target_emotion]}")
        
        lines.append("")
        lines.append("GPT SELECTION:")
        if 'gpt_reasoning' in result:
            lines.append(f"  {result['gpt_reasoning']}")
        else:
            lines.append(f"  Selected {EMOTION_NAMES[result['selected_emotion']]}")
        
        return "\n".join(lines)
    
    def _get_emotion_prompt(self, emotion: str) -> str:
        """Get emotion prompt for LLM"""
        prompts = {
            "J": "Use optimistic, positive tone",
            "S": "Use empathetic, understanding tone",
            "A": "Use firm, assertive tone",
            "F": "Use cautious, concerned tone",
            "Su": "Use engaging, creative tone",
            "D": "Use disappointed but professional tone",
            "N": "Use balanced, professional tone"
        }
        return prompts.get(emotion, "Use professional tone")
    
    def _create_transition_context(self, state: Dict[str, Any]) -> TransitionContext:
        """Create transition context from state"""
        round_num = self.negotiation_round
        
        # Determine negotiation phase
        if round_num <= 3:
            phase = 'early'
        elif round_num <= 8:
            phase = 'middle'
        elif state.get('crisis_indicator', False):
            phase = 'crisis'
        else:
            phase = 'late'
        
        # Calculate recent success rate
        recent_success_rate = 0.5
        if self.recent_successes:
            window = min(5, len(self.recent_successes))
            recent_success_rate = np.mean(self.recent_successes[-window:])
        
        # Extract gap size from state
        gap_size = state.get('gap_size', 0)
        
        return TransitionContext(
            current_emotion=self.current_emotion,
            debtor_emotion=state.get('debtor_emotion', 'N'),
            negotiation_phase=phase,
            round_number=round_num,
            emotional_history=self.emotion_history[-5:] if self.emotion_history else [],
            debt_amount=state.get('debt_amount', 0),
            recent_success_rate=recent_success_rate,
            gap_size=gap_size
        )
    
    def update_model(self, negotiation_result: Dict[str, Any]) -> None:
        """Minimal update - just track success"""
        success = negotiation_result.get('success', False)
        
        # Update recent successes
        self.recent_successes.append(success)
        if len(self.recent_successes) > 10:
            self.recent_successes = self.recent_successes[-10:]
        
        if success:
            self.performance['successful_transitions'] += 1
        
        # Update Game Theory agent
        self.game_theory_agent.update_success(success)
        
        # Update emotion history
        if 'emotion_history' in negotiation_result and negotiation_result['emotion_history']:
            self.emotion_history = negotiation_result['emotion_history'].copy()
        
        # Optional: Update Online RL if trajectory data is available
        if 'agent_predictions_history' in negotiation_result:
            emotion_history = negotiation_result.get('emotion_sequence', [])
            context_history = negotiation_result.get('context_history', [])
            
            if len(emotion_history) > 1 and success:

                 # Calculate collection rate for this negotiation
                # MATCH VANILLA METHOD: collection_rate = min(1.0, target_days / actual_days)
                creditor_target = negotiation_result.get('creditor_target_days', 30)
                final_days = negotiation_result.get('collection_days', creditor_target)
                
                # Collection rate using vanilla method (Higher is better)
                if final_days > 0 and creditor_target > 0:
                    collection_rate = min(1.0, creditor_target / final_days)
                else:
                    collection_rate = 0.0
                
                base_reward = 1.0 / (1 + np.log(len(emotion_history)))

                macro_reward = collection_rate * base_reward
                
                # Clip to reasonable range
                macro_reward = np.clip(macro_reward, 0.0, 1.0)
    
                self.online_rl_agent.learn_from_trajectory(
                    emotion_trajectory=emotion_history,
                    context_history=context_history,
                    success=success,
                    macro_reward=macro_reward
                )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get simple statistics"""
        success_rate = 0.0
        if self.performance['total_transitions'] > 0:
            success_rate = self.performance['successful_transitions'] / self.performance['total_transitions']
        
        return {
            'current_emotion': self.current_emotion,
            'round_number': self.negotiation_round,
            'emotion_history': self.emotion_history[-5:],
            'performance': {
                'success_rate': success_rate,
                'total_transitions': self.performance['total_transitions'],
                'agent_selections': dict(self.performance['agent_selections'])
            },
            'orchestrator_stats': self.orchestrator.get_stats()
        }
    
    def reset(self) -> None:
        """Reset for new negotiation"""
        self.current_emotion = 'N'
        self.negotiation_round = 0
        self.recent_successes = []
        self.emotion_history = []
        
        # Reset agents but keep learned knowledge
        self.game_theory_agent.last_debtor_emotion = None
        self.game_theory_agent.last_agent_emotion = None
        self.game_theory_agent.last_successful = True



def run_gpt_orchestrator_experiment(
    scenarios: List[Dict[str, Any]],
    iterations: int = 5,
    model_creditor: str = "gpt-4o-mini",
    model_debtor: str = "gpt-4o-mini",
    debtor_emotion: str = "neutral",
    debtor_model_type: str = "vanilla",
    max_dialog_len: int = 30,
    out_dir: str = "results/gpt_orchestrator"
) -> Dict[str, Any]:
    """Run GPT-4o-mini Orchestrator experiment with detailed logging"""
    
    from llm.negotiator_multiagent import DebtNegotiator
    import os
    
    # Create output directory
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    # Create model
    model = GPTEnhancedBayesianTransitionModel(
        exploration_rate=0.2,
        agent_exploration_rate=0.1,
        gpt_model="gpt-4o-mini"
    )
    
    # Main results structure
    results = {
        'experiment_type': 'gpt_orchestrator',
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'config': {
            'iterations': iterations,
            'scenarios': len(scenarios),
            'model_creditor': model_creditor,
            'model_debtor': model_debtor,
            'debtor_emotion': debtor_emotion,
            'max_dialog_len': max_dialog_len,
            'mode': 'llm_reasoning_only'
        },
        'detailed_negotiations': [],  # Will contain ALL negotiation details
        'summary_statistics': {},
        'gpt_usage': {}
    }
    
    all_negotiation_results = []
    
    print("="*80)
    print("🤖 GPT-4o-mini ORCHESTRATOR EXPERIMENT")
    print("="*80)
    print(f"Iterations: {iterations}")
    print(f"Scenarios: {len(scenarios)}")
    print(f"Mode: Detailed logging enabled")
    print("="*80)
    
    for iteration in range(iterations):
        print(f"\n🔄 Iteration {iteration + 1}/{iterations}")
        
        for scenario_idx, scenario in enumerate(scenarios):
            print(f"\n  🎭 Scenario {scenario_idx + 1}/{len(scenarios)}: {scenario['id']}")
            
            # Create negotiator
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
            all_negotiation_results.append(result)
            
            # ===== EXTRACT DETAILED NEGOTIATION DATA =====
            negotiation_details = {
                'negotiation_id': f"iter{iteration+1}_scen{scenario_idx+1}",
                'iteration': iteration + 1,
                'scenario_id': scenario['id'],
                'timestamp': datetime.now().strftime("%H:%M:%S"),
                'outcome': {
                    'final_state': result.get('final_state', 'unknown'),
                    'success': result.get('success', False),
                    'collection_days': result.get('collection_days', 0) if result.get('collection_days') is not None else 0,
                    'creditor_target_days': result.get('creditor_target_days', 30),
                    'collection_rate': min(1.0, result.get('creditor_target_days', 30) / max(1, result.get('collection_days', 30) if result.get('collection_days') is not None else 30)),  # MATCH VANILLA
                    'negotiation_rounds': result.get('negotiation_rounds', 0),
                    'final_gap': result.get('final_gap', 0),
                    'initial_gap': result.get('initial_gap', 0)
                },
                'emotion_trajectory': {
                    'final_emotion_sequence': result.get('emotion_sequence', []),
                    'emotion_history': result.get('emotion_history', []),
                    'debtor_emotions': result.get('debtor_emotion_history', []),
                    'total_transitions': len(result.get('emotion_sequence', [])) - 1 if result.get('emotion_sequence') else 0
                },
                'agent_predictions_history': [],
                'conversation_history': [],
                'orchestrator_decisions': [],
                'learning_updates': []
            }
            
            # 1. Extract agent predictions for each round
            if 'agent_predictions_history' in result:
                for round_idx, round_preds in enumerate(result['agent_predictions_history']):
                    round_details = {
                        'round': round_idx + 1,
                        'predictions': [],
                        'orchestrator_decision': {},
                        'actual_emotion_used': None
                    }
                    
                    # Extract each agent's prediction
                    for pred in round_preds.get('predictions', []):
                        agent_pred = {
                            'agent_name': pred.get('agent', 'unknown'),
                            'recommended_emotion': pred.get('target', 'N'),
                            'confidence': pred.get('confidence', 0.5),
                            'reasoning': pred.get('reasoning', ''),
                            'value_score': pred.get('transition_value', 0.0)
                        }
                        round_details['predictions'].append(agent_pred)
                    
                    # Get actual emotion used (from sequence)
                    if result.get('emotion_sequence') and round_idx < len(result['emotion_sequence']):
                        round_details['actual_emotion_used'] = result['emotion_sequence'][round_idx]
                    
                    negotiation_details['agent_predictions_history'].append(round_details)
            
            # 2. Extract conversation history
            if 'dialog_history' in result:
                for dialog_round, dialog in enumerate(result['dialog_history']):
                    conversation_round = {
                        'round': dialog_round + 1,
                        'creditor_message': dialog.get('creditor', ''),
                        'debtor_message': dialog.get('debtor', ''),
                        'creditor_emotion': dialog.get('creditor_emotion', 'N'),
                        'debtor_emotion': dialog.get('debtor_emotion', 'N'),
                        'temperature': dialog.get('temperature', 0.7)
                    }
                    negotiation_details['conversation_history'].append(conversation_round)
            
            # 3. Extract orchestrator decisions from emotion_config
            if 'emotion_config' in result:
                emo_config = result['emotion_config']
                orchestrator_info = {
                    'final_selected_emotion': emo_config.get('emotion', 'N'),
                    'emotion_name': emo_config.get('emotion_name', 'Neutral'),
                    'confidence': emo_config.get('confidence', 0.5),
                    'transition': emo_config.get('transition', 'N->N'),
                    'explanation': emo_config.get('explanation', ''),
                    'policy_mode': emo_config.get('policy_mode', 'unknown'),
                    'temperature': emo_config.get('temperature', 0.7)
                }
                
                # Add GPT analysis if available
                if 'gpt_analysis' in emo_config:
                    orchestrator_info['gpt_analysis'] = {
                        'reasoning': emo_config['gpt_analysis'].get('reasoning', ''),
                        'selection_mechanism': emo_config['gpt_analysis'].get('selection_mechanism', '')
                    }
                
                # Add agent agreement stats
                if 'agent_predictions' in emo_config:
                    agent_agreement = []
                    for agent_pred in emo_config['agent_predictions']:
                        agent_agreement.append({
                            'agent': agent_pred.get('agent', 'unknown'),
                            'recommended': agent_pred.get('target', 'N'),
                            'agreed': agent_pred.get('target', 'N') == emo_config.get('emotion', 'N')
                        })
                    orchestrator_info['agent_agreement'] = agent_agreement
                
                negotiation_details['orchestrator_decisions'].append(orchestrator_info)
            
            # 4. Extract learning updates
            if 'context_history' in result and 'emotion_sequence' in result:
                learning_updates = []
                for i in range(len(result['context_history'])):
                    if i < len(result.get('emotion_sequence', [])) - 1:
                        update = {
                            'round': i + 1,
                            'context': result['context_history'][i],
                            'transition': f"{result['emotion_sequence'][i]}->{result['emotion_sequence'][i+1]}",
                            'agent_predictions': []
                        }
                        learning_updates.append(update)
                negotiation_details['learning_updates'] = learning_updates
            
            # Add to results
            results['detailed_negotiations'].append(negotiation_details)
            
            # Update model
            model.update_model(result)
            
            # Clean up GPU memory after each negotiation (important for offline models)
            try:
                negotiator.cleanup_models()
            except Exception as e:
                print(f"⚠️ Cleanup warning: {e}")
            
            # Print brief summary
            outcome = "✅" if result.get('final_state') == 'accept' else "❌"
            days = result.get('collection_days', 'N/A')
            print(f"    {outcome} Result: {result.get('final_state')} | Days: {days}")
            
            # Print emotion summary if available
            if 'emotion_sequence' in result and result['emotion_sequence']:
                emotions = "→".join(result['emotion_sequence'][-5:])  # Last 5 emotions
                print(f"    🎭 Emotions: {emotions}")
    
    # ===== CALCULATE SUMMARY STATISTICS =====
    successful = [r for r in all_negotiation_results if r.get('final_state') == 'accept']
    success_rate = len(successful) / len(all_negotiation_results) if all_negotiation_results else 0
    
    # Collection rates - MATCH VANILLA METHOD
    collection_rates = []
    for r in successful:
        target_days = r.get('creditor_target_days', 30)
        actual_days = r.get('collection_days', target_days)
        if target_days > 0 and actual_days > 0:
            collection_rate = min(1.0, target_days / actual_days)  # Higher is better
            collection_rates.append(collection_rate)
    
    # Rounds
    negotiation_rounds = [r.get('negotiation_rounds', 0) for r in all_negotiation_results]
    
    # Emotion transition analysis
    all_transitions = []
    for negot in results['detailed_negotiations']:
        seq = negot['emotion_trajectory']['final_emotion_sequence']
        if seq and len(seq) > 1:
            for i in range(len(seq) - 1):
                all_transitions.append(f"{seq[i]}→{seq[i+1]}")
    
    from collections import Counter
    transition_counts = Counter(all_transitions)
    
    # Agent performance analysis
    agent_performance = {}
    for negot in results['detailed_negotiations']:
        for round_preds in negot['agent_predictions_history']:
            actual_emotion = round_preds.get('actual_emotion_used')
            if actual_emotion:
                for pred in round_preds['predictions']:
                    agent = pred['agent_name']
                    if agent not in agent_performance:
                        agent_performance[agent] = {'total': 0, 'correct': 0}
                    agent_performance[agent]['total'] += 1
                    if pred['recommended_emotion'] == actual_emotion:
                        agent_performance[agent]['correct'] += 1
    
    # Compile summary statistics
    import numpy as np
    
    results['summary_statistics'] = {
        'success_rate': float(success_rate),
        'collection_rate': {
            'mean': float(np.mean(collection_rates)) if collection_rates else 0,
            'std': float(np.std(collection_rates)) if collection_rates else 0,
            'min': float(min(collection_rates)) if collection_rates else 0,
            'max': float(max(collection_rates)) if collection_rates else 0,
            'median': float(np.median(collection_rates)) if collection_rates else 0,
            'n_samples': len(collection_rates)
        },
        'negotiation_rounds': {
            'mean': float(np.mean(negotiation_rounds)) if negotiation_rounds else 0,
            'std': float(np.std(negotiation_rounds)) if negotiation_rounds else 0,
            'min': int(min(negotiation_rounds)) if negotiation_rounds else 0,
            'max': int(max(negotiation_rounds)) if negotiation_rounds else 0,
            'median': float(np.median(negotiation_rounds)) if negotiation_rounds else 0,
            'n_samples': len(negotiation_rounds)
        },
        'emotion_transitions': {
            'total_transitions': len(all_transitions),
            'unique_transitions': len(transition_counts),
            'most_common_transitions': dict(transition_counts.most_common(10)),
            'transition_entropy': float(-sum((count/len(all_transitions)) * np.log2(count/len(all_transitions)) 
                                          for count in transition_counts.values())) if all_transitions else 0
        },
        'agent_performance': {
            agent: {
                'accuracy': perf['correct'] / perf['total'] if perf['total'] > 0 else 0,
                'total_predictions': perf['total'],
                'correct_predictions': perf['correct']
            } for agent, perf in agent_performance.items()
        },
        'counts': {
            'total_negotiations': len(all_negotiation_results),
            'successful_negotiations': len(successful),
            'failed_negotiations': len(all_negotiation_results) - len(successful),
            'total_dialog_rounds': sum(negotiation_rounds)
        }
    }
    
    # GPT usage statistics
    results['gpt_usage'] = model.orchestrator.get_stats()
    
    # ===== ADD STATISTICAL ANALYSIS WITH 95% CONFIDENCE INTERVALS =====
    results = enhance_results_with_statistics(
        results, 
        all_negotiation_results, 
        scenarios, 
        method="bootstrap"
    )
    
    # ===== SAVE COMPLETE RESULTS =====
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = f"{out_dir}/complete_results_{timestamp}.json"
    
    with open(result_file, "w", encoding='utf-8') as f:
        json.dump(results, f, indent=2, default=str)
    
    # ===== PRINT SUMMARY =====
    print("\n" + "="*80)
    print("📊 EXPERIMENT RESULTS SUMMARY")
    print("="*80)
    
    stats = results['summary_statistics']
    print(f"\n🎯 PERFORMANCE METRICS:")
    
    # Print statistical analysis with CIs if available
    if 'statistical_analysis' in results:
        print(format_ci_results(results['statistical_analysis']))
    else:
        print(f"  Success Rate:        {stats['success_rate']:.1%}")
        print(f"  Collection Rate:     {stats['collection_rate']['mean']:.3f} ± {stats['collection_rate']['std']:.3f}")
        print(f"  Negotiation Rounds:  {stats['negotiation_rounds']['mean']:.1f} ± {stats['negotiation_rounds']['std']:.1f}")
    
    print(f"\n🎭 EMOTION ANALYSIS:")
    print(f"  Total Transitions:   {stats['emotion_transitions']['total_transitions']}")
    print(f"  Unique Transitions:  {stats['emotion_transitions']['unique_transitions']}")
    
    if stats['emotion_transitions']['most_common_transitions']:
        print(f"  Top 3 Transitions:")
        for trans, count in list(stats['emotion_transitions']['most_common_transitions'].items())[:3]:
            print(f"    {trans}: {count} times")
    
    print(f"\n🤖 AGENT PERFORMANCE:")
    for agent, perf in stats['agent_performance'].items():
        print(f"  {agent}: {perf['accuracy']:.1%} ({perf['correct_predictions']}/{perf['total_predictions']})")
    
    print(f"\n📈 DETAILED COUNTS:")
    print(f"  Negotiations: {stats['counts']['total_negotiations']} total")
    print(f"  Successful:   {stats['counts']['successful_negotiations']}")
    print(f"  Failed:       {stats['counts']['failed_negotiations']}")
    print(f"  Dialog Rounds:{stats['counts']['total_dialog_rounds']}")
    
    print(f"\n💾 Complete results saved to: {result_file}")
    print(f"   Contains {len(results['detailed_negotiations'])} detailed negotiations")
    
    return results