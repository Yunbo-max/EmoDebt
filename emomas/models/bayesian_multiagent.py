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
from utils.statistical_analysis import enhance_results_with_statistics, format_ci_results
warnings.filterwarnings('ignore')

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
                confidence=0.4,
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




class BayesianTransitionOrchestrator:
    """
    Bayesian orchestrator for optimizing emotional transitions
    Uses Bayesian probability to combine agent recommendations
    """
    
    def __init__(self, exploration_rate: float = 0.3, alpha: float = 1.0, exploration_decay: float = 0.99):
        self.exploration_rate = exploration_rate
        self.alpha = alpha
        self.exploration_decay = exploration_decay
        
        # Agent reliability tracking - Dirichlet priors
        self.agent_reliability = defaultdict(lambda: np.ones(N_EMOTIONS) * 2.0)
        
        # Context-specific learning
        self.context_success_counts = defaultdict(lambda: np.zeros((N_EMOTIONS, N_EMOTIONS)))
        self.context_failure_counts = defaultdict(lambda: np.zeros((N_EMOTIONS, N_EMOTIONS)))
        
        # Reward history for adaptive learning
        self.reward_history = []
        self.learning_history = []
    
    def learn_from_trajectory(self, agent_performance: Dict[str, Dict[str, Any]], 
                     success: bool, negotiation_rounds: int,collection_rate: float = 0.0):
        """
        基于整个谈判轨迹的可靠性更新
        """
        print(f"\n🎯 DEBUG learn_from_trajectory called:")
        print(f"  Success: {success}")
        print(f"  Negotiation rounds: {negotiation_rounds}")
        
        # 1. 计算宏观奖励/惩罚
        if success:
            # 成功谈判：基础奖励
            base_reward = 1.0 / (1 + np.log(max(1, negotiation_rounds)))
            # 回收率平方使高回收率获得更高奖励
            macro_reward = base_reward*collection_rate
            print(f"  Macro reward: {macro_reward}")
        else:
            # 失败谈判：显著惩罚（比奖励更强！）
            macro_penalty = -0.3  # 固定惩罚，比奖励更强
            print(f"  Macro penalty: {macro_penalty}")
        
        # 2. 计算每个Agent的表现分数
        agent_scores = {}
        for agent_name, perf in agent_performance.items():
            if perf['total_predictions'] > 0:
                # 准确率 = 正确预测次数 / 总预测次数
                accuracy = perf['correct_predictions'] / perf['total_predictions']
                
                # 置信度加权准确率 = 正确预测的总置信度 / 总置信度
                if perf['total_confidence'] > 0:
                    confidence_weighted_accuracy = perf['correct_confidence'] / perf['total_confidence']
                else:
                    confidence_weighted_accuracy = accuracy
                
                # Score = Accuracy × ConfidenceWeightedAccuracy
                agent_scores[agent_name] = accuracy * confidence_weighted_accuracy
                agent_scores[agent_name] = max(agent_scores[agent_name], 0.01)
            else:
                agent_scores[agent_name] = 0.01
        
        # 3. 归一化Agent分数
        total_score = sum(agent_scores.values())
        
        if total_score > 0:
            normalized_scores = {agent: score/total_score for agent, score in agent_scores.items()}
        else:
            normalized_scores = {agent: 1.0/len(agent_scores) for agent in agent_scores.keys()}
        
        # 4. 更新所有context的Agent可靠性 - 强化版本
        update_count = 0
        for agent_name, normalized_score in normalized_scores.items():
            for (stored_agent, context_key), reliability_dist in self.agent_reliability.items():
                if stored_agent == agent_name:
                    # 获取该Agent最常预测的情感
                    most_likely_emotion_idx = np.argmax(reliability_dist)
                    old_value = reliability_dist[most_likely_emotion_idx]
                    
                    if success:
                        # 成功谈判：显著正更新
                        update_value = normalized_score * macro_reward * 2.0  # 加倍奖励
                        reliability_dist[most_likely_emotion_idx] += update_value
                        
                        update_count += 1
                        print(f"  ✅ SUCCESS: {agent_name} in {context_key}: {old_value:.3f} → {reliability_dist[most_likely_emotion_idx]:.3f} (+{update_value:.3f})")
                    else:
                        # 失败谈判：显著负更新
                        # 惩罚 = 基础惩罚 × 该Agent的表现分数（表现越差，惩罚越大）
                        penalty_strength = 1.0 - normalized_score  # 表现越差，惩罚强度越大
                        penalty = macro_penalty * penalty_strength * 1.5  # 加倍惩罚
                        
                        # 直接减去惩罚值（而不是乘法）
                        reliability_dist[most_likely_emotion_idx] += penalty
                        
                        # 确保可靠性不低于最小值
                        reliability_dist[most_likely_emotion_idx] = max(0.01, reliability_dist[most_likely_emotion_idx])
                        
                        update_count += 1
                        print(f"  ❌ FAILURE: {agent_name} in {context_key}: {old_value:.3f} → {reliability_dist[most_likely_emotion_idx]:.3f} ({penalty:.3f})")
        
        print(f"  Total updates applied: {update_count}")
        
        # 5. 记录学习事件
        learning_event = {
            'type': 'trajectory_learning',
            'success': success,
            'macro_update': macro_reward if success else macro_penalty,
            'agent_scores': agent_scores,
            'normalized_scores': normalized_scores,
            'negotiation_rounds': negotiation_rounds,
            'agent_performance': dict(agent_performance),
            'agents': list(agent_performance.keys()),
            'selected_by': [agent for agent, perf in agent_performance.items() 
                        if perf.get('correct_predictions', 0) / max(perf.get('total_predictions', 1), 1) > 0.5]
        }
        
        self.learning_history.append(learning_event)
        print(f"  📝 Learning event recorded (success={success})")
    
    def combine_transition_predictions(self, predictions: List[AgentPrediction],
                                context: TransitionContext) -> Dict[str, Any]:
        """
        SUM of (reliability × confidence) for each emotion
        """
        context_key = self._context_to_key(context)
        
        # Track scores for each emotion
        emotion_scores = defaultdict(float)
        
        for prediction in predictions:
            agent_name = prediction.agent_name
            target_emotion = prediction.target_emotion
            agent_confidence = prediction.confidence
            
            # Get agent reliability for this context
            reliability_key = (agent_name, context_key)
            reliability_dist = self.agent_reliability[reliability_key]
            reliability_probs = reliability_dist / reliability_dist.sum()
            
            # Get reliability for this specific emotion
            emotion_idx = EMOTION_TO_IDX[target_emotion]
            emotion_reliability = reliability_probs[emotion_idx]
            
            # Score = reliability × confidence
            score = emotion_reliability * agent_confidence
            
            # Add to emotion's total score
            emotion_scores[target_emotion] += score
        
        # Also need to consider emotions NOT recommended?
        # With sum, we can just use the scores as-is
        
        # Find emotion with highest total score
        if emotion_scores:
            selected_emotion = max(emotion_scores, key=emotion_scores.get)
            max_score = emotion_scores[selected_emotion]
            
            # Normalize scores to get probabilities
            total_score = sum(emotion_scores.values())
            probabilities = {e: score/total_score for e, score in emotion_scores.items()}
            confidence = max_score / total_score
        else:
            # Fallback: uniform over all emotions
            selected_emotion = np.random.choice(EMOTIONS)
            probabilities = {e: 1.0/N_EMOTIONS for e in EMOTIONS}
            confidence = 1.0/N_EMOTIONS
        
        # Find agreeing agents
        agreeing_agents = [
            p.agent_name for p in predictions 
            if p.target_emotion == selected_emotion
        ]
        
        return {
            'selected_emotion': selected_emotion,
            'selected_probability': float(confidence),
            'transition_probabilities': probabilities,
            'confidence': float(confidence),
            'exploration': False,
            'agent_agreement': {
                'agreement_score': len(agreeing_agents) / len(predictions),
                'num_agreeing': len(agreeing_agents),
                'total_agents': len(predictions),
                'agreeing_agents': agreeing_agents
            },
            'selection_mechanism': 'sum_reliability_x_confidence'
        }
    

            
    def _context_to_key(self, context: TransitionContext) -> str:
        """Create context key for reliability tracking"""
        key_parts = [
            f"from_{context.current_emotion}",
            f"debtor_{context.debtor_emotion}",
            f"phase_{context.negotiation_phase}",
            f"gap_{'large' if context.gap_size > 20 else 'small'}"
        ]
        return "_".join(key_parts)
    
    def get_context_success_rates(self, context: TransitionContext) -> Dict[str, float]:
        """Get success rates for transitions from current context"""
        context_key = self._context_to_key(context)
        current_idx = EMOTION_TO_IDX[context.current_emotion]
        
        success_counts = self.context_success_counts[context_key][current_idx]
        failure_counts = self.context_failure_counts[context_key][current_idx]
        
        success_rates = {}
        for i in range(N_EMOTIONS):
            total = success_counts[i] + failure_counts[i]
            if total > 0:
                success_rates[IDX_TO_EMOTION[i]] = success_counts[i] / total
            else:
                success_rates[IDX_TO_EMOTION[i]] = 0.5
        
        return success_rates
    
    def get_learning_history_summary(self) -> Dict[str, Any]:
        """Get summary of learning history"""
        if not self.learning_history:
            return {'total_events': 0, 'success_rate': 0.0}
        
        total = len(self.learning_history)
        successes = sum(1 for event in self.learning_history if event['success'])
        
        # Calculate agent accuracy
        agent_accuracy = defaultdict(list)
        for event in self.learning_history:
            for agent in event['agents']:
                agent_correct = agent in event['selected_by']
                agent_accuracy[agent].append(agent_correct)
        
        accuracy_stats = {}
        for agent, correct_list in agent_accuracy.items():
            accuracy_stats[agent] = {
                'accuracy': np.mean(correct_list) if correct_list else 0.0,
                'total_predictions': len(correct_list),
                'correct_predictions': sum(correct_list)
            }
        
        return {
            'total_events': total,
            'success_rate': successes / total if total > 0 else 0.0,
            'agent_accuracy': accuracy_stats,
            'recent_events': self.learning_history[-5:] if self.learning_history else []
        }

class BayesianTransitionModel:
    """
    Main model: Bayesian optimization of emotional transitions
    Combines three specialized agents to optimize emotional state changes
    """
    
    def __init__(self, exploration_rate: float = 0.2, agent_exploration_rate: float = 0.1):
        # Initialize specialized agents (HMM removed)
        self.game_theory_agent = TransitionGameTheoryAgent(exploration_rate=agent_exploration_rate)
        self.online_rl_agent = OnlineRLTransitionModel(exploration_rate=0.2)  # Using Online RL instead of HMM
        self.coherence_agent = EmotionalCoherenceAgent()
        
        # Initialize Bayesian orchestrator
        self.orchestrator = BayesianTransitionOrchestrator(exploration_rate)
        
        # State tracking
        self.current_emotion = 'N'  # Start with Neutral
        self.emotion_history = []
        self.negotiation_round = 0
        self.recent_successes = []
        
        # Performance tracking
        self.performance = {
            'total_transitions': 0,
            'successful_transitions': 0,
            'agent_selections': defaultdict(int)
        }
        
        # Negative emotion set for policy selection
        self.negative_emotions = {'S', 'A', 'F', 'D'}  # Sadness, Anger, Fear, Disgust
        self.negativity_threshold = 2  # k in your equation
    
    def _learn_from_trajectory(self, emotion_trajectory: List[str], 
                         predictions_history: List[Dict[str, Any]],
                         success: bool, negotiation_rounds: int,collection_rate: float) -> None:
        """
        基于整个情绪轨迹的学习
        """
        print(f"\n🔍 DEBUG _learn_from_trajectory called:")
        print(f"  Emotion trajectory length: {len(emotion_trajectory)}")
        print(f"  Predictions history length: {len(predictions_history)}")
        print(f"  Success: {success}")
        print(f"  Negotiation rounds: {negotiation_rounds}")
        
        if not emotion_trajectory or not predictions_history:
            print(f"  ⚠️ Skipping: Empty data")
            return
        
        # Print first few items to verify structure
        for i in range(min(3, len(predictions_history))):
            print(f"  Round {i} predictions: {len(predictions_history[i].get('predictions', []))} items")

        if len(emotion_trajectory) == 0 or len(predictions_history) == 0:
            print(f"⚠️ No trajectory data: emotions={len(emotion_trajectory)}, predictions={len(predictions_history)}")
            return
        
        print(f"\n🎯 Trajectory Learning: {len(predictions_history)} rounds, success={success}")
        
        # 1. 统计每个Agent在整个轨迹中的表现
        agent_performance = defaultdict(lambda: {
            'total_predictions': 0,
            'correct_predictions': 0,
            'total_confidence': 0.0,
            'correct_confidence': 0.0
        })
        
        for round_idx, round_record in enumerate(predictions_history):
            if not isinstance(round_record, dict):
                print(f"⚠️ Invalid round record at index {round_idx}: {type(round_record)}")
                continue
                
            # Extract predictions and selected emotion
            round_predictions = round_record.get('predictions', [])
            selected_emotion = round_record.get('selected_emotion', 'N')
            
            if not round_predictions:
                continue
                
            # Verify we have emotion data for this round
            actual_emotion = None
            if round_idx < len(emotion_trajectory):
                actual_emotion = emotion_trajectory[round_idx]
            else:
                # Fallback to selected emotion if no trajectory
                actual_emotion = selected_emotion
            
            # Update agent performance
            for pred_data in round_predictions:
                if not isinstance(pred_data, dict):
                    continue
                    
                agent_name = pred_data.get('agent', 'Unknown')
                target_emotion = pred_data.get('target', 'N')
                confidence = pred_data.get('confidence', 0.5)
                
                agent_performance[agent_name]['total_predictions'] += 1
                agent_performance[agent_name]['total_confidence'] += confidence
                
                if target_emotion == actual_emotion:
                    agent_performance[agent_name]['correct_predictions'] += 1
                    agent_performance[agent_name]['correct_confidence'] += confidence
        
        # Call orchestrator with performance data
        if agent_performance:
            print(f"  Agent performance summary:")
            for agent, perf in agent_performance.items():
                if perf['total_predictions'] > 0:
                    accuracy = perf['correct_predictions'] / perf['total_predictions']
                    print(f"    {agent}: {accuracy:.1%} accuracy ({perf['correct_predictions']}/{perf['total_predictions']})")
            
            self.orchestrator.learn_from_trajectory(
                agent_performance=agent_performance,
                success=success,
                negotiation_rounds=negotiation_rounds,
                collection_rate=collection_rate  # NEW: Pass collection rate
            )
        else:
            print(f"  ⚠️ No agent performance data collected")
    
    def select_emotion(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Select next emotion with Bayesian-optimized transition"""
        self.negotiation_round += 1
        
        # Create transition context
        context = self._create_transition_context(state)
        
        # Get predictions from all agents
        predictions = [
            self.game_theory_agent.predict(context),
            self.online_rl_agent.predict(context),
            self.coherence_agent.predict(context)
        ]
        
        # Bayesian combination
        bayesian_result = self.orchestrator.combine_transition_predictions(predictions, context)
        
        selected_emotion = bayesian_result['selected_emotion']
        
        # Update state
        previous_emotion = self.current_emotion
        self.current_emotion = selected_emotion
        self.emotion_history.append(selected_emotion)
        
        # Track agent contributions
        for pred in predictions:
            if pred.target_emotion == selected_emotion:
                self.performance['agent_selections'][pred.agent_name] += 1
        self.performance['total_transitions'] += 1
        
        # Build explanation
        explanation = self._build_transition_explanation(predictions, bayesian_result, context, previous_emotion)
        
        # Determine temperature
        temperature = self._calculate_temperature(bayesian_result['confidence'], context)
        
        # Emotion prompts
        emotion_prompts = {
            "J": "Use an optimistic and positive tone, expressing confidence",
            "S": "Use an empathetic and understanding tone, acknowledging difficulty",
            "A": "Use a firm and assertive tone, emphasizing urgency and importance",
            "F": "Use a cautious and concerned tone, highlighting potential consequences",
            "Su": "Use an engaging and unexpected approach, introducing creative solutions",
            "D": "Use a disappointed tone, expressing concern while remaining professional",
            "N": "Use a balanced and professional tone, focusing on facts and practical solutions"
        }
        
        # Simplified agent predictions format
        agent_predictions_formatted = [
            {
                'agent': p.agent_name,
                'target': p.target_emotion,
                'confidence': float(p.confidence),
                'reasoning': p.reasoning[:100]  # First 100 chars only
            } for p in predictions
        ]
        
        # RETURN ONLY WHAT'S NEEDED
        return {
            "emotion": selected_emotion,
            "emotion_name": EMOTION_NAMES[selected_emotion],
            "emotion_text": emotion_prompts.get(selected_emotion, "Use a professional tone"),
            "temperature": temperature,
            "confidence": float(bayesian_result['confidence']),
            "explanation": explanation,
            "transition": f"{previous_emotion} → {selected_emotion}",
            "agent_predictions": agent_predictions_formatted,
            "policy_mode": "bayesian"
        }
    
    def _count_recent_negativity(self, context: TransitionContext) -> int:
        """Count recent negative debtor emotions (your ∑ indicator function)"""
        if len(context.emotional_history) < 3:  # n in your equation
            return 0
        
        # Use last 3 emotions
        recent_history = context.emotional_history[-3:]
        return sum(1 for emotion in recent_history if emotion in self.negative_emotions)
    
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
    
    def _calculate_temperature(self, confidence: float, context: TransitionContext) -> float:
        """Calculate temperature for response generation"""
        base_temp = 0.7
        
        # Adjust based on confidence
        if confidence > 0.8:
            temperature = base_temp * 0.5  # High confidence → deterministic
        elif confidence < 0.4:
            temperature = base_temp * 1.5  # Low confidence → exploratory
        else:
            temperature = base_temp
        
        # Adjust based on phase
        if context.negotiation_phase == 'crisis':
            temperature *= 0.7  # More deterministic in crises
        elif context.negotiation_phase == 'early':
            temperature *= 1.2  # More exploratory early on
        
        return np.clip(temperature, 0.1, 1.0)
    
    def _build_transition_explanation(self, predictions: List[AgentPrediction],
                                bayesian_result: Dict[str, Any],
                                context: TransitionContext,
                                previous_emotion: str) -> str:
        """Build human-readable explanation of the transition"""
        lines = []
        
        lines.append("🧠 EMO-MAS TRANSITION OPTIMIZATION:")
        lines.append(f"Transition: {EMOTION_NAMES[previous_emotion]} → {EMOTION_NAMES[bayesian_result['selected_emotion']]}")
        
        # FIX: Safely check for agent_agreement and total_agents
        agent_agreement = bayesian_result.get('agent_agreement', {})
        total_agents = agent_agreement.get('total_agents', len(predictions))
        lines.append(f"Mode: {'Bayesian Orchestrator' if total_agents > 1 else 'Game Theory Only'}")
        
        lines.append(f"Context: {EMOTION_NAMES[context.debtor_emotion]} debtor, {context.negotiation_phase} phase, Round {context.round_number}")
        lines.append("")
        
        if len(predictions) > 1:
            lines.append("AGENT RECOMMENDATIONS:")
            for pred in predictions:
                lines.append(f"  • {pred.agent_name}: {EMOTION_NAMES[pred.target_emotion]}")
                lines.append(f"    Confidence: {pred.confidence:.2f}, Value: {pred.transition_value:.2f}")
                lines.append(f"    Reasoning: {pred.reasoning}")
            
            lines.append("")
            lines.append("BAYESIAN COMBINATION:")
            lines.append(f"  Selected: {EMOTION_NAMES[bayesian_result['selected_emotion']]}")
            lines.append(f"  Confidence: {bayesian_result.get('confidence', 0.5):.2f}")
            lines.append(f"  Exploration: {'Yes' if bayesian_result.get('exploration', False) else 'No'}")
            
            # Show top transition probabilities (if available)
            probs = bayesian_result.get('transition_probabilities', {})
            if probs:
                top_transitions = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:3]
                lines.append("  Top Probabilities:")
                for emotion, prob in top_transitions:
                    lines.append(f"    {EMOTION_NAMES[emotion]}: {prob:.3f}")
        else:
            lines.append("GAME THEORY STRATEGY:")
            lines.append(f"  Selected: {EMOTION_NAMES[bayesian_result['selected_emotion']]}")
            lines.append(f"  Payoff-based decision for non-adversarial context")
        
        return "\n".join(lines)
    
    def update_model(self, negotiation_result: Dict[str, Any]) -> None:
        """Update model based on negotiation outcome with trajectory-based learning"""
        print(f"\n🔍 DEBUG update_model:")
        print(f"  Keys: {list(negotiation_result.keys())}")
        
        success = negotiation_result.get('success', False)
        negotiation_rounds = negotiation_result.get('negotiation_rounds', 1)
        
        # Update recent successes
        self.recent_successes.append(success)
        if len(self.recent_successes) > 10:
            self.recent_successes = self.recent_successes[-10:]
        
        if success:
            self.performance['successful_transitions'] += 1
        
        # Update Game Theory agent success tracking
        self.game_theory_agent.update_success(success)
        
        # Update emotion history from negotiator
        if 'emotion_history' in negotiation_result and negotiation_result['emotion_history']:
            passed_history = negotiation_result['emotion_history']
            self.emotion_history = passed_history.copy()
        
        # 1. Only use trajectory-based learning for orchestrator
        if 'agent_predictions_history' in negotiation_result:
            # Get emotion history
            emotion_history = negotiation_result.get('emotion_sequence', [])
            context_history = negotiation_result.get('context_history', [])
            
            if len(emotion_history) > 1:
                print('yunbo2 - Emotion sequence found!')

                # MATCH VANILLA METHOD: collection_rate = min(1.0, target_days / actual_days)
                creditor_target = negotiation_result.get('creditor_target_days', 30)
                final_days = negotiation_result.get('collection_days') or creditor_target
                
                # Collection rate using vanilla method (Higher is better)
                if final_days > 0 and creditor_target > 0:
                    collection_rate = min(1.0, creditor_target / final_days)
                else:
                    collection_rate = 0.0
                
                base_reward = 1.0 / (1 + np.log(len(emotion_history)))

                

                if success:
                    macro_reward = collection_rate * base_reward
                
                    # Clip to reasonable range
                    macro_reward = np.clip(macro_reward, 0.0, 1.0)
                else:
                    macro_reward = 0.0

                # Online RL learning from trajectory
                self.online_rl_agent.learn_from_trajectory(
                    emotion_trajectory=emotion_history,
                    context_history=context_history,
                    success=success,
                    macro_reward=macro_reward
                )
                
                # We need transitions, so skip first emotion
                emotion_trajectory = emotion_history[1:]  # Transitions from round to round
                
                # Get predictions history
                predictions_history = negotiation_result['agent_predictions_history']
                
                # DEBUG: Print alignment
                print(f"  Emotions: {len(emotion_history)} total, {len(emotion_trajectory)} transitions")
                print(f"  Predictions: {len(predictions_history)} rounds")
                
                # ALIGNMENT FIX: If predictions start from round 1 (not 0)
                # Skip the first prediction if it's for round 0 or doesn't match
                if len(predictions_history) > len(emotion_trajectory):
                    print(f"  Trimming predictions: removing first item")
                    predictions_history = predictions_history[1:]  # Skip first prediction
                
                # They should now be the same length
                if len(emotion_trajectory) != len(predictions_history):
                    print(f"⚠️ Still mismatch: {len(emotion_trajectory)} emotions vs {len(predictions_history)} predictions")
                    # Truncate to shorter one
                    min_len = min(len(emotion_trajectory), len(predictions_history))
                    emotion_trajectory = emotion_trajectory[:min_len]
                    predictions_history = predictions_history[:min_len]
                
                print(f"  Final alignment: {len(emotion_trajectory)} emotions vs {len(predictions_history)} predictions")
                
                # CALL THE LEARNING FUNCTION!
                self._learn_from_trajectory(
                    emotion_trajectory=emotion_trajectory,
                    predictions_history=predictions_history,
                    success=success,
                    negotiation_rounds=negotiation_rounds,
                    collection_rate=collection_rate  # NEW: Pass collection rate
                )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive model statistics"""
        # Create dummy context for statistics
        current_context = TransitionContext(
            current_emotion=self.current_emotion,
            debtor_emotion='N',
            negotiation_phase='middle',
            round_number=self.negotiation_round,
            emotional_history=self.emotion_history[-5:] if self.emotion_history else [],
            debt_amount=0,
            recent_success_rate=np.mean(self.recent_successes) if self.recent_successes else 0.5,
            gap_size=0
        )
        
        success_rates = self.orchestrator.get_context_success_rates(current_context)
        learning_summary = self.orchestrator.get_learning_history_summary()
        
        return {
            'current_emotion': self.current_emotion,
            'current_emotion_name': EMOTION_NAMES[self.current_emotion],
            'round_number': self.negotiation_round,
            'emotion_history': self.emotion_history[-10:],
            'performance': {
                'success_rate': (
                    self.performance['successful_transitions'] / 
                    self.performance['total_transitions'] 
                    if self.performance['total_transitions'] > 0 else 0.0
                ),
                'total_transitions': self.performance['total_transitions'],
                'successful_transitions': self.performance['successful_transitions'],
                'agent_selections': dict(self.performance['agent_selections'])
            },
            'learning': {
                'exploration_rate': self.orchestrator.exploration_rate,
                'learning_events': len(self.orchestrator.learning_history),
                'context_success_rates': success_rates,
                'learning_summary': learning_summary
            },
            # Removed HMM state section
        }
    
    def reset(self) -> None:
        """Reset model state for new negotiation"""
        # Only reset current emotion and negotiation state
        self.current_emotion = 'N'
        self.negotiation_round = 0
        self.recent_successes = []
        
        # Reset agent states but keep learned knowledge
        self.game_theory_agent.last_debtor_emotion = None
        self.game_theory_agent.last_agent_emotion = None
        self.game_theory_agent.last_successful = True
        
        # Reset Online RL experience buffer
        self.online_rl_agent.experience_buffer = []

# ========== For backward compatibility ==========
# BaseEmotionModel interface implementation
class BaseBayesianTransitionModel(BaseEmotionModel):
    """Wrapper for compatibility with existing negotiator code"""
    
    def __init__(self, exploration_rate: float = 0.2):
        self.model = BayesianTransitionModel(exploration_rate)
        self.agent_name = "BayesianTransition"
    
    def select_emotion(self, state: Dict[str, Any]) -> Dict[str, Any]:
        return self.model.select_emotion(state)
    
    def update_model(self, negotiation_result: Dict[str, Any]) -> None:
        self.model.update_model(negotiation_result)
    
    def get_stats(self) -> Dict[str, Any]:
        return self.model.get_stats()
    
    def reset(self) -> None:
        self.model.reset()



def run_bayesian_transition_experiment(
    scenarios: List[Dict[str, Any]],
    iterations: int = 10,
    model_creditor: str = "gpt-4o-mini",
    model_debtor: str = "gpt-4o-mini",
    debtor_emotion: str = "neutral",
    debtor_model_type: str = "vanilla",
    max_dialog_len: int = 30,
    out_dir: str = "results/bayesian_transition"
) -> Dict[str, Any]:
    """Run Bayesian Transition Optimization experiment with detailed logging"""
    
    from llm.negotiator_multiagent import DebtNegotiator
    import os
    
    # Create output directory
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    # Create model
    model = BayesianTransitionModel(exploration_rate=0.2)
    
    # Main results structure
    results = {
        'experiment_type': 'bayesian_transition_optimization',
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'config': {
            'iterations': iterations,
            'scenarios': len(scenarios),
            'model_creditor': model_creditor,
            'model_debtor': model_debtor,
            'debtor_emotion': debtor_emotion,
            'max_dialog_len': max_dialog_len,
            'mode': 'bayesian_orchestration'
        },
        'detailed_negotiations': [],  # Will contain ALL negotiation details
        'summary_statistics': {},
        'transition_analysis': {}
    }
    
    all_negotiation_results = []
    
    print("="*80)
    print("🧠 BAYESIAN TRANSITION OPTIMIZATION EXPERIMENT")
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
                    'collection_days': result.get('collection_days') or 0,
                    'creditor_target_days': result.get('creditor_target_days', 30),
                    'collection_rate': min(1.0, result.get('creditor_target_days', 30) / max(1, result.get('collection_days') or 30)),  # MATCH VANILLA
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
                'bayesian_decisions': [],
                'learning_updates': []
            }
            
            # 1. Extract agent predictions for each round
            if 'agent_predictions_history' in result:
                for round_idx, round_preds in enumerate(result['agent_predictions_history']):
                    round_details = {
                        'round': round_idx + 1,
                        'predictions': [],
                        'bayesian_decision': {},
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
                    
                    # Get Bayesian decision details
                    if 'bayesian_analysis' in result:
                        round_details['bayesian_decision'] = {
                            'selected_emotion': result['bayesian_analysis'].get('selected_emotion', 'N'),
                            'confidence': result['bayesian_analysis'].get('confidence', 0.5),
                            'exploration': result['bayesian_analysis'].get('exploration', False),
                            'agent_agreement': result['bayesian_analysis'].get('agent_agreement', {})
                        }
                    
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
            
            # 3. Extract Bayesian decisions from emotion_config
            if 'emotion_config' in result:
                emo_config = result['emotion_config']
                bayesian_info = {
                    'final_selected_emotion': emo_config.get('emotion', 'N'),
                    'emotion_name': emo_config.get('emotion_name', 'Neutral'),
                    'confidence': emo_config.get('confidence', 0.5),
                    'transition': emo_config.get('transition', 'N->N'),
                    'explanation': emo_config.get('explanation', ''),
                    'policy_mode': emo_config.get('policy_mode', 'unknown'),
                    'temperature': emo_config.get('temperature', 0.7)
                }
                
                # Add Bayesian analysis if available
                if 'bayesian_analysis' in emo_config:
                    bayesian_info['bayesian_analysis'] = {
                        'selection_mechanism': emo_config['bayesian_analysis'].get('selection_mechanism', ''),
                        'transition_probabilities': emo_config['bayesian_analysis'].get('transition_probabilities', {}),
                        'agent_reliability': emo_config['bayesian_analysis'].get('agent_reliability', {})
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
                    bayesian_info['agent_agreement'] = agent_agreement
                
                negotiation_details['bayesian_decisions'].append(bayesian_info)
            
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
        actual_days = r.get('collection_days') or target_days
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
    
    # Calculate transition success rates
    transition_success_counts = defaultdict(int)
    transition_total_counts = defaultdict(int)
    
    for negot in results['detailed_negotiations']:
        seq = negot['emotion_trajectory']['final_emotion_sequence']
        success = negot['outcome']['success']
        
        if seq and len(seq) > 1:
            for i in range(len(seq) - 1):
                transition = f"{seq[i]}→{seq[i+1]}"
                transition_total_counts[transition] += 1
                if success:
                    transition_success_counts[transition] += 1
    
    transition_success_rates = {
        trans: success_count / transition_total_counts[trans]
        for trans, success_count in transition_success_counts.items()
        if transition_total_counts[trans] > 0
    }
    
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
    
    # Transition analysis
    results['transition_analysis'] = {
        'transition_counts': dict(transition_counts),
        'transition_success_counts': dict(transition_success_counts),
        'transition_success_rates': transition_success_rates,
        'most_common_transitions': transition_counts.most_common(10),
        'most_successful_transitions': sorted(
            transition_success_rates.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:10]
    }
    
    # Bayesian model statistics
    results['model_statistics'] = model.get_stats()
    
    # ===== ADD STATISTICAL ANALYSIS WITH 95% CONFIDENCE INTERVALS =====
    results = enhance_results_with_statistics(
        results, 
        all_negotiation_results, 
        scenarios, 
        method="bootstrap"
    )
    
    # ===== SAVE COMPLETE RESULTS =====
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = f"{out_dir}/bayesian_transition_{timestamp}.json"
    
    with open(result_file, "w", encoding='utf-8') as f:
        json.dump(results, f, indent=2, default=str)
    
    # ===== PRINT SUMMARY WITH CONFIDENCE INTERVALS =====
    print("\n" + "="*80)
    print("📊 BAYESIAN EXPERIMENT RESULTS SUMMARY")
    print("="*80)
    
    # Print statistical analysis with CIs
    if 'statistical_analysis' in results:
        print(format_ci_results(results['statistical_analysis']))
        print("="*80)
    print("="*80)
    
    stats = results['summary_statistics']
    print(f"\n🎯 PERFORMANCE METRICS:")
    print(f"  Success Rate:        {stats['success_rate']:.1%}")
    print(f"  Collection Rate:     {stats['collection_rate']['mean']:.3f} ± {stats['collection_rate']['std']:.3f}")
    print(f"  Negotiation Rounds:  {stats['negotiation_rounds']['mean']:.1f} ± {stats['negotiation_rounds']['std']:.1f}")
    
    print(f"\n🎭 EMOTION ANALYSIS:")
    print(f"  Total Transitions:   {stats['emotion_transitions']['total_transitions']}")
    print(f"  Unique Transitions:  {stats['emotion_transitions']['unique_transitions']}")
    
    if stats['emotion_transitions']['most_common_transitions']:
        print(f"  Top 3 Transitions:")
        for trans, count in list(stats['emotion_transitions']['most_common_transitions'].items())[:3]:
            success_rate = results['transition_analysis']['transition_success_rates'].get(trans, 0)
            print(f"    {trans}: {count} times ({success_rate:.1%} success)")
    
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