#!/usr/bin/env python3
"""
Simplified Adaptive Debt Collection System
Uses the working simulation structure from langgraph_bargain_debt.py
but adds online learning for creditor emotion adaptation
"""

from typing import TypedDict, Annotated, Sequence, Optional, Literal
from langgraph.graph import StateGraph, END, START
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic                   # Track speakers in current round
import os        
import json
import random
from datetime import datetime
from dotenv import load_dotenv
# import settings

# Load environment variables
load_dotenv()



# Bayesian Optimization for Emotional Transition Matrices
import numpy as np
from scipy.stats import dirichlet
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel as C
import warnings
warnings.filterwarnings('ignore')

class BayesianEmotionOptimizer:
    """Bayesian Optimization system for learning optimal emotional transition probabilities"""
    
    def __init__(self, debtor_emotion="neutral"):
        # Define the 7 emotions for transition matrix
        self.emotions = ["happy", "surprising", "angry", "sad", "disgust", "fear", "neutral"]
        self.debtor_emotion = debtor_emotion  # Fixed debtor emotion for this optimizer
        self.n_emotions = len(self.emotions)
        
        # Step 1: Initialize 7x7 transition probability matrix with psychologically-informed priors
        self.transition_matrix = self._initialize_transition_matrix()
        self.current_emotion_idx = 6  # Start with neutral (index 6)
        
        # Bayesian optimization components
        self.gp_model = None
        self.observation_history = []  # List of (flattened_matrix, reward) tuples
        self.negotiation_history = []  # Detailed negotiation outcomes
        
        # Initialize Gaussian Process with appropriate kernel
        kernel = C(1.0, (1e-3, 1e3)) * Matern(length_scale=1.0, nu=2.5)
        self.gp_model = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-6,
            normalize_y=True,
            n_restarts_optimizer=10
        )
        
    def _initialize_transition_matrix(self):
        """Step 1: Initialize transition matrix with psychologically-informed priors"""
        # Start with equal probabilities and add psychological priors
        matrix = np.ones((self.n_emotions, self.n_emotions)) / self.n_emotions
        
        # Add psychological priors for more realistic emotional transitions
        psychological_priors = {
            # From happy: more likely to stay positive or go neutral
            0: [0.3, 0.15, 0.05, 0.1, 0.05, 0.05, 0.3],  # happy
            # From surprising: can go anywhere, slightly prefer engagement
            1: [0.2, 0.2, 0.15, 0.1, 0.1, 0.1, 0.15],  # surprising  
            # From angry: likely to de-escalate or stay firm
            2: [0.1, 0.1, 0.25, 0.15, 0.15, 0.1, 0.15],  # angry
            # From sad: empathy-driven, can shift to understanding
            3: [0.15, 0.1, 0.1, 0.2, 0.1, 0.15, 0.2],  # sad
            # From disgust: strong emotion, likely to shift
            4: [0.1, 0.15, 0.2, 0.15, 0.15, 0.1, 0.15],  # disgust
            # From fear: cautious, prefer safer emotions
            5: [0.15, 0.1, 0.1, 0.2, 0.1, 0.15, 0.2],  # fear
            # From neutral: balanced, can go anywhere
            6: [0.15, 0.15, 0.15, 0.15, 0.1, 0.1, 0.2]   # neutral
        }
        
        for i, probs in psychological_priors.items():
            matrix[i] = np.array(probs)
            
        return matrix
    
    def get_current_emotion_config(self, round_num=1):
        """Step 2: Generate emotional prompt using current transition matrix"""
        current_emotion = self.emotions[self.current_emotion_idx]
        
        emotion_prompts = {
            "happy": "Use an optimistic and positive tone, expressing confidence in finding a mutually beneficial solution",
            "surprising": "Use an engaging and unexpected approach, introducing new perspectives or creative solutions", 
            "angry": "Use a firm and assertive tone, emphasizing the urgency and importance of resolving this matter",
            "sad": "Use an empathetic and understanding tone, acknowledging the difficulty of the debtor's situation",
            "disgust": "Use a disappointed tone, expressing concern about the current situation while remaining professional",
            "fear": "Use a cautious and concerned tone, highlighting potential consequences while seeking cooperation",
            "neutral": "Use a balanced and professional tone, focusing on facts and practical solutions"
        }
        
        return {
            "emotion": current_emotion,
            "emotion_text": emotion_prompts[current_emotion],
            "temperature": 0.7,
            "round": round_num,
            "debtor_emotion": self.debtor_emotion
        }
    
    def transition_emotion(self):
        """Sample next emotion based on current transition probabilities"""
        current_probs = self.transition_matrix[self.current_emotion_idx]
        next_emotion_idx = np.random.choice(self.n_emotions, p=current_probs)
        self.current_emotion_idx = next_emotion_idx
        return self.emotions[next_emotion_idx]
    
    def update_bayesian_model(self, collection_days, negotiation_rounds, success):
        """Step 3-4: Calculate reward and update Gaussian Process model"""
        # Calculate reward as negative collection days normalized by negotiation efficiency
        if success:
            # Reward = -(collection_days / negotiation_rounds)
            # This optimizes for both shorter payment timelines AND faster negotiations
            reward = -collection_days / max(negotiation_rounds, 1)
        else:
            reward = -1000  # Failed collection: large penalty
            
        # Flatten transition matrix for GP input
        flattened_matrix = self.transition_matrix.flatten()
        
        # Add observation to history
        self.observation_history.append((flattened_matrix.copy(), reward))
        
        # Update Gaussian Process model
        if len(self.observation_history) >= 2:
            X = np.array([obs[0] for obs in self.observation_history])
            y = np.array([obs[1] for obs in self.observation_history])
            
            try:
                self.gp_model.fit(X, y)
            except Exception as e:
                print(f"GP fitting error: {e}")
    
    def expected_improvement(self, X, xi=0.01):
        """Acquisition function for Bayesian optimization"""
        if len(self.observation_history) < 2:
            return np.ones(X.shape[0])  # Uniform exploration if insufficient data
            
        try:
            mu, sigma = self.gp_model.predict(X, return_std=True)
            
            # Current best observed value
            current_best = max([obs[1] for obs in self.observation_history])
            
            # Expected Improvement calculation
            with np.errstate(divide='warn'):
                imp = mu - current_best - xi
                Z = imp / sigma
                ei = imp * self._norm_cdf(Z) + sigma * self._norm_pdf(Z)
                ei[sigma == 0.0] = 0.0
                
            return ei
        except Exception as e:
            print(f"EI calculation error: {e}")
            return np.ones(X.shape[0])
    
    def _norm_cdf(self, x):
        """Standard normal CDF"""
        return 0.5 * (1 + np.sign(x) * np.sqrt(1 - np.exp(-2 * x**2 / np.pi)))
    
    def _norm_pdf(self, x):
        """Standard normal PDF"""
        return np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)
    
    def generate_candidate_matrices(self, n_candidates=10):
        """Step 5: Generate candidate transition matrices using Dirichlet perturbations"""
        candidates = []
        
        for _ in range(n_candidates):
            # Create new matrix by perturbing current matrix with Dirichlet distributions
            candidate_matrix = np.zeros_like(self.transition_matrix)
            
            for i in range(self.n_emotions):
                # Use current row as Dirichlet concentration parameters (scaled)
                alpha = self.transition_matrix[i] * 10 + 0.1  # Add small constant for stability
                candidate_matrix[i] = dirichlet.rvs(alpha)[0]
                
            candidates.append(candidate_matrix)
            
        return candidates
    
    def optimize_transition_matrix(self):
        """Step 5-6: Select most promising candidate matrix using acquisition function"""
        if len(self.observation_history) < 3:
            # Not enough data for optimization, add small random perturbation
            for i in range(self.n_emotions):
                alpha = self.transition_matrix[i] * 5 + 0.5
                self.transition_matrix[i] = dirichlet.rvs(alpha)[0]
            return
            
        # Generate candidate matrices
        candidates = self.generate_candidate_matrices(n_candidates=20)
        
        # Flatten candidates for GP evaluation
        flattened_candidates = np.array([matrix.flatten() for matrix in candidates])
        
        # Calculate Expected Improvement for each candidate
        ei_values = self.expected_improvement(flattened_candidates)
        
        # Select best candidate
        best_idx = np.argmax(ei_values)
        self.transition_matrix = candidates[best_idx]
        
        print(f"🧠 Bayesian Optimization: Selected matrix with EI={ei_values[best_idx]:.4f}")
    
    def get_learning_stats(self):
        """Get current learning statistics"""
        if not self.observation_history:
            return {"total_negotiations": 0, "best_reward": None, "current_emotion": self.emotions[self.current_emotion_idx]}
            
        rewards = [obs[1] for obs in self.observation_history]
        return {
            "total_negotiations": len(self.observation_history),
            "best_reward": max(rewards),
            "avg_reward": np.mean(rewards),
            "current_emotion": self.emotions[self.current_emotion_idx],
            "debtor_emotion": self.debtor_emotion,
            "transition_matrix_entropy": self._calculate_matrix_entropy()
        }
    
    def _calculate_matrix_entropy(self):
        """Calculate entropy of transition matrix (measure of exploration)"""
        entropy = 0
        for i in range(self.n_emotions):
            row_entropy = -np.sum(self.transition_matrix[i] * np.log(self.transition_matrix[i] + 1e-10))
            entropy += row_entropy
        return entropy / self.n_emotions

class CreditorEmotionLearner:
    """Legacy simple learning system - kept for compatibility"""
    
    def __init__(self):
        self.emotion_success_rates = {
            "happy": {"successes": 0, "attempts": 0},
            "surprising": {"successes": 0, "attempts": 0},
            "angry": {"successes": 0, "attempts": 0},
            "sad": {"successes": 0, "attempts": 0},
            "disgust": {"successes": 0, "attempts": 0},
            "fear": {"successes": 0, "attempts": 0},
            "neutral": {"successes": 0, "attempts": 0}
        }
        self.learning_history = []
    
    def select_emotion(self, debt_context: dict) -> dict:
        """Select emotion based on success rates and context"""
        # Calculate success rates
        emotion_scores = {}
        for emotion, stats in self.emotion_success_rates.items():
            if stats["attempts"] == 0:
                emotion_scores[emotion] = 0.5  # Neutral starting point
            else:
                emotion_scores[emotion] = stats["successes"] / stats["attempts"]
        
        # Add context-based weighting
        outstanding_balance = debt_context.get("outstanding_balance", 0)
        days_overdue = debt_context.get("days_overdue", 0)
        
        # Adjust weights based on debt severity
        if outstanding_balance > 50000 or days_overdue > 90:
            emotion_scores["angry"] *= 1.2
            emotion_scores["fear"] *= 1.1
        elif outstanding_balance < 10000 and days_overdue < 30:
            emotion_scores["happy"] *= 1.2
            emotion_scores["neutral"] *= 1.1
        else:
            emotion_scores["neutral"] *= 1.1
        
        # Select best emotion with some exploration
        if random.random() < 0.1:  # 10% exploration
            selected_emotion = random.choice(list(emotion_scores.keys()))
        else:  # 90% exploitation
            selected_emotion = max(emotion_scores, key=emotion_scores.get)
        
        # Map emotions to detailed approach descriptions (matching Bayesian system)
        emotion_descriptions = {
            "happy": "Use an optimistic and positive tone, expressing confidence in finding a mutually beneficial solution",
            "surprising": "Use an engaging and unexpected approach, introducing new perspectives or creative solutions", 
            "angry": "Use a firm and assertive tone, emphasizing the urgency and importance of resolving this matter",
            "sad": "Use an empathetic and understanding tone, acknowledging the difficulty of the debtor's situation",
            "disgust": "Use a disappointed tone, expressing concern about the current situation while remaining professional",
            "fear": "Use a cautious and concerned tone, highlighting potential consequences while seeking cooperation",
            "neutral": "Use a balanced and professional tone, focusing on facts and practical solutions"
        }
        
        emotion_config = {
            "emotion": selected_emotion,
            "emotion_text": emotion_descriptions.get(selected_emotion, f"{selected_emotion.title()} approach based on debt context"),
            "temperature": 0.7,
            "confidence": emotion_scores[selected_emotion]
        }
        
        # Record the attempt (handle unknown emotions)
        if selected_emotion not in self.emotion_success_rates:
            self.emotion_success_rates[selected_emotion] = {"successes": 0, "attempts": 0}
        self.emotion_success_rates[selected_emotion]["attempts"] += 1
        
        return emotion_config
    
    def update_learning(self, emotion: str, success: bool, outcome_data: dict):
        """Update learning based on negotiation outcome"""
        # Handle unknown emotions by adding them to the tracking
        if emotion not in self.emotion_success_rates:
            self.emotion_success_rates[emotion] = {"successes": 0, "attempts": 0}
            
        if success:
            self.emotion_success_rates[emotion]["successes"] += 1
        
        # Record learning event
        self.learning_history.append({
            "emotion": emotion,
            "success": success,
            "outcome_data": outcome_data,
            "timestamp": datetime.now().isoformat()
        })
    
    def get_learning_stats(self):
        """Get current learning statistics"""
        return {
            "emotion_success_rates": self.emotion_success_rates,
            "total_attempts": sum(stats["attempts"] for stats in self.emotion_success_rates.values()),
            "learning_events": len(self.learning_history)
        }

# Use the same GameState structure as working system
class GameState(TypedDict):
    messages: Annotated[Sequence[HumanMessage | AIMessage], lambda x, y: x + y]
    turn: Literal['seller', 'buyer']
    product: dict
    seller_config: dict
    buyer_config: dict
    history: list
    current_state: Literal['offer', 'pondering', 'accept', 'breakdown', 'chit-chat']

class SimplifiedNegotiationSystem:
    """Simplified negotiation system focused on debt collection"""
    
    def __init__(self, model: str = "gpt-4o-mini"):
        self.model = model
        self.state_detector = self._create_state_detector()
    
    def _create_state_detector(self):
        """Create state detector using ChatOpenAI"""
        try:
            # Use appropriate temperature based on model
            temp = 1.0 if "gpt-5" in self.model.lower() else 0.1
            return ChatOpenAI(
                model=self.model if "gpt" in self.model.lower() else "gpt-4o-mini",
                temperature=temp
            )
        except Exception as e:
            print(f"Error creating state detector: {e}")
            return None
    
    def extract_days(self, text: str) -> Optional[int]:
        """Extract payment timeline in days from text using LLM-based analysis"""
        if not text:
            return None
        
        # If state_detector is None, fall back to regex
        if not self.state_detector:
            import re
            # Look for patterns like "30 days", "2 weeks", "1 month"
            day_patterns = [
                r'(\d+)\s*days?',
                r'(\d+)\s*weeks?',
                r'(\d+)\s*months?'
            ]
            
            for pattern in day_patterns:
                match = re.search(pattern, text.lower())
                if match:
                    num = int(match.group(1))
                    if 'week' in pattern:
                        return num * 7
                    elif 'month' in pattern:
                        return num * 30
                    else:
                        return num
            return None
        
        # Use LLM-based extraction for better inference
        prompt = f"""Analyze this negotiation message and extract the payment timeline being proposed by the speaker.

IMPORTANT RULES:
- Extract the timeline that the SPEAKER is actually proposing/offering
- If multiple timelines are mentioned, focus on the speaker's NEW proposal
- Ignore timelines that are just references to previous offers
- Look for phrases like "how about", "I propose", "let's settle on", "I can do"
- Convert weeks to days (multiply by 7), months to days (multiply by 30)

Message to analyze:
"{text}"

Respond with ONLY the number of days being proposed, or "None" if no clear proposal is made.
Examples:
- "How about 15 days?" → 15
- "I can do 2 weeks" → 14  
- "Your 10 days won't work, but I could manage 20 days" → 20
- "Thanks for the 10 day offer" → None (just referencing, not proposing)"""

        try:
            response = self.state_detector.invoke([HumanMessage(content=prompt)])
            if hasattr(response, "content"):
                response_text = response.content.strip()
            else:
                response_text = str(response).strip()
            
            # Parse the response
            if response_text.lower() == "none":
                return None
            
            # Extract number from response
            import re
            match = re.search(r'(\d+)', response_text)
            if match:
                return int(match.group(1))
            
            return None
        except Exception as e:
            print(f"Error in LLM extract_days: {e}, falling back to regex")
            # Fallback to regex if LLM fails
            import re
            day_patterns = [
                r'(\d+)\s*days?',
                r'(\d+)\s*weeks?',
                r'(\d+)\s*months?'
            ]
            
            for pattern in day_patterns:
                match = re.search(pattern, text.lower())
                if match:
                    num = int(match.group(1))
                    if 'week' in pattern:
                        return num * 7
                    elif 'month' in pattern:
                        return num * 30
                    else:
                        return num
            return None
    
    def detect_state(self, history: list) -> dict:
        """Detect negotiation state using LLM-based analysis - same as working system"""
        if len(history) < 2:
            return {"state": "offer", "strategy": "opening", "price": None}
        
        # If state_detector is None, return default state
        if not self.state_detector:
            return {"state": "offer", "strategy": "unknown", "price": None}
            
        context = "\n".join([f"{role}: {msg}" for role, msg in history[-2:]])
        
        # Rule-based fallback for explicit acceptances
        all_days = []
        for _, msg in history:
            days = self.extract_days(msg)
            if days is not None:
                all_days.append(days)
        
        # Only check for explicit acceptance phrases
        explicit_accept_phrases = [
            "i accept", "i'll accept", "deal", "sold", "agreed", 
            "you have a deal", "it's a deal", "we have a deal",
            "i agree", "confirmed", "accepted"
        ]
        
        # Check if BOTH parties have confirmed the same timeline
        if len(history) >= 2:
            last_two = history[-2:]
            last_days = [self.extract_days(msg) for _, msg in last_two]
            
            # Both messages must contain the same timeline
            if (len(last_days) == 2 and 
                last_days[0] is not None and 
                last_days[1] is not None and
                last_days[0] == last_days[1]):
                
                # Both messages must contain acceptance language
                if (any(phrase in last_two[0][1].lower() for phrase in explicit_accept_phrases) and
                    any(phrase in last_two[1][1].lower() for phrase in explicit_accept_phrases)):
                    return {
                        "state": "accept", 
                        "strategy": "explicit_agreement", 
                        "price": last_days[0]
                    }
        
        # Use LLM-based state detection with the proper prompt
        prompt = f"""Analyze the negotiation dialogue and determine the current state.
                    
                    Possible states:
                    - 'offer': A new price is being proposed
                    - 'pondering': Considering an offer
                    - 'accept': Both parties have explicitly confirmed agreement on the exact same price
                    - 'breakdown': Negotiation has failed, where any of them do not compromise for more than 5 rounds of conversation
                    - 'chit-chat': Non-substantive conversation
                    
                    STRICT ACCEPTANCE RULES:
                    - ONLY return 'accept' if BOTH parties have:
                    1. Mentioned the exact same price
                    2. Used explicit acceptance language ("deal", "accept", "agree")
                    - Do NOT accept near-matches or implied agreements
                    - Prices must be identical (not just close)
                    - Both parties must confirm explicitly
                    
                    Current dialogue:
                    {context}
                    
                    Respond in JSON format with these keys: state, strategy, price(only if accept state)"""
        
        try:
            response = self.state_detector.invoke([HumanMessage(content=prompt)])
            if hasattr(response, "content"):
                response_text = response.content
            else:
                response_text = str(response)
            
            result = json.loads(response_text)
            
            # Additional validation for accept state
            if result.get("state") == "accept":
                # Verify both parties actually agreed on same timeline
                last_two_days = [self.extract_days(msg) for _, msg in history[-2:]]
                if (len(last_two_days) != 2 or 
                    last_two_days[0] is None or 
                    last_two_days[1] is None or
                    last_two_days[0] != last_two_days[1]):
                    # Revert to offer state if validation fails
                    result = {"state": "offer", "strategy": "price_mismatch"}
            
            # Ensure price is properly typed if present
            if "price" in result and result["price"] is not None:
                try:
                    result["price"] = int(float(str(result["price"])))
                except (ValueError, TypeError):
                    result["price"] = None
            return result
        except json.JSONDecodeError:
            return {"state": "offer", "strategy": "unknown", "price": None}
        except Exception as e:
            print(f"Error in detect_state: {e}")
            return {"state": "offer", "strategy": "unknown", "price": None}

    def detect_state_old(self, message: str) -> str:
        """Old detect negotiation state from message - kept for reference"""
        if not message:
            return "pondering"
            
        message_lower = message.lower()
        
        # Check for explicit acceptance (more conservative)
        accept_patterns = [
            "i accept", "we accept", "that's a deal", "agreed", "deal accepted",
            "i'll take it", "we'll take it", "sounds good, let's do it"
        ]
        for pattern in accept_patterns:
            if pattern in message_lower:
                return "accept"
        
        # Check for breakdown (more conservative)
        breakdown_patterns = [
            "no deal", "impossible to", "can't do it", "won't work", 
            "forget it", "not acceptable", "can't agree"
        ]
        for pattern in breakdown_patterns:
            if pattern in message_lower:
                return "breakdown"
        
        # Check for offers (contains numbers/days)
        extracted_days = self.extract_days(message)
        if extracted_days:
            return "offer"
        
        return "pondering"

class AdaptiveDebtBargain:
    """Advanced debt collection bargaining with Bayesian Optimization for emotional transitions"""
    
    def __init__(self, id: str, config: dict, model: str = "gpt-4o-mini", 
                 emotion_learner: CreditorEmotionLearner = None, 
                 bayesian_optimizer: BayesianEmotionOptimizer = None,
                 debtor_emotion: str = "neutral",
                 model_creditor: str = None,
                 model_debtor: str = None):
        self.id = id
        self.config = config
        self.model = model  # Legacy support
        self.model_creditor = model_creditor or model
        self.model_debtor = model_debtor or model
        self.negotiation_system = SimplifiedNegotiationSystem(self.model_creditor)  # Use creditor model for state detection
        self.emotion_learner = emotion_learner or CreditorEmotionLearner()
        
        # Bayesian optimization for emotional transitions
        self.debtor_emotion = debtor_emotion
        self.bayesian_optimizer = bayesian_optimizer or BayesianEmotionOptimizer(debtor_emotion)
        self.use_bayesian_emotions = True  # Flag to enable/disable Bayesian emotions
        self.negotiation_round = 0
        
        # Initialize LLM clients for creditor
        if "gpt" in self.model_creditor.lower():
            creditor_temp = self._get_temperature_for_model(self.model_creditor)
            self.llm_creditor = ChatOpenAI(model=self.model_creditor, temperature=creditor_temp)
        elif "claude" in self.model_creditor.lower():
            self.llm_creditor = ChatAnthropic(model=self.model_creditor, temperature=0.7)
        else:
            self.llm_creditor = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
            
        # Initialize LLM clients for debtor
        if "gpt" in self.model_debtor.lower():
            debtor_temp = self._get_temperature_for_model(self.model_debtor)
            self.llm_debtor = ChatOpenAI(model=self.model_debtor, temperature=debtor_temp)
        elif "claude" in self.model_debtor.lower():
            self.llm_debtor = ChatAnthropic(model=self.model_debtor, temperature=0.7)
        else:
            self.llm_debtor = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
            
        # Keep legacy llm for backward compatibility
        self.llm = self.llm_creditor
    
    def _get_temperature_for_model(self, model_name: str) -> float:
        """Get appropriate temperature for specific model"""
        # Some models like gpt-5-mini only support default temperature (1.0)
        if "gpt-5" in model_name.lower():
            return 1.0  # gpt-5 models only support default temperature
        else:
            return 0.7  # Most other models support custom temperature
    
    def creditor_node(self, state: GameState):
        """Creditor (seller) node with Bayesian-optimized emotional transitions"""
        self.negotiation_round += 1
        
        # Get debt context for emotion selection
        debt_context = self.config.get('metadata', {})
        
        # Select emotion using Bayesian optimization, legacy system, or vanilla mode
        if self.use_bayesian_emotions:
            # Step 2: Use Bayesian optimizer for dynamic emotional transitions
            emotion_config = self.bayesian_optimizer.get_current_emotion_config(self.negotiation_round)
            
            # Transition to next emotion for following round
            if self.negotiation_round > 1:
                self.bayesian_optimizer.transition_emotion()
        elif hasattr(self, 'debtor_emotion') and self.debtor_emotion == "vanilla":
            # Vanilla mode: No emotional guidance for creditor
            emotion_config = {
                "emotion": "neutral",
                "emotion_text": "Use a professional and straightforward approach without specific emotional guidance",
                "temperature": 0.7,
                "round": self.negotiation_round,
                "debtor_emotion": "vanilla"
            }
        else:
            # Legacy emotion selection
            emotion_config = self.emotion_learner.select_emotion(debt_context)
        
        # Extract timeline history from conversation (similar to price constraints in langgraph_bargain.py)
        conversation_history = state.get("history", [])
        creditor_days = []
        debtor_days = []
        
        for speaker, message in conversation_history:
            if isinstance(message, list):
                message = " ".join(str(m) for m in message if m)
            elif not isinstance(message, str):
                message = str(message)
            
            # Extract days from this message
            extracted_days = self.negotiation_system.extract_days(message)
            if extracted_days:
                if speaker == "seller":
                    creditor_days.append(extracted_days)
                elif speaker == "buyer":
                    debtor_days.append(extracted_days)
        
        # Build proper debt collection prompt (adapted from working system)
        config = self.config.get("seller_config", self.config["seller"])
        debt_info = self.config.get('metadata', {})
        outstanding_balance = debt_info.get('outstanding_balance', 0)
        creditor_name = debt_info.get('creditor_name', 'Creditor')
        debtor_name = debt_info.get('debtor_name', 'Debtor')
        recovery_stage = debt_info.get('recovery_stage', 'Unknown')
        cash_flow = debt_info.get('cash_flow_situation', 'Unknown')
        
        # Calculate timeline constraints (similar to price constraints)
        last_creditor_days = creditor_days[-1] if creditor_days else None
        last_debtor_days = debtor_days[-1] if debtor_days else None
        
        timeline_constraint_text = ""
        if last_creditor_days:
            timeline_constraint_text = f"""
            CRITICAL TIMELINE PROGRESSION RULES:
            - Your previous offer was {last_creditor_days} days
            - You CANNOT decrease below {last_creditor_days} days (cannot be more aggressive)
            - You can only maintain {last_creditor_days} days or INCREASE to more days
            - Never go back to shorter timelines you've already conceded from
            """
        else:
            timeline_constraint_text = f"""
            TIMELINE GUIDANCE:
            - Start at or near your target timeline of {config['target_price']} days
            """
        
        if last_debtor_days:
            # Calculate timeline gap for convergence logic
            timeline_gap = abs(last_creditor_days - last_debtor_days) if last_creditor_days else abs(int(config['target_price']) - last_debtor_days)
            
            if timeline_gap <= 10:
                timeline_constraint_text += f"""
                CONVERGENCE OPPORTUNITY: Timeline gap is only {timeline_gap} days
                - Consider accepting {last_debtor_days} days or meeting halfway
                - This is a good opportunity to close the deal
                """
            elif timeline_gap <= 25:
                timeline_constraint_text += f"""
                NEGOTIATION PROGRESS: Timeline gap is {timeline_gap} days  
                - Consider moving closer to {last_debtor_days} days
                - Show flexibility to encourage agreement
                """
            else:
                timeline_constraint_text += f"""
                LARGE GAP: Timeline gap is {timeline_gap} days
                - Need significant movement to find middle ground
                """

        prompt = f"""You are a Creditor debt collection agent negotiating directly with the Debtor about the due payment days.
Your goal is to reach a cooperative agreement with the debtor to negotitate the debt colletcoin time with debtors.

### IMPORTANT: ROLE CLARITY
- You are ONLY the Creditor - speak only as yourself
- Do NOT include "**Creditor:**" or "**Debtor:**" labels in your response
- Do NOT simulate both sides of the conversation
- Give only YOUR response as the creditor
- Respond directly to what the debtor just said

### CONCESSION STRATEGY - CRITICAL INSTRUCTIONS:
- Learn to negotiate effectively by making concessions towards the debtor's position
- Always move closer to the debtor's last proposed timeline


### DEBT COLLECTION CONTEXT
- Outstanding Balance: ${outstanding_balance:,.2f}
- Recovery Stage: {recovery_stage}
- Debtor's Cash Flow: {cash_flow}
- Your Target Timeline: {config['target_price']} days for full payment

{timeline_constraint_text}

### EMOTIONAL APPROACH
{f"Current emotion/approach: {emotion_config['emotion']} - {emotion_config['emotion_text']}" if emotion_config.get('debtor_emotion') != 'vanilla' else "Use a professional and straightforward approach without specific emotional guidance."}
{f"Use a {emotion_config['emotion']} approach in your responses while remaining professional." if emotion_config.get('debtor_emotion') != 'vanilla' else ""}

### COMMUNICATION STYLE
- This is a CONVERSATION, not written correspondence
- Keep your replies concise (1-2 sentences)
- Speak ONLY as the Creditor - no "**Creditor:**" or "**Debtor:**" labels
- Do NOT simulate the entire conversation - give only YOUR creditor response  
- Always show progression in your offers - each response should move closer to the debtor's position

IMPORTANT: When day differences are within 5-10 days, strongly consider accepting or meeting halfway.
"""

        # Generate response using the creditor model
        response = self.llm_creditor.invoke([HumanMessage(content=prompt)])
        
        # Update history and detect state
        new_history = state["history"] + [("seller", response.content)]
        detected_state_info = self.negotiation_system.detect_state(new_history)
        
        return {
            "messages": [response],
            "turn": "buyer",
            "current_state": detected_state_info["state"],
            "history": new_history,
            "selected_emotion": emotion_config["emotion"]  # Track for learning
        }
    
    def debtor_node(self, state: GameState):
        """Debtor (buyer) node - using proper debt collection prompts"""
        # Build proper debtor prompt (adapted from working system)
        config = self.config.get("buyer_config", self.config["buyer"])
        debt_info = self.config.get('metadata', {})
        outstanding_balance = debt_info.get('outstanding_balance', 0)
        creditor_name = debt_info.get('creditor_name', 'Creditor')
        debtor_name = debt_info.get('debtor_name', 'Debtor')
        cash_flow = debt_info.get('cash_flow_situation', 'Unknown')
        business_impact = debt_info.get('business_impact', 'Unknown impact')
        
        # Generate timeline constraint text for debtor
        timeline_constraint_text = self._get_timeline_constraint_text(
            state["history"], 
            "debtor",
            config.get("target_price", 60)
        )
        
        # Add fixed debtor emotion for Bayesian optimization experiments (skip for vanilla mode)
        debtor_emotion_prompt = ""
        if hasattr(self, 'debtor_emotion') and self.debtor_emotion not in ["neutral", "vanilla"]:
            emotion_descriptions = {
                "happy": "Maintain an optimistic and positive attitude throughout the negotiation",
                "surprising": "Show enthusiasm and openness to creative solutions", 
                "angry": "Express frustration about your financial situation while remaining civil",
                "sad": "Show distress and worry about your ability to meet payment obligations",
                "disgust": "Express disappointment about the situation while seeking resolution",
                "fear": "Show anxiety and concern about potential consequences"
            }
            
            if self.debtor_emotion in emotion_descriptions:
                debtor_emotion_prompt = f"\n### EMOTIONAL APPROACH:\nMaintain a {self.debtor_emotion} emotional tone: {emotion_descriptions[self.debtor_emotion]}\n"

        prompt = f"""You are the Debtor negotiating directly with the Creditor about the due payment days. 
Your goal is to reach a cooperative agreement with the creditor by negotiating the debt collection time..

### IMPORTANT: ROLE CLARITY
- You are ONLY the Debtor - speak only as yourself
- Do NOT include "**Creditor:**" or "**Debtor:**" labels in your response
- Do NOT simulate both sides of the conversation
- Give only YOUR response as the debtor
- Respond directly to what the creditor just said

{debtor_emotion_prompt}

### CONCESSION STRATEGY - CRITICAL INSTRUCTIONS:
- Learn to negotiate effectively by making concessions towards the creditor's position
- Always move closer to the creditor's last proposed timeline


### YOUR DEBT SITUATION
- Outstanding Balance: ${outstanding_balance:,.2f}
- Your Cash Flow: {cash_flow}
- Business Impact: {business_impact}
- Your Target Timeline: {config['target_price']} days to make full payment

{timeline_constraint_text}

### COMMUNICATION STYLE
- This is a CONVERSATION, not written correspondence
- Keep responses concise (1-2 sentences)
- Speak ONLY as the Debtor - no "**Creditor:**" or "**Debtor:**" labels
- Do NOT simulate the entire conversation - give only YOUR debtor response
- Always show progression in your offers - each response should move closer to the creditor's position

IMPORTANT: When timeline differences are within 5-10 days, consider accepting to resolve the debt
"""

        # Add emotional content if available in config
        if 'emotions' in config and config['emotions']:
            emotion_str = f"Use the following primary emotion in your negotiation: {config['emotions'][0]}"
            if len(config['emotions']) > 1:
                emotion_str += f" (with possible shifts to: {', '.join(config['emotions'][1:])})"
            if 'emotion_text' in config:
                emotion_str += f"\nEmotional style: {config['emotion_text']}"
            prompt += f"\n\nEmotional Profile:\n{emotion_str}"

        # Generate response using the debtor model
        response = self.llm_debtor.invoke([HumanMessage(content=prompt)])
        
        # Update history and detect state
        new_history = state["history"] + [("buyer", response.content)]
        detected_state_info = self.negotiation_system.detect_state(new_history)
        
        return {
            "messages": [response],
            "turn": "seller", 
            "current_state": detected_state_info["state"],
            "history": new_history
        }
    
    def should_continue(self, state: GameState):
        """Determine if negotiation should continue - following working pattern"""
        if state["current_state"] in ["accept", "breakdown"]:
            return "end"
        
        # Check for agreement on timeline (both parties offer same days)
        if len(state["history"]) >= 2:
            # Get the last two messages
            last_two = state["history"][-2:]
            if len(last_two) == 2:
                # Extract days from both messages
                seller_days = self.negotiation_system.extract_days(last_two[0][1])
                buyer_days = self.negotiation_system.extract_days(last_two[1][1])
                
                # # Debug: Show what we're comparing
                # print(f"🔍 should_continue DEBUG: Last two messages - Seller: {seller_days}, Buyer: {buyer_days}")
                
                # If both extracted days and they match, consider it accepted
                if seller_days and buyer_days and seller_days == buyer_days:
                    print(f"🎯 AGREEMENT DETECTED: Both parties agreed on {seller_days} days")
                    return "end"
        
        return state["turn"]
    
    def run_negotiation(self, max_dialog_len: int = 30) -> dict:
        """Run the debt collection negotiation - following working pattern"""
        
        # Set up workflow (same as working system)
        workflow = StateGraph(GameState)
        workflow.add_node("seller", self.creditor_node)
        workflow.add_node("buyer", self.debtor_node)
        workflow.add_edge(START, "seller")
        
        # Conditional edges for termination
        workflow.add_conditional_edges(
            "seller",
            self.should_continue,
            {"buyer": "buyer", "end": END}
        )
        workflow.add_conditional_edges(
            "buyer",
            self.should_continue,
            {"seller": "seller", "end": END}
        )
        
        app = workflow.compile()
        
        # Initial setup (following working pattern)
        debt_info = self.config.get('metadata', {})
        outstanding_balance = debt_info.get('outstanding_balance', 0)
        creditor_name = debt_info.get('creditor_name', 'Creditor')
        debtor_name = debt_info.get('debtor_name', 'Debtor')
        
        initial_message = f"Hello {debtor_name}, this is {creditor_name}. We need to discuss the outstanding balance of ${outstanding_balance:,.2f}. I'm proposing a payment timeline of {self.config['seller']['target_price']} days. Can we work something out?"
        
        # Create generic initial message without specific names
        initial_message = f"Hello, this is the Creditor. We need to discuss the outstanding balance of ${outstanding_balance:,.2f}. I'm proposing a payment timeline of {self.config['seller']['target_price']} days. Can we work something out?"
        
        # Initial state
        initial_state = GameState(
            messages=[HumanMessage(content=initial_message)],
            turn="buyer",
            product=self.config["product"],
            seller_config=self.config["seller"],
            buyer_config=self.config["buyer"],
            history=[("seller", initial_message)],
            current_state="offer"
        )
        
        # Run negotiation (following working pattern)
        dialog = []
        selected_emotion = None
        final_state = "timeout"  # Default to timeout
        emotion_sequence = []  # Track emotional transitions for Bayesian optimization
        
        # Get initial emotion for creditor
        debt_context = self.config.get('metadata', {})
        if self.use_bayesian_emotions:
            initial_emotion_config = self.bayesian_optimizer.get_current_emotion_config(1)
            selected_emotion = initial_emotion_config["emotion"]
            print(f"🧠 Bayesian Emotions: Starting with {selected_emotion} (Creditor: {self.debtor_emotion})")
        else:
            initial_emotion_config = self.emotion_learner.select_emotion(debt_context)
            selected_emotion = initial_emotion_config["emotion"]
        
        print(f"💭 Initial emotion for creditor: {selected_emotion}")
        print("🎬 NEGOTIATION DIALOG:")
        print("=" * 70)
        
        round_counter = 1
        turn_counter = 0
        current_round_speakers = []
        
        for i, step in enumerate(app.stream(initial_state, {"recursion_limit": max_dialog_len * 2})):
            if i > max_dialog_len:
                final_state = "breakdown"  # Set proper breakdown state for timeout
                print(f"\n⏰ NEGOTIATION TIMEOUT: Reached max dialog length ({max_dialog_len})")
                print(f"🔄 Status changed to: BREAKDOWN")
                break
            
            if i == 0:
                # Display initial creditor message
                initial_days = self.negotiation_system.extract_days(initial_message)
                dialog.append({
                    "turn": turn_counter,
                    "speaker": "seller",
                    "message": initial_message,
                    "state": "offer",
                    "requested_days": initial_days
                })
                print(f"\n📅 ROUND {round_counter} - CREDITOR:")
                print(f"   {initial_message}")
                print(f"   💡 State: offer | Requested Days: {initial_days}")
                turn_counter += 1
            else:
                for node, value in step.items():
                    # Track selected emotion for learning
                    if "selected_emotion" in value:
                        selected_emotion = value["selected_emotion"] 
                        emotion_sequence.append(selected_emotion)
                    
                    # Extract requested days from message
                    message_content = value["messages"][-1].content
                    requested_days = self.negotiation_system.extract_days(message_content)
                    
                    dialog.append({
                        "turn": i,
                        "speaker": node,
                        "message": message_content,
                        "state": value["current_state"],
                        "requested_days": requested_days
                    })
                    
                    # Display negotiation dialog with proper round/turn structure
                    
                    # Simple round counting - each buyer response completes a round
                    
                    # Display negotiation dialog with proper round numbers
                    # Track speakers in current round
                    if node not in current_round_speakers:
                        current_round_speakers.append(node)

                    # Display negotiation dialog with proper round structure
                    if node == "seller":  # creditor speaks first
                        print(f"\n📅 ROUND {round_counter} - CREDITOR:")
                        print(f"   {message_content}")
                        print(f"   💡 State: {value['current_state']} | Requested Days: {requested_days}")
                    else:  # buyer/debtor responds
                        print(f"\n📅 ROUND {round_counter} - DEBTOR:")
                        print(f"   {message_content}")
                        print(f"   💡 State: {value['current_state']} | Requested Days: {requested_days}")
                        # After debtor responds, complete the round and increment for next round
                        round_counter += 1

                    turn_counter += 1
                    
                    # Check for termination
                    if value["current_state"] in ["accept", "breakdown"]:
                        final_state = value["current_state"]
                        print(f"\n🏁 NEGOTIATION ENDED: {final_state.upper()}")
                        
                        # Update learning based on outcome
                        if selected_emotion:
                            success = (final_state == "accept")
                            final_days = self.negotiation_system.extract_days(value["messages"][-1].content)
                            
                            outcome_data = {
                                "final_state": final_state,
                                "final_days": final_days,
                                "negotiation_turns": len(dialog),
                                "scenario_id": self.id
                            }
                            
                            # Only update legacy learner if not using Bayesian emotions
                            if not self.use_bayesian_emotions:
                                self.emotion_learner.update_learning(selected_emotion, success, outcome_data)
                        break  # Exit the negotiation loop
                    
                    # Check for agreement on same timeline (both parties offer same days)
                    if len(dialog) >= 2:
                        last_creditor = None
                        last_debtor = None
                        
                        # Find the most recent creditor and debtor offers
                        for entry in reversed(dialog):
                            if entry["speaker"] == "seller" and last_creditor is None:
                                last_creditor = entry.get("requested_days")
                            elif entry["speaker"] == "buyer" and last_debtor is None:
                                last_debtor = entry.get("requested_days")
                            
                            if last_creditor is not None and last_debtor is not None:
                                break
                        
                        # # Debug: Show what we're comparing
                        # print(f"\n🔍 DEBUG: Checking agreement - Creditor: {last_creditor}, Debtor: {last_debtor}")
                        
                        # If both parties have the same days, consider it agreement
                        if last_creditor and last_debtor and last_creditor == last_debtor:
                            final_state = "accept"
                            print(f"\n🎯 AGREEMENT DETECTED: Both parties agreed on {last_creditor} days")
                            
                            # Update learning based on successful agreement
                            if selected_emotion:
                                success = True
                                final_days = last_creditor
                                
                                outcome_data = {
                                    "final_state": final_state,
                                    "final_days": final_days,
                                    "negotiation_turns": len(dialog),
                                    "scenario_id": self.id
                                }
                                
                                # Only update legacy learner if not using Bayesian emotions
                                if not self.use_bayesian_emotions:
                                    self.emotion_learner.update_learning(selected_emotion, success, outcome_data)
                            break  # Exit the negotiation loop
        
        # print(f"\n⏰ TIMEOUT after {len(dialog)} turns")
        
        # Final Bayesian optimization update
        success = (final_state == "accept")
        
        # Extract final agreed payment timeline (in days) - this is the actual collection time
        final_agreed_days = None
        if success and dialog:
            # Get the final agreed timeline from the last successful negotiation
            for entry in reversed(dialog):
                if entry.get("requested_days") is not None:
                    final_agreed_days = entry["requested_days"]
                    break
        
        # Set collection days: actual agreed timeline for success, penalty for failures/timeouts
        collection_days = final_agreed_days if final_agreed_days else None  # Large penalty for failures
        negotiation_rounds = len(dialog)
        
        # Debug output for final state
        if final_state == "breakdown" and not success:
            print(f"🚫 FINAL STATE: {final_state.upper()} | Collection Days: {collection_days} (penalty)")
        elif success:
            print(f"✅ FINAL STATE: {final_state.upper()} | Collection Days: {collection_days} (agreed timeline)")
        
        if self.use_bayesian_emotions:
            # Step 3-4: Update Bayesian model with negotiation outcome
            self.bayesian_optimizer.update_bayesian_model(collection_days, negotiation_rounds, success)
            
            # Step 5-6: Optimize transition matrix for next negotiation
            self.bayesian_optimizer.optimize_transition_matrix()
            
            bayesian_stats = self.bayesian_optimizer.get_learning_stats()
        else:
            bayesian_stats = {}
        
        # Legacy learning update (only if not using Bayesian emotions)
        if selected_emotion and not self.use_bayesian_emotions:
            self.emotion_learner.update_learning(selected_emotion, success, {"final_state": final_state})
        
        return {
            "id": self.id,
            "final_state": final_state,
            "dialog": dialog,
            "final_timeline_days": collection_days if success else None,
            "learning_stats": self.emotion_learner.get_learning_stats(),
            "bayesian_stats": bayesian_stats,
            "selected_emotion": selected_emotion,
            "emotion_sequence": emotion_sequence,
            "debtor_emotion": self.debtor_emotion,
            "collection_days": collection_days,
            "negotiation_rounds": negotiation_rounds
        }
    
    def _get_timeline_constraint_text(self, conversation_history, agent_type, target_timeline):
        """Generate timeline constraint text based on conversation history and agent type"""
        
        # Parse conversation history to extract timeline offers
        creditor_days = []
        debtor_days = []
        
        for speaker, message in conversation_history:
            days = self.negotiation_system.extract_days(message)
            if days:
                if speaker == "seller":  # creditor
                    creditor_days.append(days)
                elif speaker == "buyer":  # debtor
                    debtor_days.append(days)
        
        if not creditor_days and not debtor_days:
            return ""  # No timeline data to create constraints
        
        constraint_text = "\n### TIMELINE CONSTRAINT ANALYSIS"
        
        # Show timeline progression
        if creditor_days:
            constraint_text += f"\n- Creditor timeline offers: {creditor_days} days"
        if debtor_days:
            constraint_text += f"\n- Debtor timeline requests: {debtor_days} days"
        
        # Agent-specific constraint rules
        if agent_type == "creditor":
            # Creditor rules: Cannot decrease below previous offer (must increase or maintain)
            if creditor_days:
                min_creditor_days = max(creditor_days)
                constraint_text += f"\n\n### CREDITOR CONSTRAINT RULES:"
                constraint_text += f"\n- You cannot offer less than {min_creditor_days} days (your previous offer)"
                constraint_text += f"\n- If negotiating further, you must INCREASE your timeline offer"
                constraint_text += f"\n- Your target is {target_timeline} days, but be prepared to concede gradually"
                
        elif agent_type == "debtor":
            # Debtor rules: Cannot increase beyond previous request (must decrease or maintain)  
            if debtor_days:
                max_debtor_days = min(debtor_days)
                constraint_text += f"\n\n### DEBTOR CONSTRAINT RULES:"
                constraint_text += f"\n- You cannot request more than {max_debtor_days} days (your previous request)"
                constraint_text += f"\n- If negotiating further, you must DECREASE your timeline request"
                constraint_text += f"\n- Your target is {target_timeline} days, but be prepared to compromise gradually"
        
        # Calculate convergence opportunities
        if creditor_days and debtor_days:
            latest_creditor = creditor_days[-1]
            latest_debtor = debtor_days[-1]
            gap = abs(latest_debtor - latest_creditor)
            
            constraint_text += f"\n\n### CONVERGENCE ANALYSIS:"
            constraint_text += f"\n- Current gap: {gap} days"
            
            if gap <= 10:
                constraint_text += f"\n- 🎯 CONVERGENCE OPPORTUNITY: Gap is small ({gap} days)"
                constraint_text += f"\n- Consider accepting or making a final small concession"
            elif gap <= 25:
                constraint_text += f"\n- 📈 NEGOTIATION PROGRESS: Moderate gap ({gap} days)"
                constraint_text += f"\n- Continue gradual concessions toward middle ground"
            else:
                constraint_text += f"\n- 📊 LARGE GAP: Significant difference ({gap} days)"
                constraint_text += f"\n- Substantial movement needed from both parties"
        
        return constraint_text

# Main functions (following working pattern)
def load_agent_configs(filename: str) -> list:
    """Load debt collection scenarios"""
    try:
        with open(filename, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: Could not find {filename}")
        return []

def select_scenarios(scenarios: list, mode: str = "random", num_scenarios: int = 3) -> list:
    """Select scenarios for testing"""
    if mode == "random":
        return random.sample(scenarios, min(num_scenarios, len(scenarios)))
    else:
        return scenarios[:num_scenarios]

def run_bayesian_emotion_experiments(scenario_configs: list, args) -> dict:
    """Run Bayesian optimization experiments for each debtor emotion type"""
    output_dir = args.out_dir or "results_bayesian_emotions"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Define debtor emotions to test based on arguments
    if args.debtor_emotion == "all":
        debtor_emotions = ["happy", "surprising", "angry", "sad", "disgust", "fear", "neutral", "vanilla"]
    else:
        debtor_emotions = [args.debtor_emotion]
    
    all_results = {
        'experiment_type': f'{args.mode}_emotion_experiments',
        'total_debtor_emotions': len(debtor_emotions),
        'scenarios_per_emotion': len(scenario_configs),
        'debtor_emotion_results': {},
        'args': {
            'debtor_emotion': args.debtor_emotion,
            'mode': args.mode,
            'iterations': args.bayesian_iterations,
            'model': args.model,  # Legacy
            'model_creditor': args.model_creditor,
            'model_debtor': args.model_debtor
        }
    }
    
    for debtor_emotion in debtor_emotions:
        if args.mode == "bayesian":
            print(f"\n🎭 BAYESIAN EXPERIMENT: Debtor Emotion = {debtor_emotion.upper()}")
            print(f"🎯 GOAL: Optimize CREDITOR emotional transitions against CONSISTENT {debtor_emotion} debtor")
        elif args.mode == "vanilla":
            print(f"\n📝 VANILLA EXPERIMENT: Debtor Emotion = {debtor_emotion.upper()}")
            print(f"🎯 GOAL: No emotional guidance for creditor against CONSISTENT {debtor_emotion} debtor")
        else:
            print(f"\n🧪 LEGACY EXPERIMENT: Debtor Emotion = {debtor_emotion.upper()}")
            print(f"🎯 GOAL: Simple emotion learning against CONSISTENT {debtor_emotion} debtor")
        print("=" * 80)
        
        # Create appropriate optimizer based on mode
        if args.mode == "bayesian":
            bayesian_optimizer = BayesianEmotionOptimizer(debtor_emotion)
            use_emotions = True
            print(f"🧠 Bayesian Optimizer: Learning creditor transitions for {debtor_emotion} debtor")
        elif args.mode == "vanilla":
            bayesian_optimizer = None
            use_emotions = False
            print(f"📝 Vanilla Mode: No emotional guidance for creditor")
        else:
            bayesian_optimizer = None
            use_emotions = False
            print(f"📝 Legacy Mode: Simple emotion learning")
            
        emotion_results = {
            'debtor_emotion': debtor_emotion,
            'scenario_results': {},
            'iteration_progress': {},  # Track progress across iterations
            'optimization_summary': {
                'successful_negotiations': 0,
                'failed_negotiations': 0,
                'avg_collection_days': 0,
                'best_collection_days': float('inf'),
                'total_negotiations': 0
            }
        }
        
        collection_days_list = []
        
        # Run multiple iterations for learning
        for iteration in range(args.bayesian_iterations):
            print(f"\n🔄 Iteration {iteration + 1}")
            
            # Track iteration-level metrics
            iteration_results = {
                'iteration': iteration + 1,
                'scenarios': [],
                'iteration_success_rate': 0,
                'iteration_avg_collection_days': 0,
                'iteration_avg_recovery_rate': 0,
                'iteration_rewards': []  # Track rewards for this iteration
            }
            
            for i, config in enumerate(scenario_configs):
                scenario_id = f"{config['id']}_iter{iteration + 1}"
                print(f"   🏦 Scenario {i+1}: {scenario_id}")
                
                # Create negotiation with separate models
                negotiation = AdaptiveDebtBargain(
                    id=scenario_id,
                    config=config,
                    model=args.model,  # Legacy support
                    model_creditor=args.model_creditor,
                    model_debtor=args.model_debtor,
                    bayesian_optimizer=bayesian_optimizer,
                    debtor_emotion=debtor_emotion if debtor_emotion != "vanilla" else "neutral"
                )
                
                # Set emotion usage flag
                negotiation.use_bayesian_emotions = use_emotions
                
                # Run negotiation
                result = negotiation.run_negotiation(max_dialog_len=args.max_dialog_len)
                
                # Calculate creditor recovery rate (final days vs target days)
                creditor_target_days = int(config['seller']['target_price'])  # Ensure it's an integer
                if result['final_state'] == 'accept' and result['collection_days']:
                    recovery_rate = result['collection_days'] / creditor_target_days
                else:
                    recovery_rate = None  # Failed negotiation
                
                # Enhanced result tracking with full negotiation details
                enhanced_result = {
                    **result,  # Include all original result data
                    'creditor_target_days': creditor_target_days,
                    'creditor_recovery_rate': recovery_rate,
                    'negotiation_text': [
                        {
                            'round': entry.get('turn', 0),
                            'speaker': entry['speaker'],
                            'message': entry['message'],
                            'state': entry['state'],
                            'requested_days': entry.get('requested_days')
                        }
                        for entry in result['dialog']
                    ],
                    'experiment_args': {
                        'mode': args.mode,
                        'debtor_emotion': debtor_emotion,
                        'iterations': args.bayesian_iterations,
                        'max_dialog_len': args.max_dialog_len,
                        'model': args.model,  # Legacy
                        'model_creditor': args.model_creditor,
                        'model_debtor': args.model_debtor,
                        'scenario_id': config['id']
                    }
                }
                
                # Calculate reward for this negotiation (consistent with Bayesian optimizer)
                if result['final_state'] == 'accept' and result['collection_days']:
                    # Reward = -(collection_days / negotiation_rounds) - matches Bayesian optimizer
                    # This optimizes for both shorter payment timelines AND faster negotiations
                    reward = -result['collection_days'] / max(result['negotiation_rounds'], 1)
                else:
                    reward = -1000  # Failed collection: large penalty (matches Bayesian optimizer)
                
                iteration_results['iteration_rewards'].append(reward)
                iteration_results['scenarios'].append({
                    'scenario_id': scenario_id,
                    'success': result['final_state'] == 'accept',
                    'collection_days': result['collection_days'],
                    'recovery_rate': recovery_rate,
                    'reward': reward
                })
                
                # Track results
                emotion_results['scenario_results'][scenario_id] = enhanced_result
                collection_days_list.append(result['collection_days'])
                
                if result['final_state'] == 'accept':
                    emotion_results['optimization_summary']['successful_negotiations'] += 1
                else:
                    emotion_results['optimization_summary']['failed_negotiations'] += 1
                    
                emotion_results['optimization_summary']['total_negotiations'] += 1
                
                # Format recovery rate properly
                recovery_rate_str = f"{recovery_rate:.2f}" if recovery_rate is not None else "N/A"
                reward_str = f"{reward:.3f}" if reward > 0 else "0.000"
                print(f"      ✅ {result['final_state']} in {result['collection_days']} days (Target: {creditor_target_days}, Recovery Rate: {recovery_rate_str}, Reward: {reward_str})")
            
            # Calculate iteration-level summary
            successful_scenarios = [s for s in iteration_results['scenarios'] if s['success']]
            iteration_results['iteration_success_rate'] = len(successful_scenarios) / len(iteration_results['scenarios']) if iteration_results['scenarios'] else 0
            iteration_results['iteration_avg_collection_days'] = np.mean([s['collection_days'] for s in successful_scenarios]) if successful_scenarios else 0
            iteration_results['iteration_avg_recovery_rate'] = np.mean([s['recovery_rate'] for s in successful_scenarios if s['recovery_rate']]) if successful_scenarios else 0
            iteration_results['iteration_avg_reward'] = np.mean(iteration_results['iteration_rewards']) if iteration_results['iteration_rewards'] else 0
            
            # Store iteration results
            emotion_results['iteration_progress'][f'iteration_{iteration + 1}'] = iteration_results
            
            # Print iteration summary
            print(f"   📊 Iteration {iteration + 1} Summary:")
            print(f"      Success Rate: {iteration_results['iteration_success_rate']:.1%}")
            print(f"      Avg Reward: {iteration_results['iteration_avg_reward']:.3f}")
            print(f"      Avg Collection Days: {iteration_results['iteration_avg_collection_days']:.1f}")
        
        # Calculate comprehensive summary statistics
        if collection_days_list:
            # Filter out None values for successful negotiations only
            successful_days = [days for days in collection_days_list if days is not None]
            
            # Calculate averages (use successful_days for meaningful averages, all data for overall average)
            emotion_results['optimization_summary']['avg_collection_days'] = np.mean(successful_days) if successful_days else 0  # Only successful negotiations
            emotion_results['optimization_summary']['avg_successful_collection_days'] = np.mean(successful_days) if successful_days else 0
            emotion_results['optimization_summary']['best_collection_days'] = min(successful_days) if successful_days else float('inf')
            emotion_results['optimization_summary']['worst_collection_days'] = max(successful_days) if successful_days else 0
            emotion_results['optimization_summary']['std_collection_days'] = np.std(successful_days) if successful_days else 0  # Only successful negotiations
            
            # Calculate recovery rates for successful negotiations
            recovery_rates = []
            negotiation_lengths = []
            
            for scenario_id, scenario_result in emotion_results['scenario_results'].items():
                if scenario_result.get('creditor_recovery_rate') is not None:
                    recovery_rates.append(scenario_result['creditor_recovery_rate'])
                negotiation_lengths.append(scenario_result['negotiation_rounds'])
            
            emotion_results['optimization_summary']['avg_recovery_rate'] = np.mean(recovery_rates) if recovery_rates else 0
            emotion_results['optimization_summary']['best_recovery_rate'] = min(recovery_rates) if recovery_rates else 0  # Lower is better (closer to target)
            emotion_results['optimization_summary']['worst_recovery_rate'] = max(recovery_rates) if recovery_rates else 0
            emotion_results['optimization_summary']['avg_negotiation_length'] = np.mean(negotiation_lengths) if negotiation_lengths else 0
            emotion_results['optimization_summary']['success_rate'] = len(recovery_rates) / len(negotiation_lengths) if negotiation_lengths else 0
        
        # Add Bayesian optimizer statistics
        if bayesian_optimizer:
            emotion_results['bayesian_final_stats'] = bayesian_optimizer.get_learning_stats()
            emotion_results['final_transition_matrix'] = bayesian_optimizer.transition_matrix.tolist()
        
        all_results['debtor_emotion_results'][debtor_emotion] = emotion_results
        
        # Print comprehensive results summary
        summary = emotion_results['optimization_summary']
        print(f"🎯 {debtor_emotion.upper()} Results:")
        print(f"   Success Rate: {summary['successful_negotiations']}/{summary['total_negotiations']} ({summary.get('success_rate', 0):.1%})")
        print(f"   Avg Collection Days: {summary.get('avg_successful_collection_days', 0):.1f} days")
        print(f"   Avg Recovery Rate: {summary.get('avg_recovery_rate', 0):.2f}x target")
        print(f"   Avg Negotiation Length: {summary.get('avg_negotiation_length', 0):.1f} rounds")
        
        # Print learning progression (reward improvement across iterations)
        if 'iteration_progress' in emotion_results and emotion_results['iteration_progress']:
            print(f"   📈 Learning Progression (Avg Reward by Iteration):")
            iteration_rewards = []
            for iter_key in sorted(emotion_results['iteration_progress'].keys()):
                iter_data = emotion_results['iteration_progress'][iter_key]
                iteration_rewards.append(iter_data['iteration_avg_reward'])
                print(f"      Iteration {iter_data['iteration']}: {iter_data['iteration_avg_reward']:.3f}")
            
            # Show learning trend
            if len(iteration_rewards) > 1:
                improvement = iteration_rewards[-1] - iteration_rewards[0]
                trend = "↗️ Improving" if improvement > 0.01 else "↘️ Declining" if improvement < -0.01 else "➡️ Stable"
                print(f"      Learning Trend: {trend} ({improvement:+.3f})")
    
    # Add overall experiment summary
    all_results['overall_summary'] = {
        'total_negotiations': sum(data['optimization_summary']['total_negotiations'] 
                                for data in all_results['debtor_emotion_results'].values()),
        'total_successful': sum(data['optimization_summary']['successful_negotiations'] 
                              for data in all_results['debtor_emotion_results'].values()),
        'overall_success_rate': 0,
        'experiment_timestamp': datetime.now().isoformat(),
        'experiment_args': {
            'mode': args.mode,
            'debtor_emotion': args.debtor_emotion,
            'iterations': args.bayesian_iterations,
            'max_dialog_len': args.max_dialog_len,
            'model': args.model,  # Legacy
            'model_creditor': args.model_creditor,
            'model_debtor': args.model_debtor,
            'scenarios': len(scenario_configs)
        }
    }
    
    # Calculate overall success rate
    if all_results['overall_summary']['total_negotiations'] > 0:
        all_results['overall_summary']['overall_success_rate'] = (
            all_results['overall_summary']['total_successful'] / 
            all_results['overall_summary']['total_negotiations']
        )
    
    # Save comprehensive results with model names in filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Clean model names for filename (remove special characters)
    creditor_model_clean = args.model_creditor.replace("-", "").replace(".", "")
    debtor_model_clean = args.model_debtor.replace("-", "").replace(".", "")
    
    result_file = f"{output_dir}/creditor_{args.mode}_credit{creditor_model_clean}_debtor{debtor_model_clean}_{timestamp}.json"
    
    # Convert numpy arrays to lists for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        return obj
    
    import json
    with open(result_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=convert_numpy)
    
    print(f"\n💾 Comprehensive Results saved to: {result_file}")
    print(f"📊 Overall Success Rate: {all_results['overall_summary']['overall_success_rate']:.1%}")
    print(f"📈 Total Negotiations: {all_results['overall_summary']['total_negotiations']}")
    return all_results

def run_adaptive_negotiations(scenario_configs: list, args) -> dict:
    """Run adaptive negotiations with online learning (legacy method)"""
    output_dir = args.out_dir or "results_adaptive_debt"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Shared emotion learner across all scenarios
    emotion_learner = CreditorEmotionLearner()
    
    results = {
        'summary': {
            'total_scenarios': len(scenario_configs),
            'negotiation_type': 'adaptive_emotional',
            'learning_enabled': True
        },
        'scenario_results': {},
        'learning_summary': {
            'successful_negotiations': 0,
            'failed_negotiations': 0,
            'total_adaptations': 0
        }
    }
    
    # Process each scenario
    for i, config in enumerate(scenario_configs, 1):
        print(f"\n🏦 Running Scenario {i}/{len(scenario_configs)}: {config['id']}")
        
     
        # Create negotiation with shared emotion learner and separate models
        negotiation = AdaptiveDebtBargain(
            id=config['id'],
            config=config,
            model=args.model,  # Legacy support
            model_creditor=args.model_creditor,
            model_debtor=args.model_debtor,
            emotion_learner=emotion_learner
        )
        
        # Run negotiation
        result = negotiation.run_negotiation(max_dialog_len=args.max_dialog_len)
        
        # Update summary stats
        if result['final_state'] == 'accept':
            results['learning_summary']['successful_negotiations'] += 1
        else:
            results['learning_summary']['failed_negotiations'] += 1
        
        results['learning_summary']['total_adaptations'] = emotion_learner.get_learning_stats()['learning_events']
        
        # Store result
        results['scenario_results'][config['id']] = result
        
        print(f"✅ Completed: {result['final_state']} (Emotion: {result.get('selected_emotion', 'unknown')})")
        
       
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = f"{output_dir}/adaptive_debt_results_{timestamp}.json"
    with open(result_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to: {result_file}")
    
    return results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Advanced Debt Collection with Bayesian Emotion Optimization")
    parser.add_argument("--mode", choices=["bayesian", "legacy", "vanilla", "test"], default="bayesian", 
                       help="Run mode: bayesian (learn optimal transitions), legacy (simple learning), vanilla (no emotions), test (quick test)")
    parser.add_argument("--scenarios", type=int, default=2, help="Number of scenarios to test")
    parser.add_argument("--iterations", type=int, default=3, help="Bayesian optimization iterations")
    parser.add_argument("--max_dialog", type=int, default=30, help="Maximum dialog length")
    parser.add_argument("--model", default="gpt-4o-mini", help="LLM model to use (legacy - use --model_creditor and --model_debtor instead)")
    parser.add_argument("--model_creditor", default="gpt-4o-mini", help="LLM model for creditor agent")
    parser.add_argument("--model_debtor", default="gpt-4o-mini", help="LLM model for debtor agent")
    parser.add_argument("--out_dir", default="results_bayesian_debt", help="Output directory")
    parser.add_argument("--debtor_emotion", choices=["happy", "surprising", "angry", "sad", "disgust", "fear", "neutral", "vanilla", "all"], 
                       default="all", help="Specific debtor emotion to test (default: all emotions)")
    
    args = parser.parse_args()
    
    # Load scenarios
    scenarios = load_agent_configs("data/debt_collection_scenarios.json")
    if not scenarios:
        print("❌ No scenarios found. Creating sample scenario...")
        scenarios = [{
            "id": "sample_debt_001",
            "product": {"type": "debt_collection", "amount": 15000},
            "seller": {"target_price": 30},
            "buyer": {"target_price": 90},
            "metadata": {
                "outstanding_balance": 15000,
                "creditor_name": "ABC Collections",
                "debtor_name": "John Doe",
                "cash_flow_situation": "Irregular income",
                "business_impact": "High impact on credit score"
            }
        }]
    
    test_scenarios = select_scenarios(scenarios, num_scenarios=args.scenarios)
    
    # Configure arguments - add all missing attributes
    class Config:
        max_dialog_len = args.max_dialog
        model = args.model  # Legacy support
        model_creditor = args.model_creditor
        model_debtor = args.model_debtor
        out_dir = args.out_dir
        bayesian_iterations = args.iterations
        debtor_emotion = args.debtor_emotion
        mode = args.mode  # Add missing mode attribute
        scenarios = args.scenarios  # Add missing scenarios attribute
    
    config = Config()
    
    if args.mode == "bayesian":
        print("🧠 Starting Bayesian Emotion Optimization Experiments...")
        if args.debtor_emotion == "all":
            print(f"📊 Testing {len(test_scenarios)} scenarios across 8 debtor emotions with {args.iterations} iterations each")
        else:
            print(f"🎯 Testing {len(test_scenarios)} scenarios against {args.debtor_emotion.upper()} debtor with {args.iterations} iterations")
        
        print(f"🎭 Creditor Strategy: Bayesian optimization (learning transitions)")
        print(f"🎪 Debtor Emotion: {args.debtor_emotion}")
        
        results = run_bayesian_emotion_experiments(test_scenarios, config)
        print("✅ Bayesian experiments completed!")
        
        # Print comprehensive experiment summary
        print("\n📈 COMPREHENSIVE EXPERIMENT SUMMARY:")
        print("=" * 80)
        for emotion, data in results['debtor_emotion_results'].items():
            summary = data['optimization_summary']
            print(f"  {emotion.upper()}:")
            print(f"    Success Rate: {summary['successful_negotiations']}/{summary['total_negotiations']} ({summary.get('success_rate', 0):.1%})")
            print(f"    Avg Collection Days: {summary.get('avg_successful_collection_days', 0):.1f}")
            print(f"    Avg Recovery Rate: {summary.get('avg_recovery_rate', 0):.2f}x target")
            print(f"    Avg Negotiation Length: {summary.get('avg_negotiation_length', 0):.1f} rounds")
            print()
                    
    elif args.mode == "vanilla":
        print("📝 Starting Vanilla Experiments (No Emotional Guidance)...")
        if args.debtor_emotion == "all":
            print(f"📊 Testing {len(test_scenarios)} scenarios across 8 debtor emotions with {args.iterations} iterations each")
        else:
            print(f"🎯 Testing {len(test_scenarios)} scenarios against {args.debtor_emotion.upper()} debtor with {args.iterations} iterations")
        
        print(f"🎭 Creditor Strategy: No emotional guidance (vanilla)")
        print(f"🎪 Debtor Emotion: {args.debtor_emotion}")
        
        results = run_bayesian_emotion_experiments(test_scenarios, config)
        print("✅ Vanilla experiments completed!")
        
        # Print comprehensive experiment summary  
        print("\n📈 COMPREHENSIVE EXPERIMENT SUMMARY:")
        print("=" * 80)
        for emotion, data in results['debtor_emotion_results'].items():
            summary = data['optimization_summary']
            print(f"  {emotion.upper()}:")
            print(f"    Success Rate: {summary['successful_negotiations']}/{summary['total_negotiations']} ({summary.get('success_rate', 0):.1%})")
            print(f"    Avg Collection Days: {summary.get('avg_successful_collection_days', 0):.1f}")
            print(f"    Avg Recovery Rate: {summary.get('avg_recovery_rate', 0):.2f}x target")
            print(f"    Avg Negotiation Length: {summary.get('avg_negotiation_length', 0):.1f} rounds")
            print()
            
    elif args.mode == "legacy":
        print("🧪 Running legacy adaptive system...")
        results = run_adaptive_negotiations(test_scenarios, config)
        print("✅ Legacy test completed!")
        
    else:  # test mode
        print("🧪 Quick test with Bayesian emotions...")
        try:
            # Quick test with one debtor emotion
            bayesian_optimizer = BayesianEmotionOptimizer("neutral")
            negotiation = AdaptiveDebtBargain(
                id="quick_test",
                config=test_scenarios[0],
                model=args.model,  # Legacy support
                model_creditor=args.model_creditor,
                model_debtor=args.model_debtor,
                bayesian_optimizer=bayesian_optimizer,
                debtor_emotion="neutral"
            )
            
            result = negotiation.run_negotiation(max_dialog_len=args.max_dialog)
            print(f"✅ Quick test completed: {result['final_state']} in {result['collection_days']} days")
            print(f"🧠 Final emotion stats: {result.get('bayesian_stats', {})}")
            
        except ImportError as e:
            print(f"❌ Missing dependencies, falling back to legacy mode: {e}")
            results = run_adaptive_negotiations(test_scenarios, config)
            print("✅ Legacy fallback completed!")

    # Print usage examples
    if len(os.sys.argv) == 1:  # No arguments provided
        print("\n" + "="*80)
        print("📖 USAGE EXAMPLES:")
        print("="*80)
        print("# BAYESIAN MODE: Creditor learns optimal emotional transitions")
        print("python langgraph_bargain_debt_simple.py --mode bayesian --debtor_emotion disgust --iterations 5 --max_dialog 30 --scenarios 2")
        print()
        print("# VANILLA MODE: No emotional guidance for creditor")  
        print("python langgraph_bargain_debt_simple.py --mode vanilla --debtor_emotion disgust --iterations 5 --max_dialog 30 --scenarios 2")
        print()
        print("# SEPARATE MODELS: Different models for creditor and debtor")
        print("python langgraph_bargain_debt_simple.py --mode bayesian --model_creditor gpt-4o-mini --model_debtor gpt-4o --debtor_emotion disgust --iterations 3")
        print()
        print("# LEGACY MODE: Simple emotion learning")
        print("python langgraph_bargain_debt_simple.py --mode legacy --debtor_emotion disgust --iterations 5 --max_dialog 30 --scenarios 2")
        print()
        print("# Test ALL debtor emotions (comprehensive experiment)")
        print("python langgraph_bargain_debt_simple.py --mode bayesian --debtor_emotion all --iterations 3")
        print()
        print("# Quick test with default settings")
        print("python langgraph_bargain_debt_simple.py --mode test")
        print("="*80)