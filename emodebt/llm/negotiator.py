"""
Debt negotiation logic
"""

from typing import Dict, List, Any, Optional
from langgraph.graph import StateGraph, END, START
from langchain_core.messages import HumanMessage, AIMessage
from llm.llm_wrapper import LLMWrapper
import re

class GameState(Dict):
    """Game state for negotiation"""
    messages: List
    turn: str
    product: Dict
    seller_config: Dict
    buyer_config: Dict
    history: List
    current_state: str

class DebtNegotiator:
    """Debt negotiator using emotion models"""
    
    def __init__(
        self,
        config: Dict[str, Any],
        emotion_model: Any,  # BaseEmotionModel
        model_creditor: str = "gpt-4o-mini",
        model_debtor: str = "gpt-4o-mini",
        debtor_emotion: str = "neutral"
    ):
        self.config = config
        self.emotion_model = emotion_model
        self.debtor_emotion = debtor_emotion
        
        # Initialize LLMs
        self.llm_creditor = LLMWrapper(model_creditor, "creditor")
        self.llm_debtor = LLMWrapper(model_debtor, "debtor")
        
        # State tracking
        self.negotiation_round = 0
        self.emotion_sequence = []
    
    
    def llm_judge_agreement(self, creditor_message: str, debtor_message: str, creditor_days: int = None, debtor_days: int = None) -> dict:
        """Use LLM as a third judge to determine if agreement is reached"""
        
        judge_prompt = f"""You are an impartial JUDGE analyzing a debt collection negotiation to determine if the parties have reached an agreement.

### CONVERSATION CONTEXT:
Creditor's last message: "{creditor_message}"
Debtor's last message: "{debtor_message}"

### YOUR TASK:
Analyze these messages and determine if both parties have reached a mutual agreement on the payment timeline.

### CRITICAL BUSINESS RULE:
🔴 AUTOMATIC AGREEMENT: If the difference between proposed timelines is ≤ 5 days AND there is NO explicit rejection language, this is considered an AGREEMENT.

### AGREEMENT CRITERIA:
1. **Automatic**: Timeline difference ≤ 5 days without explicit rejection ("I can't", "won't work", "impossible")
2. **Explicit acceptance**: "I accept", "that works", "agreed", "deal", "sounds good"
3. **Compromise language**: "let's settle on", "how about we meet at", "I can live with"
4. **Implicit acceptance**: Positive acknowledgment or moving forward with terms

### EXTRACTED TIMELINES:
- Creditor proposed: {creditor_days} days
- Debtor proposed: {debtor_days} days
- Difference: {abs(creditor_days - debtor_days) if creditor_days and debtor_days else 'Unknown'} days

### RESPONSE FORMAT:
Respond with ONLY a JSON object:
{{
  "agreement_reached": true/false,
  "final_days": number or null,
  "reasoning": "Brief explanation of your decision",
  "confidence": "high/medium/low"
}}

### EXAMPLES:
- Difference ≤ 5 days without rejection → agreement_reached: true
- If creditor says "96 days works for me" and debtor says "Great, 96 days it is" → agreement_reached: true
- If creditor says "How about 50 days?" and debtor says "I need at least 80 days" → agreement_reached: false

Judgment:"""
        
        try:
            # Use creditor LLM as judge (could be any LLM)
            response = self.llm_creditor.invoke([HumanMessage(content=judge_prompt)])
            
            # Try to parse JSON response
            import json
            import re
            
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response.content, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                return {
                    "agreement": result.get("agreement_reached", False),
                    "final_days": result.get("final_days"),
                    "reasoning": result.get("reasoning", "No reasoning provided"),
                    "confidence": result.get("confidence", "low")
                }
            else:
                # Fallback parsing
                content_lower = response.content.lower()
                if "agreement_reached\":\"true" in content_lower or "agreement_reached\": true" in content_lower:
                    return {"agreement": True, "reasoning": "JSON parsing failed, used text analysis", "confidence": "low"}
                else:
                    return {"agreement": False, "reasoning": "JSON parsing failed, used text analysis", "confidence": "low"}
                    
        except Exception as e:
            print(f"        ⚠️  Judge LLM failed: {e}")
            return {"agreement": False, "reasoning": f"LLM error: {e}", "confidence": "low"}

    def extract_days(self, text: str) -> Optional[int]:
        """Extract payment timeline in days from text"""
        if not text:
            return None
        
        # Look for patterns like "30 days", "2 weeks", "1 month"
        import re
        patterns = [
            r'(\d+)\s*days?',
            r'(\d+)\s*weeks?', 
            r'(\d+)\s*months?'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text.lower())
            if matches:
                num = int(matches[-1])  # Take the last match
                if 'week' in pattern:
                    return num * 7
                elif 'month' in pattern:
                    return num * 30
                else:
                    return num
        return None
    
    def detect_debtor_emotion(self, debtor_message: str) -> str:
        """Detect debtor's emotion from their message using LLM"""
        if not debtor_message:
            return "neutral"
        
        emotion_prompt = f"""Analyze the emotional tone of this debt negotiation message and classify it into ONE of these categories:

EMOTIONS: happy, surprising, angry, sad, disgust, fear, neutral

MESSAGE TO ANALYZE:
"{debtor_message}"

Respond with ONLY the emotion word (e.g., "angry" or "sad"):"""
        
        try:
            response = self.llm_debtor.invoke([HumanMessage(content=emotion_prompt)])
            detected_emotion = response.content.strip().lower()
            
            # Validate emotion
            valid_emotions = ['happy', 'surprising', 'angry', 'sad', 'disgust', 'fear', 'neutral']
            if detected_emotion in valid_emotions:
                return detected_emotion
            else:
                return "neutral"
        except Exception as e:
            print(f"        ⚠️  Emotion detection failed: {e}")
            return "neutral"
    
    def creditor_node(self, state: GameState):
        """Creditor (seller) node"""
        self.negotiation_round += 1
        
        # Check if emotion model needs debtor emotion (HMM-based models do, evolutionary/bayesian don't)
        model_type = type(self.emotion_model).__name__.lower()
        needs_debtor_emotion = any(keyword in model_type for keyword in ['hmm', 'markov', 'interaction', 'sequential'])
        
        # Detect current debtor emotion only if the model needs it
        detected_debtor_emotion = self.debtor_emotion  # Default to initial setting
        if needs_debtor_emotion:
            conversation_history = state.get("history", [])
            if conversation_history:
                # Get the most recent debtor message
                for speaker, message in reversed(conversation_history):
                    if speaker == "buyer":
                        detected_debtor_emotion = self.detect_debtor_emotion(message)
                        break
            print(f"        🔍 Detected debtor emotion: {detected_debtor_emotion} (required for {model_type})")
        else:
            print(f"        ⚡ Skipping emotion detection (not needed for {model_type})")
        
        # Get emotion from model
        model_state = {
            'round': self.negotiation_round,
            'debtor_emotion': detected_debtor_emotion if needs_debtor_emotion else self.debtor_emotion,
            'current_emotion': self.emotion_sequence[-1] if self.emotion_sequence else 'neutral'
        }
        
        # 📊 Print model input state
        if needs_debtor_emotion:
            print(f"        📊 Model State Input: Round {model_state['round']}, Debtor: {detected_debtor_emotion} (detected), Current: {model_state['current_emotion']}")
            if detected_debtor_emotion != self.debtor_emotion:
                print(f"           💡 Debtor emotion changed: {self.debtor_emotion} → {detected_debtor_emotion}")
        else:
            print(f"        📊 Model State Input: Round {model_state['round']}, Debtor: {self.debtor_emotion} (fixed), Current: {model_state['current_emotion']}")
            print(f"           🚀 Evolutionary/Bayesian model - using fixed debtor emotion for efficiency")
        
        emotion_config = self.emotion_model.select_emotion(model_state)
        creditor_emotion = emotion_config['emotion']
        self.emotion_sequence.append(creditor_emotion)
        
        # 🧠 Print the designed emotion from the model
        model_type = type(self.emotion_model).__name__
        print(f"        🧠 {model_type} designed emotion: {creditor_emotion} (Round {self.negotiation_round})")
        
        # Show additional model-specific details
        if 'temperature' in emotion_config:
            print(f"           Temperature: {emotion_config['temperature']:.3f}")
        if 'policy_generation' in emotion_config:
            print(f"           Policy Gen: {emotion_config['policy_generation']}")
        if 'using_best_policy' in emotion_config:
            best_policy_text = "✅ Best" if emotion_config['using_best_policy'] else "🎲 Exploring"
            print(f"           Policy Type: {best_policy_text}")
        if 'confidence' in emotion_config:
            print(f"           Confidence: {emotion_config['confidence']}")
        if 'reasoning' in emotion_config:
            print(f"           Reasoning: {emotion_config['reasoning']}")
        if 'emotion_text' in emotion_config:
            print(f"           Prompt Style: {emotion_config['emotion_text']}")
        print(f"           Emotion Sequence: {self.emotion_sequence[-5:]}...")  # Show last 5 emotions
        
        # Build prompt
        config = self.config.get("seller_config", self.config["seller"])
        debt_info = self.config.get('metadata', {})
        outstanding_balance = debt_info.get('outstanding_balance', 0)
        
        # Extract timeline history from conversation
        conversation_history = state.get("history", [])
        creditor_days = []
        debtor_days = []
        
        for speaker, message in conversation_history:
            days = self.extract_days(message)
            if days:
                if speaker == "seller":
                    creditor_days.append(days)
                elif speaker == "buyer":
                    debtor_days.append(days)
        
        # Build constraint text with anti-copying logic
        timeline_text = ""
        if creditor_days:
            last_creditor = creditor_days[-1]
            timeline_text = f"Your previous offer was {last_creditor} days. "
        
        if debtor_days:
            last_debtor = debtor_days[-1]
            timeline_text += f"The debtor requested {last_debtor} days. "
            
            # Calculate reasonable concession range
            if creditor_days:
                current_gap = abs(last_debtor - creditor_days[-1])
                if current_gap > 10:
                    # Large gap: move 20-40% toward debtor
                    min_concession = creditor_days[-1] + int(current_gap * 0.2)
                    max_concession = creditor_days[-1] + int(current_gap * 0.4)
                    timeline_text += f"\n\nNEGOTIATION CONSTRAINT: Make a reasonable concession between {min_concession}-{max_concession} days. Do NOT copy their exact number ({last_debtor} days)."
                elif current_gap > 5:
                    # Medium gap: move halfway
                    target = (creditor_days[-1] + last_debtor) // 2
                    timeline_text += f"\n\nNEGOTIATION CONSTRAINT: Consider offering around {target} days (halfway point). Do NOT copy their exact number ({last_debtor} days)."
                else:
                    # Small gap: can accept or make small adjustment
                    timeline_text += f"\n\nNEGOTIATION CONSTRAINT: The gap is small ({current_gap} days). You can accept or make a small adjustment."
            else:
                # First creditor response
                target_days = int(config['target_price'])
                if last_debtor > target_days * 1.5:
                    # Their ask is too high, make a firm counter
                    counter_offer = min(target_days * 1.3, last_debtor * 0.7)
                    timeline_text += f"\n\nNEGOTIATION CONSTRAINT: Their request ({last_debtor} days) is too high. Counter with around {int(counter_offer)} days."
                else:
                    timeline_text += f"\n\nNEGOTIATION CONSTRAINT: Make a reasonable counter-offer. Do NOT immediately accept {last_debtor} days."
        
        prompt = f"""You are a PROFESSIONAL Creditor debt collection agent negotiating payment timeline with the Debtor.

### CRITICAL NEGOTIATION RULES:
🚫 NEVER copy the debtor's exact number - this shows weakness
📉 Move GRADUALLY toward their position (not all at once) 
💪 Show you are negotiating, not just accepting
🎯 Your goal: Minimize payment days while reaching agreement

### ROLE CLARITY
- You are ONLY the Creditor - speak only as yourself
- Do NOT include "**Creditor:**" or "**Debtor:**" labels
- Give only YOUR response as the creditor (1-2 sentences max)

### DEBT COLLECTION CONTEXT
- Outstanding Balance: ${outstanding_balance:,.2f}
- Your Target Timeline: {config['target_price']} days for full payment
- Recovery Stage: {debt_info.get('recovery_stage', 'Collection')}

### CURRENT SITUATION
{timeline_text}

### EMOTIONAL APPROACH
{emotion_config['emotion_text']}

### EXAMPLES OF GOOD NEGOTIATION:
✅ "I understand your situation, but {config['target_price']} days is too long. How about 45 days?"
✅ "That's closer, but I need payment sooner. Can you do 35 days?"
✅ "Let's meet in the middle at 40 days to resolve this quickly."

❌ DON'T: "I accept your 90 days." (too weak)
❌ DON'T: "90 days works for me." (copying their number)

Respond now with your negotiation counter-offer:"""
        
        # Generate response
        response = self.llm_creditor.invoke(
            [HumanMessage(content=prompt)],
            temperature=emotion_config.get('temperature', 0.7)
        )
        
        # Print the actual creditor message
        print(f"        💬 Creditor says: \"{response.content}\"")
        
        # Update history
        new_history = state["history"] + [("seller", response.content)]
        
        # Detect current state - check for agreement but prevent copying detection
        current_state = "offer"
        if len(new_history) >= 2:
            # Get last creditor and debtor messages
            last_creditor_days = self.extract_days(response.content)
            
            # Look for the most recent debtor offer
            debtor_days = None
            for speaker, message in reversed(new_history[:-1]):  # Exclude current message
                if speaker == "buyer":
                    debtor_days = self.extract_days(message)
                    if debtor_days:
                        break
            
            # Use LLM-based state detection instead of rule-based copying detection
            if last_creditor_days and debtor_days:
                # Check if they are close enough for potential agreement
                difference = abs(last_creditor_days - debtor_days)
                if difference <= 5:
                    current_state = "accept"
                    print(f"        🎯 CONVERGENCE: Creditor {last_creditor_days} vs Debtor {debtor_days} (diff: {difference} days)")
                else:
                    print(f"        📊 Gap: Creditor {last_creditor_days} vs Debtor {debtor_days} (diff: {difference} days)")
        
        return {
            "messages": [response],
            "turn": "buyer",  # Next turn goes to buyer
            "current_state": current_state,
            "history": new_history,
            "creditor_emotion": creditor_emotion
        }
    
    def debtor_node(self, state: GameState):
        """Debtor (buyer) node"""
        config = self.config.get("buyer_config", self.config["buyer"])
        debt_info = self.config.get('metadata', {})
        outstanding_balance = debt_info.get('outstanding_balance', 0)
        
        # Debtor emotion prompt
        emotion_prompt = ""
        if self.debtor_emotion != "neutral":
            emotions = {
                "happy": "Maintain an optimistic attitude",
                "surprising": "Show enthusiasm",
                "angry": "Express frustration while remaining civil",
                "sad": "Show distress",
                "disgust": "Express disappointment",
                "fear": "Show anxiety"
            }
            if self.debtor_emotion in emotions:
                emotion_prompt = f"\nYour emotional tone: {emotions[self.debtor_emotion]}"
        
        prompt = f"""You are the Debtor negotiating directly with the Creditor about the due payment days. 
Your goal is to reach a cooperative agreement with the creditor by negotiating the debt collection time.

### IMPORTANT: ROLE CLARITY
- You are ONLY the Debtor - speak only as yourself
- Do NOT include "**Creditor:**" or "**Debtor:**" labels in your response
- Do NOT simulate both sides of the conversation
- Give only YOUR response as the debtor
- Respond directly to what the creditor just said

{emotion_prompt}

### CONCESSION STRATEGY - CRITICAL INSTRUCTIONS:
- Learn to negotiate effectively by making GRADUAL concessions towards the creditor's position
- Do NOT immediately match or accept very low offers from creditor
- Make small steps: if creditor offers 40 days and you want 120 days, try 100 days next
- Always try to get REASONABLE time to pay while showing good faith
- Only accept if the difference is 5 days or less

### YOUR DEBT SITUATION
- Outstanding Balance: ${outstanding_balance:,.2f}
- Your Cash Flow: {debt_info.get('cash_flow_situation', 'Tight budget')}
- Business Impact: {debt_info.get('business_impact', 'Significant impact')}
- Your Target Timeline: {config['target_price']} days to make full payment

### COMMUNICATION STYLE
- This is a CONVERSATION, not written correspondence
- Remember to keep responses concise (1-2 sentences)!
- Speak ONLY as the Debtor - no "**Creditor:**" or "**Debtor:**" labels
- Do NOT simulate the entire conversation - give only YOUR debtor response
- Always show progression in your offers - each response should move closer to the creditor's position

IMPORTANT: When timeline differences are within 5-10 days, consider accepting to resolve the debt
"""
        
        response = self.llm_debtor.invoke([HumanMessage(content=prompt)])
        
        new_history = state["history"] + [("buyer", response.content)]
        
        # Detect current state - check for agreement based on PREVIOUS offers  
        current_state = "offer"
        if len(new_history) >= 2:
            # Get the current debtor offer
            last_debtor_days = self.extract_days(response.content)
            
            # Look for the most recent creditor offer
            creditor_days = None
            for speaker, message in reversed(new_history[:-1]):  # Exclude current message
                if speaker == "seller":
                    creditor_days = self.extract_days(message)
                    if creditor_days:
                        break
            
            # Check for agreement based on LLM analysis - within reasonable tolerance
            if last_debtor_days and creditor_days:
                difference = abs(last_debtor_days - creditor_days)
                if difference <= 5:
                    current_state = "accept"
                    print(f"        🎯 CONVERGENCE: Debtor {last_debtor_days} vs Creditor {creditor_days} (diff: {difference} days)")
                else:
                    print(f"        📊 Gap: Debtor {last_debtor_days} vs Creditor {creditor_days} (diff: {difference} days)")
        
        return {
            "messages": [response],
            "turn": "seller",  # Next turn goes to seller
            "current_state": current_state,
            "history": new_history
        }
    
    def should_continue(self, state: GameState):
        """Determine if negotiation should continue"""
        if state["current_state"] in ["accept", "breakdown"]:
            return "end"
        
        # Check for agreement on timeline - within 5 days tolerance but not exact copying
        if len(state["history"]) >= 2:
            last_two = state["history"][-2:]
            seller_days = self.extract_days(last_two[0][1])
            buyer_days = self.extract_days(last_two[1][1])
            
            # Agreement only if within 5 days AND not exact copying
            if (seller_days and buyer_days and 
                abs(seller_days - buyer_days) <= 5 and
                seller_days != buyer_days):  # Prevent exact copying from triggering end
                print(f"        🎯 CONVERGENCE: Seller {seller_days} vs Buyer {buyer_days} (within 5 days)")
                return "end"
            elif seller_days and buyer_days and seller_days == buyer_days:
                # Check if this is legitimate agreement (both moved to same number) or copying
                history_length = len(state["history"])
                if history_length >= 4:  # If negotiation has progressed, exact match is ok
                    print(f"        🎯 FINAL AGREEMENT: Both agreed on {seller_days} days after negotiation")
                    return "end"
                else:
                    print(f"        ⚠️  Exact match too early - continuing negotiation")
        
        # Return the next turn
        next_turn = state.get("turn", "buyer")
        return next_turn if next_turn in ["buyer", "seller"] else "buyer"
    
    def run_negotiation(self, max_dialog_len: int = 30) -> Dict[str, Any]:
        """Run a complete negotiation"""
        # Set up workflow
        workflow = StateGraph(GameState)
        workflow.add_node("seller", self.creditor_node)
        workflow.add_node("buyer", self.debtor_node)
        workflow.add_edge(START, "seller")
        
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
        
        # Initial state
        debt_info = self.config.get('metadata', {})
        outstanding_balance = debt_info.get('outstanding_balance', 0)
        
        initial_message = f"Hello, this is the Creditor. We need to discuss the outstanding balance of ${outstanding_balance:,.2f}. I'm proposing a payment timeline of {self.config['seller']['target_price']} days."
        
        initial_state = GameState(
            messages=[HumanMessage(content=initial_message)],
            turn="seller",  # Start with seller since workflow.add_edge(START, "seller")
            product=self.config["product"],
            seller_config=self.config["seller"],
            buyer_config=self.config["buyer"],
            history=[],  # Start with empty history, let the nodes populate it
            current_state="offer"
        )
        
        # Run negotiation
        dialog = []
        final_state = "breakdown"  # Initialize with default value
        final_state = "breakdown"  # Initialize with default value
        
        for i, step in enumerate(app.stream(initial_state, {"recursion_limit": max_dialog_len * 2})):
            if i > max_dialog_len:
                break
            
            for node, value in step.items():
                message_content = value["messages"][-1].content
                requested_days = self.extract_days(message_content)
                
                dialog.append({
                    "turn": i + 1,
                    "speaker": node,
                    "message": message_content,
                    "state": value["current_state"],
                    "requested_days": requested_days
                })
                
                # Display progress
                speaker_name = "Creditor" if node == "seller" else "Debtor"
                state_emoji = "✅" if value["current_state"] == "accept" else "❌" if value["current_state"] == "breakdown" else "💬"
                print(f"        Turn {i+1} - {speaker_name}: {requested_days} days {state_emoji}")
                
                # Update final state from negotiation
                if value["current_state"] in ["accept", "breakdown"]:
                    final_state = value["current_state"]
                    break
        
        # Use LLM Judge Agent to determine final agreement
        if len(dialog) >= 2:
            # Get the last two messages for judge analysis
            last_creditor_msg = None
            last_debtor_msg = None
            last_creditor_days = None
            last_debtor_days = None
            
            # Find last creditor and debtor messages
            for entry in reversed(dialog):
                if entry["speaker"] == "seller" and last_creditor_msg is None:
                    last_creditor_msg = entry["message"]
                    last_creditor_days = entry["requested_days"]
                elif entry["speaker"] == "buyer" and last_debtor_msg is None:
                    last_debtor_msg = entry["message"]
                    last_debtor_days = entry["requested_days"]
                
                if last_creditor_msg and last_debtor_msg:
                    break
            
            # Use LLM Judge to analyze agreement
            if last_creditor_msg and last_debtor_msg:
                judge_result = self.llm_judge_agreement(
                    last_creditor_msg, last_debtor_msg, 
                    last_creditor_days, last_debtor_days
                )
                
                if judge_result["agreement"]:
                    final_state = "accept"
                    print(f"        🤖 LLM Judge: {judge_result['reasoning']} (confidence: {judge_result['confidence']})")
                    # Use judge's suggested final days or calculate average
                    if judge_result.get("final_days"):
                        final_days = judge_result["final_days"]
                    elif last_creditor_days and last_debtor_days:
                        final_days = round((last_creditor_days + last_debtor_days) / 2)
                else:
                    final_state = "breakdown"
                    print(f"        🤖 LLM Judge: {judge_result['reasoning']} (confidence: {judge_result['confidence']})")
        
        final_days = None
        
        # Extract final agreed days if accepted
        if final_state == "accept" and len(dialog) >= 2:
            # Get the last creditor and debtor offers to see if they match
            last_seller_days = None
            last_buyer_days = None
            
            for entry in reversed(dialog):
                if entry["speaker"] == "seller" and entry["requested_days"] and last_seller_days is None:
                    last_seller_days = entry["requested_days"]
                elif entry["speaker"] == "buyer" and entry["requested_days"] and last_buyer_days is None:
                    last_buyer_days = entry["requested_days"]
                
                if last_seller_days is not None and last_buyer_days is not None:
                    break
            
            # Check if they agreed within 5 days tolerance (like original system)
            if last_seller_days and last_buyer_days and abs(last_seller_days - last_buyer_days) <= 5:
                # Use the average of the two offers as the final agreed timeline
                final_days = round((last_seller_days + last_buyer_days) / 2)
                print(f"      ✅ CLOSE AGREEMENT: Creditor {last_seller_days} vs Debtor {last_buyer_days} → Final: {final_days} days")
            elif last_seller_days:
                # Use the last creditor offer as the agreed timeline
                final_days = last_seller_days
                print(f"      ✅ ACCEPTED: Creditor's {final_days} days accepted")
            elif last_buyer_days:
                # Use the last debtor offer
                final_days = last_buyer_days
                print(f"      ✅ ACCEPTED: Debtor's {final_days} days accepted")
        
        # If no clear agreement but state shows accept, try to extract from dialog
        elif final_state == "accept" and len(dialog) > 0:
            # Get the very last offer made
            for entry in reversed(dialog):
                if entry["requested_days"]:
                    final_days = entry["requested_days"]
                    print(f"      ✅ ACCEPTED: Final timeline {final_days} days")
                    break
        
        # Calculate final metrics
        collection_days = final_days if final_state == "accept" else None
        creditor_target_days = int(self.config['seller']['target_price'])
        total_rounds = len(dialog)
        
        result = {
            "scenario_id": self.config['id'],
            "final_state": final_state,
            "collection_days": collection_days,
            "final_days": final_days,  # Add explicit final_days for compatibility
            "creditor_target_days": creditor_target_days,
            "negotiation_rounds": total_rounds,
            "dialog_length": total_rounds,
            "emotion_sequence": self.emotion_sequence.copy(),
            "dialog": dialog,  # Include full dialog for analysis
            "success": final_state == "accept"
        }
        
        # Debug output for verification
        print(f"      📊 Final Result: {final_state} | Days: {final_days} | Rounds: {total_rounds}")
        
        # Reset for next negotiation
        self.negotiation_round = 0
        self.emotion_sequence = []
        if hasattr(self.emotion_model, 'reset'):
            self.emotion_model.reset()
        
        return result