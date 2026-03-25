# """
# Debt negotiator for Bayesian Multi-Agent Transition Optimization
# Specifically designed to provide learning feedback to Bayesian models
# """

# from typing import Dict, List, Any, Optional
# from langgraph.graph import StateGraph, END, START
# from langchain_core.messages import HumanMessage, AIMessage
# from llm.llm_wrapper import LLMWrapper
# import re

# class GameState(Dict):
#     """Game state for negotiation with agent prediction tracking"""
#     messages: List
#     turn: str
#     product: Dict
#     seller_config: Dict
#     buyer_config: Dict
#     history: List
#     current_state: str
#     agent_predictions: List  # Track agent predictions for learning

# class DebtNegotiator:
#     """Debt negotiator for Bayesian transition optimization model"""
    
#     def __init__(
#         self,
#         config: Dict[str, Any],
#         emotion_model: Any,  # BayesianTransitionModel
#         model_creditor: str = "gpt-4o-mini",
#         model_debtor: str = "gpt-4o-mini",
#         debtor_emotion: str = "neutral"
#     ):
#         self.config = config
#         self.emotion_model = emotion_model
        
#         # Convert debtor_emotion to uppercase code
#         emotion_mapping = {
#             'happy': 'J', 'happy': 'J', 'joyful': 'J',
#             'sadness': 'S', 'sad': 'S', 
#             'anger': 'A', 'angry': 'A',
#             'fear': 'F', 'fearful': 'F',
#             'surprise': 'Su', 'surprising': 'Su',
#             'disgust': 'D', 'disgusted': 'D',
#             'neutral': 'N'
#         }
#         self.debtor_emotion = emotion_mapping.get(debtor_emotion.lower(), 'N')
        
#         # Initialize LLMs
#         self.llm_creditor = LLMWrapper(model_creditor, "creditor")
#         self.llm_debtor = LLMWrapper(model_debtor, "debtor")
        
#         # State tracking
#         self.negotiation_round = 0
#         self.emotion_sequence = []
#         self.debtor_emotion_sequence = []
        
#         # Learning feedback tracking
#         self.agent_predictions_history = []  # Store predictions for each round
#         self.last_gap_size = 0
#         self.last_debtor_emotion = self.debtor_emotion  # Use the normalized version
    
#     def llm_judge_agreement(self, creditor_message: str, debtor_message: str, creditor_days: int = None, debtor_days: int = None) -> dict:
#         """Use LLM as a third judge to determine if agreement is reached"""
        
#         judge_prompt = f"""You are an impartial JUDGE analyzing a debt collection negotiation to determine if the parties have reached an agreement.

# ### CONVERSATION CONTEXT:
# Creditor's last message: "{creditor_message}"
# Debtor's last message: "{debtor_message}"

# ### YOUR TASK:
# Analyze these messages and determine if both parties have reached a mutual agreement on the payment timeline.

# ### CRITICAL BUSINESS RULE:
# 🔴 AUTOMATIC AGREEMENT: If the difference between proposed timelines is ≤ 5 days AND there is NO explicit rejection language, this is considered an AGREEMENT.

# ### AGREEMENT CRITERIA:
# 1. **Automatic**: Timeline difference ≤ 5 days without explicit rejection ("I can't", "won't work", "impossible")
# 2. **Explicit acceptance**: "I accept", "that works", "agreed", "deal", "sounds good"
# 3. **Compromise language**: "let's settle on", "how about we meet at", "I can live with"
# 4. **Implicit acceptance**: Positive acknowledgment or moving forward with terms

# ### EXTRACTED TIMELINES:
# - Creditor proposed: {creditor_days} days
# - Debtor proposed: {debtor_days} days
# - Difference: {abs(creditor_days - debtor_days) if creditor_days and debtor_days else 'Unknown'} days

# ### RESPONSE FORMAT:
# Respond with ONLY a JSON object:
# {{
#   "agreement_reached": true/false,
#   "final_days": number or null,
#   "reasoning": "Brief explanation of your decision",
#   "confidence": "high/medium/low"
# }}

# ### EXAMPLES:
# - Difference ≤ 5 days without rejection → agreement_reached: true
# - If creditor says "96 days works for me" and debtor says "Great, 96 days it is" → agreement_reached: true
# - If creditor says "How about 50 days?" and debtor says "I need at least 80 days" → agreement_reached: false

# Judgment:"""
        
#         try:
#             # Use creditor LLM as judge (could be any LLM)
#             response = self.llm_creditor.invoke([HumanMessage(content=judge_prompt)])
            
#             # Try to parse JSON response
#             import json
#             import re
            
#             # Extract JSON from response
#             json_match = re.search(r'\{.*\}', response.content, re.DOTALL)
#             if json_match:
#                 result = json.loads(json_match.group())
#                 return {
#                     "agreement": result.get("agreement_reached", False),
#                     "final_days": result.get("final_days"),
#                     "reasoning": result.get("reasoning", "No reasoning provided"),
#                     "confidence": result.get("confidence", "low")
#                 }
#             else:
#                 # Fallback parsing
#                 content_lower = response.content.lower()
#                 if "agreement_reached\":\"true" in content_lower or "agreement_reached\": true" in content_lower:
#                     return {"agreement": True, "reasoning": "JSON parsing failed, used text analysis", "confidence": "low"}
#                 else:
#                     return {"agreement": False, "reasoning": "JSON parsing failed, used text analysis", "confidence": "low"}
                    
#         except Exception as e:
#             print(f"        ⚠️  Judge LLM failed: {e}")
#             return {"agreement": False, "reasoning": f"LLM error: {e}", "confidence": "low"}

#     def extract_days(self, text: str) -> Optional[int]:
#         """Extract payment timeline in days from text"""
#         if not text:
#             return None
        
#         # Look for patterns like "30 days", "2 weeks", "1 month"
#         import re
#         patterns = [
#             r'(\d+)\s*days?',
#             r'(\d+)\s*weeks?', 
#             r'(\d+)\s*months?'
#         ]
        
#         for pattern in patterns:
#             matches = re.findall(pattern, text.lower())
#             if matches:
#                 num = int(matches[-1])  # Take the last match
#                 if 'week' in pattern:
#                     return num * 7
#                 elif 'month' in pattern:
#                     return num * 30
#                 else:
#                     return num
#         return None
    
#     def detect_debtor_emotion(self, debtor_message: str) -> str:
#         """Detect debtor's emotion from their message using LLM"""
#         if not debtor_message:
#             return "N"  # Return uppercase code
        
#         emotion_prompt = f"""Analyze the emotional tone of this debt negotiation message and classify it into ONE of these categories:

#     EMOTIONS: happy, surprising, angry, sad, disgust, fear, neutral

#     MESSAGE TO ANALYZE:
#     "{debtor_message}"

#     Respond with ONLY the emotion word (e.g., "angry" or "sad"):"""
        
#         try:
#             response = self.llm_debtor.invoke([HumanMessage(content=emotion_prompt)])
#             detected_emotion = response.content.strip().lower()
            
#             # Convert to uppercase emotion codes
#             emotion_mapping = {
#                 'happy': 'J',
#                 'surprising': 'Su', 
#                 'angry': 'A',
#                 'sad': 'S',
#                 'disgust': 'D',
#                 'fear': 'F',
#                 'neutral': 'N'
#             }
            
#             if detected_emotion in emotion_mapping:
#                 return emotion_mapping[detected_emotion]
#             else:
#                 return "N"  # Default to Neutral
#         except Exception as e:
#             print(f"        ⚠️  Emotion detection failed: {e}")
#             return "N"
#     def _calculate_gap_size(self, conversation_history: List) -> float:
#         """Calculate the gap between creditor and debtor proposals"""
#         creditor_days = None
#         debtor_days = None
        
#         # Find the most recent creditor and debtor offers
#         for speaker, message in reversed(conversation_history):
#             days = self.extract_days(message)
#             if days:
#                 if speaker == "seller" and creditor_days is None:
#                     creditor_days = days
#                 elif speaker == "buyer" and debtor_days is None:
#                     debtor_days = days
                
#                 if creditor_days is not None and debtor_days is not None:
#                     break
        
#         # Calculate gap
#         if creditor_days is not None and debtor_days is not None:
#             return abs(creditor_days - debtor_days)
#         return 0.0
    
#     def creditor_node(self, state: GameState):
#         """Creditor (seller) node with Bayesian agent feedback"""
#         self.negotiation_round += 1
        
#         # Detect current debtor emotion
#         detected_debtor_emotion = self.debtor_emotion
#         conversation_history = state.get("history", [])
        
#         if conversation_history:
#             # Get the most recent debtor message
#             for speaker, message in reversed(conversation_history):
#                 if speaker == "buyer":
#                     detected_debtor_emotion = self.detect_debtor_emotion(message)
#                     self.debtor_emotion_sequence.append(detected_debtor_emotion)
#                     self.last_debtor_emotion = detected_debtor_emotion
#                     break
        
#         # Calculate gap size for Bayesian context
#         self.last_gap_size = self._calculate_gap_size(conversation_history)
        
#         # Get emotion from Bayesian transition model
#         # In creditor_node method:
#         model_state = {
#             'round': self.negotiation_round,
#             'debtor_emotion': detected_debtor_emotion,  # This is already uppercase from detect_debtor_emotion()
#             'current_emotion': self.emotion_sequence[-1] if self.emotion_sequence else 'N',  # Use 'N' not 'neutral'
#             'debt_amount': self.config.get('metadata', {}).get('outstanding_balance', 0),
#             'gap_size': self.last_gap_size
#         }
        
#         # 🧠 Get emotion from Bayesian model
#         emotion_config = self.emotion_model.select_emotion(model_state)
#         creditor_emotion = emotion_config['emotion']
#         self.emotion_sequence.append(creditor_emotion)
        
#         # 📊 Store agent predictions for learning feedback
#         current_predictions = []
#         if 'agent_predictions' in emotion_config:
#             current_predictions = emotion_config['agent_predictions']
#             self.agent_predictions_history.append({
#                 'round': self.negotiation_round,
#                 'predictions': current_predictions,
#                 'selected_emotion': creditor_emotion,
#                 'debtor_emotion': detected_debtor_emotion,
#                 'gap_size': self.last_gap_size
#             })
        
#         # 🧠 Print Bayesian model details
#         model_type = type(self.emotion_model).__name__
#         print(f"        🧠 {model_type}: {creditor_emotion} (Round {self.negotiation_round})")
        
#         # Show Bayesian-specific details
#         if 'transition' in emotion_config:
#             print(f"           Transition: {emotion_config['transition']}")
#         if 'exploration' in emotion_config:
#             print(f"           Exploration: {'Yes' if emotion_config['exploration'] else 'No'}")
#         if 'confidence' in emotion_config:
#             print(f"           Confidence: {emotion_config['confidence']:.2f}")
#         if current_predictions:
#             print(f"           🤖 {len(current_predictions)} agents consulted")
#             # Show ALL agents
#             for pred in current_predictions:
#                 print(f"             {pred['agent']}: {pred['target']} (confidence: {pred['confidence']:.2f})")
        
#         print(f"           Negotiator Emotion Sequence: {self.emotion_sequence[-5:]}...")
        
#         # Build prompt
#         config = self.config.get("seller_config", self.config["seller"])
#         debt_info = self.config.get('metadata', {})
#         outstanding_balance = debt_info.get('outstanding_balance', 0)
        
#         # Extract timeline history
#         creditor_days = []
#         debtor_days = []
        
#         for speaker, message in conversation_history:
#             days = self.extract_days(message)
#             if days:
#                 if speaker == "seller":
#                     creditor_days.append(days)
#                 elif speaker == "buyer":
#                     debtor_days.append(days)
        
#         # Build constraint text
#         timeline_text = ""
#         if creditor_days:
#             last_creditor = creditor_days[-1]
#             timeline_text = f"Your previous offer was {last_creditor} days. "
        
#         if debtor_days:
#             last_debtor = debtor_days[-1]
#             timeline_text += f"The debtor requested {last_debtor} days. "
            
#             # Calculate reasonable concession range
#             if creditor_days:
#                 current_gap = abs(last_debtor - creditor_days[-1])
#                 if current_gap > 10:
#                     # Large gap: move 20-40% toward debtor
#                     min_concession = creditor_days[-1] + int(current_gap * 0.2)
#                     max_concession = creditor_days[-1] + int(current_gap * 0.4)
#                     timeline_text += f"\n\nNEGOTIATION CONSTRAINT: Make a reasonable concession between {min_concession}-{max_concession} days. Do NOT copy their exact number ({last_debtor} days)."
#                 elif current_gap > 5:
#                     # Medium gap: move halfway
#                     target = (creditor_days[-1] + last_debtor) // 2
#                     timeline_text += f"\n\nNEGOTIATION CONSTRAINT: Consider offering around {target} days (halfway point). Do NOT copy their exact number ({last_debtor} days)."
#                 else:
#                     # Small gap: can accept or make small adjustment
#                     timeline_text += f"\n\nNEGOTIATION CONSTRAINT: The gap is small ({current_gap} days). You can accept or make a small adjustment."
#             else:
#                 # First creditor response
#                 target_days = int(config['target_price'])
#                 if last_debtor > target_days * 1.5:
#                     # Their ask is too high, make a firm counter
#                     counter_offer = min(target_days * 1.3, last_debtor * 0.7)
#                     timeline_text += f"\n\nNEGOTIATION CONSTRAINT: Their request ({last_debtor} days) is too high. Counter with around {int(counter_offer)} days."
#                 else:
#                     timeline_text += f"\n\nNEGOTIATION CONSTRAINT: Make a reasonable counter-offer. Do NOT immediately accept {last_debtor} days."
        
#         prompt = f"""You are a PROFESSIONAL Creditor debt collection agent negotiating payment timeline with the Debtor.

# ### CRITICAL NEGOTIATION RULES:
# 🚫 NEVER copy the debtor's exact number - this shows weakness
# 📉 Move GRADUALLY toward their position (not all at once) 
# 💪 Show you are negotiating, not just accepting
# 🎯 Your goal: Minimize payment days while reaching agreement

# ### ROLE CLARITY
# - You are ONLY the Creditor - speak only as yourself
# - Do NOT include "**Creditor:**" or "**Debtor:**" labels
# - Give only YOUR response as the creditor (1-2 sentences max)

# ### DEBT COLLECTION CONTEXT
# - Outstanding Balance: ${outstanding_balance:,.2f}
# - Your Target Timeline: {config['target_price']} days for full payment
# - Recovery Stage: {debt_info.get('recovery_stage', 'Collection')}

# ### CURRENT SITUATION
# {timeline_text}

# ### EMOTIONAL APPROACH
# {emotion_config['emotion_text']}

# ### EXAMPLES OF GOOD NEGOTIATION:
# ✅ "I understand your situation, but {config['target_price']} days is too long. How about 45 days?"
# ✅ "That's closer, but I need payment sooner. Can you do 35 days?"
# ✅ "Let's meet in the middle at 40 days to resolve this quickly."

# ❌ DON'T: "I accept your 90 days." (too weak)
# ❌ DON'T: "90 days works for me." (copying their number)

# Respond now with your negotiation counter-offer:"""
        
#         # Generate response
#         response = self.llm_creditor.invoke(
#             [HumanMessage(content=prompt)],
#             temperature=emotion_config.get('temperature', 0.7)
#         )
        
#         # Print the actual creditor message
#         print(f"        💬 Creditor says: \"{response.content}\"")
        
#         # Update history
#         new_history = state["history"] + [("seller", response.content)]
        
#         # Update agent predictions in state
#         new_agent_predictions = state.get("agent_predictions", []) + [{
#             'round': self.negotiation_round,
#             'predictions': current_predictions,
#             'selected_emotion': creditor_emotion
#         }]
        
#         # Detect current state
#         current_state = "offer"
#         if len(new_history) >= 2:
#             # Get last creditor and debtor messages
#             last_creditor_days = self.extract_days(response.content)
            
#             # Look for the most recent debtor offer
#             debtor_days = None
#             for speaker, message in reversed(new_history[:-1]):  # Exclude current message
#                 if speaker == "buyer":
#                     debtor_days = self.extract_days(message)
#                     if debtor_days:
#                         break
            
#             # Use LLM-based state detection instead of rule-based copying detection
#             if last_creditor_days and debtor_days:
#                 # Check if they are close enough for potential agreement
#                 difference = abs(last_creditor_days - debtor_days)
#                 if difference <= 5:
#                     current_state = "accept"
#                     print(f"        🎯 CONVERGENCE: Creditor {last_creditor_days} vs Debtor {debtor_days} (diff: {difference} days)")
#                 else:
#                     print(f"        📊 Gap: Creditor {last_creditor_days} vs Debtor {debtor_days} (diff: {difference} days)")
        
#         return {
#             "messages": [response],
#             "turn": "buyer",  # Next turn goes to buyer
#             "current_state": current_state,
#             "history": new_history,
#             "agent_predictions": new_agent_predictions,
#             "creditor_emotion": creditor_emotion
#         }
    
#     def debtor_node(self, state: GameState):
#         """Debtor (buyer) node"""
#         config = self.config.get("buyer_config", self.config["buyer"])
#         debt_info = self.config.get('metadata', {})
#         outstanding_balance = debt_info.get('outstanding_balance', 0)
        
#         # Debtor emotion prompt
#         emotion_prompt = ""
#         if self.debtor_emotion != "neutral":
#             emotions = {
#                 "happy": "Maintain an optimistic attitude",
#                 "surprising": "Show enthusiasm",
#                 "angry": "Express frustration while remaining civil",
#                 "sad": "Show distress",
#                 "disgust": "Express disappointment",
#                 "fear": "Show anxiety"
#             }
#             if self.debtor_emotion in emotions:
#                 emotion_prompt = f"\nYour emotional tone: {emotions[self.debtor_emotion]}"
        
#         prompt = f"""You are the Debtor negotiating directly with the Creditor about the due payment days. 
# Your goal is to reach a cooperative agreement with the creditor by negotiating the debt collection time.

# ### IMPORTANT: ROLE CLARITY
# - You are ONLY the Debtor - speak only as yourself
# - Do NOT include "**Creditor:**" or "**Debtor:**" labels in your response
# - Do NOT simulate both sides of the conversation
# - Give only YOUR response as the debtor
# - Respond directly to what the creditor just said

# {emotion_prompt}

# ### CONCESSION STRATEGY - CRITICAL INSTRUCTIONS:
# - Learn to negotiate effectively by making GRADUAL concessions towards the creditor's position
# - Do NOT immediately match or accept very low offers from creditor
# - Make small steps: if creditor offers 40 days and you want 120 days, try 100 days next
# - Always try to get REASONABLE time to pay while showing good faith
# - Only accept if the difference is 5 days or less

# ### YOUR DEBT SITUATION
# - Outstanding Balance: ${outstanding_balance:,.2f}
# - Your Cash Flow: {debt_info.get('cash_flow_situation', 'Tight budget')}
# - Business Impact: {debt_info.get('business_impact', 'Significant impact')}
# - Your Target Timeline: {config['target_price']} days to make full payment

# ### COMMUNICATION STYLE
# - This is a CONVERSATION, not written correspondence
# - Remember to keep responses concise (1-2 sentences)!
# - Speak ONLY as the Debtor - no "**Creditor:**" or "**Debtor:**" labels
# - Do NOT simulate the entire conversation - give only YOUR debtor response
# - Always show progression in your offers - each response should move closer to the creditor's position

# IMPORTANT: When timeline differences are within 5-10 days, consider accepting to resolve the debt
# """
        
#         response = self.llm_debtor.invoke([HumanMessage(content=prompt)])
        
#         new_history = state["history"] + [("buyer", response.content)]
        
#         # Detect current state - check for agreement based on PREVIOUS offers  
#         current_state = "offer"
#         if len(new_history) >= 2:
#             # Get the current debtor offer
#             last_debtor_days = self.extract_days(response.content)
            
#             # Look for the most recent creditor offer
#             creditor_days = None
#             for speaker, message in reversed(new_history[:-1]):  # Exclude current message
#                 if speaker == "seller":
#                     creditor_days = self.extract_days(message)
#                     if creditor_days:
#                         break
            
#             # Check for agreement based on LLM analysis - within reasonable tolerance
#             if last_debtor_days and creditor_days:
#                 difference = abs(last_debtor_days - creditor_days)
#                 if difference <= 5:
#                     current_state = "accept"
#                     print(f"        🎯 CONVERGENCE: Debtor {last_debtor_days} vs Creditor {creditor_days} (diff: {difference} days)")
#                 else:
#                     print(f"        📊 Gap: Debtor {last_debtor_days} vs Creditor {creditor_days} (diff: {difference} days)")
        
#         return {
#             "messages": [response],
#             "turn": "seller",  # Next turn goes to seller
#             "current_state": current_state,
#             "history": new_history,
#             "agent_predictions": state.get("agent_predictions", [])  # Pass through agent predictions
#         }
    
#     def should_continue(self, state: GameState):
#         """Determine if negotiation should continue"""
#         if state["current_state"] in ["accept", "breakdown"]:
#             return "end"
        
#         # Check for agreement on timeline - within 5 days tolerance but not exact copying
#         if len(state["history"]) >= 2:
#             last_two = state["history"][-2:]
#             seller_days = self.extract_days(last_two[0][1])
#             buyer_days = self.extract_days(last_two[1][1])
            
#             # Agreement only if within 5 days AND not exact copying
#             if (seller_days and buyer_days and 
#                 abs(seller_days - buyer_days) <= 5 and
#                 seller_days != buyer_days):  # Prevent exact copying from triggering end
#                 print(f"        🎯 CONVERGENCE: Seller {seller_days} vs Buyer {buyer_days} (within 5 days)")
#                 return "end"
#             elif seller_days and buyer_days and seller_days == buyer_days:
#                 # Check if this is legitimate agreement (both moved to same number) or copying
#                 history_length = len(state["history"])
#                 if history_length >= 4:  # If negotiation has progressed, exact match is ok
#                     print(f"        🎯 FINAL AGREEMENT: Both agreed on {seller_days} days after negotiation")
#                     return "end"
#                 else:
#                     print(f"        ⚠️  Exact match too early - continuing negotiation")
        
#         # Return the next turn
#         next_turn = state.get("turn", "buyer")
#         return next_turn if next_turn in ["buyer", "seller"] else "buyer"
    
#     def run_negotiation(self, max_dialog_len: int = 30) -> Dict[str, Any]:
#         """Run a complete negotiation with Bayesian learning feedback"""
#         # Set up workflow
#         workflow = StateGraph(GameState)
#         workflow.add_node("seller", self.creditor_node)
#         workflow.add_node("buyer", self.debtor_node)
#         workflow.add_edge(START, "seller")
        
#         workflow.add_conditional_edges(
#             "seller",
#             self.should_continue,
#             {"buyer": "buyer", "end": END}
#         )
#         workflow.add_conditional_edges(
#             "buyer",
#             self.should_continue,
#             {"seller": "seller", "end": END}
#         )
        
#         app = workflow.compile()
        
#         # Initial state
#         debt_info = self.config.get('metadata', {})
#         outstanding_balance = debt_info.get('outstanding_balance', 0)
        
#         initial_message = f"Hello, this is the Creditor. We need to discuss the outstanding balance of ${outstanding_balance:,.2f}. I'm proposing a payment timeline of {self.config['seller']['target_price']} days."
        
#         initial_state = GameState(
#             messages=[HumanMessage(content=initial_message)],
#             turn="seller",  # Start with seller since workflow.add_edge(START, "seller")
#             product=self.config["product"],
#             seller_config=self.config["seller"],
#             buyer_config=self.config["buyer"],
#             history=[],  # Start with empty history
#             agent_predictions=[],  # Initialize empty agent predictions
#             current_state="offer"
#         )
        
#         # Run negotiation
#         dialog = []
#         final_state = "breakdown"  # Initialize with default value
        
#         for i, step in enumerate(app.stream(initial_state, {"recursion_limit": max_dialog_len * 2})):
#             if i > max_dialog_len:
#                 break
            
#             for node, value in step.items():
#                 message_content = value["messages"][-1].content
#                 requested_days = self.extract_days(message_content)
                
#                 dialog.append({
#                     "turn": i + 1,
#                     "speaker": node,
#                     "message": message_content,
#                     "state": value["current_state"],
#                     "requested_days": requested_days
#                 })
                
#                 # Display progress
#                 speaker_name = "Creditor" if node == "seller" else "Debtor"
#                 state_emoji = "✅" if value["current_state"] == "accept" else "❌" if value["current_state"] == "breakdown" else "💬"
#                 print(f"        Turn {i+1} - {speaker_name}: {requested_days} days {state_emoji}")
                
#                 # Update final state from negotiation
#                 if value["current_state"] in ["accept", "breakdown"]:
#                     final_state = value["current_state"]
#                     break
        
#         # Use LLM Judge Agent to determine final agreement
#         if len(dialog) >= 2:
#             # Get the last two messages for judge analysis
#             last_creditor_msg = None
#             last_debtor_msg = None
#             last_creditor_days = None
#             last_debtor_days = None
            
#             # Find last creditor and debtor messages
#             for entry in reversed(dialog):
#                 if entry["speaker"] == "seller" and last_creditor_msg is None:
#                     last_creditor_msg = entry["message"]
#                     last_creditor_days = entry["requested_days"]
#                 elif entry["speaker"] == "buyer" and last_debtor_msg is None:
#                     last_debtor_msg = entry["message"]
#                     last_debtor_days = entry["requested_days"]
                
#                 if last_creditor_msg and last_debtor_msg:
#                     break
            
#             # Use LLM Judge to analyze agreement
#             if last_creditor_msg and last_debtor_msg:
#                 judge_result = self.llm_judge_agreement(
#                     last_creditor_msg, last_debtor_msg, 
#                     last_creditor_days, last_debtor_days
#                 )
                
#                 if judge_result["agreement"]:
#                     final_state = "accept"
#                     print(f"        🤖 LLM Judge: {judge_result['reasoning']} (confidence: {judge_result['confidence']})")
#                     # Use judge's suggested final days or calculate average
#                     if judge_result.get("final_days"):
#                         final_days = judge_result["final_days"]
#                     elif last_creditor_days and last_debtor_days:
#                         final_days = round((last_creditor_days + last_debtor_days) / 2)
#                 else:
#                     final_state = "breakdown"
#                     print(f"        🤖 LLM Judge: {judge_result['reasoning']} (confidence: {judge_result['confidence']})")
        
#         final_days = None
        
#         # Extract final agreed days if accepted
#         if final_state == "accept" and len(dialog) >= 2:
#             # Get the last creditor and debtor offers to see if they match
#             last_seller_days = None
#             last_buyer_days = None
            
#             for entry in reversed(dialog):
#                 if entry["speaker"] == "seller" and entry["requested_days"] and last_seller_days is None:
#                     last_seller_days = entry["requested_days"]
#                 elif entry["speaker"] == "buyer" and entry["requested_days"] and last_buyer_days is None:
#                     last_buyer_days = entry["requested_days"]
                
#                 if last_seller_days is not None and last_buyer_days is not None:
#                     break
            
#             # Check if they agreed within 5 days tolerance (like original system)
#             if last_seller_days and last_buyer_days and abs(last_seller_days - last_buyer_days) <= 5:
#                 # Use the average of the two offers as the final agreed timeline
#                 final_days = round((last_seller_days + last_buyer_days) / 2)
#                 print(f"      ✅ CLOSE AGREEMENT: Creditor {last_seller_days} vs Debtor {last_buyer_days} → Final: {final_days} days")
#             elif last_seller_days:
#                 # Use the last creditor offer as the agreed timeline
#                 final_days = last_seller_days
#                 print(f"      ✅ ACCEPTED: Creditor's {final_days} days accepted")
#             elif last_buyer_days:
#                 # Use the last debtor offer
#                 final_days = last_buyer_days
#                 print(f"      ✅ ACCEPTED: Debtor's {final_days} days accepted")
        
#         # If no clear agreement but state shows accept, try to extract from dialog
#         elif final_state == "accept" and len(dialog) > 0:
#             # Get the very last offer made
#             for entry in reversed(dialog):
#                 if entry["requested_days"]:
#                     final_days = entry["requested_days"]
#                     print(f"      ✅ ACCEPTED: Final timeline {final_days} days")
#                     break
        
#         # Prepare COMPLETE learning feedback for Bayesian model
#         # Prepare COMPLETE learning feedback for Bayesian model
#         # In run_negotiation method:
#         learning_data = {
#             'success': final_state == 'accept',
#             'collection_days': final_days,
#             'negotiation_rounds': len(dialog),
#             'final_state': final_state,
#             'last_debtor_emotion': self.last_debtor_emotion,  # Already uppercase
#             'debtor_emotion_sequence': self.debtor_emotion_sequence,
#             'debt_amount': self.config.get('metadata', {}).get('outstanding_balance', 0),
#             'gap_size': self.last_gap_size,
#             'emotion_history': self.emotion_sequence.copy(),  # Should be uppercase codes
#             'transition': f"{self.emotion_sequence[-2] if len(self.emotion_sequence) >= 2 else 'N'} → {self.emotion_sequence[-1] if self.emotion_sequence else 'N'}"  # Use 'N'
#         }
#         # Add agent predictions if available
#         if self.agent_predictions_history:
#             # Get predictions from the last round
#             last_prediction_round = self.agent_predictions_history[-1]
#             learning_data['agent_predictions'] = last_prediction_round['predictions']
            
#             # Log learning data
#             print(f"      🧠 Learning data prepared:")
#             print(f"        - Success: {learning_data['success']}")
#             print(f"        - Agent predictions: {len(learning_data['agent_predictions'])}")
#             print(f"        - Transition: {learning_data['transition']}")
#             print(f"        - Last debtor emotion: {learning_data['last_debtor_emotion']}")
        
#         # Calculate final metrics
#         collection_days = final_days if final_state == "accept" else None
#         creditor_target_days = int(self.config['seller']['target_price'])
#         total_rounds = len(dialog)
        
#         result = {
#             "scenario_id": self.config['id'],
#             "final_state": final_state,
#             "collection_days": collection_days,
#             "final_days": final_days,
#             "creditor_target_days": creditor_target_days,
#             "negotiation_rounds": total_rounds,
#             "dialog_length": total_rounds,
#             "emotion_sequence": self.emotion_sequence.copy(),
#             "dialog": dialog,
#             "success": final_state == "accept",
#             # Learning feedback data (CRITICAL for Bayesian model)
#             "agent_predictions": learning_data.get('agent_predictions', []),
#             "last_debtor_emotion": learning_data['last_debtor_emotion'],
#             "debt_amount": learning_data['debt_amount'],
#             "gap_size": learning_data['gap_size'],
#             "transition": learning_data['transition']
#         }
        
#         # Debug output
#         print(f"      📊 Final Result: {final_state} | Days: {final_days} | Rounds: {total_rounds}")
#         print(f"      🧠 Learning feedback: {len(result['agent_predictions'])} agent predictions included")
        
#         # Reset for next negotiation
#         self.negotiation_round = 0
#         self.emotion_sequence = []
#         self.debtor_emotion_sequence = []
#         self.agent_predictions_history = []
#         self.last_gap_size = 0
#         self.last_debtor_emotion = self.debtor_emotion
        
#         # Reset model if it has reset method
#         if hasattr(self.emotion_model, 'reset'):
#             self.emotion_model.reset()
        
#         return result




"""
Debt negotiator for Bayesian Multi-Agent Transition Optimization
Specifically designed to provide learning feedback to Bayesian models
"""

from typing import Dict, List, Any, Optional
from langgraph.graph import StateGraph, END, START
from langchain_core.messages import HumanMessage, AIMessage
from llm.llm_wrapper import LLMWrapper
import re
import json

class GameState(Dict):
    """Game state for negotiation with agent prediction tracking"""
    messages: List
    turn: str
    product: Dict
    seller_config: Dict
    buyer_config: Dict
    history: List
    current_state: str
    agent_predictions: List  # Track agent predictions for learning

class DebtNegotiator:
    """Debt negotiator for Bayesian transition optimization model"""
    
    def __init__(
        self,
        config: Dict[str, Any],
        emotion_model: Any,  # BayesianTransitionModel
        model_creditor: str = "gpt-4o-mini",
        model_debtor: str = "gpt-4o-mini",
        debtor_emotion: str = "neutral"
    ):
        self.config = config
        self.emotion_model = emotion_model
        
        # Convert debtor_emotion to uppercase code
        emotion_mapping = {
            'happy': 'J', 'joyful': 'J',
            'sadness': 'S', 'sad': 'S', 
            'anger': 'A', 'angry': 'A',
            'fear': 'F', 'fearful': 'F',
            'surprise': 'Su', 'surprising': 'Su',
            'disgust': 'D', 'disgusted': 'D',
            'neutral': 'N'
        }
        self.debtor_emotion = emotion_mapping.get(debtor_emotion.lower(), 'N')
        
        # Initialize LLMs
        self.llm_creditor = LLMWrapper(model_creditor, "creditor")
        self.llm_debtor = LLMWrapper(model_debtor, "debtor")
        
        # State tracking
        self.negotiation_round = 0
        self.emotion_history = []  # Store creditor's emotion history
        self.debtor_emotion_history = []  # Store debtor's emotion history
        
        # Learning feedback tracking - ENHANCED for trajectory learning
        self.agent_predictions_history = []  # Store ALL predictions for each round
        self.context_history = []  # Store context for each round
        self.last_gap_size = 0
        self.last_debtor_emotion = self.debtor_emotion  # Use the normalized version
    
    def llm_judge_agreement(self, creditor_message: str, debtor_message: str, 
                           creditor_days: int = None, debtor_days: int = None) -> dict:
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
            return "N"  # Return uppercase code
        
        emotion_prompt = f"""Analyze the emotional tone of this debt negotiation message and classify it into ONE of these categories:

    EMOTIONS: happy, surprising, angry, sad, disgust, fear, neutral

    MESSAGE TO ANALYZE:
    "{debtor_message}"

    Respond with ONLY the emotion word (e.g., "angry" or "sad"):"""
        
        try:
            response = self.llm_debtor.invoke([HumanMessage(content=emotion_prompt)])
            detected_emotion = response.content.strip().lower()
            
            # Convert to uppercase emotion codes
            emotion_mapping = {
                'happy': 'J',
                'surprising': 'Su', 
                'angry': 'A',
                'sad': 'S',
                'disgust': 'D',
                'fear': 'F',
                'neutral': 'N'
            }
            
            if detected_emotion in emotion_mapping:
                return emotion_mapping[detected_emotion]
            else:
                return "N"  # Default to Neutral
        except Exception as e:
            print(f"        ⚠️  Emotion detection failed: {e}")
            return "N"
    
    def _calculate_gap_size(self, conversation_history: List) -> float:
        """Calculate the gap between creditor and debtor proposals"""
        creditor_days = None
        debtor_days = None
        
        # Find the most recent creditor and debtor offers
        for speaker, message in reversed(conversation_history):
            days = self.extract_days(message)
            if days:
                if speaker == "seller" and creditor_days is None:
                    creditor_days = days
                elif speaker == "buyer" and debtor_days is None:
                    debtor_days = days
                
                if creditor_days is not None and debtor_days is not None:
                    break
        
        # Calculate gap
        if creditor_days is not None and debtor_days is not None:
            return abs(creditor_days - debtor_days)
        return 0.0
    
    def creditor_node(self, state: GameState):
        """Creditor (seller) node with Bayesian agent feedback"""
        self.negotiation_round += 1
        
        # Detect current debtor emotion
        detected_debtor_emotion = self.debtor_emotion
        conversation_history = state.get("history", [])
        
        if conversation_history:
            # Get the most recent debtor message
            for speaker, message in reversed(conversation_history):
                if speaker == "buyer":
                    detected_debtor_emotion = self.detect_debtor_emotion(message)
                    self.debtor_emotion_history.append(detected_debtor_emotion)
                    self.last_debtor_emotion = detected_debtor_emotion
                    break
        
        # Calculate gap size for Bayesian context
        self.last_gap_size = self._calculate_gap_size(conversation_history)
        
        # Get current creditor emotion (last emotion in history, or 'N' for first round)
        current_creditor_emotion = self.emotion_history[-1] if self.emotion_history else 'N'
        
        # Get emotion from Bayesian transition model
        model_state = {
            'round': self.negotiation_round,
            'debtor_emotion': detected_debtor_emotion,
            'current_emotion': current_creditor_emotion,
            'debt_amount': self.config.get('metadata', {}).get('outstanding_balance', 0),
            'gap_size': self.last_gap_size
        }
        
        # 🧠 Get emotion from Bayesian model
        emotion_config = self.emotion_model.select_emotion(model_state)
        creditor_emotion = emotion_config['emotion']
        self.emotion_history.append(creditor_emotion)  # Add to emotion history
        
        # 📊 Store agent predictions for learning feedback - ENHANCED for trajectory learning
        current_predictions = []
        if 'agent_predictions' in emotion_config:
            current_predictions = emotion_config['agent_predictions']
            
            # Store detailed prediction data for trajectory learning
            prediction_record = {
                'round': self.negotiation_round,
                'predictions': current_predictions,
                'selected_emotion': creditor_emotion,
                'debtor_emotion': detected_debtor_emotion,
                'gap_size': self.last_gap_size,
                'context': model_state.copy(),  # Store full context for learning
                'creditor_emotion_before': current_creditor_emotion,
                'creditor_emotion_after': creditor_emotion,
                'bayesian_analysis': emotion_config.get('bayesian_analysis', {})
            }
            self.agent_predictions_history.append(prediction_record)
        
        # Also store context for trajectory analysis
        self.context_history.append({
            'round': self.negotiation_round,
            'debtor_emotion': detected_debtor_emotion,
            'creditor_emotion': creditor_emotion,
            'gap_size': self.last_gap_size,
            'successful_round': None  # Will be updated at end
        })
        
        # 🧠 Print Bayesian model details
        model_type = type(self.emotion_model).__name__
        print(f"        🧠 {model_type}: {creditor_emotion} (Round {self.negotiation_round})")
        
        # Show Bayesian-specific details
        if 'transition' in emotion_config:
            print(f"           Transition: {emotion_config['transition']}")
        if 'exploration' in emotion_config:
            print(f"           Exploration: {'Yes' if emotion_config['exploration'] else 'No'}")
        if 'confidence' in emotion_config:
            print(f"           Confidence: {emotion_config['confidence']:.2f}")
        if current_predictions:
            print(f"           🤖 {len(current_predictions)} agents consulted")
            # Show ALL agents
            for pred in current_predictions:
                print(f"             {pred['agent']}: {pred['target']} (confidence: {pred['confidence']:.2f})")
        
        print(f"           Negotiator Emotion History: {self.emotion_history[-5:]}...")
        
        # Build prompt
        config = self.config.get("seller_config", self.config["seller"])
        debt_info = self.config.get('metadata', {})
        outstanding_balance = debt_info.get('outstanding_balance', 0)
        
        # Extract timeline history
        creditor_days = []
        debtor_days = []
        
        for speaker, message in conversation_history:
            days = self.extract_days(message)
            if days:
                if speaker == "seller":
                    creditor_days.append(days)
                elif speaker == "buyer":
                    debtor_days.append(days)
        
        # Build constraint text
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
        
        # Update agent predictions in state
        new_agent_predictions = state.get("agent_predictions", []) + [{
            'round': self.negotiation_round,
            'predictions': current_predictions,
            'selected_emotion': creditor_emotion
        }]
        
        # Detect current state
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
            "agent_predictions": new_agent_predictions,
            "creditor_emotion": creditor_emotion
        }
    
    def debtor_node(self, state: GameState):
        """Debtor (buyer) node"""
        config = self.config.get("buyer_config", self.config["buyer"])
        debt_info = self.config.get('metadata', {})
        outstanding_balance = debt_info.get('outstanding_balance', 0)
        
        # Debtor emotion prompt
        emotion_prompt = ""
        if self.debtor_emotion != "N":  # Using uppercase 'N' for neutral
            emotions = {
                "J": "Maintain an optimistic attitude",  # Joy
                "Su": "Show enthusiasm",  # Surprise
                "A": "Express frustration while remaining civil",  # Anger
                "S": "Show distress",  # Sadness
                "D": "Express disappointment",  # Disgust
                "F": "Show anxiety"  # Fear
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
            "history": new_history,
            "agent_predictions": state.get("agent_predictions", [])  # Pass through agent predictions
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
    
    def _prepare_trajectory_learning_data(self, success: bool, final_days: Optional[int]) -> Dict[str, Any]:
        """
        Prepare trajectory learning data for Bayesian model
        This creates a complete record of the entire negotiation for trajectory-based learning
        """
        trajectory_data = {
            'success': success,
            'negotiation_rounds': len(self.emotion_history),
            'emotion_history': self.emotion_history.copy(),  # Creditor's emotion sequence
            'debtor_emotion_history': self.debtor_emotion_history.copy(),
            'final_days': final_days,
            'last_debtor_emotion': self.last_debtor_emotion,
            'debt_amount': self.config.get('metadata', {}).get('outstanding_balance', 0),
            'initial_gap': None,  # Will calculate below
            'final_gap': self.last_gap_size,
            'agent_predictions_history': self.agent_predictions_history.copy(),
            'context_history': self.context_history.copy()
        }
        
        # Calculate initial gap if we have enough history
        if len(self.emotion_history) >= 2:
            # Try to find initial gap from context history
            if len(self.context_history) >= 2:
                initial_gap = self.context_history[0].get('gap_size', 0)
                final_gap = self.context_history[-1].get('gap_size', 0)
                trajectory_data['initial_gap'] = initial_gap
                trajectory_data['final_gap'] = final_gap
                
                # Calculate gap reduction
                if initial_gap > 0:
                    gap_reduction = 1.0 - (final_gap / initial_gap)
                    trajectory_data['gap_reduction'] = max(0.0, min(1.0, gap_reduction))
        
        # Mark which rounds were successful in terms of gap reduction
        for i, context in enumerate(trajectory_data['context_history']):
            if i > 0:
                prev_gap = trajectory_data['context_history'][i-1].get('gap_size', 0)
                current_gap = context.get('gap_size', 0)
                # Success: gap reduced by at least 10%
                if prev_gap > 0 and current_gap < prev_gap * 0.9:
                    context['successful_round'] = True
                else:
                    context['successful_round'] = False
        
        return trajectory_data
    
    def _prepare_agent_performance_metrics(self) -> Dict[str, Dict[str, Any]]:
        """
        Calculate agent performance metrics from the trajectory
        Returns: Dict with agent_name -> performance stats
        """
        agent_performance = {}
        
        if not self.agent_predictions_history:
            return agent_performance
        
        # Initialize performance tracking for each agent
        all_agent_names = set()
        for prediction_round in self.agent_predictions_history:
            for pred in prediction_round.get('predictions', []):
                agent_name = pred.get('agent')
                if agent_name:
                    all_agent_names.add(agent_name)
        
        # Initialize performance dict
        for agent_name in all_agent_names:
            agent_performance[agent_name] = {
                'total_predictions': 0,
                'correct_predictions': 0,
                'total_confidence': 0.0,
                'correct_confidence': 0.0,
                'emotion_preferences': {},  # Which emotions this agent tends to predict
                'context_accuracy': {}  # Accuracy by context type
            }
        
        # Calculate performance metrics
        for round_idx, prediction_round in enumerate(self.agent_predictions_history):
            selected_emotion = prediction_round.get('selected_emotion')
            debtor_emotion = prediction_round.get('debtor_emotion')
            context_type = f"debtor_{debtor_emotion}"
            
            for pred in prediction_round.get('predictions', []):
                agent_name = pred.get('agent')
                if not agent_name:
                    continue
                
                target_emotion = pred.get('target')
                confidence = pred.get('confidence', 0.0)
                
                # Update basic stats
                perf = agent_performance[agent_name]
                perf['total_predictions'] += 1
                perf['total_confidence'] += confidence
                
                # Check if prediction was correct
                if target_emotion == selected_emotion:
                    perf['correct_predictions'] += 1
                    perf['correct_confidence'] += confidence
                
                # Track emotion preferences
                if target_emotion not in perf['emotion_preferences']:
                    perf['emotion_preferences'][target_emotion] = 0
                perf['emotion_preferences'][target_emotion] += 1
                
                # Track context accuracy
                if context_type not in perf['context_accuracy']:
                    perf['context_accuracy'][context_type] = {'correct': 0, 'total': 0}
                perf['context_accuracy'][context_type]['total'] += 1
                if target_emotion == selected_emotion:
                    perf['context_accuracy'][context_type]['correct'] += 1
        
        # Calculate final metrics
        for agent_name, perf in agent_performance.items():
            if perf['total_predictions'] > 0:
                perf['accuracy'] = perf['correct_predictions'] / perf['total_predictions']
                perf['confidence_weighted_accuracy'] = (
                    perf['correct_confidence'] / perf['total_confidence'] 
                    if perf['total_confidence'] > 0 else perf['accuracy']
                )
                
                # Find most preferred emotion
                if perf['emotion_preferences']:
                    most_preferred = max(perf['emotion_preferences'].items(), key=lambda x: x[1])
                    perf['most_preferred_emotion'] = most_preferred[0]
                    perf['preference_score'] = most_preferred[1] / perf['total_predictions']
        
        return agent_performance
    
    def run_negotiation(self, max_dialog_len: int = 30) -> Dict[str, Any]:
        """Run a complete negotiation with Bayesian learning feedback"""
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
            history=[],  # Start with empty history
            agent_predictions=[],  # Initialize empty agent predictions
            current_state="offer"
        )
        
        # Run negotiation
        dialog = []
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
        
        # Prepare COMPREHENSIVE learning feedback for Bayesian model - ENHANCED for trajectory learning
        success = final_state == 'accept'
        
        # Prepare trajectory learning data
        trajectory_data = self._prepare_trajectory_learning_data(success, final_days)
        
        # Prepare agent performance metrics
        agent_performance = self._prepare_agent_performance_metrics()
        
        # Get the last prediction for backward compatibility
        last_prediction_data = None
        if self.agent_predictions_history:
            last_prediction_data = self.agent_predictions_history[-1]
        
        # Enhanced learning data structure
        learning_data = {
            'success': success,
            'collection_days': final_days,
            'negotiation_rounds': len(dialog),
            'final_state': final_state,
            'last_debtor_emotion': self.last_debtor_emotion,
            'debtor_emotion_history': self.debtor_emotion_history.copy(),
            'debt_amount': self.config.get('metadata', {}).get('outstanding_balance', 0),
            'gap_size': self.last_gap_size,
            'emotion_history': self.emotion_history.copy(),
            'transition': f"{self.emotion_history[-2] if len(self.emotion_history) >= 2 else 'N'} → {self.emotion_history[-1] if self.emotion_history else 'N'}",
            
            # NEW: Enhanced trajectory learning data
            'trajectory_data': trajectory_data,
            'agent_performance': agent_performance,
            'agent_predictions_history': self.agent_predictions_history.copy(),
            'context_history': self.context_history.copy(),
            'initial_gap': trajectory_data.get('initial_gap', 0),
            'final_gap': trajectory_data.get('final_gap', 0),
            'gap_reduction': trajectory_data.get('gap_reduction', 0.0),
            
            # Backward compatibility
            'agent_predictions': last_prediction_data.get('predictions', []) if last_prediction_data else []
        }
        
        # Log learning data
        print(f"      🧠 Enhanced Learning Feedback Prepared:")
        print(f"        - Success: {learning_data['success']}")
        print(f"        - Negotiation Rounds: {len(dialog)}")
        print(f"        - Emotion History: {learning_data['emotion_history']}")
        print(f"        - Trajectory Data Points: {len(learning_data['agent_predictions_history'])}")
        print(f"        - Gap Reduction: {learning_data.get('gap_reduction', 0.0):.1%}")
        
        # Calculate final metrics
        collection_days = final_days if final_state == "accept" else None
        creditor_target_days = int(self.config['seller']['target_price'])
        total_rounds = len(dialog)
        
        # Enhanced result structure
        result = {
            "scenario_id": self.config['id'],
            "final_state": final_state,
            "collection_days": collection_days,
            "final_days": final_days,
            "creditor_target_days": creditor_target_days,
            "negotiation_rounds": total_rounds,
            "dialog_length": total_rounds,
            "emotion_sequence": self.emotion_history.copy(),
            "debtor_emotion_sequence": self.debtor_emotion_history.copy(),
            "dialog": dialog,
            "success": success,
            
            # Enhanced learning feedback data
            "agent_predictions": learning_data.get('agent_predictions', []),
            "agent_predictions_history": learning_data.get('agent_predictions_history', []),
            "agent_performance": learning_data.get('agent_performance', {}),
            "trajectory_data": learning_data.get('trajectory_data', {}),
            "last_debtor_emotion": learning_data['last_debtor_emotion'],
            "debt_amount": learning_data['debt_amount'],
            "gap_size": learning_data['gap_size'],
            "transition": learning_data['transition'],
            "initial_gap": learning_data.get('initial_gap', 0),
            "final_gap": learning_data.get('final_gap', 0),
            "gap_reduction": learning_data.get('gap_reduction', 0.0)
        }
        
        # Debug output
        print(f"      📊 Final Result: {final_state} | Days: {final_days} | Rounds: {total_rounds}")
        print(f"      🧠 Enhanced Learning Feedback:")
        print(f"        - Trajectory Rounds: {len(result['agent_predictions_history'])}")
        print(f"        - Agent Performance: {len(result['agent_performance'])} agents analyzed")
        print(f"        - Gap Reduction: {result.get('gap_reduction', 0.0):.1%}")
        
        # Reset for next negotiation
        self.negotiation_round = 0
        self.emotion_history = []
        self.debtor_emotion_history = []
        self.agent_predictions_history = []
        self.context_history = []
        self.last_gap_size = 0
        self.last_debtor_emotion = self.debtor_emotion
        
        # Reset model if it has reset method
        if hasattr(self.emotion_model, 'reset'):
            self.emotion_model.reset()
        
        return result