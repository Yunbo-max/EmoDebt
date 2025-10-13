import csv
import json
import argparse
import random
from enum import Enum
from datetime import datetime

# Define emotions (Ekman's 6 + neutral)
class Emotion(Enum):
    NEUTRAL = "neutral"
    ANGER = "anger"
    DISGUST = "disgust"
    FEAR = "fear"
    HAPPINESS = "happiness"
    SADNESS = "sadness"
    SURPRISE = "surprise"

# Emotion descriptions (for natural language output)
EMOTION_DESCRIPTIONS = {
    Emotion.NEUTRAL: ["calmly", "neutrally", "evenly"],
    Emotion.ANGER: ["angrily", "furiously", "irritably"],
    Emotion.DISGUST: ["disgustedly", "contemptuously", "sneeringly"],
    Emotion.FEAR: ["fearfully", "nervously", "anxiously"],
    Emotion.HAPPINESS: ["happily", "cheerfully", "joyfully"],
    Emotion.SADNESS: ["sadly", "gloomily", "mournfully"],
    Emotion.SURPRISE: ["surprisedly", "astonishedly", "shockedly"]
}

# Updated Emotion system with intensity levels
EMOTIONS = {
    "neutral": ["calm", "balanced", "measured"],
    "anger": ["angry", "frustrated", "irritated"],
    "disgust": ["disgusted", "contemptuous", "repulsed"],
    "fear": ["afraid", "nervous", "anxious"],
    "happiness": ["happy", "pleased", "content"],
    "sadness": ["sad", "disappointed", "gloomy"],
    "surprise": ["surprised", "astonished", "shocked"]
}

def parse_csv(file_path):
    """Parse CSV file containing debt collection scenarios"""
    with open(file_path, mode='r', encoding='utf-8', errors='ignore') as file:
        reader = csv.DictReader(file)
        data = [row for row in reader]
        return data

def get_emotion_profile(n_emotions: int = 3) -> dict:
    """Returns a profile with multiple emotions and their descriptors"""
    selected_emotions = random.sample(list(EMOTIONS.keys()), n_emotions)
    
    emotion_texts = []
    for emotion in selected_emotions:
        descriptors = random.sample(EMOTIONS[emotion], min(2, len(EMOTIONS[emotion])))
        emotion_texts.append(f"{' and '.join(descriptors)} ({emotion})")
    
    return {
        "emotions": selected_emotions,
        "emotion_text": "Initially " + ", then ".join(emotion_texts)
    }

def create_debt_description(row):
    """Create a detailed description of the debt scenario"""
    description = f"""
Debt Collection Scenario:
- Credit Type: {row['Credit Type']}
- Original Amount: ${float(row['Original Amount (USD)']):,.2f}
- Outstanding Balance: ${float(row['Outstanding Balance (USD)']):,.2f}
- Days Overdue: {row['Days Overdue']} days
- Purchase Purpose: {row['Purchase Purpose']}
- Reason for Overdue: {row['Reason for Overdue']}
- Business Sector: {row['Business Sector']}
- Collateral: {row['Collateral']}
- Recovery Stage: {row['Recovery Stage']}
- Cash Flow Situation: {row['Cash Flow Situation']}
- Business Impact: {row['Business Impact Description']}
- Proposed Solution: {row['Proposed Solution']}
- Recovery Probability: {row['Recovery Probability (%)']}%
- Interest Accrued: ${float(row['Interest Accrued (USD)']):,.2f}
""".strip()
    return description

def calculate_negotiation_targets(row):
    """Calculate negotiation targets for days - creditor wants shorter timeline, debtor wants longer"""
    creditor_target_days = int(row['Creditor Target Days'])
    debtor_target_days = int(row['Debtor Target Days'])
    
    # Add some variation around the base targets
    creditor_variation = random.uniform(0.8, 1.2)
    debtor_variation = random.uniform(0.8, 1.2)
    
    creditor_target = max(1, int(creditor_target_days * creditor_variation))
    debtor_target = max(creditor_target + 1, int(debtor_target_days * debtor_variation))
    
    return creditor_target, debtor_target



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate debt collection negotiation scenarios')
    parser.add_argument('--debt_csv', default='./data/credit_recovery_scenarios.csv',
                        help='Path to debt collection CSV file')
    parser.add_argument('--n_trial_per_debt', type=int, default=2,
                        help='Number of negotiation trials per debt scenario')
    parser.add_argument('--n_emotions', type=int, default=3,
                        help='Number of emotions per agent')
    parser.add_argument('--out_fn', default='data/debt_collection_scenarios.json',
                        help='Output JSON file path')
    args = parser.parse_args()

    # Parse the debt collection CSV
    debt_scenarios = parse_csv(args.debt_csv)
    game_settings = []

    for idx, debt_row in enumerate(debt_scenarios):
        # Skip empty rows or invalid data
        if not debt_row.get('Creditor Name') or not debt_row.get('Debtor Name'):
            continue
        
        for trial in range(args.n_trial_per_debt):
            # Calculate negotiation targets (days)
            creditor_target, debtor_target = calculate_negotiation_targets(debt_row)
            
            # Create unique scenario ID
            scenario_id = f"debt_{idx}_{trial}"
            
            # Create debt description (acts as "product")
            debt_description = create_debt_description(debt_row)
            
            # Build the game configuration
            game_conf = {
                'id': scenario_id,
                'product': {
                    'name': f"{debt_row['Credit Type']} - {debt_row['Creditor Name']} vs {debt_row['Debtor Name']}",
                    'category': 'debt_collection',
                    'description': debt_description
                }
            }
            
            # Creditor (seller) configuration - wants shorter payment timeline
            creditor_emotions = get_emotion_profile(n_emotions=args.n_emotions)
            outstanding_balance = float(debt_row['Outstanding Balance (USD)'])

            
            game_conf['seller'] = {
                'target_price': str(creditor_target),  # Target days for payment
                'emotions': creditor_emotions['emotions'],
                'emotion_text': creditor_emotions['emotion_text']
            }
            
            # Debtor (buyer) configuration - wants longer payment timeline
            debtor_emotions = get_emotion_profile(n_emotions=args.n_emotions)
            
            game_conf['buyer'] = {
                'target_price': str(debtor_target),  # Target days for payment
                'emotions': debtor_emotions['emotions'],
                'emotion_text': debtor_emotions['emotion_text']
            }
            
            # Add metadata for analysis
            game_conf['metadata'] = {
                'creditor_name': debt_row['Creditor Name'],
                'debtor_name': debt_row['Debtor Name'],
                'original_amount': float(debt_row['Original Amount (USD)']),
                'outstanding_balance': outstanding_balance,
                'days_overdue': int(debt_row['Days Overdue']),
                'recovery_stage': debt_row['Recovery Stage'],
                'cash_flow_situation': debt_row['Cash Flow Situation'],
                'recovery_probability': float(debt_row['Recovery Probability (%)']),
                'business_impact': debt_row['Business Impact Description'],
                'proposed_solution': debt_row['Proposed Solution'],
                'creditor_original_target_days': int(debt_row['Creditor Target Days']),
                'debtor_original_target_days': int(debt_row['Debtor Target Days'])
            }
            
            game_settings.append(game_conf)

    # Save to JSON file
    with open(args.out_fn, 'w', encoding='utf-8') as f:
        json.dump(game_settings, f, indent=4, ensure_ascii=False)

    print(f"Generated {len(game_settings)} debt collection negotiation scenarios")
    print(f"Saved to: {args.out_fn}")
    
    # Print some statistics
    print(f"\nStatistics:")
    print(f"- Total debt scenarios processed: {len(debt_scenarios)}")
    print(f"- Trials per scenario: {args.n_trial_per_debt}")
    print(f"- Total negotiation scenarios: {len(game_settings)}")
    
    # Sample scenario info
    if game_settings:
        sample = game_settings[0]
        print(f"\nSample scenario:")
        print(f"- ID: {sample['id']}")
        print(f"- Creditor target: {sample['seller']['target_price']} days")
        print(f"- Debtor target: {sample['buyer']['target_price']} days")
        print(f"- Outstanding balance: ${sample['metadata']['outstanding_balance']:,.2f}")
        print(f"- Creditor emotions: {sample['seller']['emotions']}")
        print(f"- Debtor emotions: {sample['buyer']['emotions']}")