#!/usr/bin/env python3
"""
Test dataset-specific prompts across different scenario types
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm.prompt_templates import PromptTemplates

def test_prompt_templates():
    """Test all dataset-specific prompt templates"""
    
    # Sample configuration for testing
    sample_config = {
        'target_price': 30,
        'min_price': 15,
        'max_price': 60
    }
    
    sample_emotion_config = {
        'emotion_text': 'Use a calm and professional tone',
        'temperature': 0.7
    }
    
    timeline_text = "Previous negotiations have established some baseline expectations."
    
    # Test different scenario types with appropriate metadata
    scenario_configs = {
        'debt': {
            'outstanding_balance': 15000.0,
            'creditor_name': 'ABC Collections',
            'debtor_name': 'Small Business Inc',
            'recovery_stage': 'Early Collection',
            'business_sector': 'Retail',
            'reason_for_overdue': 'Cash flow issues'
        },
        'disaster': {
            'disaster_type': 'Earthquake',
            'survivor_condition': 'Trapped with minor injuries',
            'estimated_endurance': 90,
            'critical_needs': 'Water, medical attention',
            'rescue_eta': 45
        },
        'student': {
            'student_age': 16,
            'student_background': 'High achiever with test anxiety',
            'situation_faced': 'Important exam tomorrow',
            'student_wanted_bedtime': '2:00 AM',
            'primary_annoyance_reason': 'Fear of failing the test'
        },
        'medical': {
            'patient_age': 65,
            'patient_condition': 'Coronary artery disease',
            'required_surgery': 'Bypass surgery',
            'urgency_level': 'High',
            'days_on_waitlist': 45,
            'patient_reason_for_urgency': 'Chest pain worsening'
        }
    }
    
    print("🎭 Testing Dataset-Specific Prompt Templates")
    print("=" * 60)
    
    for scenario_type, debt_info in scenario_configs.items():
        print(f"\n### {scenario_type.upper()} SCENARIO ###")
        
        # Test scenario detection
        detected_type = PromptTemplates.detect_scenario_type(debt_info)
        print(f"Detected Type: {detected_type} (Expected: {scenario_type})")
        
        # Test creditor prompt
        print(f"\n--- CREDITOR PROMPT ({scenario_type}) ---")
        creditor_prompt = PromptTemplates.get_creditor_prompt(
            scenario_type, sample_config, sample_emotion_config, timeline_text, debt_info
        )
        
        # Print first few lines of the prompt to show it's different
        prompt_lines = creditor_prompt.split('\\n')[:8]
        for line in prompt_lines:
            print(f"  {line}")
        print(f"  ... [truncated {len(prompt_lines) - 8} more lines]")
        
        # Test debtor prompt  
        print(f"\n--- DEBTOR PROMPT ({scenario_type}) ---")
        debtor_prompt = PromptTemplates.get_debtor_prompt(
            scenario_type, sample_config, "Show determination but remain respectful", debt_info
        )
        
        prompt_lines = debtor_prompt.split('\\n')[:8]
        for line in prompt_lines:
            print(f"  {line}")
        print(f"  ... [truncated {len(prompt_lines) - 8} more lines]")
        
        # Test emotion detection prompt
        emotion_prompt = PromptTemplates.get_emotion_detection_prompt(scenario_type)
        print(f"\n--- EMOTION DETECTION ({scenario_type}) ---")
        print(f"  {emotion_prompt.split('MESSAGE TO ANALYZE:')[0]}...")
        
        print(f"\n{'='*40}")

def demonstrate_context_differences():
    """Show how different contexts change the prompt content"""
    
    print("\\n🔍 CONTEXT-SPECIFIC EXAMPLES")
    print("=" * 50)
    
    # Same configuration, different contexts
    config = {'target_price': 30}
    emotion_config = {'emotion_text': 'Be assertive but understanding'}
    timeline = "Current gap is significant"
    
    contexts = {
        'debt': {'outstanding_balance': 10000, 'business_sector': 'Manufacturing'},
        'disaster': {'disaster_type': 'Flood', 'survivor_condition': 'Hypothermia risk'},
        'student': {'student_age': 14, 'situation_faced': 'Big presentation tomorrow'},
        'medical': {'patient_condition': 'Appendicitis', 'urgency_level': 'Medium'}
    }
    
    for scenario_type, metadata in contexts.items():
        print(f"\\n{scenario_type.upper()} - Key Differences:")
        
        creditor_prompt = PromptTemplates.get_creditor_prompt(
            scenario_type, config, emotion_config, timeline, metadata
        )
        
        # Extract role description (first paragraph typically)
        role_section = creditor_prompt.split('###')[0].strip()
        print(f"  Role: {role_section[:100]}...")
        
        # Find scenario-specific vocabulary
        if 'debt' in scenario_type and 'creditor' in creditor_prompt.lower():
            print("  ✓ Uses debt collection vocabulary")
        elif 'disaster' in scenario_type and ('rescue' in creditor_prompt.lower() or 'survivor' in creditor_prompt.lower()):
            print("  ✓ Uses disaster rescue vocabulary")
        elif 'student' in scenario_type and 'sleep' in creditor_prompt.lower():
            print("  ✓ Uses sleep health vocabulary")
        elif 'medical' in scenario_type and ('surgery' in creditor_prompt.lower() or 'patient' in creditor_prompt.lower()):
            print("  ✓ Uses medical scheduling vocabulary")

if __name__ == "__main__":
    test_prompt_templates()
    demonstrate_context_differences()
    
    print("\\n✅ Dataset-specific prompt testing complete!")
    print("\\nNow you can run experiments with different dataset types:")
    print("  python experiments/run_all_datasets.py --dataset_type debt --scenarios 3")
    print("  python experiments/run_all_datasets.py --dataset_type disaster --scenarios 3") 
    print("  python experiments/run_all_datasets.py --dataset_type student --scenarios 3")
    print("  python experiments/run_all_datasets.py --dataset_type medical --scenarios 3")