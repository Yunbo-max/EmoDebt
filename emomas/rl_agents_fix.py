# Fixed RL agents - key changes needed in rl_agents.py

# 1. Import fixes (already done)
from llm.negotiator import DebtNegotiator  # Move to top imports

# 2. Variable consistency fixes needed in both DQN and Q-learning:

# Line ~652 and ~865: Change failed_negotiations to failed
# FROM: failed_negotiations = [n for n in all_negotiations if n.get('final_state') != 'accept']
# TO: failed = [n for n in all_negotiations if n.get('final_state') != 'accept']

# 3. Fix references to failed_negotiations throughout:
# - In performance dict: 'failed_negotiations': len(failed)
# - In analysis dict avg_rounds_failed: use failed instead of failed_negotiations  
# - In failure_reasons loop: for result in failed:
# - In summary writing: if failed: ... count/len(failed)*100

# 4. Make sure both functions have this missing code block (after model.update_model):

"""
            iteration_results['scenario_results'].append(result)
            all_negotiations.append(result)
            
            # Update model with result
            model.update_model(result)
        
        # Store iteration results  
        results['iteration_results'][f'iteration_{iteration+1}'] = iteration_results
"""

# 5. Ensure success_patterns uses successful and failed, not successful_negotiations/failed_negotiations:

"""
    results['analysis'] = {
        'failure_breakdown': failure_reasons,
        'success_patterns': {
            'avg_rounds_successful': float(np.mean([len(r.get('dialog', [])) for r in successful])) if successful else 0,
            'avg_rounds_failed': float(np.mean([len(r.get('dialog', [])) for r in failed])) if failed else 0
        }
    }
"""

print("RL Agents fixes needed:\n1. failed_negotiations -> failed\n2. Add missing iteration storage\n3. Fix variable references in summary writing")