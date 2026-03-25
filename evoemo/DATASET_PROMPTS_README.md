# Dataset-Specific Prompts for Multi-Domain Negotiation

This system now supports **dataset-specific prompts** that automatically adapt to different negotiation contexts. The prompts change based on the type of data you're working with, making conversations more realistic and domain-appropriate.

## 🎭 Supported Dataset Types

### 1. **Debt Collection** (`debt`)
- **Context**: Business debt recovery negotiations
- **Roles**: Creditor (collection agent) ↔ Debtor (business owner)
- **Goal**: Negotiate payment timeline for outstanding debt
- **Key Vocabulary**: Outstanding balance, recovery stage, cash flow, business impact
- **Sample**: "I understand your situation, but 60 days is too long. How about 35 days?"

### 2. **Disaster Rescue** (`disaster`) 
- **Context**: Emergency rescue coordination
- **Roles**: RoboDog Coordinator ↔ Disaster Survivor  
- **Goal**: Coordinate realistic rescue timeline vs. survivor endurance
- **Key Vocabulary**: ETA, endurance, critical needs, survivor condition
- **Sample**: "Our team is 20 minutes out. Can you hold on with your current supplies?"

### 3. **Student Sleep Health** (`student`)
- **Context**: Educational sleep schedule negotiation
- **Roles**: Sleep Health AI ↔ Student
- **Goal**: Balance academic needs with healthy sleep schedules
- **Key Vocabulary**: Bedtime, academic pressure, sleep quality, study schedule
- **Sample**: "I understand the project stress, but 11 PM gives you 8 hours sleep for tomorrow's test."

### 4. **Medical Surgery Scheduling** (`medical`)
- **Context**: Hospital surgery scheduling
- **Roles**: Hospital Scheduler ↔ Patient
- **Goal**: Balance medical urgency with surgeon/resource availability
- **Key Vocabulary**: Surgery wait time, urgency level, surgeon availability, medical condition
- **Sample**: "I can offer you a slot in 45 days with Dr. Smith, or 25 days with our associate surgeon."

## 🔄 How It Works

### Automatic Dataset Detection
The system automatically detects dataset type based on CSV column names and metadata:

```python
# Debt detection
if 'creditor', 'debtor', 'overdue', 'balance' in columns:
    scenario_type = "debt"

# Disaster detection  
if 'disaster', 'survivor', 'rescue', 'eta' in columns:
    scenario_type = "disaster"

# Student detection
if 'student', 'bedtime', 'sleep' in columns:
    scenario_type = "student"
    
# Medical detection
if 'patient', 'surgery', 'medical', 'surgeon' in columns:
    scenario_type = "medical"
```

### Dynamic Prompt Generation
Each dataset type gets specialized prompts:

```python
from llm.prompt_templates import PromptTemplates

# Automatically uses appropriate prompts based on scenario type
creditor_prompt = PromptTemplates.get_creditor_prompt(
    scenario_type="disaster",  # or "debt", "student", "medical"
    config=negotiation_config,
    emotion_config=emotion_settings,
    timeline_text=current_situation,
    debt_info=scenario_metadata
)
```

## 🚀 Usage Examples

### Running Different Dataset Types
```bash
# Debt collection scenarios
python experiments/run_all_datasets.py --dataset_type debt --scenarios 5 --model_creditor "deepseek-7b"

# Disaster rescue scenarios  
python experiments/run_all_datasets.py --dataset_type disaster --scenarios 3 --model_creditor "gpt-4o-mini"

# Student sleep negotiations
python experiments/run_all_datasets.py --dataset_type student --scenarios 4 --model_debtor "qwen-8b"

# Medical scheduling
python experiments/run_all_datasets.py --dataset_type medical --scenarios 2 --iterations 3
```

### Manual Dataset Type Override
```bash
# Force specific dataset type (overrides auto-detection)
python experiments/run_all_datasets.py --dataset_type student data/hospital_surgery_scenarios.csv --scenarios 3
```

### Testing All Model Types with Different Datasets
```bash
# Vanilla model with disaster scenarios
python experiments/run_all_datasets.py --model_type vanilla --dataset_type disaster --scenarios 3

# Prompt model with student scenarios  
python experiments/run_all_datasets.py --model_type prompt --dataset_type student --scenarios 3

# Bayesian model with medical scenarios
python experiments/run_all_datasets.py --model_type bayesian --dataset_type medical --scenarios 3
```

## 📊 Dataset Files

The system works with these CSV files in the `/data` folder:

- `credit_recovery_scenarios.csv` → **debt** type
- `disaster_survivor_scenarios.csv` → **disaster** type  
- `education_sleep_scenarios.csv` → **student** type
- `hospital_surgery_scenarios.csv` → **medical** type

## 🎯 Model Integration

All model types now support dataset-specific prompts:

### ✅ **Vanilla Model** (`models/vanilla_model.py`)
- Uses `DebtNegotiator` with dataset-specific prompts
- Automatically adapts to scenario type

### ✅ **Prompt Model** (`models/prompt_model.py`) 
- Uses `DebtNegotiator` with dataset-specific prompts
- Random emotion selection + appropriate context

### ✅ **Bayesian Model** (`models/bayesian_multiagent.py`)
- Uses `DebtNegotiatorMultiagent` with dataset-specific prompts
- Emotion optimization + context adaptation

### ✅ **All LLM Models**
- DeepSeek-7B, GPT-4o-mini, Qwen-8B all work with dataset-specific prompts
- Same model can negotiate across domains with appropriate context

## 🧪 Testing Dataset Prompts

```bash
# Test all prompt templates
python test_dataset_prompts.py

# This will show you:
# - How prompts differ between dataset types
# - Scenario detection accuracy  
# - Context-specific vocabulary usage
# - Role adaptation examples
```

## 💡 Advanced Features

### Custom Scenario Types
You can extend the system by adding new scenario types:

```python
# In prompt_templates.py
@staticmethod
def _get_custom_creditor_prompt(config, emotion_config, timeline_text, debt_info):
    return f"""You are a [CUSTOM_ROLE] negotiating [CUSTOM_CONTEXT]..."""

# Register in get_creditor_prompt()
if scenario_type == "custom":
    return PromptTemplates._get_custom_creditor_prompt(...)
```

### Mixed Dataset Experiments  
```bash
# Compare same model across different domains
python experiments/run_all_datasets.py --dataset_type debt --scenarios 5 --model_creditor deepseek-7b
python experiments/run_all_datasets.py --dataset_type disaster --scenarios 5 --model_creditor deepseek-7b
python experiments/run_all_datasets.py --dataset_type student --scenarios 5 --model_creditor deepseek-7b
```

## 🔍 Debugging Dataset Detection

If your dataset isn't detected correctly:

```python
from llm.prompt_templates import PromptTemplates

# Check what the system detects
metadata = {"your": "scenario", "metadata": "here"}
detected_type = PromptTemplates.detect_scenario_type(metadata)
print(f"Detected: {detected_type}")

# Force a specific type in your experiment
python experiments/run_all_datasets.py --dataset_type disaster your_file.csv
```

## 📈 Benefits

1. **Domain Expertise**: Each domain gets appropriate professional vocabulary
2. **Realistic Conversations**: Prompts match real-world contexts
3. **Cross-Domain Testing**: Same models work across multiple domains  
4. **Automatic Adaptation**: No manual prompt engineering needed
5. **Extensible**: Easy to add new domains and contexts

Now your negotiation system can realistically simulate:
- 💰 **Business debt recovery**
- 🆘 **Emergency rescue coordination** 
- 😴 **Educational sleep health**
- 🏥 **Medical appointment scheduling**

All with the same underlying models but context-appropriate conversations!