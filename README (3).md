# Integrated AI System: ML + DL + LLM

## Overview
This project combines **Machine Learning (ML)**, **Deep Learning (DL)**, and **Large Language Models (LLM)** to create an end-to-end AI system that makes predictions and explains them in natural language.

## Models Used

### 1. Machine Learning Model: Random Forest Classifier
- **Algorithm**: Random Forest (Ensemble Learning)
- **Configuration**: 100 trees, max depth of 5
- **Strengths**: 
  - Robust to overfitting
  - Handles non-linear relationships well
  - Provides feature importance
  - Fast inference time

### 2. Deep Learning Model: Neural Network
- **Architecture**: 
  - Input Layer
  - Hidden Layer 1: 64 neurons + ReLU + Dropout (30%)
  - Hidden Layer 2: 32 neurons + ReLU + Dropout (30%)
  - Hidden Layer 3: 16 neurons + ReLU
  - Output Layer: Softmax activation
- **Strengths**:
  - Can learn complex patterns
  - Better for high-dimensional data
  - Adaptive feature learning

### 3. Large Language Model: hugging face
- **Purpose**: Generate human-readable explanations
- **API**: hugging face API token(Free tier)
- **Function**: Translates technical predictions into understandable insights

## Why ML vs DL?

### Random Forest (ML) chosen for:
- **Interpretability**: Easy to understand feature importance
- **Efficiency**: Fast training and prediction
- **Robustness**: Works well with small to medium datasets
- **No feature scaling required**: Handles raw data well

### Neural Network (DL) chosen for:
- **Complexity handling**: Can model intricate patterns
- **Scalability**: Performs better as data grows
- **End-to-end learning**: Automatically learns features
- **Flexibility**: Easy to extend to more complex architectures

### Hybrid Approach Benefits:
- Use the model with **higher confidence** for final prediction
- Compare performance metrics side-by-side
- Get best of both worlds: interpretability + complexity

## LLM Prompt Design

### Prompt Structure:
```
1. Input Features: Describe the data point
2. Model Comparison: Show both model predictions and confidence
3. Performance Metrics: Include accuracy and F1 scores
4. Task: Request clear explanation focusing on:
   - What the prediction means
   - Why one model was chosen over the other
   - Confidence level interpretation
```

### Design Principles:
- **Context-rich**: Provide all relevant information
- **Structured**: Organized sections for clarity
- **Action-oriented**: Clear task definition
- **Constrained**: Request 2-3 sentences to avoid verbosity


You can replace with your own CSV dataset.

## Expected Output Format
```json
{
  "prediction": "Class name or label",
  "confidence": "XX.XX%",
  "ml_vs_dl_comparison": "ML: XX% | DL: XX% | Selected: Model name",
  "llm_explanation": "Natural language explanation of the prediction"
}
```

## Installation & Setup

### Prerequisites
- Python 3.8 or higher
- Google Gemini API key (Free)

### Steps:
1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd integrated-ai-system
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt (install requirements seperately)
   ```

3. **Get hugging face API Key** (FREE)
   - paste it in .env file

4. **Run the system**
   ```bash
   python ai_integrated_system.py
   ```

5. **Enter API key when prompted**


## Performance Metrics

Both models are evaluated on:
- **Accuracy**: Overall correctness
- **Precision**: Accuracy of positive predictions
- **Recall**: Coverage of actual positives
- **F1 Score**: Harmonic mean of precision and recall

## Architecture

```
┌─────────────┐
│   CSV Data  │
└──────┬──────┘
       │
       ├─────────────────────────────┐
       │                             │
       ▼                             ▼
┌──────────────┐              ┌──────────────┐
│  ML Model    │              │  DL Model    │
│  (Random     │              │  (Neural     │
│   Forest)    │              │   Network)   │
└──────┬───────┘              └──────┬───────┘
       │                             │
       └─────────────┬───────────────┘
                     │
                     ▼
              ┌──────────────┐
              │  Confidence  │
              │  Comparison  │
              └──────┬───────┘
                     │
                     ▼
              ┌──────────────┐
              │     LLM      │
              │   (Gemini)   │
              └──────┬───────┘
                     │
                     ▼
              ┌──────────────┐
              │ JSON Output  │
              │ + Explanation│
              └──────────────┘
```

## Project Structure
```
integrated-ai-system/
│
├── ai_integrated_system.py    # Main application
├── requirements.txt           # Python dependencies
├── README.md                  # This file
├── demo_video.mp4            # Demo recording
└── diagrams/
    ├── architecture.png      # System architecture
    └── workflow.png          # Process workflow
```

## Features
✅ ML Model (Random Forest)
✅ DL Model (Neural Network)
✅ Performance comparison
✅ Hybrid prediction system
✅ LLM-powered explanations
✅ JSON formatted output
✅ Interactive prediction mode
✅ Free API usage


