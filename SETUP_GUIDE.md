# Step-by-Step Guide: Running the Project in VS Code

## Prerequisites Setup

### Step 1: Install Python
1. Download Python 3.8 or higher from https://www.python.org/downloads/
2. During installation, **CHECK** "Add Python to PATH"
3. Verify installation:
   ```bash
   python --version
   ```

### Step 2: Install VS Code
1. Download from https://code.visualstudio.com/
2. Install the **Python extension** by Microsoft:
   - Open VS Code
   - Click Extensions icon (or Ctrl+Shift+X)
   - Search "Python"
   - Install the one by Microsoft

---

## Project Setup in VS Code

### Step 3: Create Project Folder
1. Open VS Code
2. Click: File → Open Folder
3. Create a new folder: `integrated-ai-system`
4. Select it

### Step 4: Copy All Project Files
Copy these files into your project folder:
- `ai_integrated_system.py` (main code)
- `requirements.txt` (dependencies)
- `README.md` (documentation)
- `create_diagrams.py` (diagram generator)
- `SETUP_GUIDE.md` (this file)

### Step 5: Open Integrated Terminal
1. In VS Code, press: **Ctrl + `** (backtick)
2. Or: View → Terminal

---

## Get FREE Gemini API Key

### Step 6: Get Google Gemini API Key
1. Go to: https://makersuite.google.com/app/apikey
   OR: https://aistudio.google.com/app/apikey

2. Sign in with Google account

3. Click "Create API Key"

4. Click "Create API key in new project"

5. **COPY** the API key (starts with "AIza...")

6. **SAVE** it somewhere safe - you'll need it!

---

## Install Dependencies

### Step 7: Create Virtual Environment (Recommended)
In the VS Code terminal:

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac/Linux
python3 -m venv venv
source venv/bin/activate
```

You should see `(venv)` in your terminal prompt.

### Step 8: Install Required Packages
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This will install:
- pandas (data handling)
- numpy (numerical operations)
- scikit-learn (ML models)
- tensorflow (DL models)
- google-generativeai (LLM)
- matplotlib (visualizations)
- seaborn (charts)

**Wait** for installation to complete (2-5 minutes).

---

## Generate Diagrams

### Step 9: Create Diagrams
```bash
python create_diagrams.py
```

This creates:
- `architecture_diagram.png`
- `workflow_diagram.png`
- `model_comparison.png`

---

## Run the Main Application

### Step 10: Run the System
```bash
python ai_integrated_system.py
```

### Step 11: Enter API Key
When prompted:
```
Enter your Google Gemini API key: 
```
Paste your API key (the one you saved earlier) and press Enter.

### Step 12: Watch the Training
The system will:
1. ✓ Load Iris dataset
2. ✓ Split data (train/test)
3. ✓ Train Random Forest model
4. ✓ Train Neural Network
5. ✓ Evaluate both models
6. ✓ Show performance metrics

### Step 13: See Demo Predictions
The system automatically shows 3 sample predictions with:
- Prediction result
- Confidence score
- ML vs DL comparison
- LLM explanation

### Step 14: Try Interactive Mode
Enter your own values:
```
Enter features: 5.1, 3.5, 1.4, 0.2
```

The system will:
1. Process your input
2. Run both ML and DL models
3. Compare predictions
4. Generate LLM explanation
5. Output JSON result

---

## Understanding the Output

### Example Output:
```json
{
  "prediction": "Iris-setosa",
  "confidence": "98.50%",
  "ml_vs_dl_comparison": "ML Model: 98.50% confidence | DL Model: 95.20% confidence | Selected: ML (Random Forest)",
  "llm_explanation": "The system predicts this flower is an Iris-setosa with high confidence. The Random Forest model was chosen over the Neural Network because it showed 98.5% confidence versus 95.2%. Given these strong metrics and the models' overall accuracy of around 95%, we can be very confident in this classification."
}
```

### What each field means:
- **prediction**: The predicted class name
- **confidence**: How sure the selected model is
- **ml_vs_dl_comparison**: Comparison of both models
- **llm_explanation**: Human-readable explanation from Gemini

---

## Common Issues & Solutions

### Issue 1: "pip not found"
**Solution:**
```bash
python -m pip install -r requirements.txt
```

### Issue 2: "Module not found"
**Solution:**
```bash
pip install --upgrade pip
pip install -r requirements.txt --force-reinstall
```

### Issue 3: TensorFlow Installation Error (Windows)
**Solution:**
```bash
pip install tensorflow-cpu
```

### Issue 4: "Invalid API key"
**Solution:**
- Get a new API key from https://makersuite.google.com/app/apikey
- Make sure you copied the entire key
- Try pasting in a text file first to check

### Issue 5: Virtual Environment Not Activating
**Windows PowerShell:**
```bash
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
venv\Scripts\activate
```

---

## Testing Your Setup

### Quick Test Checklist:
- [ ] Python installed and in PATH
- [ ] VS Code with Python extension
- [ ] Virtual environment created
- [ ] Dependencies installed successfully
- [ ] Gemini API key obtained
- [ ] Diagrams generated
- [ ] Main script runs without errors
- [ ] Demo predictions work
- [ ] Interactive mode accepts input

---

## Recording Demo Video

### What to Record:
1. **Show the code** in VS Code
2. **Run the script**: `python ai_integrated_system.py`
3. **Enter API key**
4. **Show model training** output
5. **Show evaluation metrics**
6. **Show demo predictions** (all 3 samples)
7. **Try interactive input** (2-3 examples)
8. **Show JSON outputs**

### Recording Tools (Free):
- **Windows**: Xbox Game Bar (Win+G)
- **Mac**: QuickTime Player (Cmd+Shift+5)
- **All Platforms**: OBS Studio (https://obsproject.com/)

### Tips:
- Keep video under 5 minutes
- Speak clearly while explaining
- Show the terminal output clearly
- Highlight the JSON output
- Explain what each model does

---

## Submission Checklist

### Files to Upload to GitHub:
- [ ] `ai_integrated_system.py`
- [ ] `requirements.txt`
- [ ] `README.md`
- [ ] `create_diagrams.py`
- [ ] `SETUP_GUIDE.md`

### Files to Upload to Drive:
- [ ] `architecture_diagram.png`
- [ ] `workflow_diagram.png`
- [ ] `model_comparison.png`
- [ ] Demo video (MP4)
- [ ] README.md

### GitHub Upload Steps:
1. Create repository on GitHub
2. Initialize git in VS Code terminal:
   ```bash
   git init
   git add .
   git commit -m "Initial commit: Integrated AI System"
   git branch -M main
   git remote add origin <your-repo-url>
   git push -u origin main
   ```

---

## Need Help?

### Resources:
- **Python**: https://docs.python.org/3/
- **Scikit-learn**: https://scikit-learn.org/
- **TensorFlow**: https://www.tensorflow.org/
- **Gemini API**: https://ai.google.dev/

### Common Commands Reference:
```bash
# Activate environment
venv\Scripts\activate          # Windows
source venv/bin/activate       # Mac/Linux

# Run main script
python ai_integrated_system.py

# Generate diagrams
python create_diagrams.py

# Install package
pip install package-name

# Check installed packages
pip list

# Deactivate environment
deactivate
```

---

## Success! 🎉

You've successfully set up and run the Integrated AI System!

Next steps:
1. Experiment with different inputs
2. Record your demo video
3. Upload to GitHub
4. Submit to Drive

Good luck with your assignment! 🚀
