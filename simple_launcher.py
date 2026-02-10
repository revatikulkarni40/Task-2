"""
Simple Launcher - Just drag CSV and run!
"""
from auto_flexible_ai_system import AutoFlexibleAISystem
import json

def simple_run(csv_file, hf_api_key):
    """
    Simple function to run with any CSV file
    """
    try:
        # Initialize
        ai_system = AutoFlexibleAISystem(hf_api_key=hf_api_key)
        
        # Auto-load data
        data = ai_system.load_and_prepare_data(csv_file)
        
        # Auto-train
        X_train, X_test, y_train, y_test = ai_system.prepare_for_training(data)
        ai_system.train_ml_model(X_train, y_train)
        ai_system.train_dl_model(X_train, y_train, len(ai_system.target_names))
        ml_metrics, dl_metrics = ai_system.evaluate_models(X_test, y_test)
        
        print(f"\n✅ System ready! Features: {', '.join(ai_system.feature_names)}")
        
        return ai_system, ml_metrics, dl_metrics
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return None, None, None

def predict_json(ai_system, input_values, ml_metrics, dl_metrics):
    """
    Get prediction in JSON format
    """
    output = ai_system.get_final_output(input_values, ml_metrics, dl_metrics)
    return json.dumps(output, indent=2)

def interactive_mode(ai_system, ml_metrics, dl_metrics):
    """
    Interactive prediction mode
    """
    print(f"\n🎮 INTERACTIVE MODE - JSON OUTPUT")
    print("="*50)
    print(f"Features: {', '.join(ai_system.feature_names)}")
    print("Enter values separated by commas, or 'quit' to exit\n")
    
    while True:
        user_input = input("🔮 Enter values: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("👋 Goodbye!")
            break
        
        try:
            input_data = [float(x.strip()) for x in user_input.split(',')]
            
            if len(input_data) != len(ai_system.feature_names):
                print(f"❌ Need exactly {len(ai_system.feature_names)} values")
                continue
            
            result = predict_json(ai_system, input_data, ml_metrics, dl_metrics)
            print("\n📊 Result:")
            print(result)
            print()
            
        except ValueError:
            print("❌ Please enter valid numbers")
        except Exception as e:
            print(f"❌ Error: {str(e)}")

# Main execution
if __name__ == "__main__":
    print("🚀 SIMPLE AI LAUNCHER")
    print("="*30)
    
    # Get inputs
    csv_file = input("📁 CSV file path: ").strip()
    hf_api_key = input("🔑 HuggingFace API key: ").strip()
    
    if not csv_file or not hf_api_key:
        print("❌ Both CSV file and API key required!")
        exit()
    
    # Initialize system
    print("\n🔄 Setting up AI system...")
    ai_system, ml_metrics, dl_metrics = simple_run(csv_file, hf_api_key)
    
    if ai_system:
        # Start interactive mode
        interactive_mode(ai_system, ml_metrics, dl_metrics)
    else:
        print("❌ Failed to initialize system")