import logging
import torch
import sys

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

print(f"Python: {sys.version}")
print(f"Torch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")

try:
    import transformers
    print(f"Transformers: {transformers.__version__}")
    from transformers import VitsModel, AutoTokenizer
    print("VitsModel imported successfully.")
except ImportError as e:
    print(f"Import Error: {e}")
    sys.exit(1)
except Exception as e:
    print(f"General Error: {e}")
    sys.exit(1)

try:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model on {device}...")
    
    model_name = "facebook/mms-tts-tha" # Try 3-letter code
    print(f"Loading {model_name}...")
    
    # model = VitsModel.from_pretrained("facebook/mms-tts-th").to(device)
    # tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-th")
    
    from transformers import VitsModel, AutoTokenizer
    # model = AutoModel.from_pretrained(model_name).to(device)
    model = VitsModel.from_pretrained(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    print(f"{model_name} loaded successfully!")
    
    # Test generation
    text = "สวัสดีครับ ทดสอบเสียง"
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model(**inputs).waveform
    print(f"Generated waveform shape: {output.shape}")
    
except Exception as e:
    print(f"Failed to load/run model: {e}")
    import traceback
    traceback.print_exc()
