import sys
import traceback

print(f"Python: {sys.version}")
print(f"Executable: {sys.executable}")

try:
    print("Importing torch...")
    import torch
    print(f"Torch version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
except ImportError as e:
    print(f"Failed to import torch: {e}")

try:
    print("Importing TTS...")
    import TTS
    print(f"TTS package: {TTS.__file__}")
    
    print("Importing TTS.api...")
    from TTS.api import TTS as TTS_API
    print("TTS.api imported successfully")
except Exception as e:
    print("FAILED to import TTS:")
    traceback.print_exc()
