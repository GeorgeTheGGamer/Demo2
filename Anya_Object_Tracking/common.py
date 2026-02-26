"""TFLite runtime import shim with fallback support."""

import sys

try:
    # from tflite_runtime.interpreter import Interpreter
    from ai_edge_litert.interpreter import Interpreter
    USING_TFLITE_RUNTIME = True
except ImportError:
    try:
        from tensorflow.lite.python.interpreter import Interpreter
        USING_TFLITE_RUNTIME = False
    except ImportError:
        raise ImportError(
            "Neither tflite_runtime nor tensorflow.lite could be imported. "
            "Please install one of them:\n"
            "  - For Raspberry Pi: pip install tflite-runtime\n"
            "  - For development: pip install tensorflow"
        )


def get_interpreter(model_path, num_threads=None):
    """Create and configure a TFLite Interpreter instance.
    
    Args:
        model_path (str): Path to the .tflite model file.
        num_threads (int, optional): Number of threads for inference.
    
    Returns:
        Interpreter: Configured TFLite interpreter instance.
    
    Raises:
        FileNotFoundError: If model_path does not exist.
        ValueError: If model is invalid.
    """
    import os
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    if num_threads:
        interpreter = Interpreter(
            model_path=model_path,
            num_threads=num_threads
        )
    else:
        interpreter = Interpreter(model_path=model_path)
    
    interpreter.allocate_tensors()
    
    return interpreter


def get_runtime_info():
    """Get information about which TFLite runtime is being used.
    
    Returns:
        dict: Runtime information with 'using_tflite_runtime' and 'backend' keys.
    """
    return {
        'using_tflite_runtime': USING_TFLITE_RUNTIME,
        'backend': 'tflite_runtime' if USING_TFLITE_RUNTIME else 'tensorflow.lite'
    }
