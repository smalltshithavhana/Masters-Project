\
import subprocess
import tensorflow as tf

print("TensorFlow version:", tf.__version__)
print("GPUs:", tf.config.list_physical_devices("GPU"))
try:
    out = subprocess.check_output(["nvidia-smi"], text=True)
    print("nvidia-smi:\n", out)
except Exception as e:
    print("nvidia-smi not available:", e)
