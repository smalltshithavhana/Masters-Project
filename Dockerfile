\
# GPU-ready TensorFlow base image (includes CUDA + cuDNN)
FROM tensorflow/tensorflow:latest-gpu

WORKDIR /workspace
COPY . /workspace

RUN python -m pip install --upgrade pip && \
    pip install -r requirements.txt

CMD ["python", "run_experiment1.py", "--outputs", "Outputs"]
