#!/bin/bash

# Attiva venv se necessario
source venv/bin/activate

echo "========================================================"
echo " STARTING MODEL DOWNLOAD "
echo "========================================================"

# Verifica dipendenze
if ! command -v huggingface-cli &> /dev/null; then
    echo "Installo huggingface_hub..."
    pip install huggingface_hub
fi

# Configurazioni
MODELS=("3B" "7B" "14B")
BASE_DIR="models"

mkdir -p $BASE_DIR

for SIZE in "${MODELS[@]}"; do
    echo "--------------------------------------------------------"
    echo " Processing size: ${SIZE}"
    echo "--------------------------------------------------------"

    # 1. Download Modello BASE (Qwen2.5-Instruct)
    BASE_MODEL_NAME="Qwen/Qwen2.5-${SIZE}-Instruct"
    BASE_LOCAL_DIR="${BASE_DIR}/Qwen2.5-${SIZE}-Instruct"
    
    echo " >> Downloading Base Model: ${BASE_MODEL_NAME}..."
    huggingface-cli download ${BASE_MODEL_NAME} --local-dir ${BASE_LOCAL_DIR} --exclude "*.pth" "*.bin"

    # 2. Download Modello ADAPTER (TokenSkip-GSM8K)
    ADAPTER_MODEL_NAME="hemingkx/TokenSkip-Qwen2.5-${SIZE}-Instruct-GSM8K"
    ADAPTER_LOCAL_DIR="${BASE_DIR}/TokenSkip-Qwen2.5-${SIZE}-Instruct-GSM8K"
    
    echo " >> Downloading Adapter: ${ADAPTER_MODEL_NAME}..."
    huggingface-cli download ${ADAPTER_MODEL_NAME} --local-dir ${ADAPTER_LOCAL_DIR}

    echo " >> Done for ${SIZE}."
done

echo "========================================================"
echo " ALL DOWNLOADS COMPLETED."
echo "========================================================"





















































































































 






























