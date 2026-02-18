#!/bin/bash

# Nome della cartella del progetto e del virtual environment
REPO_DIR="TokenSkip"
VENV_NAME="venv"
REQ_FILE="$REPO_DIR/requirements.txt"

echo "--- Inizio configurazione esperimento ---"

# 1. Verifica se la cartella TokenSkip esiste
if [ ! -d "$REPO_DIR" ]; then
    echo "ERRORE: La cartella '$REPO_DIR' non è stata trovata."
    echo "Assicurati che lo script sia posizionato accanto alla cartella del repository."
    exit 1
fi

# 2. Verifica se il file requirements.txt esiste
if [ ! -f "$REQ_FILE" ]; then
    echo "ERRORE: Il file $REQ_FILE non esiste."
    exit 1
fi

# 3. Creazione del Virtual Environment (venv)
echo "Creazione dell'ambiente virtuale ($VENV_NAME)..."
python3 -m venv $VENV_NAME

# 4. Attivazione dell'ambiente
echo "Attivazione dell'ambiente..."
source $VENV_NAME/bin/activate

# 5. Aggiornamento pip
echo "Aggiornamento di pip..."
pip install --upgrade pip

# 6. Installazione dei requisiti
echo "Installazione delle dipendenze da $REQ_FILE..."
# I requisiti includono pacchetti come torch, vllm e transformers
pip install -r $REQ_FILE

echo "--- Installazione completata con successo! ---"
echo "Per iniziare a lavorare, attiva l'ambiente con il comando:"
echo "source $VENV_NAME/bin/activate"
