#!/bin/bash

echo "Starting Kairos..."

# Check Ollama is installed
if ! command -v ollama &> /dev/null; then
    echo "Ollama not found. Please install from https://ollama.com"
    exit 1
fi

# Check mistral is pulled
if ! ollama list | grep -q "mistral"; then
    echo "Pulling Mistral model..."
    ollama pull mistral
fi

# Start Ollama in background
echo "Starting Ollama..."
ollama serve &> /dev/null &
sleep 2

# Start Streamlit
echo "Starting Kairos at http://localhost:8501"
python3 -m streamlit run app/streamlit_app.py
