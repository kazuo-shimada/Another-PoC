# 📦 SIVA: Smart Inventory & Vision Agent (PoC)

An edge-AI Proof of Concept that bridges computer vision and local language models to automate enterprise inventory management and procurement. 

Built entirely for offline inference, SIVA demonstrates how visual data can be seamlessly orchestrated into actionable business workflows without relying on external cloud APIs.

## 🚀 The Business Value
In traditional warehouse or retail environments, cycle counting and procurement are manual, time-intensive processes. SIVA proposes an automated pipeline:
1. **Detect:** A computer vision node (YOLOv8) analyzes a physical space (e.g., a stockroom shelf) and identifies current inventory.
2. **Analyze:** The raw visual data is aggregated and mapped against a target quota.
3. **Act:** A local Large Language Model (Llama 3) orchestrates the data to automatically draft a professional purchase order for the deficient items.

*Note: This repository serves as a functional Proof of Concept. In a production enterprise environment, the hardcoded quotas would be replaced with real-time SQL database queries.*

## 🏗️ Architecture & Tech Stack
* **Frontend UI:** `Gradio` (for rapid, interactive web deployment)
* **Vision Node:** `YOLOv8` via `ultralytics` (Object detection)
* **AI Orchestrator:** `LangChain` (Prompt engineering and pipeline routing)
* **Local Inference Engine:** `Llama.cpp` running Meta's Llama-3-8B-Instruct (GGUF format for optimized Apple Silicon/Mac performance)

## 🛠️ Quick Start Guide

### Prerequisites
* Mac OS environment recommended.
* Download the [Meta-Llama-3-8B-Instruct-Q4_K_M.gguf](https://huggingface.co/bartowski/Meta-Llama-3-8B-Instruct-GGUF/resolve/main/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf?download=true) model file.
* Place the downloaded `.gguf` file directly in the root directory of this project.

### Installation
Clone this repository and set up your virtual environment:

# Clone the repo
git clone (https://github.com/YOUR_USERNAME/SIVA-inventory-agent.git)](https://github.com/kazuo-shimada/Another-PoC)

    cd SIVA-inventory-agent

# Create and activate the virtual environment
    python3 -m venv siva_env
    source siva_env/bin/activate

# Install the required dependencies
    pip3 install gradio ultralytics opencv-python langchain langchain-community langchain-core llama-cpp-python pandas pydantic

Launching the Application
With your environment active and the .gguf model in the project folder, start the Gradio server:

    python3 app.py

Open the provided local URL (typically http://127.0.0.1:7860) in your browser. Upload an image of a desk, pantry, or shelf to see the automated procurement pipeline in action.

💡 Future Enterprise Scaling
To take this PoC to production, the following architecture upgrades would be implemented:

Swap Gradio for a React frontend and FastAPI backend.

Connect the LangChain orchestrator to a PostgreSQL database for live quota tracking.

Deploy the YOLO vision node to edge cameras (e.g., AWS Panorama) for continuous monitoring.


---
