# HCSS Cost Code Conversational Agent

This project implements a Microsoft Teams chatbot for construction cost analytics.
Users can ask natural-language questions about cost codes, production rates, and seasonal vs overall metrics.

The system consists of:
- A Teams bot (Node.js)
- A Flask backend (Python) for data access and agent logic
- Azure Blob Storage for analytics JSON files

## Architecture Overview

```text
Microsoft Teams
      ↓
Node.js Bot (bot-node/index.js)
      ↓
Flask API (main.py)
      ↓
Agent Logic (hcss_mcp_server.py)
      ↓
Azure Blob Storage (processed JSON data)

Repository Structure
HCSS-Bot/
├── bot-node/
│   ├── index.js              # Teams bot entry point
│   ├── package.json          # Node dependencies
│   ├── package-lock.json
│   └── .env                  # Bot environment variables
│
├── main.py                   # Flask API (agent + data endpoints)
├── hcss_mcp_server.py        # Conversational agent logic
├── HCSS_analysis.py          # Data processing / analytics
├── cost_code_descriptions_info.csv
├── .env                      # Python environment variables
└── README.md

Prerequisites
System
Python 3.9+
Node.js 18+
npm
Azure Blob Storage account
Microsoft Bot Framework registration (for Teams)
Bot Framework Emulator (for local testing)

Environment Variables
Create a .env file in the project root (Flask/Python):
AZURE_STORAGE_CONNECTION_STRING=your_azure_connection_string
OPENAI_API_KEY=your_openai_key

Create a separate .env inside bot-node/ if your bot requires it.
Python Backend Setup
Create virtual environment:
python -m venv venv
Activate it:
Mac/Linux:
source venv/bin/activate
Windows:
venv\Scripts\activate

Install dependencies:
pip install -r requirements.txt
Start Flask server:
python main.py
Flask will start on:
http://localhost:8000


Node.js Bot Setup
Install dependencies:
cd bot-node
npm install
Start the bot:
npm start
The bot will listen on:
http://localhost:3978/api/messages

Testing Locally (Without Teams)
Use Bot Framework Emulator:
Open Bot Framework Emulator
Connect to:
http://localhost:3978/api/messages
