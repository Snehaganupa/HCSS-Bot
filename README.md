**HCSS Cost Code Conversational Agent**

This project implements a Teams chatbot for construction cost analytics.
Users can ask natural-language questions about cost codes, production rates, and seasonal vs overall metrics.
The system consists of:
A Teams bot (Node.js)
A Flask backend (Python) for data access and agent logic
Azure Blob Storage for analytics JSON files

**Architecture Overview**

Microsoft Teams
     ↓
Node.js Bot (bot-node/index.js)
     ↓
Flask API (main.py)
     ↓
Agent Logic (hcss_mcp_server.py)
     ↓
Azure Blob Storage (processed JSON data)


**Repository Structure**

├── bot-node/
│   ├── index.js              # Teams bot entry point
│   ├── package.json          # Node dependencies
│   ├── package-lock.json
│   ├── node_modules/         # Installed via npm 
│   └── .env                  # Bot environment variables
│
├── main.py                   # Flask API (agent + data endpoints)
├── hcss_mcp_server.py        # Conversational agent logic
├── HCSS_analysis.py          # Data processing / analytics
├── cost_code_descriptions_info.csv
├── requirements.txt
├── .env                      # Python environment variables
└── README.md


**Prerequisites**
**System**
Python 3.9+
Node.js 18+
npm
Azure Blob Storage account
Microsoft Bot Framework registration (for Teams)
Bot Framework Emulator (for local testing)

**Environment Variables**
Create a .env file in the project root.
**Flask / Python**
AZURE_STORAGE_CONNECTION_STRING=your_azure_connection_string
OPENAI_API_KEY=your_openai_key

**Python Backend Setup**
1. Create virtual environment
python -m venv venv
Mac/Linux: source venv/bin/activate   # Windows: venv\Scripts\activate
2. Install dependencies
pip install -r requirements.txt
3. Start Flask server
python main.py
Flask will start on:
http://localhost:8000

**Node.js Bot Setup**
1. Install dependencies
npm install
2. Start the bot
node index.js
The bot will listen on:
http://localhost:3978/api/messages

**Testing Locally (Without Teams)**
Use Bot Framework Emulator:
Open Bot Emulator
Connect to:
http://localhost:3978/api/messages

