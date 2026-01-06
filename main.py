import os
import json
import csv
from io import BytesIO, StringIO
from typing import List, Dict, Any
from collections import defaultdict

from dotenv import load_dotenv

from hcss_mcp_server import get_session_state
from hcss_mcp_server import answer_cost_query

load_dotenv()  # automatically loads variables from .env or .environment

from flask import request, jsonify
from flask_openapi3 import OpenAPI, Info

from azure.storage.blob import BlobServiceClient

import HCSS_analysis

# NEW: OpenAI

from openai import OpenAI

# Temporary in-memory conversation state (you had this already, not used yet)
session_state = defaultdict(dict)

# -----------------------------
# App setup
# -----------------------------
info = Info(title="HCSS Cost Codes API", version="1.0.0")
app = OpenAPI(__name__, info=info)

# -----------------------------
# Azure Blob setup
# -----------------------------
AZURE_STORAGE_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
CONTAINER_NAME = os.getenv("CONTAINER_NAME", "hcss-data")
# Excel location in your Storage container (not used in current JSON flow,
# but we keep it if you decide to reuse Excel later)
BLOB_EXCEL_PATH = os.getenv("BLOB_EXCEL_PATH", "combined_files/HCSS_new_combined_file.xlsx")

blob_service_client = None
container_client = None

if not AZURE_STORAGE_CONNECTION_STRING:
    print("⚠️ AZURE_STORAGE_CONNECTION_STRING not set. Storage integration will be disabled.")
else:
    try:
        blob_service_client = BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)
        container_client = blob_service_client.get_container_client(CONTAINER_NAME)
        if not container_client.exists():
            container_client.create_container()
        print(f"✅ Connected to Azure Blob container: {CONTAINER_NAME}")
    except Exception as e:
        print(f"⚠️ Failed to init Azure Blob client: {e}")
        container_client = None

def _download_blob_to_file(blob_path: str, local_path: str) -> None:
    if not container_client:
        raise RuntimeError("Azure storage not configured")
    bc = container_client.get_blob_client(blob_path)
    with open(local_path, "wb") as f:
        f.write(bc.download_blob().readall())


def _upload_json_blob(blob_path: str, obj: Any) -> None:
    if not container_client:
        raise RuntimeError("Azure storage not configured")
    bc = container_client.get_blob_client(blob_path)
    payload = json.dumps(obj, indent=2).encode("utf-8")
    bc.upload_blob(BytesIO(payload), overwrite=True)


def _read_json_blob(blob_path: str) -> Any:
    if not container_client:
        raise RuntimeError("Azure storage not configured")
    bc = container_client.get_blob_client(blob_path)
    return json.loads(bc.download_blob().readall().decode("utf-8"))


# ============================================================
# 🔹 SHARED NORMALIZATION HELPERS (used by /data and MCP tools)
# ============================================================
import re


def normalize_text(text: str) -> str:
    """Lowercase, remove non-alphanumerics, collapse spaces."""
    return re.sub(r'\s+', ' ', re.sub(r'[^a-z0-9\s]', '', str(text or '').lower())).strip()


def normalize_season(val: str) -> str:
    s = (val or "").strip().lower()
    s = s.replace("–", "-").replace("—", "-")  # long dashes
    s = s.replace("_", "-").replace("  ", " ").strip()
    s_compact = s.replace("-", "").replace(" ", "")
    if ("non" in s_compact and "rain" in s_compact) or ("dry" in s_compact):
        return "nonrainy"
    if "rain" in s_compact:
        return "rainy"
    return s_compact  # fallback (could be '')


def remap_record(rec: dict) -> dict:
    """
    Normalize keys & add helper fields (same idea as your original /data logic).
    """
    r = {(k or "").lower().strip(): v for k, v in rec.items()}

    # unify expected keys
    if "jobcode" in r and "job_code" not in r:
        r["job_code"] = r["jobcode"]
    if "season type" in r and "season" not in r:
        r["season"] = r["season type"]
    if "seasontype" in r and "season" not in r:
        r["season"] = r["seasontype"]

    # normalized helper fields for robust comparisons
    r["_season_norm"] = normalize_season(r.get("season", ""))
    r["_unit_norm"] = normalize_text(r.get("unit", ""))
    r["_cc_norm"] = normalize_text(r.get("cost_code", ""))
    r["_jc_norm"] = normalize_text(r.get("job_code", ""))
    r["_desc_norm"] = normalize_text(r.get("cost_code_description", ""))

    return r
def fuzzy_match_tokens(desc: str, query: str) -> bool:
    """
    TRUE if all query tokens appear somewhere in the normalized description,
    even if they are glued together (like 'grassingseedmulch').
    Order does not matter.
    """
    desc_n = normalize_text(desc)           # e.g. "grassingseedmulch"
    query_tokens = normalize_text(query).split()  # ["grassing","seed","mulch"]
    return all(t in desc_n for t in query_tokens)



# ============================================================
# ROUTES
# ============================================================

@app.get("/")
def root():
    return {"message": "HCSS Azure API up"}, 200


@app.get("/envtest")
def envtest():
    return {
        "AZURE_STORAGE_CONNECTION_STRING": bool(AZURE_STORAGE_CONNECTION_STRING),
        "CONTAINER_NAME": CONTAINER_NAME,
        "BLOB_EXCEL_PATH": BLOB_EXCEL_PATH
    }, 200


@app.route("/build", methods=["GET", "POST"])
def build():
    import pandas as pd, os, json
    from io import BytesIO

    if not container_client:
        return {"error": "Azure storage not configured"}, 500

    # --------------------- STEP 1: Get all Excel blobs ---------------------
    print("▶ Fetching Excel files from Azure...")
    blob_list = [
        b.name for b in container_client.list_blobs()
        if b.name.startswith("CostCodeDetailReport_")
    ]
    if not blob_list:
        return {"error": "No CostCodeDetailReport_*.xlsx files found"}, 404

    dfs = []
    for blob_name in blob_list:
        blob_client = container_client.get_blob_client(blob_name)
        blob_data = blob_client.download_blob().readall()
        df = pd.read_excel(BytesIO(blob_data))
        dfs.append(df)
        print(f"✅ Loaded {blob_name} with {len(df)} rows")

    combined_df = pd.concat(dfs, ignore_index=True)
    temp_combined = "/tmp/HCSS_combined.xlsx"
    combined_df.to_excel(temp_combined, index=False)
    print(f"📊 Combined total rows: {len(combined_df)}")

    # --------------------- STEP 2: Run overall (non-seasonal) analysis ---------------------
    print("▶ Running overall (non-seasonal) analysis...")
    overall_jobs = [
        (['Job Code', 'Cost Code', 'Cost Code Description', 'Unit'], 'main_data_analyzed_jc.json'),
        (['Cost Code', 'Cost Code Description', 'Unit'], 'main_data_analyzed_cc.json')
    ]

    for groupby_cols, out_name in overall_jobs:
        df_result = HCSS_analysis.main(groupby_cols, temp_combined)
        if isinstance(df_result, pd.DataFrame) and not df_result.empty:
            records = df_result.to_dict(orient="records")
            _upload_json_blob(out_name, records)
            print(f"✅ Uploaded {out_name} ({len(records)} rows)")
        else:
            print(f"⚠️  No data produced for {out_name}")

    # --------------------- STEP 3: Run season-wise analysis ---------------------
    print("▶ Running season-wise analysis via process_all_years()...")
    HCSS_analysis.process_all_years(blob_list, "/tmp/seasonal")

    # --------------------- STEP 4: Convert seasonal Excel → JSON + Upload ---------------------
    print("▶ Checking for seasonal Excel outputs from process_all_years()...")
    seasonal_files = {
        "/tmp/seasonal_cost_code_season.xlsx": "seasonal_cost_code_season.json",
        "/tmp/seasonal_job_code_season.xlsx": "seasonal_job_code_season.json",
    }

    for local_xlsx, blob_json in seasonal_files.items():
        if os.path.exists(local_xlsx):
            df = pd.read_excel(local_xlsx)
            if not df.empty:
                records = df.to_dict(orient="records")
                _upload_json_blob(blob_json, records)
                print(f"✅ Uploaded {blob_json} ({len(records)} rows) to Azure.")
            else:
                print(f"⚠️  {local_xlsx} is empty, skipping upload.")
        else:
            print(f"⚠️  Seasonal file not found locally: {local_xlsx}")

    print("✅ All analyses complete and uploaded to Azure.")
    return {"status": "ok"}


LOCAL_CACHE_PATH = "/tmp/main_data_analyzed_cc.json"


@app.get("/data")
def data():
    """
    Your existing /data logic, unchanged in behavior – just using
    the shared normalize_* and remap_record helpers.
    """
    # ------------------------ Inputs ------------------------
    cost_code = (request.args.get("cost_code") or "").strip()
    unit = (request.args.get("unit") or "").strip()
    job_code = (request.args.get("job_code") or "").strip()
    season_param = (request.args.get("season") or "").strip()
    desc_param = (request.args.get("cost_code_description") or "").strip()
    match_type = (request.args.get("match") or "exact").strip().lower()
    force_type = (request.args.get("type") or "").strip().lower()
    debug = request.args.get("debug") == "1"

    if not container_client:
        return {"error": "Azure storage not configured"}, 500

    # Decide dataset: if season is provided → use seasonal
    data_type = "season" if season_param or force_type == "season" else "overall"

    # ------------------------ Choose blob ------------------------
    if data_type == "season":
        blob_name = "seasonal_job_code_season.json" if job_code else "seasonal_cost_code_season.json"
    else:
        blob_name = "main_data_analyzed_jc.json" if job_code else "main_data_analyzed_cc.json"

    # ------------------------ Load ------------------------
    blob_client = container_client.get_blob_client(blob_name)
    records = json.loads(blob_client.download_blob().readall().decode("utf-8"))
    if not isinstance(records, list) or not records:
        return {"message": f"No records found in {blob_name}"}, 200

    # ------------------------ Normalize & helpers ------------------------
    normalized = [remap_record(r) for r in records]

    # ------------------------ Begin filtering (narrowing only) ------------------------
    filtered = normalized

    # cost_code (exact string compare but normalized)
    if cost_code:
        filtered = [r for r in filtered if r["_cc_norm"] == normalize_text(cost_code)]

    # unit
    if unit:
        filtered = [r for r in filtered if r["_unit_norm"] == normalize_text(unit)]

    # job_code
    if job_code:
        filtered = [r for r in filtered if r["_jc_norm"] == normalize_text(job_code)]

    # season
    if season_param:
        target_season = normalize_season(season_param)
        filtered = [r for r in filtered if r["_season_norm"] == target_season]

    # description
    if desc_param:
        q = normalize_text(desc_param)
        if match_type == "fuzzy":
            filtered = [
                r for r in filtered
                if fuzzy_match_tokens(r["_desc_norm"], desc_param)
            ]

        else:  # exact
            filtered = [r for r in filtered if r["_desc_norm"] == q]

    if not filtered:
        return {"message": "No data found for the given filters"}, 200

    # ------------------------ Debug snapshot ------------------------
    if debug:
        season_set = sorted({r.get("season", "") for r in filtered})
        return jsonify({
            "blob": blob_name,
            "count": len(filtered),
            "seasons_seen": season_set,
            "sample": filtered[:2],  # small peek
        })

    # strip helper fields before returning
    for r in filtered:
        for k in list(r.keys()):
            if k.startswith("_"):
                r.pop(k, None)

    return jsonify(filtered)
"""
def format_production_answer(data: dict) -> str:
    
    Very simple formatter for production data returned from MCP.
    You can improve this later once you see the exact JSON structure.
    
    # If MCP returns {"rows": [...]}
    rows = data.get("rows") if isinstance(data, dict) else data
    if not rows:
        return "I couldn't find any production data for that cost code."

    # Take the first row as an example record
    first = rows[0] if isinstance(rows, list) else rows

    lines = ["Here is the production information I found:"]
    for key, value in first.items():
        # only show fields that look like production metrics
        if "prod" in key.lower() or "production" in key.lower():
            lines.append(f"- {key}: {value}")

    # Fallback: if we didn't match any 'production' keys, just dump the record
    if len(lines) == 1:
        lines.append(str(first))

    return "\n".join(lines)
"""
"""
import re
import asyncio

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import json


def extract_tool_payload(tool_result):
    """"""
    Unwrap the MCP CallToolResult → Python dict.
    Works with:
    - FastMCP JSON results
    - TextContent
    - StructuredContent (.value)
    """"""
    content = getattr(tool_result, "content", None)
    if not content:
        return {}

    item = content[0]

    # Case 1: objects that expose a .value (already a dict or list)
    if hasattr(item, "value"):
        return item.value

    # Case 2: plain text content – maybe JSON, maybe not
    if hasattr(item, "text"):
        text = item.text
        try:
            return json.loads(text)
        except Exception:
            return {"text": text}

    # Fallback
    return {}



COST_CODE_PATTERN = re.compile(r"\b\d{2,3}-\d{3}\b")  # e.g. 25-105, 125-055

# Configure how to start your MCP server (hcss_mcp_server.py)
SERVER_PARAMS = StdioServerParameters(
    command="python3",              # or "python" depending on your env
    args=["hcss_mcp_server.py"],    # the MCP server script
    env=None,                       # or os.environ.copy() if needed
    cwd=None,                       # working dir; None = current
)

async def _call_mcp_tool(tool_name: str, args: dict):
    # 1) Connect to server over stdio
    async with stdio_client(SERVER_PARAMS) as (read, write):
        # 2) Create a high-level client session
        async with ClientSession(read, write) as session:
            # 3) Initialize the protocol session
            await session.initialize()
            # 4) Call the tool exposed by your FastMCP server
            result = await session.call_tool(tool_name, arguments=args)
            return result

def run_mcp(tool_name: str, args: dict):
    # Sync wrapper for Flask
    return asyncio.run(_call_mcp_tool(tool_name, args))

"""

# ============================================================
#  CHAT ENDPOINT – now uses single MCP agent tool
# ============================================================

@app.post("/agent/chat")
def agent_chat():
    payload = request.get_json(silent=True) or {}
    user_message = (payload.get("message") or "").strip()
    session_id = payload.get("session_id", "default")

    if not user_message:
        return jsonify({"answer": "Please type a question."})

    # Call MCP agent tool
    result = answer_cost_query(session_id=session_id, user_message=user_message)
    answer = result.get("answer") or "I couldn't generate an answer."

    return jsonify({"answer": answer})

# ============================================================
# ENTRY POINT
# ============================================================
if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    print(f"🚀 Flask + OpenAPI + MCP server running on http://0.0.0.0:{port}")
    app.run(host="0.0.0.0", port=port)
