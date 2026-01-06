# hcss_mcp_server.py

import os
import json
from typing import Dict, Any, List, Optional

import requests
from dotenv import load_dotenv
from openai import OpenAI
"""from mcp.server.fastmcp import FastMCP"""

load_dotenv()
from azure.storage.blob import BlobServiceClient
import pandas as pd
import difflib

DESCRIPTION_MAP = {}   # maps elaborated description → short description
ELAB_LIST = []         # list of elaborated descriptions

def load_description_info():
    global DESCRIPTION_MAP, ELAB_LIST
    try:
        df = pd.read_csv("cost_code_descriptions_info.csv")
        # Expected columns: cost_code_description, description_info

        for _, row in df.iterrows():
            short = str(row["cost_code_description"]).strip()
            info  = str(row["Description Info"]).strip()

            DESCRIPTION_MAP[info] = short
            ELAB_LIST.append(info)

        print(f"Loaded {len(ELAB_LIST)} elaborated descriptions.")
    except Exception as e:
        print("Error loading description info:", e)

load_description_info()



# -----------------------------
# MCP server + OpenAI client
# -----------------------------
"""mcp = FastMCP("hcss-costcode-tools")"""
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# -----------------------------
# In-memory session state
# -----------------------------
SESSION_STATE: Dict[str, Dict[str, Any]] = {}


def get_session_state(session_id: str) -> Dict[str, Any]:
    return SESSION_STATE.setdefault(session_id, {})


# -----------------------------
# Backend call helper (/data)
# -----------------------------
def call_hcss_data_api(args: Dict[str, Any]) -> Any:
    """
    Call your existing /data endpoint.

    Args should match query params:
      cost_code, job_code, unit, season, cost_code_description, match, type
    """
    try:
        res = requests.get("http://127.0.0.1:8000/data", params=args, timeout=30)
        return res.json()
    except Exception as e:
        return {"error": str(e), "args": args}


# -----------------------------
# Cost-code helpers
# -----------------------------
def normalize_rows(raw: Any) -> List[Dict[str, Any]]:
    if isinstance(raw, list):
        return raw
    if isinstance(raw, dict):
        # common patterns: {"data":[...]} or {"rows":[...]}
        for key in ("data", "rows", "results"):
            if key in raw and isinstance(raw[key], list):
                return raw[key]
    return []


def dedupe_cost_codes(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = {}
    for r in rows or []:
        cc = r.get("Cost Code") or r.get("cost_code")
        desc = r.get("Cost Code Description") or r.get("cost_code_description")
        unit = r.get("Unit") or r.get("unit")
        if not cc:
            continue
        key = (cc, desc, unit)
        if key not in seen:
            seen[key] = {"cost_code": cc, "description": desc, "unit": unit}
    return list(seen.values())


def format_cost_code_choices(choices: List[Dict[str, Any]]) -> str:
    lines = []
    for i, c in enumerate(choices, start=1):
        cc = c.get("cost_code") or "?"
        desc = c.get("description") or ""
        unit = c.get("unit") or ""
        if unit:
            lines.append(f"{i}. {cc} – {desc} ({unit})")
        else:
            lines.append(f"{i}. {cc} – {desc}")
    return "\n".join(lines)


# -----------------------------
# 1) Interpret query (LLM-only)
# -----------------------------
def interpret_query(user_message: str) -> Optional[Dict[str, Any]]:
    """
    Use LLM to interpret the user's natural-language question and return a JSON intent:
      {
        "cost_code": string | null,
        "description": string | null,
        "filters": object,
        "metric": string,
        "explanation": string
      }
    """
    system_prompt = {
        "role": "system",
        "content": f"""
    You are an HCSS construction analytics interpreter.
    You do NOT perform calculations. You do NOT call tools.
    Your ONLY job is to interpret the user's question and return a JSON object.

    ============================
     JSON FORMAT (RETURN ONLY THIS)
    ============================
    {{
      'cost_code': string | null,
      'description': string | null,
      'filters': object,
      'metric': string,
      'explanation': string
    }}
    ============================
     RULES FOR DESCRIPTION
    ============================
    - 'description' MUST contain ONLY the cost code description / activity name.
    - Examples: 'import fill', 'excavation', 'fine grading', 'asphalt paving'.
    - NEVER include metric words (production, cost, rate, seasonal, per CY, etc.).
    - NEVER include attribute details (depth, pipe size, trench width, diameter, crew size, equipment type, etc.).
    - Reduce the user question to the closest matching ACTIVITY from the list below.

    Normalize user phrasing to one of these descriptions.

    Examples:
    - "RCP pipe installation less than six feet deep for pipe sizes 15–24 inches"
        → "RCP pipe installation"


    ============================
     RULES FOR METRIC
    ============================
    - 'metric' describes WHAT is being measured (production rate, unit cost, etc.).
    - Keep metric separate from description.

    ============================
     RULES FOR COST_CODE
    ============================
    - Only set cost_code if the user explicitly mentions a code like '25-055'.
    - Otherwise cost_code = null.

    ============================
     CRITICAL RULES FOR FILTERS (MUST OBEY)
    ============================
    You MUST NOT create ANY filters that are NOT supported by the backend.

    VALID FILTER NAMES (ONLY these):
    - cost_code
    - cost_code_description
    - job_code
    - job_code_description
    - unit
    - season
    - type
    - expected_only
    - match

    If the user mentions unsupported attributes such as:
    - depth
    - trench depth
    - pipe size (e.g. 15–24 inches)
    - diameter
    - soil type
    - traffic
    - crew size
    - conditions
    - weather (other than rainy/non-rainy)
    - excavation width
    - equipment model
    - etc.

    → DO NOT add these to filters.
    → Use them ONLY to help identify the activity.

    EXAMPLE (VERY IMPORTANT):
    User: "What is my average production rate for RCP pipe installation less than six feet deep for pipe sizes 15–24 inches?"

    CORRECT OUTPUT:
        description = "RCP pipe installation"
        metric = "average production rate"
        filters = {{}}

    INCORRECT OUTPUT (NEVER DO THIS):
        filters = {{
            "depth": "less than six feet",
            "pipe_sizes": "15-24 inches"
        }}

    ============================
     FINAL RULES
    ============================
    - If nothing else is specified → filters = {{}}
    - Return ONLY the JSON. No extra text.
    """
    }

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[system_prompt, {"role": "user", "content": user_message}],
        response_format={"type": "json_object"},
        temperature=0,
    )

    msg = resp.choices[0].message
    print(msg)
    try:
        intent = msg.parsed  # available because of response_format=json_object
        return intent
    except Exception:
        try:
            return json.loads(msg.content or "")
        except Exception:
            return None


# -----------------------------
# 2) Get metrics & summarize
# -----------------------------
def summarize_metrics(user_message: str, metric: str, cost_code: str, backend_rows: Any) -> str:
    """
    NEW VERSION — bulletproof metric handling.
    Automatically classifies metric as cost or productivity,
    filters backend rows before sending to LLM, so the model cannot mix metrics.
    """
    # ✅ ALWAYS normalize first
    rows_list = normalize_rows(backend_rows)
    # ------------------------------------------------------
    # 1. Detect metric type from user message
    # ------------------------------------------------------
    metric_l = (metric or "").lower()
    user_l = (user_message or "").lower()

    prod_words = [
        "production", "productivity", "production rate",
        "shift productivity", "yards per", "per hour", "shift"
    ]
    cost_words = [
        "cost", "unit cost", "cost per", "labor cost", "equipment cost"
    ]

    is_prod = any(w in metric_l or w in user_l for w in prod_words)
    is_cost = (not is_prod) and any(w in metric_l or w in user_l for w in cost_words)

    if is_prod:
        metric_type = "productivity"
    elif is_cost:
        metric_type = "cost"
    else:
        metric_type = "other"

    # ------------------------------------------------------
    # 2. Filter backend rows BEFORE passing to LLM
    #    (prevents incorrect mixing of cost/productivity)
    # ------------------------------------------------------
    filtered_rows = []
    # Does user explicitly request shift productivity?
    wants_shift = ("shift" in metric_l) or ("shift" in user_l)

    for row in rows_list or []:
        if metric_type == "cost":
            keep = {}

            # ALWAYS KEEP SEASON (critical for grouping rainy/non-rainy)
            if "season" in row:
                keep["season"] = row["season"]

            # ALWAYS KEEP UNIT
            if "unit" in row:
                keep["unit"] = row["unit"]

            # KEEP ALL COST FIELDS
            for k, v in row.items():
                if "cost" in k.lower():
                    keep[k] = v

        elif metric_type == "productivity":
            keep = {}
            # ALWAYS KEEP SEASON
            if "season" in row:
                keep["season"] = row["season"]

            # KEEP UNIT
            if "unit" in row:
                keep["unit"] = row["unit"]
            # Include normal productivity unless it's shift-only
            for k, v in row.items():
                kl = k.lower()

                # include regular productivity
                if "productivity" in kl and "shift" not in kl:
                    keep[k] = v

                # include shift productivity ONLY if user asked for shift
                if wants_shift and "shift_productivity" in kl:
                    keep[k] = v

        else:
            keep = row  # fallback if unclear
        filtered_rows.append(keep)

    # ------------------------------------------------------
    # 3. Build system prompt with strict instructions
    # ------------------------------------------------------
    system_prompt = {
        "role": "system",
        "content": f"""
You are an HCSS cost analytics summarizer.

METRIC_TYPE = '{metric_type}'
USER_METRIC = '{metric}'

You are given FILTERED_RESULTS which contains ONLY the fields relevant
to the metric type. You MUST NOT mention or invent any fields that are
not present in FILTERED_RESULTS.

GLOBAL RULES (apply to ALL metric types):
- By default, you MUST use ACTUAL values only.
- NEVER return any fields whose name contains "expected"
  (like mean_of_expected_..., median_of_expected_..., mode_of_expected_...)
  UNLESS:
    • the user explicitly asks for expected, planned, target, future,
      estimate, estimated, prediction, predicted, projected, or a comparison
      between actual and expected, OR
    • USER_MESSAGE == "expected_followup" (see SPECIAL EXPECTED-FOLLOWUP RULE below).
- MEAN-ONLY RULE:
    • By default, you MUST use ONLY mean_* fields (mean_of_...).
    • You MUST NOT include ANY median_of_* or mode_of_* fields
      unless the user explicitly says they want "median", "mode",
      "distribution", "spread", "percentile", or similar wording.
    • This applies to BOTH actual_* and expected_* fields.
- If expected_* fields exist in FILTERED_RESULTS and the user has NOT explicitly
  asked for expected values and USER_MESSAGE is NOT "expected_followup", then:

    1. You MUST NOT ask “Would you like to see expected values?” inside each row
      or inside each season block. 
    2. You MUST return ONLY ACTUAL fields in this response if it is related to seasonal data.
        Example style:
        Non-Rainy Season :
        Mean Actual Production Rate: 402.33 
        Rainy Season :
        Mean Actual Production Rate: 45.33
    3. You MUST return in this format if it is related to overall data.
        Example style:
        Mean Actual Production Rate: 402.33 
        Mean Actual Production Rate: 45.33 
    4. Display units as:
        - For productivity, Units per Labor Hour like 45.33 CY per labor hour.
        - For shift productivity, Units per Shift.
        - For labor and equipment costs, as $123 per Unit.
    5. After ALL rows have been listed, at the very end of the entire response,
      you MUST add ONE SINGLE question:
            "Would you like to see the expected values?"
    5. Immediately after that question, you MUST append this hidden HTML comment
       on its own line:
         <!-- expected_pending -->
    6. You MUST NOT show any expected_* numbers in this response.
     
SPECIAL EXPECTED-FOLLOWUP RULE:
If USER_MESSAGE == "expected_followup":
    - Ignore all actual_* fields completely.
    • You MUST format expected values EXACTLY the same way you formatted actual values along with displaying unit.
    Example style:
    You MUST return in this format if it is related to seasonal data.
        Non-Rainy Season :
        Mean Expected Production Rate: 402.33 
        Rainy Season :
        Mean Expected Production Rate: 45.33
    You MUST return in this format if it is related to overall data.
        Example style:
        Mean Expected Production Rate: 402.33 
        Mean Expected Production Rate: 45.33 
    . Display units as:
        - For productivity, Units per Labor Hour like 45.33 CY per labor hour.
        - For shift productivity, Units per Shift.
        - For labor and equipment costs, as $123 per Unit.
    • NEVER show median_of_expected_* or mode_of_expected_* fields unless the user explicitly
      asked for median or mode (which they did not).
    • At the end of the response, append this hidden HTML comment on its own line:
        <!-- expected_done -->
Both for actual and expected values, When providing averages, medians, and modes, include the value with its specific unit (e.g., "The average actual productivity for cost code 35-200 in unit EA is 0.05 Units per Labor Hour").

RULES:
- If METRIC_TYPE='cost':
    • If the user's question mentions “cost”, “unit cost”, “cost per…”, “labor cost”, or “equipment cost”, follow these rules: 
    • PRIMARY METRIC (always return): 
    - Mean Actual Unit Equipment Cost (mean_of_actual_unit_equipment_cost) 
    - Mean Actual Unit Labor Cost (mean_of_actual_unit_labor_cost) 
    → These two combined represent the user’s REAL cost. 
    
    • DO NOT RETURN: - ANY mode values (mode_of_…) 
    - Any median values, unless the user explicitly says “median” or relevant terms. 
    - ANY expected cost fields (expected_…) unless explicitly requests like expected or future or relevant terms. 
    - ANY productivity fields 
    - ANY unit-of-measurement fields unless user asked

- If METRIC_TYPE='productivity':
    • Return ONLY productivity fields (already filtered), don't include shift productivity by default.
    • Include actual values.
    DO NOT RETURN
    • ANY cost fields
    . ANY expected values unless user explicitly request like expected or future or relevant terms. 
    . Shift productivity fields unless explicitly requested by user.

- - If METRIC_TYPE='other':
    • Summarize FILTERED_RESULTS safely using the GLOBAL RULES above.
    • Prefer ACTUAL fields over EXPECTED fields.
    • Do NOT output any expected_* fields or values unless the user explicitly
      asks for expected, target, planned, future, or a comparison.

- SEASON LABELING RULE:
- You MUST ONLY display season headings ("Rainy Season", "Non-Rainy Season", etc.)
  when the user's question explicitly contains the word "seasonal, rainy, non rainy".

- You MUST NEVER guess or apply seasonal grouping unless the user explicitly asked for it.

Your answer MUST be short, clean, and formatted clearly.
        """
    }

    # ------------------------------------------------------
    # 4. Context: only filtered clean data (safe!)
    # ------------------------------------------------------
    assistant_context = {
        "role": "assistant",
        "content": f"FILTERED_RESULTS:\n{json.dumps(filtered_rows)[:8000]}"
    }

    user_msg = {"role": "user", "content": user_message}

    # ------------------------------------------------------
    # 5. Call OpenAI
    # ------------------------------------------------------
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[system_prompt, assistant_context, user_msg],
        temperature=0,
    )

    answer = resp.choices[0].message.content or ""
    return answer




# ============================================================
# MAIN MCP TOOL
# ============================================================
"""@mcp.tool()"""
def answer_cost_query(session_id: str, user_message: str) -> dict:

    """
    Orchestrates:
      1) Selection mode (user replying with a number or cost code)
      2) Expected-values follow-up ("yes", "ok", etc.)
      3) New question → interpret intent, query backend, summarize
    """
    state = get_session_state(session_id)
    user_lower = user_message.lower().strip()
    # If user sends a non-numeric, non-cost-code message,
    # treat it as a NEW question and reset multi-select mode
    if not user_message.strip().isdigit() and "-" not in user_message:
        state["allow_multi_select"] = False
        state["pending_choices"] = None

    # =============================
    # FALLBACK ELABORATED SELECTION
    # =============================
    elab_pending = state.get("pending_elab_choices")

    if elab_pending and not state.get("pending_elab_consumed"):

        user = user_message.strip()

        # ----------------------------
        # User selects an elaborated match
        # ----------------------------
        if user.isdigit():
            idx = int(user) - 1

            if 0 <= idx < len(elab_pending):
                chosen_elab = elab_pending[idx]
                chosen_short = DESCRIPTION_MAP.get(chosen_elab)

                # Update the interpreted description to corrected short form ("WM", "RCP", etc.)
                description = chosen_short

                # Mark fallback selection as consumed
                state["pending_elab_consumed"] = True
                state["pending_elab_choices"] = None

                # ----------------------------
                # Retry fuzzy backend search with the corrected description
                # ----------------------------
                params = state.get("last_search_params", {}).copy()

                # CRITICAL FIX: Remove stale cost_code so backend returns ALL fuzzy matches
                params.pop("cost_code", None)

                params["cost_code_description"] = chosen_short
                params["match"] = "fuzzy"

                # FIX: enable seasonal mode if user said "rainy" or "seasonal"
                user_lower = user_message.lower()
                if "rainy" in user_lower or "seasonal" in user_lower:
                    params["type"] = "season"

                backend_rows = call_hcss_data_api(params)
                rows = normalize_rows(backend_rows)

                # ----------------------------
                # Now follow NORMAL multi-choice behavior
                # ----------------------------
                choices = dedupe_cost_codes(rows)

                # MULTIPLE COST CODES → ask user to choose
                if len(choices) > 1:
                    list_text = format_cost_code_choices(choices)
                    state["pending_choices"] = choices
                    state["pending_consumed"] = False
                    state["allow_multi_select"] = True  # ✅ NEW

                    return {
                        "answer": (
                            "I found several matching activities:\n\n"
                            f"{list_text}\n\n"
                            "Please reply with a number or cost code."
                        ),
                        "used_tools": False,
                    }

                # ============================================
                # EXACTLY ONE COST CODE → SAFETY CHECK
                # ============================================
                if len(choices) == 1:

                    backend_desc = (choices[0].get("description") or "").lower()
                    user_desc = (state.get("last_description") or "").lower()

                    # ----------------------------------------------
                    # SAFETY RULE:
                    # Do NOT auto-select a single cost code unless
                    # backend_desc EXACTLY MATCHES the interpreted description.
                    #
                    # This prevents incorrect auto-selection when the user gives
                    # a long messy description ("24 inches less than 10 ft deep")
                    # and fuzzy search returns only 1 backend row.
                    # ----------------------------------------------
                    if backend_desc != user_desc:
                        list_text = format_cost_code_choices(choices)

                        state["pending_choices"] = choices
                        state["pending_consumed"] = False

                        return {
                            "answer": (
                                "I found a possible match for your description:\n\n"
                                f"{list_text}\n\n"
                                "Please reply with the number or cost code to confirm."
                            ),
                            "used_tools": False,
                        }

                    # ----------------------------------------------
                    # EXACT MATCH → SAFE TO AUTO-SELECT
                    # ----------------------------------------------
                    chosen_cc = choices[0]["cost_code"]

                    summary = summarize_metrics(
                        user_message=state.get("last_user_question", user_message),
                        metric=state.get("last_metric"),
                        cost_code=chosen_cc,
                        backend_rows=backend_rows,
                    )

                    summary = (
                        summary.replace("<!-- expected_pending -->", "")
                        .replace("<!-- expected_done -->", "")
                    )

                    return {
                        "answer": summary,
                        "used_tools": True,
                    }

    # ========================================================
    # 1. SELECTION MODE: user is replying with a number or code
    # ========================================================
    pending = state.get("pending_choices")
    if pending and not state.get("pending_consumed"):
        state["pending_consumed"] = True

        user = user_message.strip()
        chosen_code = None
        chosen_desc_forced = None  # only when user picked a specific numbered line

        # A) numeric option → specific choice (cost code + description)
        if user.isdigit():
            idx = int(user) - 1
            if 0 <= idx < len(pending):
                chosen_code = pending[idx]["cost_code"]
                chosen_desc_forced = pending[idx].get("description")

        # B) cost code text → user typed the cost code directly
        if not chosen_code:
            for c in pending:
                if c["cost_code"] in user:
                    chosen_code = c["cost_code"]
                    state["last_description"] = c.get("description")
                    break

        if not chosen_code:
            return {
                "answer": "Please reply with a number or cost code from the list.",
                "used_tools": False,
            }

        # Clear pending choices
        # Clear pending choices ONLY if multi-select is not allowed
        if not state.get("allow_multi_select"):
            state["pending_choices"] = None

        state["pending_consumed"] = False

        # Retrieve previous context
        metric = state.get("last_metric", "requested metric")
        base_params = state.get("last_search_params") or state.get("last_filters", {})

        # Backend query using previous filters + chosen cost code
        params = base_params.copy()
        params["cost_code"] = chosen_code

        # If user picked a specific numbered choice, force that description exactly
        if chosen_desc_forced:
            params["cost_code_description"] = chosen_desc_forced
            params["match"] = "exact"

        backend_rows = call_hcss_data_api(params)
        rows_list = normalize_rows(backend_rows)

        # Save for expected follow-up
        state["last_query_params"] = params.copy()
        state["last_cost_code"] = chosen_code
        state["last_metric"] = metric

        # If multiple rows and user selected only the COST CODE,
        # summarize each row separately so we keep each description.
        if rows_list and isinstance(rows_list, list) and len(rows_list) > 1 and not chosen_desc_forced:
            parts = []
            QUESTION = "Would you like to see the expected values?"

            for r in rows_list:
                desc = (
                    r.get("Cost Code Description")
                    or r.get("cost_code_description")
                    or r.get("description")
                    or ""
                )

                summary = summarize_metrics(
                    user_message=state.get("last_user_question", user_message),
                    metric=metric,
                    cost_code=chosen_code,
                    backend_rows=[r],   # only this row
                )

                # Track expected-followup markers and strip them
                if "<!-- expected_pending -->" in summary:
                    state["awaiting_expected"] = True
                elif "<!-- expected_done -->" in summary:
                    state["awaiting_expected"] = False

                cleaned_summary = (
                    summary
                    .replace("<!-- expected_pending -->", "")
                    .replace("<!-- expected_done -->", "")
                )

                if desc:
                    parts.append(f"**{desc}**\n{cleaned_summary}")
                else:
                    parts.append(cleaned_summary)

            final_answer = "\n\n".join(parts)

            # Keep ONLY ONE copy of the follow-up question at the end
            if QUESTION in final_answer:
                pieces = final_answer.split(QUESTION)
                final_answer = "".join(pieces[:-1]) + QUESTION

            return {
                "answer": final_answer,
                "used_tools": True,
            }

        # Otherwise single row (or specific description) → one summary
        final_answer = summarize_metrics(
            user_message=state.get("last_user_question", user_message),
            metric=metric,
            cost_code=chosen_code,
            backend_rows=backend_rows,
        )

        if "<!-- expected_pending -->" in final_answer:
            state["awaiting_expected"] = True
        elif "<!-- expected_done -->" in final_answer:
            state["awaiting_expected"] = False

        cleaned = (
            final_answer
            .replace("<!-- expected_pending -->", "")
            .replace("<!-- expected_done -->", "")
        )

        return {
            "answer": cleaned,
            "used_tools": True,
        }

    # ========================================================
    # 2. EXPECTED VALUES FOLLOW-UP ("yes", "ok", "show", etc.)
    # ========================================================
    followup_words = {"yes", "yeah", "yep", "ok", "okay", "sure", "show", "show me"}
    is_followup = user_lower in followup_words

    # If we were waiting for expected but user asked a NEW question → reset
    if state.get("awaiting_expected") and not is_followup:
        state["awaiting_expected"] = False

    # If this really IS an expected-followup turn
    if state.get("awaiting_expected") and is_followup:
        params = (state.get("last_query_params") or {}).copy()
        metric_for_expected = state.get("last_metric", "requested metric")
        cc_for_expected = state.get("last_cost_code")

        if not params or not cc_for_expected:
            state["awaiting_expected"] = False
            return {
                "answer": (
                    "I couldn't retrieve the previous result to show expected values. "
                    "Please ask your question again."
                ),
                "used_tools": False,
            }

        backend_rows = call_hcss_data_api(params)
        rows_list = normalize_rows(backend_rows)

        # Deduplicate by (description, season)
        unique = {}
        for r in rows_list:
            desc = (
                r.get("Cost Code Description")
                or r.get("cost_code_description")
                or r.get("description")
                or ""
            )
            season = r.get("season", "")
            key = (desc, season)
            if key not in unique:
                unique[key] = r
        rows_list = list(unique.values())

        if rows_list and isinstance(rows_list, list) and len(rows_list) > 1:
            parts = []
            for r in rows_list:
                desc = (
                    r.get("Cost Code Description")
                    or r.get("cost_code_description")
                    or r.get("description")
                    or ""
                )

                summary = summarize_metrics(
                    user_message="expected_followup",
                    metric=metric_for_expected,
                    cost_code=cc_for_expected,
                    backend_rows=[r],   # this row's expected values
                )

                cleaned_summary = (
                    summary
                    .replace("<!-- expected_pending -->", "")
                    .replace("<!-- expected_done -->", "")
                )

                if desc:
                    parts.append(f"**{desc}**\n{cleaned_summary}")
                else:
                    parts.append(cleaned_summary)

            final_answer = "\n\n".join(parts)
        else:
            summary = summarize_metrics(
                user_message="expected_followup",
                metric=metric_for_expected,
                cost_code=cc_for_expected,
                backend_rows=rows_list,
            )
            final_answer = (
                summary
                .replace("<!-- expected_pending -->", "")
                .replace("<!-- expected_done -->", "")
            )

        state["awaiting_expected"] = False

        return {
            "answer": final_answer,
            "used_tools": True,
        }

    # ========================================================
    # 3. NORMAL MODE: new question → interpret intent
    # ========================================================
    intent = interpret_query(user_message)
    if not intent:
        return {"answer": "Sorry, I couldn't understand your request."}

    cost_code = intent.get("cost_code")
    description = intent.get("description")
    metric = intent.get("metric") or "requested metric"
    filters = intent.get("filters") or {}

    # Save last metric/description/user question
    state["last_metric"] = metric
    state["last_description"] = description
    state["last_user_question"] = user_message

    # Seasonal behavior: ONLY seasonal if user said "seasonal"
    if "rainy" in user_lower or "non rainy" in user_lower or "seasonal" in user_lower:
        filters["type"] = "season"
    else:
        filters.pop("type", None)
        filters.pop("season", None)

    # --------------------------------------------------------
    # 3A. COST CODE PROVIDED DIRECTLY
    # --------------------------------------------------------
    if cost_code:
        params = filters.copy()
        params["cost_code"] = cost_code
        backend_rows = call_hcss_data_api(params)

        final_answer = summarize_metrics(
            user_message=user_message,
            metric=metric,
            cost_code=cost_code,
            backend_rows=backend_rows,
        )

        if "<!-- expected_pending -->" in final_answer:
            state["awaiting_expected"] = True

        state["last_query_params"] = params.copy()
        state["last_cost_code"] = cost_code

        cleaned = (
            final_answer
            .replace("<!-- expected_pending -->", "")
            .replace("<!-- expected_done -->", "")
        )

        return {
            "answer": cleaned,
            "used_tools": True,
        }

    # --------------------------------------------------------
    # 3B. NO COST CODE → SEARCH BY DESCRIPTION
    # --------------------------------------------------------
    if not description:
        return {
            "answer": (
                "I couldn't identify which work item you're asking about. "
                "Please mention the activity or cost code description, like "
                "'import fill', 'excavation', or 'grading'."
            )
        }

    search_params = filters.copy()
    search_params["cost_code_description"] = description
    search_params["match"] = "fuzzy"
    state["last_search_params"] = search_params.copy()

    raw = call_hcss_data_api(search_params)
    rows = normalize_rows(raw)

    if not rows:
        # FALLBACK: match user query against elaborated descriptions
        user_desc = description.lower()

        scored = []
        for elaborated in ELAB_LIST:
            ratio = difflib.SequenceMatcher(None, user_desc, elaborated.lower()).ratio()
            scored.append((ratio, elaborated))

        scored.sort(reverse=True, key=lambda x: x[0])

        # take top 5 elaborated descriptions
        candidates = [el for score, el in scored[:5] if score > 0.25]

        if not candidates:
            return {"answer": f"No data found for '{description}'."}

        # store for selection mode
        state["pending_elab_choices"] = candidates
        state["pending_elab_consumed"] = False

        text = "\n".join(
            f"{i + 1}. {el}  ➜  {DESCRIPTION_MAP.get(el, '')}"
            for i, el in enumerate(candidates)
        )

        return {
            "answer": (
                f"I couldn't find an exact match, but here are close activities:\n\n"
                f"{text}\n\n"
                "Please reply with a number to select the correct activity."
            ),
            "used_tools": False,
        }

    choices = dedupe_cost_codes(rows)

    # Single cost code → skip selection and go straight to metrics
    if len(choices) == 1:
        chosen = choices[0]["cost_code"]
        params = state.get("last_search_params", search_params).copy()
        params["cost_code"] = chosen
        backend_rows = call_hcss_data_api(params)

        final_answer = summarize_metrics(
            user_message=user_message,
            metric=metric,
            cost_code=chosen,
            backend_rows=backend_rows,
        )

        if "<!-- expected_pending -->" in final_answer:
            state["awaiting_expected"] = True

        state["last_query_params"] = params.copy()
        state["last_cost_code"] = chosen

        cleaned = (
            final_answer
            .replace("<!-- expected_pending -->", "")
            .replace("<!-- expected_done -->", "")
        )

        return {
            "answer": cleaned,
            "used_tools": True,
        }

    # Multiple choices → ask user to choose
    list_text = format_cost_code_choices(choices)
    state["pending_choices"] = choices
    state["pending_consumed"] = False
    state["allow_multi_select"] = True  # ✅ NEW

    answer = (
        "Multiple matching cost codes found:\n\n"
        f"{list_text}\n\n"
        "Please reply with a number or cost code."
    )

    return {
        "answer": answer,
        "used_tools": True,
    }



# -----------------------------
# Run MCP server
# -----------------------------
"""if __name__ == "__main__":
    mcp.run()"""
