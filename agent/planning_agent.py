"""
agent/planning_agent.py
Planning Agent — Step 3 of the agentic pipeline.

Implements an explicit state machine:
  IDLE → DATA_LOADED → RETRIEVED → GENERATING → DONE
                                              ↘ ERROR

Uses Google Gemini Flash (free tier) to generate structured
infrastructure planning reports as validated Pydantic models.
"""

from __future__ import annotations

import json
import os
import re
import textwrap
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional

# Pydantic for structured output validation
try:
    from pydantic import BaseModel, Field, field_validator
except ImportError:
    raise ImportError("Run: pip install pydantic")

# Google Gemini SDK
try:
    import google.generativeai as genai
except ImportError:
    raise ImportError("Run: pip install google-generativeai")

from agent.data_agent import ZoneDemandStats


# ═══════════════════════════════════════════════════════════════════════
# Structured Output Model
# ═══════════════════════════════════════════════════════════════════════

class PlanningReport(BaseModel):
    """Structured EV infrastructure planning report."""

    demand_summary: str = Field(
        description="2-3 sentence summary of observed charging demand patterns."
    )
    high_load_locations: list[str] = Field(
        description="List of high-load zone identifiers with brief justification."
    )
    expansion_recommendations: list[str] = Field(
        description="3-5 actionable infrastructure expansion recommendations."
    )
    scheduling_insights: list[str] = Field(
        description="2-4 scheduling and load-balancing insights."
    )
    references: list[str] = Field(
        description="2-4 supporting standards or guidelines cited."
    )
    confidence: str = Field(
        description="'high', 'medium', or 'low' — based on data quality.",
        default="medium",
    )
    data_warnings: list[str] = Field(
        description="Any data quality warnings from preprocessing.",
        default_factory=list,
    )

    @field_validator("high_load_locations", "expansion_recommendations", "scheduling_insights", "references", "data_warnings", mode="before")
    @classmethod
    def ensure_list_of_strings(cls, v):
        """Convert single items to lists and complex objects to strings."""
        # 1. Handle single objects (str or dict) that should be lists
        if not isinstance(v, list):
            v = [v]
        
        # 2. Convert each element to string if it is a dictionary
        processed = []
        for item in v:
            if isinstance(item, dict):
                # Format: "Key1: Val1, Key2: Val2..."
                parts = []
                for key, val in item.items():
                    k_pretty = str(key).replace("_", " ").title()
                    parts.append(f"{k_pretty}: {val}")
                processed.append(" | ".join(parts))
            else:
                s_item = str(item).strip()
                # Skip literal "None" or "none" strings
                if s_item.lower() != "none" and s_item != "":
                    processed.append(s_item)
        return processed


# ═══════════════════════════════════════════════════════════════════════
# Agent State Machine
# ═══════════════════════════════════════════════════════════════════════

class AgentState(str, Enum):
    IDLE       = "idle"
    DATA_LOADED = "data_loaded"
    RETRIEVED  = "retrieved"
    GENERATING = "generating"
    DONE       = "done"
    ERROR      = "error"


@dataclass
class AgentContext:
    """Mutable state carrier passed through the pipeline."""
    stats: Optional[ZoneDemandStats] = None
    retrieved_chunks: list[str] = field(default_factory=list)
    prompt: Optional[str] = None
    raw_response: Optional[str] = None
    report: Optional[PlanningReport] = None
    error: Optional[str] = None


# ═══════════════════════════════════════════════════════════════════════
# Prompt Builder
# ═══════════════════════════════════════════════════════════════════════

def _build_prompt(stats: ZoneDemandStats, chunks: list[str]) -> str:
    """Construct the structured prompt sent to Gemini."""
    zones_str = ", ".join(str(z) for z in stats.zone_ids)
    monthly   = "\n".join(f"  - {m}: {v:.1f} kWh avg" for m, v in stats.monthly_trend.items())
    kb_text   = "\n\n".join(f"[Guideline {i+1}]\n{c}" for i, c in enumerate(chunks))
    warnings  = "\n".join(f"  - {w}" for w in stats.warnings) if stats.warnings else "  - (No specific warnings)"

    quality_note = {
        "good":   "Data quality is good. Provide confident, specific recommendations.",
        "sparse": "Data is sparse. Acknowledge uncertainty; provide conservative recommendations.",
        "noisy":  "Data contains anomalies. Note caution; recommend manual verification.",
    }.get(stats.data_quality, "Data quality is unknown.")

    prompt = textwrap.dedent(f"""
    You are ChargeWise AI, an expert EV infrastructure planning assistant.
    Analyze the following demand statistics and planning guidelines, then produce a structured JSON report.

    ## Zone Analysis
    - Zone IDs: {zones_str}
    - Total Energy (kWh): {stats.total_kwh:,.1f}
    - Average Hourly Demand (kWh): {stats.avg_hourly_kwh:.2f}
    - Peak Hour of Day: {stats.peak_hour}:00
    - Peak Day of Week: {stats.peak_day}
    - High-Load Zone: {'Yes — top 25% by volume' if stats.is_high_load else 'No'}
    - Weekday Average (kWh): {stats.weekday_avg:.2f}
    - Weekend Average (kWh): {stats.weekend_avg:.2f}
    - Anomalies Detected: {stats.anomaly_count}
    - Data Quality: {stats.data_quality.upper()}
    - Top 5 Peak Timestamps: {', '.join(stats.top_peak_timestamps)}
    - Monthly Trend:
    {monthly}

    ## Data Quality Note
    {quality_note}

    ## Data Warnings
    {warnings}

    ## Relevant EV Infrastructure Guidelines (use these to support recommendations)
    {kb_text}

    ## Task
    Based ONLY on the data above and the guidelines provided, generate a structured JSON report.
    Do NOT make up statistics not present above.
    Do NOT recommend specific vendor products.
    Cite the guideline sections when making recommendations using the format: "ChargeWise Guidelines: [Section Title]".

    Return ONLY valid JSON matching this exact schema:
    {{
      "demand_summary": "<2-3 sentence summary>",
      "high_load_locations": ["<zone id>: <brief reason>", ...],
      "expansion_recommendations": ["<recommendation 1>", "<recommendation 2>", ...],
      "scheduling_insights": ["<insight 1>", "<insight 2>", ...],
      "references": ["<standard/guideline 1>", ...],
      "confidence": "<high|medium|low>",
      "data_warnings": ["<warning if any>", ...]
    }}
    """).strip()

    return prompt


# ═══════════════════════════════════════════════════════════════════════
# JSON Extractor
# ═══════════════════════════════════════════════════════════════════════

def _extract_json(text: str) -> dict:
    """Extract the first valid JSON object from LLM output."""
    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Strip markdown code fences
    cleaned = re.sub(r"```(?:json)?", "", text).strip().rstrip("`").strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Find first {...} block
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    # ── Repair truncated JSON ───────────────────────────────────────
    # If it starts with { but doesn't end with }, try to close it
    if text.strip().startswith("{") and not text.strip().endswith("}"):
        repaired = text.strip()
        # Add a closing quote if we are inside one
        if repaired.count('"') % 2 != 0:
            repaired += '"'
        # Close any open brackets/braces
        open_braces = repaired.count("{") - repaired.count("}")
        open_brackets = repaired.count("[") - repaired.count("]")
        repaired += "]" * max(0, open_brackets)
        repaired += "}" * max(0, open_braces)
        
        try:
            return json.loads(repaired)
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Could not extract JSON from model output (length {len(text)}):\n{text[:500]}")


# ═══════════════════════════════════════════════════════════════════════
# Fallback Report (when LLM fails or data quality is too poor)
# ═══════════════════════════════════════════════════════════════════════

def _fallback_report(stats: ZoneDemandStats) -> PlanningReport:
    zones_str = ", ".join(str(z) for z in stats.zone_ids)
    return PlanningReport(
        demand_summary=(
            f"Zone(s) {zones_str} recorded a total of {stats.total_kwh:,.0f} kWh "
            f"with an average hourly demand of {stats.avg_hourly_kwh:.1f} kWh. "
            f"Data quality is {stats.data_quality}; results should be interpreted with caution."
        ),
        high_load_locations=[
            f"Zone {zones_str}: {'High-load (top 25%)' if stats.is_high_load else 'Normal load'}"
        ],
        expansion_recommendations=[
            "Conduct a manual grid impact assessment before adding new chargers.",
            "Consider Level 2 AC charger expansion in peak hours (17:00–21:00).",
            "Evaluate battery energy storage to offset peak grid draw.",
        ],
        scheduling_insights=[
            "Implement time-of-use pricing to shift demand to off-peak hours.",
            "Enable smart charging protocols (OCPP 2.0.1) for demand response.",
        ],
        references=[
            "IEA Global EV Outlook 2023",
            "NEVI Formula Program Standards (U.S. DOE)",
        ],
        confidence="low",
        data_warnings=stats.warnings or ["Fallback report — LLM generation failed."],
    )


# ═══════════════════════════════════════════════════════════════════════
# Planning Agent
# ═══════════════════════════════════════════════════════════════════════

class PlanningAgent:
    """
    Agentic workflow with explicit state management.

    Usage
    -----
    agent = PlanningAgent(api_key="...")
    report = agent.run(stats, chunks, on_state_change=st.write)
    """

    # Ordered list of model names to try (most preferred first)
    MODEL_CANDIDATES = [
        "gemini-1.5-flash",
        "gemini-1.5-flash-latest",
        "gemini-1.0-pro",
    ]

    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        self._api_key = api_key
        
        # ── Dynamic Model Discovery ─────────────────────────────────
        # Try to find what models this specific key actually has access to
        self._model_names = []
        try:
            available = genai.list_models()
            for m in available:
                if "generateContent" in m.supported_generation_methods:
                    # m.name is usually "models/gemini-1.5-flash"
                    self._model_names.append(m.name)
        except Exception:
            # Fallback to hardcoded defaults if list_models is restricted
            pass

        # Ensure we have at least the standard candidates as fallbacks
        defaults = [
            "gemini-1.5-flash",
            "gemini-1.5-flash-latest",
            "gemini-pro",
            "models/gemini-1.5-flash",
        ]
        for d in defaults:
            if d not in self._model_names:
                self._model_names.append(d)

        self.state   = AgentState.IDLE
        self.context = AgentContext()

    def _transition(
        self,
        new_state: AgentState,
        on_change: Optional[Callable[[str], None]] = None,
    ) -> None:
        self.state = new_state
        if on_change:
            labels = {
                AgentState.IDLE:       "Idle",
                AgentState.DATA_LOADED:"Analyzing demand data...",
                AgentState.RETRIEVED:  "Retrieving planning guidelines...",
                AgentState.GENERATING: "Generating infrastructure report...",
                AgentState.DONE:       "Report ready",
                AgentState.ERROR:      f"Error: {self.context.error}",
            }
            on_change(labels.get(new_state, new_state.value))

    def run(
        self,
        stats: ZoneDemandStats,
        retrieved_chunks: list[str],
        on_state_change: Optional[Callable[[str], None]] = None,
    ) -> PlanningReport:
        """
        Run the full agentic pipeline. 
        Tries multiple model candidates in sequence if execution fails.
        """
        self.context = AgentContext(stats=stats, retrieved_chunks=retrieved_chunks)

        try:
            self._transition(AgentState.DATA_LOADED, on_state_change)
            self._transition(AgentState.RETRIEVED, on_state_change)
            self._transition(AgentState.GENERATING, on_state_change)
            self.context.prompt = _build_prompt(stats, retrieved_chunks)

            # ── Loop through model candidates until one works ────────
            response = None
            last_err = None
            
            for m_name in self._model_names:
                try:
                    # Inform user which model is being attempted
                    if on_state_change:
                        on_state_change(f"Trying model: {m_name}...")
                    
                    model = genai.GenerativeModel(
                        m_name,
                        generation_config=genai.types.GenerationConfig(
                            temperature=0.3,
                            max_output_tokens=4096,
                        ),
                    )
                    # Attempt generation
                    response = model.generate_content(self.context.prompt)
                    # If we reach here, it worked!
                    break
                except Exception as e:
                    last_err = e
                    # If it's a 404 or similar, try next one
                    continue
            
            if response is None:
                raise RuntimeError(
                    f"All model candidates failed. Final error: {last_err}. "
                    "Please verify your API key access at https://aistudio.google.com/app/apikey"
                )

            self.context.raw_response = response.text

            # Step 4: Parse + validate
            raw_dict = _extract_json(self.context.raw_response)
            report   = PlanningReport(**raw_dict)
            report.data_warnings = list(set(report.data_warnings + stats.warnings))
            self.context.report  = report

            self._transition(AgentState.DONE, on_state_change)
            return report

        except Exception as exc:
            self.context.error = str(exc)
            self._transition(AgentState.ERROR, on_state_change)
            # Graceful fallback — include actual error in warnings
            fallback = _fallback_report(stats)
            fallback.data_warnings = [f"LLM Error: {str(exc)[:250]}"] + fallback.data_warnings
            self.context.report = fallback
            return fallback


# ═══════════════════════════════════════════════════════════════════════
# Smoke test (requires GEMINI_API_KEY env var)
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import os
    from agent.data_agent import compute_stats
    from agent.rag import get_knowledge_base

    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        print("Set GEMINI_API_KEY env var to run smoke test.")
    else:
        stats  = compute_stats([108])
        kb     = get_knowledge_base()
        chunks = kb.retrieve_for_stats(stats)
        agent  = PlanningAgent(api_key=api_key)
        report = agent.run(stats, chunks, on_state_change=print)
        print("\n── Report ──")
        print(report.model_dump_json(indent=2))
