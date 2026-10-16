"""Core logic for the Healthcare AI Workflow Assistant (no UI).

Four tools for clinical note workflows:
1. PHI scrubber  2. SOAP structuring  3. Follow-up task extraction  4. Red-flag screen

Runs fully offline with rule-based logic. Optionally uses the Claude API
for SOAP structuring if ANTHROPIC_API_KEY is set.
Demo only. Not a medical device. Never use real patient data.
"""
import os
import re


# ---------- PHI scrubber ----------
PHI_PATTERNS = [
    ("EMAIL", r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+"),
    ("PHONE", r"(?:\+?\d{1,3}[\s-]?)?\d{5}[\s-]?\d{5}|\b\d{3}[-.\s]\d{3}[-.\s]\d{4}\b"),
    ("DATE", r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b"),
    ("MRN", r"\b(?:MRN|UHID|ID)[:#\s]*\d{4,}\b"),
    ("NAME", r"\b(?:Mr|Mrs|Ms|Miss|Dr)\.?\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?"),
]


def scrub_phi(text):
    found = []
    for label, pat in PHI_PATTERNS:
        for m in re.finditer(pat, text):
            found.append((label, m.group(0)))
        text = re.sub(pat, f"[{label}]", text)
    return text, found


# ---------- SOAP structuring ----------
SOAP_KEYS = {
    "Subjective": ["complains", "reports", "states", "history of", "c/o", "denies", "pain for"],
    "Objective": ["bp", "hr", "temp", "spo2", "exam", "lab", "x-ray", "ecg", "on examination", "o/e"],
    "Assessment": ["diagnosis", "impression", "likely", "consistent with", "assessment", "dx"],
    "Plan": ["start", "advise", "refer", "follow up", "follow-up", "prescribe", "order", "plan", "review in"],
}


def rule_soap(text):
    out = {k: [] for k in SOAP_KEYS}
    for sent in re.split(r"(?<=[.!?])\s+|\n+", text):
        s = sent.strip()
        if not s:
            continue
        low = s.lower()
        best = max(SOAP_KEYS, key=lambda k: sum(w in low for w in SOAP_KEYS[k]))
        if any(w in low for w in SOAP_KEYS[best]):
            out[best].append(s)
        else:
            out["Subjective"].append(s)
    return out


def llm_soap(text):
    import anthropic

    client = anthropic.Anthropic()
    model = os.getenv("CLAUDE_MODEL", "claude-sonnet-5-5")
    msg = client.messages.create(
        model=model,
        max_tokens=1000,
        system=(
            "You restructure de-identified clinical notes into SOAP format. "
            "Use only facts in the note. Never add diagnoses or doses. "
            "If a section has no content, write 'Not documented'."
        ),
        messages=[{"role": "user", "content": text}],
    )
    return msg.content[0].text


# ---------- Tasks ----------
TASK_CUES = r"(follow[- ]?up|refer(?:ral)?|order|repeat|review in|schedule|arrange|check|start|stop|advise)"


def extract_tasks(text):
    tasks = []
    for sent in re.split(r"(?<=[.!?])\s+|\n+", text):
        if re.search(TASK_CUES, sent, re.I):
            tasks.append(sent.strip())
    return tasks


# ---------- Red flags ----------
RED_FLAGS = {
    "Cardiac": ["chest pain", "chest tightness", "radiating to arm"],
    "Respiratory": ["shortness of breath", "breathless", "spo2 <", "cyanosis"],
    "Neurological": ["slurred speech", "facial droop", "sudden weakness", "seizure", "worst headache"],
    "Bleeding / sepsis": ["vomiting blood", "black stool", "fever with confusion", "rigors"],
    "Mental health": ["suicidal", "self-harm"],
}


def red_flag_screen(text):
    low = text.lower()
    hits = {}
    for cat, terms in RED_FLAGS.items():
        h = [t for t in terms if t in low]
        if h:
            hits[cat] = h
    return hits


