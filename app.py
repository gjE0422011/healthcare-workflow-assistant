"""Healthcare AI Workflow Assistant: Streamlit UI. Logic lives in core.py.
Demo only. Not a medical device. Never use real patient data.
"""
import os

import streamlit as st

from core import extract_tasks, llm_soap, red_flag_screen, rule_soap, scrub_phi

# ---------- UI ----------
st.set_page_config(page_title="Healthcare AI Workflow Assistant", page_icon="🩺", layout="wide")
st.title("🩺 Healthcare AI Workflow Assistant")
st.warning("Prototype for learning and demos. Use synthetic notes only. Not for clinical decisions.")

SAMPLE = (
    "Mr. Raj Kumar, MRN 482913, seen on 12/10/2026. Complains of chest tightness for 2 days. "
    "Denies fever. BP 150/95, HR 98, SpO2 97%. ECG ordered. Likely hypertensive episode. "
    "Start amlodipine as advised by consultant. Follow up in 1 week. Refer to cardiology. "
    "Contact 98765 43210."
)

use_llm = False
with st.sidebar:
    st.header("Settings")
    has_key = bool(os.getenv("ANTHROPIC_API_KEY"))
    use_llm = st.toggle("Use Claude for SOAP structuring", value=False, disabled=not has_key)
    if not has_key:
        st.caption("Set ANTHROPIC_API_KEY to enable.")
    st.caption("Scrubbing always runs before any text leaves your machine.")

note = st.text_area("Paste a synthetic clinical note", value=SAMPLE, height=180)

if st.button("Run workflow", type="primary") and note.strip():
    clean, found = scrub_phi(note)
    t1, t2, t3, t4 = st.tabs(["1. PHI scrub", "2. SOAP note", "3. Tasks", "4. Red flags"])

    with t1:
        st.code(clean, language=None)
        if found:
            st.write("Removed items:")
            st.table([{"Type": a, "Text": b} for a, b in found])
        st.caption("Regex scrubbing is a first pass. It will miss things. Review before sharing.")

    with t2:
        if use_llm:
            try:
                st.markdown(llm_soap(clean))
            except Exception as e:
                st.error(f"Claude call failed: {e}")
        else:
            for sec, lines in rule_soap(clean).items():
                st.subheader(sec)
                st.write("\n".join(f"- {l}" for l in lines) or "Not documented")

    with t3:
        tasks = extract_tasks(clean)
        if tasks:
            for i, t in enumerate(tasks):
                st.checkbox(t, key=f"task{i}")
        else:
            st.info("No follow-up actions detected.")

    with t4:
        hits = red_flag_screen(clean)
        if hits:
            for cat, terms in hits.items():
                st.error(f"{cat}: {', '.join(terms)}")
            st.caption("Keyword match only. A clinician must assess.")
        else:
            st.success("No listed red-flag terms found. This does not mean the case is low risk.")
