# Healthcare AI Workflow Assistant

A small Streamlit prototype that cuts admin load around clinical notes.

## What it does
1. **PHI scrub.** Masks emails, phones, dates, MRNs and titled names.
2. **SOAP structuring.** Rule-based by default. Optional Claude mode.
3. **Task extraction.** Pulls follow-ups, referrals and orders into a checklist.
4. **Red-flag screen.** Keyword alerts for urgent patterns.

## Run
```bash
pip install -r requirements.txt
streamlit run app.py
```
Optional Claude mode:
```bash
export ANTHROPIC_API_KEY=your_key
```

## Limits (be honest about these)
- Regex PHI removal is not HIPAA-grade de-identification.
- Red-flag logic is keyword matching, not a validated triage tool.
- Tested on synthetic text only. No clinical validation.

## Next steps
- Replace regex with a clinical NER model for PHI.
- Build a labelled synthetic note set and report precision and recall.
- Add an evaluation script and a bias check across note styles.
- Add audit logging.

## Evaluate
```bash
python eval.py --json results.json
```
Scores PHI scrubbing, task extraction and red-flag screening against
`data/synthetic_notes.jsonl` (12 hand-written notes).

**Read the numbers with care.** The set is tiny and written by the same
person who wrote the rules. Use it as a regression test, not as proof of
real-world performance. Known failures are visible in the error list:
negated red flags ("denies chest pain"), names without a title, and
non-standard phone formats.
