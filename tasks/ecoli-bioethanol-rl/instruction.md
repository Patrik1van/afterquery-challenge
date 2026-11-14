# Task: Fed-Batch *E. coli* Bioethanol Bioreactor Control

You are a bioprocess control engineer operating an engineered *Escherichia coli* fed-batch bioreactor producing bioethanol (`EtOH`). Your objective is to maximize terminal net profit over a 36-hour batch (`Difficulty Tier 3`) without killing the culture (`viability >= 0.20`).

## How to Interact with the Bioreactor
A CLI controller `/workspace/run_step.py` is available in your working directory.

1. **Initialize / Inspect Current State:**
   ```bash
   python3 /workspace/run_step.py --status
