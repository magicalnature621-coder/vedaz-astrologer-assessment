# Vedaz AI Astrologer — Take-Home Assessment

This repo contains my submission for the assessment.

## Contents

- **`Chat_Data_for_assessment_of_applicants.json`** — the original chat dataset provided for the assessment.
- **`1_vllm_vps_hosting_guide.md`** — write-up: process for hosting a fine-tuned model on a VPS using vLLM (server setup, systemd service, nginx + SSL, security checklist).
- **`2_qwen_finetuning_guide.md`** — write-up: process for fine-tuning Qwen2.5/Qwen3 on the provided chat data (data cleaning, QLoRA training script, merging, evaluation checklist).
- **`3_astrologer_training_conversations.jsonl`** — 5 additional manually written user↔astrologer conversations for training purposes, matching the schema and safety philosophy of the original dataset (kundli/dasha analysis, empathetic tone, "please wait while I analyze" pattern, and astrological timing windows rather than fixed guaranteed dates).

## Notes

The 5 new conversations were written to stay consistent with the original dataset's guardrails — e.g. never promising an exact date, redirecting health/legal/safety issues to real-world professionals, and holding that line even under user pushback.
