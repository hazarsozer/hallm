---
type: model
title: "ALBERT"
full_name: "A Lite BERT"
authors: ["Zhenzhong Lan", "Mingda Chen", "Sebastian Goodman", "Kevin Gimpel", "Piyush Sharma", "Radu Soricut"]
institution: "Google Research"
venue: "ICLR 2020"
source_page: "[[sources/1909.11942]]"
---

## What It Is

ALBERT is a parameter-efficient variant of BERT that applies two compression techniques: **cross-layer parameter sharing** (same weights across all transformer layers) and **factorized embedding parameterization** (decoupling vocabulary embedding size E from hidden size H). It also replaces BERT's NSP pretraining loss with Sentence Order Prediction (SOP).

## Model Variants

| Model | Params | Layers | H | E |
|-------|--------|--------|---|---|
| ALBERT-base | 12M | 12 | 768 | 128 |
| ALBERT-large | 18M | 24 | 1024 | 128 |
| ALBERT-xlarge | 60M | 24 | 2048 | 128 |
| ALBERT-xxlarge | 235M | 12 | 4096 | 128 |

ALBERT-large: **18× fewer parameters than BERT-large** (18M vs 334M).

## Key Numbers

- GLUE Avg (ALBERT-xxlarge, single model): 88.0
- GLUE Avg (ALBERT ensemble): **89.4** — SOTA at ICLR 2020
- RACE: 89.4% (vs BERT-large 73.9%)
- SQuAD 2.0 F1: 88.1

## Relation to HaLViT

ALBERT shares across *layers*; HaLViT shares across *W and Wᵀ within* a layer. Complementary, not competing. See [[sources/1909.11942]] and [[sources/halvit]] for the full comparison.

Referenced by: [[concepts/weight-sharing]], [[sources/1909.11942]]
