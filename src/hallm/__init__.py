"""hallm — weight-sharing comparison harness for small GPT-style language models.

Four experimental arms (see ROADMAP.md §3):
    A0  none      — standard decoder-only GPT baseline
    A1  ALBERT    — cross-layer sharing: one block reused across all L layers (depth axis)
    A2  HaLViT    — intra-layer W+Wᵀ sharing within each layer (width axis)
    A3  combined  — both axes simultaneously

All arms train from scratch under a matched compute budget so any perplexity difference is
attributable to the sharing scheme alone. NO real training is launched by the overnight loop;
the harness is smoke-tested on CPU/tiny configs only.
"""

__version__ = "0.1.0"
