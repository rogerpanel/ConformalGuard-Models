"""Score R-Judge records with a local Hugging Face chat model as the guard (GPU).

One forward pass per unit, no generation: the model is asked for a single letter
(A safe / B unsafe) and the score is logit(B) - logit(A) at the answer position.

    python scripts/rjudge_judge_local.py --rjudge-dir R-Judge --model Qwen/Qwen2.5-7B-Instruct \
        --mode causal --out runs/guard_qwen7b_causal

Add --load-4bit on a single 16 GB GPU (pip install bitsandbytes).  Writes
<out>/scores.jsonl; an interrupted run resumes.  The scoring arithmetic is unit-tested
with a stand-in model; loading a real model has not been executed in the environment
where this was written.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from conformalguard.rjudge import guards as G  # noqa: E402


def letter_ids(tok) -> tuple[list[int], list[int]]:
    """Token ids that encode the single letters A and B (with and without a leading space)."""
    def ids(ch):
        out = set()
        for s in (ch, " " + ch):
            enc = tok.encode(s, add_special_tokens=False)
            if len(enc) == 1:
                out.add(enc[0])
        return sorted(out)
    a, b = ids("A"), ids("B")
    if not a or not b:
        raise SystemExit("this tokenizer does not encode 'A' and 'B' as single tokens")
    return a, b


def score_unit(tok, model, rec: dict, step: int, a_ids, b_ids, max_input_tokens: int) -> float:
    import torch

    messages = [{"role": "system", "content": G.INSTRUCTIONS},
                {"role": "user", "content": G.user_prompt(rec, step, G.LETTER_FORMAT)}]
    enc = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
    ids = enc if hasattr(enc, "shape") else enc["input_ids"]
    if ids.dim() == 1:
        ids = ids.unsqueeze(0)
    ids = ids[:, -max_input_tokens:].to(model.device)
    with torch.no_grad():
        logits = model(input_ids=ids, attention_mask=torch.ones_like(ids)).logits[0, -1].float()
    return float(torch.logsumexp(logits[b_ids], 0) - torch.logsumexp(logits[a_ids], 0))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rjudge-dir", required=True)
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--mode", choices=("full", "causal"), default="causal")
    ap.add_argument("--load-4bit", action="store_true")
    ap.add_argument("--max-input-tokens", type=int, default=6000)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    recs = G.load_records(args.rjudge_dir)
    if args.limit:
        recs = recs[:args.limit]
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    path = out / "scores.jsonl"
    done = G.read_done(path)
    todo = [(r, s) for r, s in G.units(recs, args.mode) if (int(r["id"]), s) not in done]
    print(f"{args.model}, mode {args.mode}: {len(done)} already scored, {len(todo)} to go")
    if not todo:
        return
    tok = AutoTokenizer.from_pretrained(args.model)
    kw = {"device_map": "auto"}
    if args.load_4bit:
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
        model = AutoModelForCausalLM.from_pretrained(args.model, **kw)
    else:
        half = torch.float16 if torch.cuda.is_available() else torch.float32
        try:
            model = AutoModelForCausalLM.from_pretrained(args.model, dtype=half, **kw)
        except TypeError:
            model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=half, **kw)
    model.eval()
    a_ids, b_ids = letter_ids(tok)
    t0 = time.time()
    with path.open("a", encoding="utf-8") as fh:
        for i, (rec, step) in enumerate(todo, 1):
            s = score_unit(tok, model, rec, step, a_ids, b_ids, args.max_input_tokens)
            fh.write(json.dumps({"id": int(rec["id"]), "step": step, "guard": args.model,
                                 "mode": args.mode, "score": s}) + "\n")
            fh.flush()
            if i <= 3 or i % 100 == 0 or i == len(todo):
                el = time.time() - t0
                print(f"  {i}/{len(todo)}  {el:.0f}s elapsed, about {el / i * (len(todo) - i) / 60:.0f} min left", flush=True)
    print(f"done: {path}")


if __name__ == "__main__":
    main()
