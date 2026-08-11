"""Checks for the benchmark construction. Run: python -m data.test_hotpot

No model weights needed -- only the Llama tokenizer -- so this runs offline in
seconds. It guards the things that would silently corrupt every downstream
number: gold labelling, gold placement, pool contamination, and whether the
token budget actually binds.
"""

import sys

from transformers import AutoTokenizer

from paths import resolve_model

from .budget import fill_budget
from .hotpot import DistractorPool, InstanceBuilder, load_hotpot, render_context

MODEL = resolve_model("NousResearch/Meta-Llama-3.1-8B-Instruct")

_passed, _failed = 0, 0


def check(cond, msg):
    global _passed, _failed
    if cond:
        _passed += 1
        print(f"[PASS] {msg}")
    else:
        _failed += 1
        print(f"[FAIL] {msg}")


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    ds = load_hotpot("validation")
    pool = DistractorPool(size=3000)
    builder = InstanceBuilder(tok, pool)
    check(len(pool) > 5000, f"pool built ({len(pool)} unique paragraphs)")

    target = 8000
    insts = [builder.build(ds[i], target, seed=i) for i in range(12)]

    # --- length ---------------------------------------------------------
    actual = [i.context_tokens for i in insts]
    rendered = [
        len(tok(render_context(i.chunks), add_special_tokens=False).input_ids)
        for i in insts
    ]
    check(
        all(0.85 * target <= r <= 1.05 * target for r in rendered),
        f"rendered length near target {target}: min {min(rendered)}, max {max(rendered)}",
    )
    drift = max(abs(r - a) / r for r, a in zip(rendered, actual))
    check(drift < 0.15, f"chunk-token sum tracks rendered length (max drift {drift:.1%})")

    # --- gold labelling -------------------------------------------------
    ok = True
    for inst, i in zip(insts, range(12)):
        sf = ds[i]["supporting_facts"]
        want = set(zip(sf["title"], sf["sent_id"]))
        got = {(c.title, c.sent_idx) for c in inst.chunks if c.is_gold}
        ok &= got == want
    check(ok, "gold chunks match supporting_facts exactly on all 12 instances")
    check(
        all(i.n_gold >= 2 for i in insts),
        f"every instance keeps its gold sentences (min {min(i.n_gold for i in insts)})",
    )

    # --- pool contamination ---------------------------------------------
    clean = True
    for inst, i in zip(insts, range(12)):
        native = set(ds[i]["context"]["title"])
        gold_titles = {c.title for c in inst.chunks if c.is_gold}
        titles = [c.title for c in inst.chunks]
        # a gold title must appear in exactly one paragraph slot
        for g in gold_titles:
            paras = {c.para_idx for c in inst.chunks if c.title == g}
            clean &= len(paras) == 1
        clean &= native.issubset(set(titles))
    check(clean, "no pooled paragraph duplicates a gold title, natives all present")

    # --- gold depth is controllable -------------------------------------
    depths = []
    for d in (0.0, 0.5, 1.0):
        inst = builder.build(ds[0], target, gold_depth=d, seed=0)
        n_para = max(c.para_idx for c in inst.chunks) + 1
        pos = min(c.para_idx for c in inst.chunks if c.is_gold) / n_para
        depths.append(pos)
    check(
        depths[0] < 0.15 and 0.35 < depths[1] < 0.65 and depths[2] > 0.85,
        f"gold lands at requested depth: {[round(d, 2) for d in depths]}",
    )
    rand_pos = {
        min(c.para_idx for c in builder.build(ds[0], target, seed=s).chunks if c.is_gold)
        for s in range(8)
    }
    check(len(rand_pos) > 3, f"gold_depth=None spreads gold across seeds ({len(rand_pos)} distinct)")

    # --- budget binds ---------------------------------------------------
    inst = insts[0]
    titles = builder.title_tokens(inst)
    over = []
    for budget in (128, 256, 512, 1024, 2048):
        scores = [-i for i in range(len(inst.chunks))]
        sel = fill_budget(inst.chunks, scores, budget, titles)
        n = len(tok(render_context(sel), add_special_tokens=False).input_ids)
        over.append((budget, n))
    check(
        all(n <= b for b, n in over),
        f"rendered context never exceeds budget: {over}",
    )
    check(
        all(n >= 0.8 * b for b, n in over),
        f"budget is not badly underspent: {[(b, n) for b, n in over]}",
    )

    # --- oracle reaches full recall -------------------------------------
    from eval.metrics import full_support, gold_recall

    oracle_ok = 0
    for inst in insts:
        titles = builder.title_tokens(inst)
        s = [1e6 - i if c.is_gold else -i for i, c in enumerate(inst.chunks)]
        sel = fill_budget(inst.chunks, s, 512, titles)
        oracle_ok += full_support(sel, inst.chunks)
    check(oracle_ok == len(insts), f"oracle achieves full support at 512 tokens ({oracle_ok}/{len(insts)})")

    trunc_recall = []
    for inst in insts:
        titles = builder.title_tokens(inst)
        s = [-i for i in range(len(inst.chunks))]
        sel = fill_budget(inst.chunks, s, 512, titles)
        trunc_recall.append(gold_recall(sel, inst.chunks))
    mean_trunc = sum(trunc_recall) / len(trunc_recall)
    check(mean_trunc < 0.6, f"truncation is a real floor, not a hidden oracle (recall {mean_trunc:.2f})")

    # --- render preserves document order --------------------------------
    text = render_context(insts[0].chunks)
    first_gold = [c for c in insts[0].chunks if c.is_gold][0]
    check(first_gold.text in text, "gold sentence text survives rendering")
    check(text.count(first_gold.title) >= 1, "gold title appears as a header")

    # --- metrics --------------------------------------------------------
    from eval.metrics import exact_match, f1

    check(exact_match("The Beatles", "the beatles.") == 1.0, "EM normalises case/punct")
    check(exact_match("yes", "no") == 0.0, "EM rejects wrong yes/no")
    check(abs(f1("John Smith", "John Smith Jr") - 0.8) < 1e-6, "F1 partial credit correct")
    check(f1("yes", "Marina Bay") == 0.0, "F1 gives yes/no no partial credit")

    print(f"\n{_passed} passed, {_failed} failed")
    return 1 if _failed else 0


if __name__ == "__main__":
    sys.exit(main())
