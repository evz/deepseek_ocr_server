#!/usr/bin/env python3
"""
Prove the incremental n-gram processor bans exactly what the stock one does.

server.py replaces transformers' NoRepeatNGramLogitsProcessor with an
incremental version because the stock one rebuilds its whole dictionary every
decode step, which dominates CPU time on long generations. That is only a
safe trade if the two ban identical tokens at every step - otherwise the
model emits different text.

This checks that directly, against the real transformers implementation, over
whole simulated generations. It needs torch and transformers but no GPU and
no model weights, so it runs anywhere the server is installed:

    python test_ngram.py
"""
import random
import sys

import torch
from transformers import NoRepeatNGramLogitsProcessor

from ocr_engine import IncrementalNoRepeatNGramLogitsProcessor

VOCAB = 256


def banned_set(processor, tokens):
    """Run one processor step and read back which tokens it banned."""
    input_ids = torch.tensor([tokens], dtype=torch.long)
    scores = torch.zeros((1, VOCAB), dtype=torch.float)
    out = processor(input_ids, scores)
    return set(torch.nonzero(torch.isinf(out[0]) & (out[0] < 0)).flatten().tolist())


def compare(name, ngram_size, prompt, steps, vocab, seed):
    """Step a generation forward token by token, comparing bans each step."""
    rng = random.Random(seed)
    incremental = IncrementalNoRepeatNGramLogitsProcessor(ngram_size)
    tokens = list(prompt)
    mismatches = 0

    for _ in range(steps):
        tokens.append(rng.randrange(vocab))
        # The stock processor is stateless, so a fresh one per step is exactly
        # what generate() effectively does.
        expected = banned_set(NoRepeatNGramLogitsProcessor(ngram_size), tokens)
        actual = banned_set(incremental, tokens)
        if expected != actual:
            mismatches += 1
            if mismatches == 1:
                print(f"    first mismatch at length {len(tokens)}: "
                      f"expected {sorted(expected)}, got {sorted(actual)}")

    status = 'ok  ' if not mismatches else 'FAIL'
    print(f"  {status} {name}: {steps} steps, {mismatches} mismatches")
    return mismatches


def check_reset(ngram_size=20):
    """A second generation must not inherit the first one's n-grams."""
    processor = IncrementalNoRepeatNGramLogitsProcessor(ngram_size)
    for length in range(1, 101):
        banned_set(processor, [5] * length)
    processor.reset()

    mismatches = 0
    for length in range(1, 41):
        tokens = [9] * length
        expected = banned_set(NoRepeatNGramLogitsProcessor(ngram_size), tokens)
        if banned_set(processor, tokens) != expected:
            mismatches += 1
    print(f"  {'ok  ' if not mismatches else 'FAIL'} reset between generations: "
          f"{mismatches} mismatches")
    return mismatches


def check_self_heal(ngram_size=20):
    """Even without reset(), a sequence that does not continue the previous
    one must be detected and rebuilt from scratch."""
    processor = IncrementalNoRepeatNGramLogitsProcessor(ngram_size)
    for length in range(1, 101):
        banned_set(processor, [5] * length)

    mismatches = 0
    for length in range(1, 41):
        tokens = [9] * length
        expected = banned_set(NoRepeatNGramLogitsProcessor(ngram_size), tokens)
        if banned_set(processor, tokens) != expected:
            mismatches += 1
    print(f"  {'ok  ' if not mismatches else 'FAIL'} self-heals without reset(): "
          f"{mismatches} mismatches")
    return mismatches


def main():
    print("incremental vs stock NoRepeatNGramLogitsProcessor:")
    failures = 0

    # A tiny vocabulary forces constant n-gram collisions, so the banning
    # path is exercised on nearly every step rather than almost never.
    failures += compare("degenerate (vocab=3)", 20, [1, 2, 3] * 10, 400, 3, 1)
    # The failure mode that motivated the generation bounds: counting upward
    # never repeats an n-gram, so nothing is ever banned.
    failures += compare("counting loop (vocab=12)", 20, list(range(12)) * 5, 400, 12, 2)
    failures += compare("realistic text (vocab=200)", 20,
                        [random.Random(9).randrange(200) for _ in range(300)],
                        400, 200, 3)
    failures += compare("prompt shorter than ngram_size", 20, [7], 60, 5, 4)
    # 35 is what infer() passes on the eval path.
    failures += compare("ngram_size=35", 35, [1, 2] * 30, 300, 4, 5)
    failures += check_reset()
    failures += check_self_heal()

    print()
    if failures:
        print(f"FAILED: {failures} mismatching steps")
        return 1
    print("PASS: identical bans at every step")
    return 0


if __name__ == '__main__':
    sys.exit(main())
