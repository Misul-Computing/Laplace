#!/usr/bin/env python3
"""Find an optimistic global mixed-precision allocation at three bits."""

import argparse
from collections import Counter, defaultdict

import numpy as np

from prototype_delta import load_trace, relative_error, softmax, stack
from prototype_oracle import quantize_rows


OPTIONS = ((2, 2), (3, 2), (4, 2), (4, 3), (4, 4), (8, 2), (8, 4), (8, 8), (16, 16))


def attention(keys, values, query, scale):
    weights = softmax(np.einsum("td,d->t", keys, query) * scale)
    return np.einsum("t,td->d", weights, values)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("trace")
    parser.add_argument("--target-bits", type=float, default=3.0)
    args = parser.parse_args()
    keys, values, queries = load_trace(args.trace)
    groups = defaultdict(list)
    for record in queries:
        groups[record[0], record[1]].append(record)

    group_results = []
    for (layer, head), records in sorted(groups.items()):
        count = records[0][2]
        first = records[0][3]
        original_k = stack(keys, layer, head, count)[first:]
        original_v = stack(values, layer, head, count)[first:]
        key_variants = {bits: quantize_rows(original_k, bits) for bits in (2, 3, 4, 8)}
        value_variants = {bits: quantize_rows(original_v, bits) for bits in (2, 3, 4, 8)}
        variants = {}
        for key_bits, value_bits in OPTIONS:
            candidate_k = original_k if key_bits == 16 else key_variants[key_bits]
            candidate_v = original_v if value_bits == 16 else value_variants[value_bits]
            errors = []
            for record in records:
                query = record[6].astype(np.float64)
                exact = attention(original_k, original_v, query, record[4])
                candidate = attention(candidate_k, candidate_v, query, record[4])
                errors.append(relative_error(candidate, exact))
            variants[key_bits, value_bits] = np.asarray(errors)
        group_results.append(variants)

    budget = int(round(args.target_bits * 2 * len(group_results)))
    infinity = float("inf")
    costs = {(k, v): k + v for k, v in OPTIONS}
    dynamic = {(0, 0): (0.0, [])}
    for index, variants in enumerate(group_results):
        updated = {}
        for (_, used), (loss, choices) in dynamic.items():
            for option, errors in variants.items():
                candidate_cost = used + costs[option]
                if candidate_cost > budget:
                    continue
                candidate_loss = loss + float(np.sum(errors * errors))
                key = index + 1, candidate_cost
                if candidate_loss < updated.get(key, (infinity, None))[0]:
                    updated[key] = candidate_loss, choices + [option]
        dynamic = updated
    _, choices = min(dynamic.values(), key=lambda item: item[0])
    selected_errors = np.concatenate([
        variants[option] for variants, option in zip(group_results, choices)
    ])
    percent = selected_errors * 100.0
    used = sum(costs[option] for option in choices)
    print(
        f"traces={percent.size} groups={len(groups)} target={args.target_bits:g} "
        f"used={used / (2 * len(groups)):.3f} metadata=free future-query-oracle"
    )
    print(
        f"mean={np.mean(percent):.3f}% p95={np.percentile(percent, 95):.3f}% "
        f"max={np.max(percent):.3f}% pass={np.mean(percent <= 2.0) * 100:.2f}%"
    )
    for option, count in sorted(Counter(choices).items()):
        print(f"K{option[0]}/V{option[1]} groups={count}")


if __name__ == "__main__":
    main()
