#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Parse continual learning logs like:
[20:57:12] Task 5/5 | Avg Acc: 58.10% | Class-IL: [43.1, 26.75, 52.9, 71.1, 96.65]

Outputs per run:
- final_avg_acc
- forgetting_mean (mean over tasks 1..T-1): max_t A[k,t] - A[k,T]
- bwt (mean over tasks 1..T-1): A[k,T] - A[k,k]
Also optionally saves accuracy matrix as CSV.

Usage examples:
  python compute_forgetting.py --inputs logs/er/log.txt logs/scope/log.txt
  python compute_forgetting.py --inputs logs_dir --recursive
  python compute_forgetting.py --inputs logs_dir --recursive --save-matrix

Author: for your SCOPE paper utilities
"""

from __future__ import annotations
import argparse
import csv
import glob
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


LINE_RE = re.compile(
    r"""
    Task\s+(?P<t>\d+)\s*/\s*(?P<T>\d+)        # Task t/T
    \s*\|\s*Avg\s*Acc:\s*(?P<avg>[-+]?\d+(\.\d+)?)\s*%   # Avg Acc: xx.xx%
    \s*\|\s*Class-IL:\s*\[(?P<cil>[^\]]*)\]  # Class-IL: [...]
    """,
    re.VERBOSE
)


def _safe_float(x: str) -> Optional[float]:
    x = x.strip()
    if not x:
        return None
    try:
        return float(x)
    except Exception:
        return None


def parse_log_file(path: str) -> Tuple[int, List[Optional[float]], List[List[float]]]:
    """
    Returns:
      T: number of tasks
      avg_acc_per_step: list length T (final is last)
      class_il: list of length T, each entry is list length t with per-task acc at stage t
    """
    avg_acc_per_step: Dict[int, float] = {}
    class_il: Dict[int, List[float]] = {}
    T_seen: Optional[int] = None

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = LINE_RE.search(line)
            if not m:
                continue

            t = int(m.group("t"))
            T = int(m.group("T"))
            T_seen = T_seen or T

            avg = float(m.group("avg"))
            cil_str = m.group("cil").strip()

            # Split numbers inside Class-IL: [...]
            # Handles "91.14999999999999" etc.
            cil_vals: List[float] = []
            if cil_str:
                parts = [p.strip() for p in cil_str.split(",")]
                for p in parts:
                    v = _safe_float(p)
                    if v is not None:
                        cil_vals.append(v)

            # Basic sanity: at stage t, should have length t
            if len(cil_vals) != t:
                # Some logs might format differently; still keep but warn-ish by padding/truncation
                if len(cil_vals) > t:
                    cil_vals = cil_vals[:t]
                else:
                    # pad with NaN-like values
                    cil_vals = cil_vals + [float("nan")] * (t - len(cil_vals))

            avg_acc_per_step[t] = avg
            class_il[t] = cil_vals

    if T_seen is None:
        raise ValueError(f"No valid lines found in: {path}")

    T = T_seen
    # Build ordered lists
    avg_list: List[Optional[float]] = [None] * T
    cil_list: List[List[float]] = [[] for _ in range(T)]
    for t in range(1, T + 1):
        avg_list[t - 1] = avg_acc_per_step.get(t, None)
        cil_list[t - 1] = class_il.get(t, [])

    return T, avg_list, cil_list


def build_acc_matrix(T: int, cil_list: List[List[float]]) -> List[List[Optional[float]]]:
    """
    A[k][t] = accuracy on task k after training stage t
    k,t are 1-indexed conceptually, matrix stored 0-indexed.

    cil_list[t-1] is list of length t: [A[1,t], A[2,t], ..., A[t,t]]
    """
    A: List[List[Optional[float]]] = [[None for _ in range(T)] for _ in range(T)]
    for t in range(1, T + 1):
        vals = cil_list[t - 1]
        for k in range(1, min(t, len(vals)) + 1):
            A[k - 1][t - 1] = vals[k - 1]
    return A


def compute_forgetting_and_bwt(A: List[List[Optional[float]]]) -> Tuple[Optional[float], Optional[float]]:
    """
    Forgetting:
      F_k = max_{t >= k} A[k,t] - A[k,T], averaged over k=1..T-1
    BWT:
      BWT = mean_{k=1..T-1} (A[k,T] - A[k,k])
    """
    T = len(A)
    if T <= 1:
        return None, None

    forgetting_vals: List[float] = []
    bwt_vals: List[float] = []

    for k in range(1, T):  # 1..T-1
        row = A[k - 1]
        final = row[T - 1]
        if final is None:
            continue

        # max over t=k..T
        hist = [row[t - 1] for t in range(k, T + 1) if row[t - 1] is not None]
        if hist:
            f_k = max(hist) - final
            forgetting_vals.append(f_k)

        diag = row[k - 1]  # A[k,k]
        if diag is not None:
            bwt_vals.append(final - diag)

    forgetting_mean = sum(forgetting_vals) / len(forgetting_vals) if forgetting_vals else None
    bwt_mean = sum(bwt_vals) / len(bwt_vals) if bwt_vals else None
    return forgetting_mean, bwt_mean


def save_matrix_csv(out_path: str, A: List[List[Optional[float]]]) -> None:
    T = len(A)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        header = ["task\\stage"] + [f"t{t}" for t in range(1, T + 1)]
        w.writerow(header)
        for k in range(1, T + 1):
            row = [f"k{k}"] + [("" if A[k - 1][t - 1] is None else f"{A[k - 1][t - 1]:.6g}") for t in range(1, T + 1)]
            w.writerow(row)


def gather_input_files(inputs: List[str], recursive: bool) -> List[str]:
    files: List[str] = []
    for p in inputs:
        if os.path.isdir(p):
            if recursive:
                for ext in ("*.txt", "*.log"):
                    files.extend(glob.glob(os.path.join(p, "**", ext), recursive=True))
            else:
                for ext in ("*.txt", "*.log"):
                    files.extend(glob.glob(os.path.join(p, ext)))
        else:
            files.append(p)
    # de-dup while keeping order
    seen = set()
    out = []
    for f in files:
        af = os.path.abspath(f)
        if af not in seen and os.path.isfile(af):
            seen.add(af)
            out.append(af)
    return out


@dataclass
class RunResult:
    path: str
    T: int
    final_avg_acc: Optional[float]
    forgetting_mean: Optional[float]
    bwt: Optional[float]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True,
                    help="One or more log files or directories.")
    ap.add_argument("--recursive", action="store_true",
                    help="If an input is a directory, search recursively for *.txt/*.log.")
    ap.add_argument("--out", default="cl_metrics.csv",
                    help="Output CSV path (default: cl_metrics.csv).")
    ap.add_argument("--save-matrix", action="store_true",
                    help="Also save per-run accuracy matrix CSV next to output.")
    args = ap.parse_args()

    log_files = gather_input_files(args.inputs, args.recursive)
    if not log_files:
        raise SystemExit("No log files found. Check --inputs / --recursive.")

    results: List[RunResult] = []

    for lf in log_files:
        try:
            T, avg_list, cil_list = parse_log_file(lf)
            A = build_acc_matrix(T, cil_list)
            forgetting_mean, bwt = compute_forgetting_and_bwt(A)
            final_avg = avg_list[-1] if avg_list and avg_list[-1] is not None else None

            results.append(RunResult(
                path=lf,
                T=T,
                final_avg_acc=final_avg,
                forgetting_mean=forgetting_mean,
                bwt=bwt
            ))

            if args.save_matrix:
                base = os.path.splitext(os.path.basename(lf))[0]
                mat_out = os.path.join(os.path.dirname(os.path.abspath(args.out)),
                                       "acc_matrices",
                                       f"{base}_acc_matrix.csv")
                save_matrix_csv(mat_out, A)

        except Exception as e:
            print(f"[WARN] Failed on {lf}: {e}")

    # Write summary CSV
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", "T", "final_avg_acc", "forgetting_mean", "bwt"])
        for r in results:
            w.writerow([
                r.path,
                r.T,
                "" if r.final_avg_acc is None else f"{r.final_avg_acc:.6g}",
                "" if r.forgetting_mean is None else f"{r.forgetting_mean:.6g}",
                "" if r.bwt is None else f"{r.bwt:.6g}",
            ])

    print(f"Saved summary to: {os.path.abspath(args.out)}")
    if args.save_matrix:
        print(f"Saved matrices under: {os.path.join(os.path.dirname(os.path.abspath(args.out)), 'acc_matrices')}")


if __name__ == "__main__":
    main()
