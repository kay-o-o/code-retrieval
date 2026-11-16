"""Baseline evaluation for the mini coding agent.

Runs a fixed set of 10 repo-local tasks against the current code and reports
success rate, tool steps, token usage and latency.

Why a baseline first: every later change (semantic retrieval, shell sandboxing)
needs a before-number, otherwise "it got better" is an opinion. The task set
deliberately mixes one-hop lookups with multi-hop questions, because the
interesting failure mode of a coding agent is not "cannot find a string" but
"found it and stopped thinking".

Usage:
    python scripts/eval_baseline.py
    python scripts/eval_baseline.py --provider ollama --max-steps 6
    python scripts/eval_baseline.py --out eval/baseline.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import mini_coding_agent as mca  # noqa: E402


TASK_FILE = ROOT / "eval" / "tasks.json"


def load_tasks():
    """Tasks live in a data file, not in this script.

    Reason, and it is a measured one: with the questions inlined here, the
    semantic index treated this file as the best answer to itself. A Chinese
    query matched the eval script because it contains the question verbatim,
    so the agent read the harness instead of the source. Keeping the task set
    as data also means the index can skip it like any other export.
    """
    return json.loads(TASK_FILE.read_text(encoding="utf-8"))["tasks"]


def judge(answer, must_groups):
    """Pass when every group has at least one hit."""
    low = (answer or "").lower()
    hits = []
    for group in must_groups:
        hit = next((k for k in group if k.lower() in low), None)
        hits.append(hit)
    return all(hits), hits


def run_task(task, args):
    agent = mca.build_agent(args)
    started = time.time()
    error = ""
    answer = ""
    try:
        answer = agent.ask(task["question"])
    except Exception as exc:  # noqa: BLE001 - a crashed task is a failed task
        error = f"{type(exc).__name__}: {exc}"
    elapsed = time.time() - started

    history = agent.session.get("history", [])
    tool_items = [item for item in history if item.get("role") == "tool"]
    usage = getattr(agent.model_client, "total_usage", {}) or {}
    passed, hits = judge(answer, task["must"])

    return {
        "id": task["id"],
        "question": task["question"],
        "note": task["note"],
        "passed": bool(passed and not error),
        "hits": hits,
        "steps": len(tool_items),
        "tools": [item.get("name") for item in tool_items],
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "completion_tokens": usage.get("completion_tokens", 0),
        "latency_s": round(elapsed, 1),
        "answer": (answer or "").strip(),
        "error": error,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run the fixed baseline task set.")
    parser.add_argument("--provider", default="openai", choices=("ollama", "openai"))
    parser.add_argument("--max-steps", type=int, default=6)
    parser.add_argument("--out", default="eval/baseline.json")
    parser.add_argument("--task", default=None, help="Run a single task id, e.g. T05")
    opts = parser.parse_args(argv)

    mca.load_env_file(ROOT / ".env")

    argv_for_agent = [
        "--provider", opts.provider,
        "--cwd", str(ROOT),
        "--approval", "never",
        "--max-steps", str(opts.max_steps),
    ]

    tasks = [t for t in load_tasks() if opts.task is None or t["id"] == opts.task]
    results = []

    print(f"任务数 {len(tasks)} | provider={opts.provider} | max_steps={opts.max_steps}\n")
    print(f"{'任务':<6}{'结果':<8}{'步数':>4}{'token':>8}{'耗时':>8}  说明")
    print("-" * 72)

    for task in tasks:
        agent_args = mca.build_arg_parser().parse_args(argv_for_agent)
        row = run_task(task, agent_args)
        results.append(row)
        flag = "PASS" if row["passed"] else "FAIL"
        tokens = row["prompt_tokens"] + row["completion_tokens"]
        print(
            f"{row['id']:<6}{flag:<8}{row['steps']:>4}{tokens:>8}{row['latency_s']:>7}s  {task['note']}"
        )
        if not row["passed"]:
            print(f"      命中 {row['hits']} | 工具 {row['tools']}")
            print(f"      回答 {row['answer'][:120] or row['error']}")
        sys.stdout.flush()

    total = len(results)
    passed = sum(1 for r in results if r["passed"])
    steps = sum(r["steps"] for r in results) / total
    tokens = sum(r["prompt_tokens"] + r["completion_tokens"] for r in results) / total
    latency = sum(r["latency_s"] for r in results) / total

    print("-" * 72)
    print(
        f"成功率 {passed}/{total} ({passed / total:.0%}) | "
        f"平均步数 {steps:.1f} | 平均 token {tokens:.0f} | 平均耗时 {latency:.1f}s"
    )

    out_path = ROOT / opts.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "provider": opts.provider,
        "max_steps": opts.max_steps,
        "summary": {
            "total": total,
            "passed": passed,
            "success_rate": round(passed / total, 3),
            "avg_steps": round(steps, 2),
            "avg_tokens": round(tokens, 1),
            "avg_latency_s": round(latency, 2),
        },
        "results": results,
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n明细已写入 {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
