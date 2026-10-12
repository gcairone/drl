import argparse
import json
import os
import shutil
import sys
from datetime import datetime

import torch

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from scripts.eval_policy import load_agent, run_episodes, summarize

BEST_CKPT = "best.pth"
BEST_META = "best.json"


def evaluate(path: str, episodes: int, max_steps: int, noise: float, seed: int) -> dict:
    """Deterministic evaluation under the fixed promotion protocol."""
    torch.manual_seed(seed)
    agent, ckpt, env_kwargs = load_agent(path)
    res = run_episodes(agent, env_kwargs, noise, False, episodes, max_steps, seed)
    s = summarize(res, max_steps)
    return {"epoch": ckpt.get("epoch"), "mean_standing_time": round(float(s["mean"]), 3),
            "sem": round(float(s["sem"]), 3), "median": round(float(s["median"]), 3),
            "survival": round(float(s["surv"]), 3), "return": round(float(s["ret"]), 1)}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Promote a checkpoint to pretrained/best.pth if it beats the current best.")
    parser.add_argument("--candidate", type=str, required=True, help="Checkpoint to consider")
    parser.add_argument("--best_dir", type=str, default="pretrained", help="Versioned folder holding the best checkpoint")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--max_steps", type=int, default=3000, help="Episode cap (control steps); long enough not to saturate")
    parser.add_argument("--noise", type=float, default=0.01, help="Reset noise scale")
    parser.add_argument("--seed", type=int, default=12345)
    args = parser.parse_args()

    protocol = {"episodes": args.episodes, "max_steps": args.max_steps, "noise": args.noise,
                "seed": args.seed, "mode": "deterministic"}
    best_path = os.path.join(args.best_dir, BEST_CKPT)
    meta_path = os.path.join(args.best_dir, BEST_META)

    cand = evaluate(args.candidate, args.episodes, args.max_steps, args.noise, args.seed)
    print(f"candidate {args.candidate}: {cand}")

    promote = True
    if os.path.isfile(best_path):
        # re-evaluate the current best under the same protocol, so the comparison stays fair
        # even if the environment or the evaluation changed since it was promoted
        best = evaluate(best_path, args.episodes, args.max_steps, args.noise, args.seed)
        print(f"current best {best_path}: {best}")
        promote = cand["mean_standing_time"] > best["mean_standing_time"]

    if not promote:
        print("Candidate is not better: best checkpoint unchanged.")
        return

    os.makedirs(args.best_dir, exist_ok=True)
    shutil.copyfile(args.candidate, best_path)
    meta = {"source": args.candidate, "promoted_at": datetime.now().isoformat(timespec="seconds"),
            "protocol": protocol, "metrics": cand}
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
        f.write("\n")
    print(f"Promoted to {best_path} (metadata in {meta_path}).")


if __name__ == "__main__":
    main()
