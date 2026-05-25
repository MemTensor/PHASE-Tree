#!/usr/bin/env python3
"""Parallel OPPU training dispatcher — one role per GPU concurrently.

Discovers all characters (roles) in the training data that exceed a minimum
sample threshold, then dispatches ``train_mt_lora.py --filter_role`` jobs
across available GPUs using a thread pool.  Each GPU trains one role at a
time; finished GPUs immediately pick up the next queued role.

Output layout::

    <output_dir>/<slugified_role>/adapter_model.safetensors
    <output_dir>/<slugified_role>/adapter_config.json
    <output_dir>/_logs/train_<role>.log

Usage::

    python evaluation/run_oppu_parallel.py \\
        --train_data phase_tree_data/processed/RAIDEN/m1_context_only/train.json \\
        --output_dir phase_tree_models/oppu/RAIDEN \\
        --gpus 0,1,2,3
"""
import argparse, json, os, subprocess, sys, time, re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def slugify(role: str) -> str:
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", role.strip())
    return s.strip("_") or "role"

def list_roles(train_path, min_samples=10):
    data = json.load(open(train_path))
    cnt = Counter(s["role"] for s in data)
    return [(r, n) for r, n in cnt.most_common() if n >= min_samples]

def train_role(gpu_id, ds, role, n_samples, args, task_idx, total):
    slug = slugify(role)
    role_dir = os.path.join(args.output_root, ds, slug)
    log_dir = os.path.join(args.output_root, ds, "_logs")
    os.makedirs(role_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    adapter = os.path.join(role_dir, "adapter_model.safetensors")
    if os.path.exists(adapter) or os.path.exists(os.path.join(role_dir, "adapter_model.bin")):
        print(f"  [GPU{gpu_id}] SKIP [{task_idx}/{total}] {ds}/{role} (exists)", flush=True)
        return 0

    log_path = os.path.join(log_dir, f"train_{slug}.log")
    print(f"  [GPU{gpu_id}] START [{task_idx}/{total}] {ds}/{role} ({n_samples} samples)", flush=True)
    t0 = time.time()

    cmd = [
        sys.executable, os.path.join(PROJECT_ROOT, "evaluation", "train_mt_lora.py"),
        "--train_data", os.path.join(PROJECT_ROOT, "phase_tree_data", "processed", ds, "m1_context_only", "train.json"),
        "--output_dir", role_dir,
        "--filter_role", role,
        "--model", args.model,
        "--lora_r", str(args.lora_r), "--lora_alpha", str(args.lora_alpha),
        "--lora_dropout", "0.05",
        "--epochs", str(args.epochs), "--lr", str(args.lr),
        "--batch_size", str(args.batch_size), "--grad_accum", str(args.grad_accum),
        "--max_seq_len", str(args.max_seq_len), "--seed", "42", "--logging_steps", "10",
    ]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["TRANSFORMERS_VERBOSITY"] = "warning"

    with open(log_path, "w") as f:
        rc = subprocess.call(cmd, env=env, stdout=f, stderr=subprocess.STDOUT)

    elapsed = time.time() - t0
    tag = "✓" if rc == 0 else f"✗ rc={rc}"
    print(f"  [GPU{gpu_id}] DONE  [{task_idx}/{total}] {ds}/{role} {tag} ({elapsed:.0f}s)", flush=True)
    return rc

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gpus", default=None, help="Comma-separated GPU IDs (default: all visible GPUs)")
    p.add_argument("--datasets", default="HPD,TheOffice,StarTrek_TNG,Friends")
    p.add_argument("--output_root", default=os.path.join(PROJECT_ROOT, "phase_tree_models", "oppu"))
    p.add_argument("--model", default="/dev/shm/phase/models/Qwen2.5-7B-Instruct")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--grad_accum", type=int, default=2)
    p.add_argument("--max_seq_len", type=int, default=4096)
    args = p.parse_args()

    if args.gpus:
        gpus = [int(g) for g in args.gpus.split(",")]
    else:
        import subprocess as _sp
        try:
            out = _sp.check_output(
                ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
                text=True
            )
            gpus = [int(line.strip()) for line in out.strip().splitlines() if line.strip()]
        except (FileNotFoundError, _sp.CalledProcessError):
            gpus = [0]
    datasets = [d.strip() for d in args.datasets.split(",")]

    tasks = []
    for ds in datasets:
        train_path = os.path.join(PROJECT_ROOT, "phase_tree_data", "processed", ds, "m1_context_only", "train.json")
        for role, n in list_roles(train_path):
            tasks.append((ds, role, n))

    print(f"\n{'═'*60}")
    print(f"  OPPU Parallel Training")
    print(f"  GPUs    : {gpus} ({len(gpus)} workers)")
    print(f"  Datasets: {datasets}")
    print(f"  Roles   : {len(tasks)} total")
    print(f"  Config  : epochs={args.epochs}, lr={args.lr}, r={args.lora_r}, bs={args.batch_size}×{args.grad_accum}")
    print(f"{'═'*60}\n")

    gpu_lock = {g: Lock() for g in gpus}
    gpu_queue = list(gpus)
    queue_lock = Lock()

    def get_gpu():
        while True:
            with queue_lock:
                if gpu_queue:
                    return gpu_queue.pop(0)
            time.sleep(1)

    def release_gpu(g):
        with queue_lock:
            gpu_queue.append(g)

    t_start = time.time()
    n_ok = n_fail = n_skip = 0
    count_lock = Lock()

    def run_task(idx, ds, role, n_samples):
        nonlocal n_ok, n_fail, n_skip
        gpu = get_gpu()
        try:
            slug = slugify(role)
            role_dir = os.path.join(args.output_root, ds, slug)
            adapter = os.path.join(role_dir, "adapter_model.safetensors")
            if os.path.exists(adapter) or os.path.exists(os.path.join(role_dir, "adapter_model.bin")):
                print(f"  [GPU{gpu}] SKIP [{idx}/{len(tasks)}] {ds}/{role}", flush=True)
                with count_lock:
                    n_skip += 1
                return
            rc = train_role(gpu, ds, role, n_samples, args, idx, len(tasks))
            with count_lock:
                if rc == 0:
                    n_ok += 1
                else:
                    n_fail += 1
        finally:
            release_gpu(gpu)

    with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        futures = []
        for i, (ds, role, n) in enumerate(tasks, 1):
            futures.append(pool.submit(run_task, i, ds, role, n))
        for f in as_completed(futures):
            f.result()

    elapsed = time.time() - t_start
    print(f"\n{'═'*60}")
    print(f"  All OPPU training complete in {elapsed/60:.1f} min")
    print(f"  Trained: {n_ok}, Skipped: {n_skip}, Failed: {n_fail}")
    for ds in datasets:
        ds_dir = os.path.join(args.output_root, ds)
        n = sum(1 for d in os.listdir(ds_dir)
                if os.path.isfile(os.path.join(ds_dir, d, "adapter_model.safetensors"))
                or os.path.isfile(os.path.join(ds_dir, d, "adapter_model.bin")))
        print(f"  {ds}: {n} adapters")
    print(f"{'═'*60}")

if __name__ == "__main__":
    main()
