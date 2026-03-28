"""K8s hypothesis sweep orchestrator.

Reads hypothesis parameter grids from ConfigMap JSON files (or local files in
k8s/hypothesis-validation/), generates the Cartesian product of parameters,
and submits one K8s Job per combination.

Usage:
    # Dry-run: print all Jobs that would be created
    python -m quant.scripts.k8s_hypothesis_sweep --dry-run

    # Submit all Jobs to the cluster
    python -m quant.scripts.k8s_hypothesis_sweep --submit

    # Submit a single hypothesis only
    python -m quant.scripts.k8s_hypothesis_sweep --submit --hypothesis regime_adaptive_momentum

    # Wait for all submitted Jobs to finish and print a summary
    python -m quant.scripts.k8s_hypothesis_sweep --submit --wait

Options:
    --dry-run            print generated Job manifests, do not apply
    --submit             create Jobs via kubectl apply
    --wait               after submitting, poll until all Jobs complete (blocking)
    --namespace NS       K8s namespace (default: hypothesis-validation)
    --hypothesis NAME    run only this hypothesis (repeatable)
    --start-date DATE    YYYY-MM-DD (default: 2020-01-01)
    --end-date DATE      YYYY-MM-DD (default: 2025-12-31)
    --is-window N        in-sample window days (default: 252)
    --oos-window N       OOS window days (default: 63)
    --step-size N        step size days (default: 63)
    --results-path PATH  results PVC path (default: /results)
    --params-dir PATH    directory containing <hypothesis>.json sweep configs
                         (default: k8s/hypothesis-validation/ relative to repo root)
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from loguru import logger


# Default parameter grid files live next to the K8s manifests
_REPO_ROOT = Path(__file__).parent.parent.parent
_DEFAULT_PARAMS_DIR = _REPO_ROOT / "k8s" / "hypothesis-validation"
_JOB_TEMPLATE = _DEFAULT_PARAMS_DIR / "backtest-job-template.yaml"

# All supported hypothesis names
_ALL_HYPOTHESES = [
    "regime_adaptive_momentum",
    "ic_weighted_ensemble",
    "statistical_arbitrage_pairs",
    "hmm_regime_detection",
    "volatility_crush",
]

# The HMM hypothesis benefits from GPU nodes
_GPU_HYPOTHESES = {"hmm_regime_detection"}


def _cartesian_grid(param_grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
    """Return the Cartesian product of all parameter lists as a list of dicts."""
    if not param_grid:
        return [{}]
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


def _job_name(hypothesis: str, params: dict[str, Any]) -> str:
    """Generate a deterministic, DNS-safe job name for a (hypothesis, params) pair."""
    params_str = json.dumps(params, sort_keys=True)
    digest = hashlib.sha256(params_str.encode()).hexdigest()[:6]
    hyp_short = hypothesis.replace("_", "-")[:24]
    return f"hyp-{hyp_short}-{digest}"


def _render_job(
    hypothesis: str,
    params: dict[str, Any],
    job_name: str,
    args: argparse.Namespace,
    template_text: str,
) -> str:
    """Fill in the Job template with specific hypothesis and param values."""
    use_gpu = hypothesis in _GPU_HYPOTHESES

    gpu_selector_block = """
      nodeSelector:
        node.kubernetes.io/instance-type: jetson-orin-nano
      tolerations:
        - key: nvidia.com/gpu
          operator: Exists
          effect: NoSchedule""" if use_gpu else ""

    gpu_image = "mirror.registry:5000/quant-rs:gpu-r36.4.0" if use_gpu else "mirror.registry:5000/quant-rs:latest"

    rendered = (
        template_text
        .replace("${JOB_NAME}", job_name)
        .replace("${HYPOTHESIS_NAME}", hypothesis)
        .replace("${RUN_PARAMS_JSON}", json.dumps(params).replace("'", "'\\''"))
        .replace("${START_DATE}", args.start_date)
        .replace("${END_DATE}", args.end_date)
        .replace("${IS_WINDOW}", str(args.is_window))
        .replace("${OOS_WINDOW}", str(args.oos_window))
        .replace("${STEP_SIZE}", str(args.step_size))
        .replace("mirror.registry:5000/quant-rs:latest", gpu_image)
    )

    # Inject GPU node selector if needed
    if use_gpu and gpu_selector_block:
        rendered = rendered.replace(
            "      volumes:",
            f"{gpu_selector_block}\n      volumes:",
            1,
        )

    return rendered


def _kubectl_apply(manifest_text: str, dry_run: bool) -> bool:
    """Apply a manifest via kubectl. Returns True on success."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(manifest_text)
        tmp_path = f.name

    cmd = ["kubectl", "apply", "-f", tmp_path]
    if dry_run:
        cmd.append("--dry-run=client")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        logger.info(result.stdout.strip())
        return True
    except subprocess.CalledProcessError as exc:
        logger.error("kubectl apply failed:\n{}", exc.stderr)
        return False
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def _wait_for_jobs(job_names: list[str], namespace: str, timeout_sec: int = 14400) -> dict[str, str]:
    """Poll until all jobs reach a terminal state. Returns {job_name: 'succeeded'|'failed'}."""
    remaining = set(job_names)
    states: dict[str, str] = {}
    deadline = time.monotonic() + timeout_sec
    poll_interval = 30

    while remaining and time.monotonic() < deadline:
        for job_name in list(remaining):
            try:
                out = subprocess.check_output(
                    ["kubectl", "get", "job", job_name, "-n", namespace,
                     "-o", "jsonpath={.status.conditions[*].type}"],
                    text=True, stderr=subprocess.DEVNULL,
                )
                if "Complete" in out:
                    states[job_name] = "succeeded"
                    remaining.discard(job_name)
                elif "Failed" in out:
                    states[job_name] = "failed"
                    remaining.discard(job_name)
            except subprocess.CalledProcessError:
                pass

        if remaining:
            logger.info("Waiting for {} jobs to complete...", len(remaining))
            time.sleep(poll_interval)

    for job_name in remaining:
        states[job_name] = "timeout"

    return states


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true", help="Print manifests, do not apply")
    parser.add_argument("--submit", action="store_true", help="Submit Jobs to cluster")
    parser.add_argument("--wait", action="store_true", help="Wait for all Jobs to finish")
    parser.add_argument("--namespace", default="hypothesis-validation")
    parser.add_argument("--hypothesis", action="append", dest="hypotheses", metavar="NAME",
                        help="Run only this hypothesis (repeatable)")
    parser.add_argument("--start-date", default="2020-01-01")
    parser.add_argument("--end-date", default="2025-12-31")
    parser.add_argument("--is-window", type=int, default=252)
    parser.add_argument("--oos-window", type=int, default=63)
    parser.add_argument("--step-size", type=int, default=63)
    parser.add_argument("--results-path", default="/results")
    parser.add_argument("--params-dir", type=Path, default=_DEFAULT_PARAMS_DIR)
    args = parser.parse_args()

    if not args.dry_run and not args.submit:
        parser.error("Specify --dry-run or --submit")

    hypotheses = args.hypotheses or _ALL_HYPOTHESES

    # Validate hypothesis names
    invalid = [h for h in hypotheses if h not in _ALL_HYPOTHESES]
    if invalid:
        logger.error("Unknown hypotheses: {}. Valid: {}", invalid, _ALL_HYPOTHESES)
        return 1

    # Load Job template
    if not _JOB_TEMPLATE.exists():
        logger.error("Job template not found: {}", _JOB_TEMPLATE)
        return 1
    template_text = _JOB_TEMPLATE.read_text()

    submitted_jobs: list[str] = []
    total_combinations = 0

    for hypothesis in hypotheses:
        # Load sweep params
        params_file = args.params_dir / f"{hypothesis}.json"
        if params_file.exists():
            sweep_config = json.loads(params_file.read_text())
            param_grid = sweep_config.get("param_grid", {})
        else:
            logger.warning("No param grid found for {} at {}, running with defaults", hypothesis, params_file)
            param_grid = {}

        combinations = _cartesian_grid(param_grid)
        logger.info("Hypothesis {}: {} parameter combinations", hypothesis, len(combinations))
        total_combinations += len(combinations)

        for params in combinations:
            job_name = _job_name(hypothesis, params)
            manifest = _render_job(hypothesis, params, job_name, args, template_text)

            if args.dry_run:
                print(f"\n--- Job: {job_name} ---")
                print(manifest)
            elif args.submit:
                logger.info("Submitting: {}", job_name)
                ok = _kubectl_apply(manifest, dry_run=False)
                if ok:
                    submitted_jobs.append(job_name)
                else:
                    logger.warning("Skipping {} — apply failed", job_name)

    logger.info("Total: {} jobs across {} hypotheses", total_combinations, len(hypotheses))

    if args.submit and submitted_jobs:
        logger.info("Submitted {} jobs", len(submitted_jobs))

        if args.wait:
            logger.info("Waiting for jobs to complete (timeout 4h)...")
            states = _wait_for_jobs(submitted_jobs, args.namespace)

            succeeded = [j for j, s in states.items() if s == "succeeded"]
            failed = [j for j, s in states.items() if s == "failed"]
            timed_out = [j for j, s in states.items() if s == "timeout"]

            logger.info("Results: {} succeeded, {} failed, {} timed out",
                        len(succeeded), len(failed), len(timed_out))

            if failed:
                logger.warning("Failed jobs: {}", failed)
            if timed_out:
                logger.warning("Timed out jobs: {}", timed_out)

            return 0 if not failed and not timed_out else 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
