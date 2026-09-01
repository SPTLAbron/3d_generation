from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent


SCRIPTS = [
    "src/analysis/evaluate_ae.py",
    "src/analysis/interpolate.py",
    "src/analysis/latent_probe.py",
    "src/analysis/latent_edit.py",
    "src/analysis/measure_latent_edits.py",
    "src/analysis/shape_optimization.py",
    "src/analysis/check_vae_latent.py",
    "src/analysis/compare_ae_vae.py",
    "src/analysis/summarize_results.py",
    "src/analysis/environment_report.py",
    "src/analysis/validate_submission.py",
]


def run(script):
    path = ROOT / script

    if not path.exists():
        print(f"\n[SKIP] {script} does not exist")
        return

    print("\n" + "=" * 72)
    print(f"RUNNING: {script}")
    print("=" * 72)

    subprocess.run(
        [sys.executable, str(path)],
        cwd=ROOT,
        check=True,
    )


def main():
    for script in SCRIPTS:
        run(script)

    print("\n" + "=" * 72)
    print("ANALYSIS PIPELINE COMPLETE")
    print("=" * 72)


if __name__ == "__main__":
    main()