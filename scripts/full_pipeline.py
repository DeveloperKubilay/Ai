import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def run_step(args: list[str]) -> None:
    command = [sys.executable, *args]
    print(f"\n>>> {' '.join(command)}")
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def file_has_content(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tam veri -> train -> repair -> retrain akisini calistirir.")
    parser.add_argument("--skip-repair", action="store_true", help="Red-team ve repair asamasini atla.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repair_path = PROJECT_ROOT / "data" / "repair_train.jsonl"

    run_step(["scripts/create_data.py"])
    run_step(["scripts/prepare_dataset.py"])
    run_step(["scripts/index.py"])
    run_step(["scripts/quick_test.py"])

    if args.skip_repair:
        return

    run_step(["scripts/redteam_repair.py", "--apply"])
    if file_has_content(repair_path):
        print("\n>>> Repair bulundu, retrain dongusu baslatiliyor")
        run_step(["scripts/prepare_dataset.py"])
        run_step(["scripts/index.py"])
        run_step(["scripts/quick_test.py"])


if __name__ == "__main__":
    main()
