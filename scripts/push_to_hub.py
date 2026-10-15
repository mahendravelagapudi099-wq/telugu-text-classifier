"""Push trained models and Gradio demo Space to Hugging Face Hub.

Safety Rules:
- Reads HF_TOKEN exclusively from the environment (or passed securely via CLI argument).
- Never hardcodes or logs tokens to disk.
- Validates model checkpoint files before uploading.
"""

import argparse
import os
import sys
from pathlib import Path
from huggingface_hub import HfApi, create_repo

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


def parse_args():
    parser = argparse.ArgumentParser(description="Upload models and spaces to Hugging Face Hub")
    parser.add_argument(
        "--model-dir",
        type=str,
        default=None,
        help="Path to local trained model directory (e.g., outputs/IndicBERTv2-MLM-only_seed123/best_model)",
    )
    parser.add_argument(
        "--model-repo",
        type=str,
        default=None,
        help="Target Hugging Face Model repo ID (e.g., 'username/indicbertv2-telugu-sentiment')",
    )
    parser.add_argument(
        "--space-dir",
        type=str,
        default="deployment/hf_space",
        help="Path to Hugging Face Space folder (default: deployment/hf_space)",
    )
    parser.add_argument(
        "--space-repo",
        type=str,
        default=None,
        help="Target Hugging Face Space repo ID (e.g., 'username/telugu-sentiment-demo')",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate parameters and local files without performing network upload",
    )
    return parser.parse_args()


def push_model(api: HfApi, model_dir: Path, repo_id: str, dry_run: bool = False):
    print(f"\n📦 Preparing model upload from: {model_dir}")
    print(f"🎯 Target Model Repo: {repo_id}")

    if not model_dir.exists() or not model_dir.is_dir():
        print(f"❌ Error: Model directory '{model_dir}' does not exist.")
        return False

    required_files = ["config.json"]
    for req in required_files:
        if not (model_dir / req).exists():
            print(f"❌ Error: Required model file '{req}' missing from {model_dir}")
            return False

    files = list(model_dir.iterdir())
    print(f"Found {len(files)} files to upload: {[f.name for f in files]}")

    if dry_run:
        print("✅ [DRY RUN] Model files validated. Skipping upload.")
        return True

    print("Creating or verifying model repository on Hugging Face...")
    create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)

    print("Uploading folder...")
    api.upload_folder(
        folder_path=str(model_dir),
        repo_id=repo_id,
        repo_type="model",
    )
    print(f"🎉 Model successfully uploaded to: https://huggingface.co/{repo_id}")
    return True


def push_space(api: HfApi, space_dir: Path, space_id: str, dry_run: bool = False):
    print(f"\n🚀 Preparing Space upload from: {space_dir}")
    print(f"🎯 Target Space Repo: {space_id}")

    if not space_dir.exists() or not space_dir.is_dir():
        print(f"❌ Error: Space directory '{space_dir}' does not exist.")
        return False

    readme_path = space_dir / "README.md"
    app_path = space_dir / "app.py"
    if not readme_path.exists():
        print(f"❌ Error: Space README.md missing from {space_dir}")
        return False
    if not app_path.exists():
        print(f"❌ Error: Space app.py missing from {space_dir}")
        return False

    if dry_run:
        print("✅ [DRY RUN] Space files validated. Skipping upload.")
        return True

    print("Creating or verifying Space repository on Hugging Face...")
    create_repo(repo_id=space_id, repo_type="space", space_sdk="gradio", exist_ok=True)

    print("Uploading Space files...")
    api.upload_folder(
        folder_path=str(space_dir),
        repo_id=space_id,
        repo_type="space",
    )
    print(f"🎉 Space successfully deployed to: https://huggingface.co/spaces/{space_id}")
    return True


def main():
    args = parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token and not args.dry_run:
        print("⚠️ Warning: Environment variable HF_TOKEN is not set.")
        print("To push to Hugging Face Hub, set your token first:")
        print("  $env:HF_TOKEN=\"hf_your_token_here\"  # On Windows PowerShell")
        print("  export HF_TOKEN=\"hf_your_token_here\" # On Linux / Mac")
        print("\nAlternatively, run with --dry-run to test configuration.")
        sys.exit(1)

    api = HfApi(token=token) if token else None

    actions_performed = 0

    if args.model_repo:
        if not args.model_dir:
            print("❌ Error: --model-repo specified but --model-dir is missing.")
            sys.exit(1)
        success = push_model(api, Path(args.model_dir), args.model_repo, dry_run=args.dry_run)
        if success:
            actions_performed += 1

    if args.space_repo:
        success = push_space(api, Path(args.space_dir), args.space_repo, dry_run=args.dry_run)
        if success:
            actions_performed += 1

    if actions_performed == 0:
        print("\nℹ️ No repositories specified to upload.")
        print("Example usages:")
        print("  python scripts/push_to_hub.py --dry-run --space-repo username/telugu-sentiment-space")
        print("  python scripts/push_to_hub.py --model-dir outputs/best_model --model-repo username/indicbertv2-telugu")
        print("  python scripts/push_to_hub.py --space-repo username/telugu-sentiment-space")


if __name__ == "__main__":
    main()
