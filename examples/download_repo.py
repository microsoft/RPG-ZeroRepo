import os
import argparse
import subprocess
from datasets import load_dataset
from tqdm import tqdm

def ensure_mirror(repo, repo_to_top_folder, mirror_root):
    top = repo_to_top_folder[repo]
    mirror = os.path.join(mirror_root, f"{top}.git")
    if not os.path.exists(mirror):
        print(f"[mirror] cloning {repo}")
        subprocess.run(["git", "clone", "--mirror",
                        f"https://github.com/{repo}.git", mirror], check=True)
    return mirror


def ensure_base_repo(repo, repo_to_top_folder, mirror_root, base_repo_dir):
    top = repo_to_top_folder[repo]
    base_path = os.path.join(base_repo_dir, top)
    if not os.path.exists(base_path):
        mirror = ensure_mirror(repo, repo_to_top_folder, mirror_root)
        print(f"[base] clone from mirror → {base_path}")
        subprocess.run(["git", "clone", "--shared", mirror, base_path], check=True)
    return base_path


def checkout_instance(repo, commit, instance_id, repo_to_top_folder, mirror_root, base_repo_dir, checkout_dir):
    base_repo = ensure_base_repo(repo, repo_to_top_folder, mirror_root, base_repo_dir)
    top = repo_to_top_folder[repo]

    inst_dir = os.path.join(checkout_dir, f"{instance_id}")
    os.makedirs(inst_dir, exist_ok=True)

    work_path = os.path.join(inst_dir, top)

    subprocess.run(["git", "clone", "--shared", base_repo, work_path], check=True)
    subprocess.run(["git", "-C", work_path, "checkout", commit], check=True)

    return work_path


def main():
    parser = argparse.ArgumentParser(description="Download and checkout SWE-bench repositories")
    parser.add_argument(
        "--mirror-root",
        type=str,
        default="/mnt/jianwen/RepoEncoder/exp/repo_strucutres/mirror_cache",
        help="Root directory for mirror cache"
    )
    parser.add_argument(
        "--base-repo-dir",
        type=str,
        default="./repo_strucutres/repos_base",
        help="Root directory for base repositories"
    )
    parser.add_argument(
        "--checkout-dir",
        type=str,
        default="./repo_strucutres/repos_instances",
        help="Root directory for checked out instances"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="princeton-nlp/SWE-bench_Verified",
        help="HuggingFace dataset name"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split to use"
    )
    parser.add_argument(
        "--instance-id",
        type=str,
        default=None,
        help="Specific instance ID to checkout (optional, checkout all if not specified)"
    )

    args = parser.parse_args()

    os.makedirs(args.mirror_root, exist_ok=True)
    os.makedirs(args.base_repo_dir, exist_ok=True)
    os.makedirs(args.checkout_dir, exist_ok=True)

    ds = load_dataset(args.dataset)
    d = ds[args.split]

    # Build repo to top folder mapping
    repo_to_top_folder = {}
    for item in tqdm(d, desc="Building repo mapping"):
        repo = item["repo"]
        if repo not in repo_to_top_folder:
            repo_name = repo.split("/")[-1]
            repo_to_top_folder[repo] = repo_name

    # Filter instances if specific instance_id is provided
    if args.instance_id:
        items_to_process = [item for item in d if item["instance_id"] == args.instance_id]
        if not items_to_process:
            print(f"[ERROR] Instance ID '{args.instance_id}' not found in dataset")
            return
    else:
        items_to_process = d

    # Checkout instances
    for item in tqdm(items_to_process, desc="Checkout commits"):
        repo = item["repo"]
        commit = item["base_commit"]
        inst = item["instance_id"]

        try:
            path = checkout_instance(
                repo=repo,
                commit=commit,
                instance_id=inst,
                repo_to_top_folder=repo_to_top_folder,
                mirror_root=args.mirror_root,
                base_repo_dir=args.base_repo_dir,
                checkout_dir=args.checkout_dir
            )
            print(f"[OK] {inst}: {repo}@{commit} → {path}")
        except Exception as e:
            print(f"[FAIL] {inst}: {e}")


if __name__ == "__main__":
    main()
