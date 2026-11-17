import argparse


def main():
    parser = argparse.ArgumentParser(description="Download a model from the Hugging Face Hub.")
    parser.add_argument("repo_id", help="Hugging Face model repository ID")
    parser.add_argument("--output-dir", required=True, help="Destination directory")
    args = parser.parse_args()

    from huggingface_hub import snapshot_download

    destination = snapshot_download(repo_id=args.repo_id, local_dir=args.output_dir)
    print(f"Downloaded model to {destination}")


if __name__ == "__main__":
    main()
