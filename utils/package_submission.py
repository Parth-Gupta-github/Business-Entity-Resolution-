"""
Submission Packaging Script for Amazon ML Challenge 2026.
Creates the exact zip archive structure required by the organizers:

<team_name>_submission.zip
├── output/
│   ├── matching_results.tsv
│   └── candidate_pairs.tsv
├── code/
│   └── business_entity_resolution/
│       ├── src/
│       ├── README.md
│       └── requirements.txt
└── Documentation_template.md
"""

import os
import sys
import zipfile
import argparse
from pathlib import Path


def create_submission_zip(team_name: str, output_dir: str = "submissions"):
    base_dir = Path(".").resolve()
    target_zip_name = f"{team_name}_submission.zip"
    os.makedirs(output_dir, exist_ok=True)
    target_zip_path = Path(output_dir) / target_zip_name

    # Validate output files exist
    output_files = [
        base_dir / "output" / "matching_results.tsv",
        base_dir / "output" / "candidate_pairs.tsv"
    ]
    for of in output_files:
        if not of.exists():
            print(f"ERROR: Required output file missing: {of}")
            sys.exit(1)

    # Validate doc template exists
    doc_template = base_dir / "Documentation_template.md"
    if not doc_template.exists():
        print(f"ERROR: Documentation_template.md missing at {doc_template}")
        sys.exit(1)

    # Validate code dir exists
    code_dir = base_dir / "code" / "business_entity_resolution"
    if not code_dir.exists():
        print(f"ERROR: Code directory missing at {code_dir}")
        sys.exit(1)

    print(f"Packaging submission for team '{team_name}' into: {target_zip_path}")
    with zipfile.ZipFile(target_zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        # 1. Output files
        z.write(base_dir / "output" / "matching_results.tsv", "output/matching_results.tsv")
        z.write(base_dir / "output" / "candidate_pairs.tsv", "output/candidate_pairs.tsv")

        # 2. Documentation template
        z.write(doc_template, "Documentation_template.md")

        # 3. Code package
        for root, dirs, files in os.walk(code_dir):
            for file in files:
                if file.endswith((".pyc", ".pyo", ".DS_Store")) or "__pycache__" in root:
                    continue
                file_path = Path(root) / file
                arc_name = os.path.relpath(file_path, base_dir)
                z.write(file_path, arc_name)

    print("Successfully created submission package:")
    with zipfile.ZipFile(target_zip_path, "r") as z:
        for info in z.infolist():
            print(f"  + {info.filename} ({info.file_size / 1024:.1f} KB)")
    print(f"\nSaved to: {target_zip_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Package final submission zip for Amazon ML Challenge 2026")
    parser.add_argument("--team", "-t", required=True, help="Your official team name")
    parser.add_argument("--out-dir", "-o", default="submissions", help="Destination folder for zip")
    args = parser.parse_args()

    create_submission_zip(args.team, args.out_dir)
