#!/usr/bin/env python3
"""Amazon ML Challenge 2026 - Submission Packager.

Packages the final deliverables into <team_name>_submission.zip according to the
official competition specifications:

<team_name>_submission.zip
├── output/
│   ├── matching_results.tsv        # final matches (leaderboard submission file)
│   └── candidate_pairs.tsv         # blocking candidate set
├── code/
│   └── business_entity_resolution/
│       ├── src/                    # source code
│       ├── README.md               # run instructions
│       └── requirements.txt        # pinned dependencies
└── Documentation_template.md       # filled methodology document
"""

import argparse
import os
import zipfile


def package_submission(team_name: str = "nixel", output_zip: str = None):
    if not output_zip:
        output_zip = f"{team_name}_submission.zip"

    # Verify required deliverables exist
    required_files = [
        "output/matching_results.tsv",
        "output/candidate_pairs.tsv",
        "student_resource/Documentation_template.md",
        "code/business_entity_resolution/run.py",
        "code/business_entity_resolution/requirements.txt",
        "code/business_entity_resolution/README.md",
    ]
    for rf in required_files:
        if not os.path.exists(rf):
            raise FileNotFoundError(f"Missing required file for submission: {rf}")

    if os.path.exists(output_zip):
        os.remove(output_zip)

    print(f"Packaging submission into '{output_zip}'...")
    with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        # 1. Output TSVs
        zf.write("output/matching_results.tsv", "output/matching_results.tsv")
        zf.write("output/candidate_pairs.tsv", "output/candidate_pairs.tsv")

        # 2. Code directory
        code_dir = "code/business_entity_resolution"
        for root, _, files in os.walk(code_dir):
            if "__pycache__" in root:
                continue
            for f in files:
                full_path = os.path.join(root, f)
                arc_name = os.path.normpath(full_path).replace("\\", "/")
                zf.write(full_path, arc_name)

        # 3. Methodology Document
        zf.write("student_resource/Documentation_template.md", "Documentation_template.md")

    zip_size_mb = os.path.getsize(output_zip) / (1024 * 1024)
    print(f"[SUCCESS] Submission packaged: {output_zip} ({zip_size_mb:.2f} MB)")
    print("\nVerified archive contents:")
    with zipfile.ZipFile(output_zip, "r") as zf:
        for name in zf.namelist():
            print(f"  - {name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Package Amazon ML Challenge Submission")
    parser.add_argument("--team-name", type=str, default="nixel", help="Team name prefix for zip archive")
    parser.add_argument("--output-zip", type=str, default=None, help="Custom output zip filename")
    args = parser.parse_args()
    package_submission(args.team_name, args.output_zip)
