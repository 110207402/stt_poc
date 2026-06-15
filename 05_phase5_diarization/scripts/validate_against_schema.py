"""
Validate dialog specs against the formal JSON Schema.

This is the strict gate referenced by the cross-review: 0 errors = ready to freeze.
"""

from __future__ import annotations
import json
import pathlib
import sys
import collections

try:
    from jsonschema import Draft202012Validator
except ImportError:
    print("Installing jsonschema...")
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "jsonschema"], check=True)
    from jsonschema import Draft202012Validator

V2_ROOT = pathlib.Path(__file__).resolve().parent.parent
SCHEMA_PATH = V2_ROOT / "design" / "dialog_spec.schema.json"
MASTER_PATH = V2_ROOT / "data" / "dialog_specs" / "all_dialogs.json"


def main():
    schema = json.loads(SCHEMA_PATH.read_text())
    validator = Draft202012Validator(schema)

    cases = json.loads(MASTER_PATH.read_text())
    print(f"Validating {len(cases)} cases against {SCHEMA_PATH.name}\n")

    error_count = 0
    cases_with_errors = 0
    error_summary = collections.Counter()
    sample_errors_per_type = collections.defaultdict(list)

    for c in cases:
        errs = list(validator.iter_errors(c))
        if errs:
            cases_with_errors += 1
            error_count += len(errs)
            for e in errs:
                # Categorize error by path + message stem
                path_str = ".".join(str(p) for p in e.absolute_path) or "<root>"
                msg_stem = e.message.split("'")[0][:60].strip()
                key = f"{path_str}: {msg_stem}"
                error_summary[key] += 1
                if len(sample_errors_per_type[key]) < 2:
                    sample_errors_per_type[key].append({
                        "case_id": c.get("case_id"),
                        "message": e.message[:200],
                    })

    print(f"=== Validation result ===")
    print(f"  Total cases:           {len(cases)}")
    print(f"  Cases with errors:     {cases_with_errors}")
    print(f"  Total errors:          {error_count}")

    if error_count == 0:
        print("\n✓ 0 schema errors — ready to freeze.")
        return 0

    print("\n=== Error categories ===")
    for key, n in error_summary.most_common():
        print(f"\n  [{n}x] {key}")
        for sample in sample_errors_per_type[key][:2]:
            print(f"     ├─ {sample['case_id']}: {sample['message'][:140]}")

    print(f"\n✗ {error_count} schema errors remain.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
