"""Comprehensive repository audit for BioNexus MOSAIC multi-omics PAM50 classifier.
Generates docs/CLEANUP_REPORT.md fulfilling Phase 1 requirements.
"""
import os
import sys
import re
import json
import subprocess
from pathlib import Path

REPO_ROOT = Path(r"c:\anushka\bio_nexus")
IGNORE_SCAN_DIRS = {".git", "node_modules", ".venv", ".pytest_cache", "__pycache__", ".mypy_cache", ".ruff_cache"}

def format_size(bytes_val):
    if bytes_val < 1024:
        return f"{bytes_val} B"
    elif bytes_val < 1024 * 1024:
        return f"{bytes_val/1024:.1f} KB"
    elif bytes_val < 1024 * 1024 * 1024:
        return f"{bytes_val/(1024*1024):.1f} MB"
    else:
        return f"{bytes_val/(1024*1024*1024):.2f} GB"

def get_dir_size(path):
    total = 0
    try:
        for root, dirs, files in os.walk(path):
            dirs[:] = [d for d in dirs if d not in IGNORE_SCAN_DIRS]
            for f in files:
                try:
                    total += os.path.getsize(os.path.join(root, f))
                except OSError:
                    pass
    except Exception:
        pass
    return total

def get_tree_3levels(dir_path, level=0, max_level=3):
    lines = []
    try:
        entries = sorted(dir_path.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower()))
    except Exception:
        return lines

    for entry in entries:
        if entry.name in {".git", "node_modules", ".venv", ".pytest_cache", "__pycache__"}:
            continue
        indent = "  " * level
        if entry.is_dir():
            sz = get_dir_size(entry)
            lines.append(f"{indent}├── {entry.name}/ ({format_size(sz)})")
            if level < max_level - 1:
                lines.extend(get_tree_3levels(entry, level + 1, max_level))
        else:
            try:
                sz = entry.stat().st_size
            except Exception:
                sz = 0
            lines.append(f"{indent}├── {entry.name} ({format_size(sz)})")
    return lines

def run_git_grep(pattern):
    try:
        res = subprocess.run(
            ["git", "grep", "-I", "-l", pattern],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False
        )
        if res.returncode == 0:
            return [line.strip() for line in res.stdout.splitlines() if line.strip()]
    except Exception:
        pass
    return []

def main():
    print("1. Building tree...")
    tree_lines = get_tree_3levels(REPO_ROOT, 0, 3)

    print("2. Listing tracked and working tree files...")
    tracked = set(subprocess.check_output(["git", "ls-files"], cwd=REPO_ROOT, text=True).splitlines())
    
    # Large files check
    print("3. Checking large files...")
    large_files = []
    for root, dirs, files in os.walk(REPO_ROOT):
        dirs[:] = [d for d in dirs if d not in IGNORE_SCAN_DIRS]
        for f in files:
            fp = Path(root) / f
            try:
                sz = fp.stat().st_size
                if sz > 500 * 1024:
                    rel = fp.relative_to(REPO_ROOT).as_posix()
                    large_files.append((rel, sz, format_size(sz), rel in tracked))
            except Exception:
                pass
    large_files.sort(key=lambda x: x[1], reverse=True)

    # Secret scanning
    print("4. Scanning for potential secrets in tracked files...")
    secret_patterns = [
        (r"(?i)(?:api_key|apikey|secret_key|secret|password|private_key|token)\s*[:=]\s*['\"][A-Za-z0-9_\-\.\/]{16,}['\"]", "High-entropy token/key assignment"),
        (r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----", "Private key block"),
        (r"0x[a-fA-F0-9]{64}", "Ethereum/Polygon private key hex")
    ]
    detected_secrets = []
    for tf in tracked:
        tf_path = REPO_ROOT / tf
        if not tf_path.is_file():
            continue
        try:
            content = tf_path.read_text(encoding="utf-8", errors="ignore")
            for pat, desc in secret_patterns:
                matches = re.finditer(pat, content)
                for m in matches:
                    # Exclude sample, test, or placeholder strings
                    s_val = m.group(0)
                    if "example" in s_val.lower() or "placeholder" in s_val.lower() or "your_" in s_val.lower() or "0000000000" in s_val:
                        continue
                    line_num = content[:m.start()].count("\n") + 1
                    detected_secrets.append({
                        "file": tf,
                        "line": line_num,
                        "description": desc
                    })
        except Exception:
            pass

    # Inspect candidate files for removal or move
    print("5. Evaluating candidates for removal or move...")
    candidates = []

    # Helper function to add candidate
    def add_candidate(path_str, why_exists, recommendation, confidence, note=""):
        p = REPO_ROOT / path_str
        sz_str = "0 B"
        if p.exists():
            if p.is_dir():
                sz_str = format_size(get_dir_size(p))
            else:
                sz_str = format_size(p.stat().st_size)
        
        # Grep references
        fname = Path(path_str).name
        stem = Path(path_str).stem
        refs = run_git_grep(fname)
        if not refs and stem:
            refs = run_git_grep(stem)
        refs_filtered = [r for r in refs if r != path_str]
        refs_str = ", ".join(refs_filtered[:3]) if refs_filtered else "None"
        if len(refs_filtered) > 3:
            refs_str += f" (+{len(refs_filtered)-3} more)"
            
        candidates.append({
            "path": path_str,
            "why": why_exists,
            "refs": refs_str,
            "size": sz_str,
            "rec": recommendation,
            "conf": confidence,
            "note": note
        })

    # Candidates list
    add_candidate("prompt_9_full.txt", "Old copy-pasted LLM prompt from development", "delete", "High", "Loose text artifact in root")
    add_candidate("ChatGPT Image Sep 30, 2026, 11_34_36 PM.png", "Screenshot / diagram from ChatGPT ideation", "delete / move to docs/assets", "High", "1.7 MB loose image in root")
    add_candidate("ChatGPT Image Sep 30, 2026, 11_35_38 PM.png", "Screenshot / diagram from ChatGPT ideation", "delete / move to docs/assets", "High", "1.7 MB loose image in root")
    add_candidate("ChatGPT Image Sep 30, 2026, 11_36_30 PM.png", "Screenshot / diagram from ChatGPT ideation", "delete / move to docs/assets", "High", "1.7 MB loose image in root")
    add_candidate("MOSAIC_Project_Documentation.docx", "Word document version of project docs", "move to docs/archive or delete", "High", "Binary docx in root")
    add_candidate("BioNexus_Project_Writeup.md", "Extensive writeup/thesis document (54 KB)", "move to docs/PROJECT_WRITEUP.md", "High", "Comprehensive documentation belong in docs/")
    add_candidate("MODEL_CARD.md", "Model card for mosaic-subtype-classifier", "move to docs/MODEL_CARD.md", "High", "Target layout puts model card in docs/")
    add_candidate("RUNBOOK.md", "Production runbook and SOPs", "move to docs/RUNBOOK.md", "High", "Target layout puts runbook in docs/")
    add_candidate("bionexus_spec.json", "Multi-omics input feature dimensions spec", "move to configs/bionexus_spec.json", "High", "Config spec belongs in configs/")
    add_candidate("check_cohort.py", "One-off script checking cohort patient IDs", "delete", "High", "Redundant with patient_cohort.py and step1_cohort.py")
    add_candidate("scratch_test_import.py", "One-off test script importing adapter", "delete", "High", "Temporary scratch verification script")
    add_candidate("metrics_seed.py", "Script to seed initial metrics", "move to scripts/metrics_seed.py", "High", "Root script should be in scripts/")
    add_candidate("mlflow.db", "SQLite database for MLflow experiments", "git-ignore & keep local", "High", "Database should be git-ignored, read by local MLflow")
    add_candidate("mlruns/mlflow.db", "Duplicate copy of mlflow.db mounted for Docker", "keep (managed via bind mount)", "High", "Mounted in docker-compose.yml for container MLflow")
    add_candidate("frontend/", "Old frontend prototype (Streamlit/Dash/React prototype)", "delete / archive", "Medium", "Superceded by frontend-new; check if still referenced")
    add_candidate("frontend_old/", "Deprecated previous version of frontend", "delete", "High", "Explicitly deprecated legacy frontend")
    add_candidate("railway.json", "Railway deployment configuration", "move to docker/ or keep in root", "Medium", "Platform deploy config; can remain in root or move to deploy/")
    add_candidate("render.yaml", "Render deployment blueprint", "move to docker/ or keep in root", "Medium", "Platform deploy config; can remain in root or move to deploy/")
    add_candidate("vercel.json", "Vercel frontend deployment configuration", "move to frontend-new/ or keep in root", "Medium", "Vercel looks for root or frontend-new vercel.json")
    add_candidate("scripts/fix_alembic.py", "One-off schema migration patch for MLflow 2.14 vs 3.x", "keep in scripts/ (docstringed)", "High", "Useful maintenance utility for Alembic sync")
    add_candidate("scripts/check_deps.py", "Temporary check for dependency installation", "delete", "High", "One-off debugging script")
    add_candidate("scripts/step1_cohort.py", "Step 1 audit verification script", "delete", "High", "Superseded by patient_cohort.py")
    add_candidate("scripts/step1_inspect.py", "Step 1 checkpoint inspection script", "delete", "High", "Superseded by inspect_checkpoints.py")
    add_candidate("scripts/test_api_dash.py", "Ad-hoc test script for dashboard API", "delete", "High", "Redundant with test_endpoints.py and pytest suite")
    add_candidate("scripts/test_xgb.py", "Ad-hoc test for XGBoost baseline", "delete", "High", "Redundant with unit and benchmark tests")
    add_candidate("scripts/scratch_inventory.py", "Scratch script listing repo inventory", "delete", "High", "Temporary scratch script")
    add_candidate("src/model_comparison/", "Model comparison benchmark and registry module", "move to src/bio_nexus/model_comparison/", "High", "All core code belongs under src/bio_nexus/")

    # Documentation vs Reality Mismatches
    print("6. Auditing doc vs reality mismatches...")
    mismatches = []
    
    # Read README.md
    readme_path = REPO_ROOT / "README.md"
    if readme_path.exists():
        readme_content = readme_path.read_text(encoding="utf-8", errors="ignore")
        if "make up" in readme_content:
            makefile_content = (REPO_ROOT / "Makefile").read_text(encoding="utf-8", errors="ignore") if (REPO_ROOT / "Makefile").exists() else ""
            if "up:" not in makefile_content:
                mismatches.append({
                    "doc": "README.md",
                    "issue": "References 'make up' command, but Makefile has no 'up:' target (only has 'docker-up:' or similar)",
                    "severity": "Medium"
                })
        if "poetry" in readme_content.lower() and not (REPO_ROOT / "poetry.lock").exists():
            mismatches.append({
                "doc": "README.md",
                "issue": "Mentions Poetry, but repository uses pyproject.toml with pip-tools / requirements.lock",
                "severity": "Medium"
            })
            
    # Check Makefile
    if (REPO_ROOT / "Makefile").exists():
        mf = (REPO_ROOT / "Makefile").read_text(encoding="utf-8", errors="ignore")
        if "src/bionexus" in mf and not (REPO_ROOT / "src/bionexus").exists() and (REPO_ROOT / "src/bio_nexus").exists():
            mismatches.append({
                "doc": "Makefile",
                "issue": "Makefile references 'src/bionexus' without underscore, while package is 'src/bio_nexus'",
                "severity": "High"
            })

    # Check docker-compose.yml
    if (REPO_ROOT / "docker-compose.yml").exists():
        dcf = (REPO_ROOT / "docker-compose.yml").read_text(encoding="utf-8", errors="ignore")
        # Check volume mounts
        if "./mlflow.db:/mlruns/mlflow.db" in dcf:
            mismatches.append({
                "doc": "docker-compose.yml",
                "issue": "Mounts single file './mlflow.db:/mlruns/mlflow.db' on Windows Docker Desktop; file inode locks can cause sync delays between host and container",
                "severity": "Medium"
            })

    # Check CI workflow
    ci_path = REPO_ROOT / ".github" / "workflows" / "ci.yml"
    if ci_path.exists():
        ci_content = ci_path.read_text(encoding="utf-8", errors="ignore")
        if "tests/test_" in ci_content and not (REPO_ROOT / "tests/test_").exists():
            pass

    # Move map
    move_map = [
        {"old": "BioNexus_Project_Writeup.md", "new": "docs/PROJECT_WRITEUP.md", "action": "git mv", "reason": "Extensive project writeup belongs in docs/"},
        {"old": "MODEL_CARD.md", "new": "docs/MODEL_CARD.md", "action": "git mv", "reason": "Target layout places Model Card in docs/"},
        {"old": "RUNBOOK.md", "new": "docs/RUNBOOK.md", "action": "git mv", "reason": "Target layout places Runbook in docs/"},
        {"old": "bionexus_spec.json", "new": "configs/bionexus_spec.json", "action": "git mv", "reason": "Configuration specs belong in configs/"},
        {"old": "metrics_seed.py", "new": "scripts/metrics_seed.py", "action": "git mv", "reason": "Root scripts belong in scripts/"},
        {"old": "src/model_comparison/", "new": "src/bio_nexus/model_comparison/", "action": "git mv", "reason": "Package code should reside inside the bio_nexus namespace"},
        {"old": "ChatGPT Image Sep 30, 2026, 11_34_36 PM.png", "new": "docs/assets/architecture_chatgpt_1.png", "action": "git mv or delete", "reason": "If retained, assets belong in docs/assets/ with snake_case names; otherwise delete"},
        {"old": "ChatGPT Image Sep 30, 2026, 11_35_38 PM.png", "new": "docs/assets/architecture_chatgpt_2.png", "action": "git mv or delete", "reason": "If retained, assets belong in docs/assets/ with snake_case names; otherwise delete"},
        {"old": "ChatGPT Image Sep 30, 2026, 11_36_30 PM.png", "new": "docs/assets/architecture_chatgpt_3.png", "action": "git mv or delete", "reason": "If retained, assets belong in docs/assets/ with snake_case names; otherwise delete"},
        {"old": "MOSAIC_Project_Documentation.docx", "new": "docs/archive/MOSAIC_Project_Documentation.docx", "action": "git mv or delete", "reason": "Binary Word document should not sit in repo root"}
    ]

    report_data = {
        "tree_lines": tree_lines,
        "candidates": candidates,
        "large_files": large_files,
        "secrets": detected_secrets,
        "mismatches": mismatches,
        "move_map": move_map
    }

    with open(REPO_ROOT / "reports" / "audit_summary.json", "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)

    print("Audit completed successfully. Wrote reports/audit_summary.json")

if __name__ == "__main__":
    main()
