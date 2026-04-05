#!/usr/bin/env python3
"""
Quick validation test for NLP Exam Solver
==========================================

Run this to verify everything is set up correctly.

Usage:
    python validate_setup.py
    python validate_setup.py --full
"""

import sys
import subprocess
import json
from pathlib import Path


def check_file(path: str, description: str) -> bool:
    """Check if a file exists."""
    if Path(path).exists():
        print(f"  [OK] {description}")
        return True
    else:
        print(f"  [FAIL] {description} - NOT FOUND")
        return False


def check_directory(path: str, description: str) -> bool:
    """Check if a directory exists."""
    if Path(path).is_dir():
        print(f"  [OK] {description}")
        return True
    else:
        print(f"  [FAIL] {description} - NOT FOUND")
        return False


def run_command(cmd: list, description: str) -> bool:
    """Run a command and check if it works."""
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print(f"  [OK] {description}")
            return True
        else:
            print(f"  [WARN] {description} - FAILED")
            return False
    except FileNotFoundError:
        print(f"  [WARN] {description} - COMMAND NOT FOUND")
        return False
    except subprocess.TimeoutExpired:
        print(f"  [WARN] {description} - TIMEOUT")
        return False


def main():
    print("=" * 50)
    print("NLP Exam Solver - Validation Check")
    print("=" * 50)
    print()

    all_passed = True

    # Check project structure
    print("[DIR] Project Structure:")
    all_passed &= check_file("Modelfile", "Modelfile (Ollama config)")
    all_passed &= check_file("install.ps1", "install.ps1 (Windows installer)")
    all_passed &= check_file("exam_solver.py", "exam_solver.py (OpenCode integration)")
    all_passed &= check_file("exam_solver_pipeline.py", "exam_solver_pipeline.py (full pipeline)")
    all_passed &= check_file("requirements.txt", "requirements.txt (dependencies)")
    all_passed &= check_directory("tests", "tests/ (test suite)")
    all_passed &= check_directory("data", "data/ (datasets)")
    print()

    # Check model files
    print("[PKG] Model Files:")
    all_passed &= check_file("data/gguf_model.zip", "GGUF model (quantized)")
    all_passed &= check_file("data/lora_adapter.zip", "LoRA adapter")
    all_passed &= check_file("data/training_dataset_final.json", "Training dataset")
    print()

    # Check Ollama
    print("[OLLAMA] Ollama:")
    all_passed &= run_command(["ollama", "--version"], "Ollama installed")
    print()

    # Check Python
    print("[PYTHON] Python:")
    all_passed &= run_command([sys.executable, "--version"], f"Python ({sys.version.split()[0]})")
    print()

    # Check requirements (if pip available)
    print("[PIP] Dependencies:")
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "list"],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            print("  [OK] pip available")
            
            # Check key packages
            packages = result.stdout.lower()
            if "requests" in packages:
                print("  [OK] requests installed")
            else:
                print("  [WARN] requests not installed (run: pip install requests)")
            if "pytest" in packages:
                print("  [OK] pytest installed")
            else:
                print("  [WARN] pytest not installed (run: pip install pytest)")
    except:
        print("  [WARN] pip not available")
    print()

    # Summary
    print("=" * 50)
    if all_passed:
        print("[SUCCESS] All checks passed!")
        print()
        print("Next steps:")
        print("  1. Run: .\\install.ps1 (on Windows)")
        print("  2. Run: ollama serve")
        print("  3. Run: python exam_solver.py --interactive")
    else:
        print("[WARNING] Some checks failed")
        print()
        print("To fix:")
        print("  - Install Ollama: https://ollama.com")
        print("  - Install dependencies: pip install -r requirements.txt")
    print("=" * 50)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())