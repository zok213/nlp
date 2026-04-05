#!/usr/bin/env python3
"""
NLP Exam Solver Pipeline
=======================

End-to-end pipeline that:
1. Reads exam question images (using vision model)
2. Extracts question text
3. Generates solution code (using fine-tuned model)
4. Tests the generated code
5. Saves results

Usage:
    python exam_solver_pipeline.py image1.jpg [image2.jpg ...]

Requirements:
    - Ollama installed and running
    - qwen2.5vl:3b pulled
    - exam-r1-solver model created
"""

import os
import sys
import json
import time
import re
import subprocess
import argparse
import tempfile
from pathlib import Path
from datetime import datetime

# Configuration
VISION_MODEL = os.getenv("VISION_MODEL", "qwen2.5vl:3b")
CODE_MODEL = os.getenv("CODE_MODEL", "exam-r1-solver")
TIMEOUT_SEC = int(os.getenv("TIMEOUT_SEC", "15"))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "2"))


def check_ollama_running():
    """Verify Ollama is running and models are available."""
    try:
        import ollama
        # Test connection
        ollama.list()
        return True
    except Exception as e:
        print(f"❌ Ollama not running: {e}")
        print("   Start with: ollama serve")
        return False


def extract_question_from_image(image_path: str, model: str = VISION_MODEL) -> dict:
    """
    Use vision model to extract exam question from image.
    
    Returns:
        dict with keys: question, raw_response, confidence
    """
    import ollama
    
    extraction_prompt = """You are reading a university NLP course exam with Python/NLTK programming questions.

CAREFULLY read the exam image and extract:
1. The exact problem statement (what code to write)
2. Any input examples provided
3. Expected output format
4. Any specific requirements (NLTK functions, edge cases, etc.)

Format as structured text suitable for a code generator. 
Do NOT write code yet - just extract the problem clearly."""

    print(f"  🔍 Extracting question from {Path(image_path).name}...")
    start_time = time.time()
    
    try:
        with open(image_path, 'rb') as f:
            image_bytes = f.read()
        
        response = ollama.chat(
            model=model,
            messages=[{
                "role": "user",
                "content": extraction_prompt,
                "images": [image_bytes]
            }],
            options={
                "temperature": 0.0,  # Deterministic
                "num_ctx": 4096,
            }
        )
        
        elapsed = time.time() - start_time
        question_text = response['message']['content']
        
        print(f"  ✅ Extraction complete ({elapsed:.1f}s)")
        
        return {
            "question": question_text,
            "raw_response": question_text,
            "confidence": "high",
            "elapsed": elapsed
        }
        
    except Exception as e:
        print(f"  ❌ Extraction failed: {e}")
        return {
            "question": None,
            "error": str(e),
            "confidence": "low"
        }


def generate_solution_code(question: str, model: str = CODE_MODEL) -> dict:
    """
    Use fine-tuned model to generate solution code.
    
    Returns:
        dict with keys: code, raw_response, elapsed
    """
    import ollama
    
    generation_prompt = f"""Given this exam question, write complete, correct, runnable Python code using NLTK where appropriate.

Question:
{question}

Requirements:
- Use NLTK library where applicable for NLP tasks
- Handle edge cases properly
- Include comments for clarity
- Wrap code in ```python ... ``` code blocks
- Code must run without errors
- Include main guard or function calls to demonstrate usage"""

    print("  🤖 Generating solution code...")
    start_time = time.time()
    
    try:
        response = ollama.chat(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert Python/NLTK programmer. Generate only working code with proper imports."
                },
                {
                    "role": "user",
                    "content": generation_prompt
                }
            ],
            options={
                "temperature": 0.05,  # Slightly deterministic
                "num_predict": 1024,  # Max tokens to generate
                "num_ctx": 4096,
            }
        )
        
        elapsed = time.time() - start_time
        code_response = response['message']['content']
        
        print(f"  ✅ Generation complete ({elapsed:.1f}s)")
        
        return {
            "code": code_response,
            "raw_response": code_response,
            "elapsed": elapsed
        }
        
    except Exception as e:
        print(f"  ❌ Generation failed: {e}")
        return {
            "code": None,
            "error": str(e),
            "elapsed": 0
        }


def extract_code_from_response(response_text: str) -> str:
    """Extract Python code from model response."""
    # Try to find code blocks
    code_blocks = re.findall(r'```(?:python)?\n(.*?)```', response_text, re.DOTALL)
    
    if code_blocks:
        # Use the longest code block
        return max(code_blocks, key=len).strip()
    
    # Fallback: return entire response if no code blocks found
    # but try to clean it up
    lines = response_text.strip().split('\n')
    code_lines = []
    in_code = False
    
    for line in lines:
        if line.startswith('```'):
            in_code = not in_code
            continue
        if in_code:
            code_lines.append(line)
    
    if code_lines:
        return '\n'.join(code_lines).strip()
    
    # Last resort: return as-is
    return response_text.strip()


def test_generated_code(code: str, timeout: int = TIMEOUT_SEC) -> dict:
    """
    Test if generated code executes without errors.
    
    Returns:
        dict with keys: success, stdout, stderr, returncode, elapsed
    """
    print("  🧪 Testing generated code...")
    start_time = time.time()
    
    # Write to temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as f:
        f.write(code)
        temp_path = f.name
    
    try:
        result = subprocess.run(
            [sys.executable, temp_path],
            capture_output=True,
            text=True,
            timeout=timeout
        )
        
        elapsed = time.time() - start_time
        
        success = result.returncode == 0 and not result.stderr
        
        if success:
            print(f"  ✅ Code executes successfully ({elapsed:.1f}s)")
        else:
            print(f"  ⚠️  Code has issues ({elapsed:.1f}s)")
            if result.stderr:
                print(f"      Error: {result.stderr[:200]}")
        
        return {
            "success": success,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode,
            "elapsed": elapsed
        }
        
    except subprocess.TimeoutExpired:
        elapsed = time.time() - start_time
        print(f"  ⚠️  Code timeout ({timeout}s)")
        return {
            "success": False,
            "stdout": "",
            "stderr": f"TIMEOUT: Code took longer than {timeout} seconds",
            "returncode": -1,
            "elapsed": elapsed
        }
        
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"  ❌ Code test error: {e}")
        return {
            "success": False,
            "stdout": "",
            "stderr": str(e),
            "returncode": -2,
            "elapsed": elapsed
        }
        
    finally:
        # Clean up temp file
        try:
            os.unlink(temp_path)
        except:
            pass


def save_solution(image_path: str, extracted_question: str, code_response: str, 
                 test_result: dict, output_dir: str = ".") -> str:
    """Save solution to file."""
    
    base_name = Path(image_path).stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = Path(output_dir) / f"{base_name}_solution_{timestamp}.py"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"""#!/usr/bin/env python3
# Auto-generated solution for: {base_name}
# Generated at: {datetime.now().isoformat()}
# Test passed: {test_result.get('success', False)}
# Execution time: {test_result.get('elapsed', 0):.1f}s

# ===== EXTRACTED QUESTION =====
{extracted_question}

# ===== GENERATED CODE =====
{test_result.get('code', code_response)}

# ===== EXECUTION OUTPUT =====
# stdout:
{test_result.get('stdout', '')[:500] if test_result.get('stdout') else '(no output)'}

# stderr:
{test_result.get('stderr', '')[:500] if test_result.get('stderr') else '(no errors)'}
""")
    
    print(f"  💾 Solution saved: {output_file}")
    return str(output_file)


def process_single_image(image_path: str, output_dir: str = ".") -> dict:
    """Process a single exam image through the full pipeline."""
    
    result = {
        "image": image_path,
        "timestamp": datetime.now().isoformat(),
        "status": "pending",
        "stages": {},
        "total_elapsed": 0
    }
    
    start_total = time.time()
    
    # Stage 1: Extract question
    print(f"\n{'='*60}")
    print(f"📄 Processing: {image_path}")
    print('='*60)
    
    extraction = extract_question_from_image(image_path)
    result["stages"]["extraction"] = extraction
    
    if not extraction.get("question"):
        result["status"] = "failed_extraction"
        result["total_elapsed"] = time.time() - start_total
        return result
    
    # Stage 2: Generate code
    generation = generate_solution_code(extraction["question"])
    result["stages"]["generation"] = generation
    
    if not generation.get("code"):
        result["status"] = "failed_generation"
        result["total_elapsed"] = time.time() - start_total
        return result
    
    # Extract clean code from response
    code = extract_code_from_response(generation["code"])
    
    # Stage 3: Test code
    test_result = test_generated_code(code)
    test_result["code"] = code  # Include the actual code
    result["stages"]["testing"] = test_result
    
    # Stage 4: Save solution
    solution_file = save_solution(
        image_path,
        extraction["question"],
        generation["code"],
        test_result,
        output_dir
    )
    result["solution_file"] = solution_file
    
    # Final status
    if test_result["success"]:
        result["status"] = "success"
    else:
        result["status"] = "failed_test"
    
    result["total_elapsed"] = time.time() - start_total
    
    # Print summary
    status_emoji = "✅" if result["status"] == "success" else "❌"
    print(f"\n{status_emoji} RESULT: {result['status']}")
    print(f"   Total time: {result['total_elapsed']:.1f}s")
    
    return result


def main():
    parser = argparse.ArgumentParser(
        description="NLP Exam Solver Pipeline - Solve exam questions end-to-end"
    )
    parser.add_argument("images", nargs="+", help="Exam image files to process")
    parser.add_argument("--output-dir", "-o", default=".", 
                        help="Output directory for solutions")
    parser.add_argument("--vision-model", default=VISION_MODEL,
                        help="Vision model to use")
    parser.add_argument("--code-model", default=CODE_MODEL,
                        help="Code generation model to use")
    parser.add_argument("--timeout", type=int, default=TIMEOUT_SEC,
                        help="Code execution timeout in seconds")
    parser.add_argument("--json-output", "-j", 
                        help="Save results to JSON file")
    
    args = parser.parse_args()
    
    # Check Ollama
    if not check_ollama_running():
        sys.exit(1)
    
    # Verify images exist
    image_files = []
    for img in args.images:
        p = Path(img)
        if not p.exists():
            print(f"❌ File not found: {img}")
            continue
        image_files.append(str(p))
    
    if not image_files:
        print("❌ No valid images to process")
        sys.exit(1)
    
    print(f"\n🎯 NLP Exam Solver Pipeline")
    print(f"   Images: {len(image_files)}")
    print(f"   Vision model: {args.vision_model}")
    print(f"   Code model: {args.code_model}")
    print(f"   Timeout: {args.timeout}s")
    
    # Process each image
    results = []
    for i, img_path in enumerate(image_files, 1):
        result = process_single_image(img_path, args.output_dir)
        results.append(result)
    
    # Summary
    print(f"\n{'='*60}")
    print(f"📊 SUMMARY")
    print('='*60)
    
    success_count = sum(1 for r in results if r["status"] == "success")
    total = len(results)
    
    print(f"   Processed: {total}")
    print(f"   ✅ Success: {success_count} ({success_count/total*100:.0f}%)")
    print(f"   ❌ Failed:  {total - success_count} ({(total-success_count)/total*100:.0f}%)")
    
    total_time = sum(r["total_elapsed"] for r in results)
    print(f"   Total time: {total_time:.1f}s")
    print(f"   Avg per image: {total_time/total:.1f}s")
    
    # Save JSON output if requested
    if args.json_output:
        with open(args.json_output, 'w', encoding='utf-8') as f:
            json.dump({
                "metadata": {
                    "timestamp": datetime.now().isoformat(),
                    "vision_model": args.vision_model,
                    "code_model": args.code_model,
                    "total_images": total,
                    "success_count": success_count,
                },
                "results": results
            }, f, indent=2, ensure_ascii=False)
        print(f"\n   📄 Results saved: {args.json_output}")
    
    print('='*60)
    
    # Exit code: 0 if all success, 1 if any failed
    sys.exit(0 if success_count == total else 1)


if __name__ == "__main__":
    main()
