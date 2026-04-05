#!/usr/bin/env python3
"""
NLP Exam Solver - OpenCode Integration
=======================================

Simple Python script to call Ollama API for exam-solving.
Use this with OpenCode's /shell or /bash commands.

Usage:
    python exam_solver.py "Your NLP question here"
    python exam_solver.py --file question.txt
    python exam_solver.py --interactive

Requirements:
    - Ollama running (ollama serve)
    - exam-r1-solver model created
"""

import os
import sys
import json
import argparse
import re
import requests
from pathlib import Path

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
MODEL = os.getenv("CODE_MODEL", "exam-r1-solver")


def call_ollama(prompt: str, model: str = MODEL, temperature: float = 0.05) -> str:
    """Call Ollama API and return response."""
    url = f"{OLLAMA_HOST}/api/generate"
    
    payload = {
        "model": model,
        "prompt": prompt,
        "temperature": temperature,
        "stream": False,
        "options": {
            "num_predict": 1024,
            "num_ctx": 4096,
        }
    }
    
    response = requests.post(url, json=payload, timeout=300)
    response.raise_for_status()
    
    return response.json().get("response", "")


def extract_code(response: str) -> str:
    """Extract Python code from model response."""
    # Try code blocks first
    code_blocks = re.findall(r'```(?:python)?\n(.*?)```', response, re.DOTALL)
    if code_blocks:
        return max(code_blocks, key=len).strip()
    
    # Fallback: find anything that looks like Python
    lines = response.strip().split('\n')
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
    
    return response.strip()


def solve_question(question: str, show_thinking: bool = True) -> dict:
    """Solve an NLP exam question."""
    
    prompt = f"""Given this exam question, write complete, correct, runnable Python code using NLTK where appropriate.

Question:
{question}

Requirements:
- Use NLTK library where applicable for NLP tasks
- Handle edge cases properly
- Include comments for clarity
- Wrap code in ```python ... ``` code blocks
- Code must run without errors
- Include main guard or function calls to demonstrate usage"""

    print(f"[ASKING] {question[:80]}...")
    
    try:
        response = call_ollama(prompt)
        code = extract_code(response)
        
        return {
            "success": True,
            "question": question,
            "code": code,
            "raw_response": response
        }
    except Exception as e:
        return {
            "success": False,
            "question": question,
            "error": str(e),
            "code": None
        }


def interactive_mode():
    """Interactive Q&A mode."""
    print("📚 NLP Exam Solver - Interactive Mode")
    print("Type your question or 'quit' to exit")
    print("-" * 50)
    
    while True:
        try:
            question = input("\n❓ Question: ").strip()
            if not question:
                continue
            if question.lower() in ('quit', 'exit', 'q'):
                break
            
            result = solve_question(question)
            if result["success"]:
                print("\n💻 Code:\n")
                print(result["code"])
            else:
                print(f"\n❌ Error: {result['error']}")
                
        except KeyboardInterrupt:
            break
        except EOFError:
            break
    
    print("\n👋 Goodbye!")


def main():
    parser = argparse.ArgumentParser(
        description="NLP Exam Solver - Generate code from questions"
    )
    parser.add_argument("question", nargs="?", help="Question to solve")
    parser.add_argument("--file", "-f", help="Read question from file")
    parser.add_argument("--interactive", "-i", action="store_true",
                        help="Interactive mode")
    parser.add_argument("--model", "-m", default=MODEL,
                        help="Ollama model to use")
    parser.add_argument("--output", "-o", help="Save code to file")
    parser.add_argument("--raw", action="store_true",
                        help="Show raw model output (no code extraction)")
    
    args = parser.parse_args()
    
    # Check Ollama is running
    try:
        requests.get(f"{OLLAMA_HOST}/api/tags", timeout=5)
    except:
        print(f"❌ Cannot connect to Ollama at {OLLAMA_HOST}")
        print("   Start with: ollama serve")
        sys.exit(1)
    
    # Interactive mode
    if args.interactive:
        interactive_mode()
        return
    
    # Get question
    question = args.question
    
    if args.file:
        question = Path(args.file).read_text(encoding='utf-8')
    
    if not question:
        parser.print_help()
        sys.exit(1)
    
    # Solve
    result = solve_question(question)
    
    if result["success"]:
        if args.raw:
            print(result["raw_response"])
        else:
            print(result["code"])
        
        if args.output:
            Path(args.output).write_text(result["code"], encoding='utf-8')
            print(f"\n💾 Saved to: {args.output}")
    else:
        print(f"❌ Error: {result['error']}")
        sys.exit(1)


if __name__ == "__main__":
    main()