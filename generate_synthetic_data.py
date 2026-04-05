#!/usr/bin/env python3
"""
Generate synthetic exam data - Using stepfun model (working!).
"""

import os
import json
import time
import random
import requests
from pathlib import Path

API_KEY = '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'
MODEL = 'stepfun/step-3.5-flash:free'  # This one works!

TOPICS = [
    "NLTK word tokenization",
    "NLTK sentence tokenization", 
    "FreqDist frequency distribution",
    "Stopword removal",
    "Porter stemmer",
    "WordNet lemmatization",
    "POS tagging",
    "Named Entity Recognition",
    "N-gram generation",
    "Text classification with Naive Bayes",
    "Regex pattern matching",
]

def generate_question(topic, difficulty):
    """Generate one synthetic question."""
    
    prompt = f"""Generate ONE NLP programming exam question.

Topic: {topic}
Difficulty: {difficulty}

Return ONLY valid JSON:
{{
  "instruction": "Write a Python/NLTK program that...",
  "input": "example input",
  "expected_output": "expected result", 
  "output": "complete working Python code with imports",
  "thinking": "Explain approach in 2 sentences"
}}

Make it realistic for an NLP course exam."""

    try:
        response = requests.post(
            'https://openrouter.ai/api/v1/chat/completions',
            headers={
                'Authorization': f'Bearer {API_KEY}',
                'Content-Type': 'application/json',
            },
            json={
                'model': MODEL,
                'messages': [
                    {'role': 'system', 'content': 'You are an NLP exam question generator.'},
                    {'role': 'user', 'content': prompt}
                ],
                'temperature': 0.7,
                'max_tokens': 600
            },
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            content = data['choices'][0]['message']['content']
            
            # Extract JSON
            import re
            match = re.search(r'\{.*\}', content, re.DOTALL)
            if match:
                return json.loads(match.group())
                    
    except Exception as e:
        print(f"Error: {e}")
    
    return None

def main():
    # Load real exams
    with open('data/real_exams.json') as f:
        real_data = json.load(f)
    
    real_questions = real_data['questions']
    print(f"Loaded {len(real_questions)} real questions")
    
    # Target: 80 synthetic (total ~110)
    target = 80
    synthetic = []
    
    print(f"Generating {target} synthetic questions with {MODEL}...")
    
    # Generate - distribute across topics
    per_topic = target // len(TOPICS)
    
    for topic in TOPICS:
        print(f"\n{topic}:", end=" ")
        
        for i in range(per_topic):
            result = generate_question(topic, "intermediate")
            
            if result and result.get('output') and 'def ' in result.get('output', ''):
                result['topic'] = topic
                result['difficulty'] = 'intermediate'
                result['_synthetic'] = True
                synthetic.append(result)
                print(".", end="", flush=True)
            else:
                print("x", end="", flush=True)
            
            time.sleep(1.5)  # Rate limit
            
            # Save progress
            if len(synthetic) % 20 == 0 and len(synthetic) > 0:
                print(f" {len(synthetic)}", end="")
    
    print(f"\nTotal synthetic: {len(synthetic)}")
    
    # Combine with real
    output = {
        'metadata': {
            'generated': len(synthetic),
            'real': len(real_questions),
            'total': len(synthetic) + len(real_questions),
            'model': MODEL
        },
        'synthetic': synthetic,
        'real': real_questions
    }
    
    with open('data/full_exam_dataset.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"Saved: data/full_exam_dataset.json")
    print(f"Total: {output['metadata']['total']}")

if __name__ == "__main__":
    main()
