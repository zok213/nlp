#!/usr/bin/env python3
"""
Extract exam questions - Ultra robust version.
"""

import os
import json
import base64
import requests
from pathlib import Path
from PIL import Image
from io import BytesIO
import time
import traceback

API_KEY = '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'
MODEL = 'nvidia/nemotron-nano-12b-v2-vl:free'
API_URL = 'https://openrouter.ai/api/v1/chat/completions'

MAX_SIZE = 800
QUALITY = 60

def convert_webp_to_small_jpeg(webp_path):
    try:
        img = Image.open(webp_path)
        img.thumbnail((MAX_SIZE, MAX_SIZE))
        
        if img.mode in ('RGBA', 'LA'):
            background = Image.new('RGB', img.size, (255, 255, 255))
            background.paste(img, mask=img.split()[-1])
            img = background
        elif img.mode != 'RGB':
            img = img.convert('RGB')
        
        buffer = BytesIO()
        img.save(buffer, format='JPEG', quality=QUALITY)
        return base64.b64encode(buffer.getvalue()).decode('utf-8')
    except Exception as e:
        return None

def extract_text(image_b64, retry=3):
    if not image_b64:
        return None, "Image conversion failed"
    
    prompt = """Extract all exam questions. Return JSON array: [{\"instruction\": \"...\", \"input\": \"...\", \"expected_output\": \"...\"}]"""

    for attempt in range(retry):
        try:
            response = requests.post(
                API_URL,
                headers={
                    'Authorization': f'Bearer {API_KEY}',
                    'Content-Type': 'application/json',
                },
                json={
                    'model': MODEL,
                    'messages': [{
                        'role': 'user',
                        'content': [
                            {'type': 'text', 'text': prompt},
                            {'type': 'image_url', 'image_url': {'url': f'data:image/jpeg;base64,{image_b64}'}}
                        ]
                    }],
                    'temperature': 0.0,
                    'max_tokens': 1500
                },
                timeout=90
            )
            
            if response.status_code != 200:
                print(f"API error {response.status_code}")
                time.sleep(5)
                continue
            
            data = response.json()
            
            # Check response structure
            if 'choices' not in data:
                print(f"No choices in response")
                print(f"Response: {str(data)[:200]}")
                continue
                
            if not data['choices']:
                print(f"Empty choices")
                continue
                
            content = data['choices'][0]['message']['content']
            return content, None
            
        except requests.exceptions.Timeout:
            print(f"Timeout (attempt {attempt+1})")
            time.sleep(3)
        except Exception as e:
            print(f"Error: {type(e).__name__}: {e}")
            time.sleep(3)
    
    return None, "All retries failed"

def main():
    data_dir = Path("./data")
    real_exams_dir = data_dir / "5 real sample"
    output_file = data_dir / "real_exams_raw.json"
    
    # Load existing
    existing = []
    if output_file.exists():
        with open(output_file, 'r') as f:
            existing = json.load(f).get('questions', [])
    
    # Find unprocessed
    processed = {q.get('_source') for q in existing if q.get('_source')}
    
    image_files = sorted(real_exams_dir.rglob("*.webp"))
    to_process = [f for f in image_files if str(f.relative_to(real_exams_dir)) not in processed]
    
    print(f"Total: {len(image_files)}, Processed: {len(processed)}, Remaining: {len(to_process)}")
    
    results = existing
    
    for i, img_path in enumerate(to_process):
        img_key = str(img_path.relative_to(real_exams_dir))
        
        print(f"[{len(results)+1}/{len(image_files)}] {img_path.name}...", end=" ")
        
        # Convert
        image_b64 = convert_webp_to_small_jpeg(img_path)
        if not image_b64:
            print("CONVERT FAILED")
            results.append({'_source': img_key, 'error': 'Image conversion failed', 'status': 'failed'})
            continue
        
        # Extract
        content, error = extract_text(image_b64)
        
        if content:
            print(f"OK ({len(content)} chars)")
            results.append({'_source': img_key, 'raw_text': content, 'status': 'success'})
        else:
            print(f"FAILED: {error}")
            results.append({'_source': img_key, 'error': error, 'status': 'failed'})
        
        # Save progress
        with open(output_file, 'w') as f:
            json.dump({'questions': results}, f, indent=2)
        
        time.sleep(2)
    
    print(f"\nDone! Total: {len(results)}")
    
    # Summary
    success = sum(1 for r in results if r.get('status') == 'success')
    failed = sum(1 for r in results if r.get('status') == 'failed')
    print(f"Success: {success}, Failed: {failed}")

if __name__ == "__main__":
    main()
