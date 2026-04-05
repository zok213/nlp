# NLP Exam Solver - Deployment Guide
## Fine-tuned LLM for solving NLP exam questions (CPU-only, offline)

---

## Overview

This system solves NLP exam questions using a fine-tuned DeepSeek-R1-Distill-Qwen-7B model. It's designed for **CPU-only Windows machines** with **8GB RAM**.

### Architecture

```
Exam Image → (Vision Model) → Question Text → (Fine-tuned Model) → Python Code → Execution
```

- **Vision**: `qwen2.5vl:3b` (extracts text from exam images)
- **Code Generation**: `exam-r1-solver` (fine-tuned 7B model in GGUF Q4_K_M format)
- **Inference Engine**: Ollama (local, offline)

### System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| OS | Windows 10 21H2+ | Windows 11 |
| RAM | 8 GB | 16 GB |
| Disk Space | 6 GB free | 10 GB free |
| CPU | x64 with AVX2 | Modern multi-core |
| GPU | Not required (CPU inference) | Optional: CUDA 12+ |

---

## Quick Start (30 minutes)

### Step 1: Install Ollama (5 minutes)

Download and install Ollama for Windows:

```powershell
# Option A: Manual download
# Visit https://ollama.com/download/OllamaSetup.exe
# Run the installer

# Option B: Using PowerShell (run as Administrator)
Invoke-WebRequest -Uri "https://ollama.com/download/OllamaSetup.exe" -OutFile "$env:TEMP\ollama-setup.exe"
Start-Process -FilePath "$env:TEMP\ollama-setup.exe" -Wait
```

**Verify installation:**
```powershell
ollama --version
# Expected: ollama version is 0.x.x
```

### Step 2: Deploy NLP Exam Solver (15 minutes)

1. **Transfer the `NLP/` folder** to your Windows machine (complete project with all files)

2. **Run the installer script** (as Administrator):
```powershell
cd D:\GitHub\projects\NLP  # Adjust to your path
.\install.ps1
```

The installer will:
- Install Ollama (if not present)
- Extract GGUF model from `data/gguf_model.zip` to `models/`
- Create Ollama model `exam-r1-solver` from `Modelfile`
- Pull vision model `qwen2.5vl:3b` (~2GB)
- Configure OpenCode integration (`.env`, `start_opencode.ps1`)
- Install Python dependencies
- Test the installation

3. **Start Ollama service** (if not already running):
```powershell
ollama serve
# Keep this terminal open (runs in foreground)
```

### Step 3: Test the System (10 minutes)

**Interactive mode:**
```powershell
python exam_solver.py --interactive
```

**Single question:**
```powershell
python exam_solver.py "Write a Python function to compute TF-IDF using NLTK"
```

**Full pipeline with image:**
```powershell
python exam_solver_pipeline.py data/5\ real\ sample/SP23/NLP301_-_PE2_-_SU_2023_663.webp
```

### Step 4: Use with OpenCode (Optional)

The installer creates `start_opencode.ps1` for seamless OpenCode integration:

```powershell
.\start_opencode.ps1
```

This will:
1. Start Ollama in background
2. Verify the API is responding
3. Set environment variables (`OLLAMA_HOST`, `CODE_MODEL`)
4. Launch OpenCode (if installed)

**In OpenCode, use these commands:**
```
/shell python exam_solver.py "Write a Python function to tokenize text using NLTK"
/shell python exam_solver_pipeline.py exam_image.jpg
/shell python exam_solver.py --interactive
```

---

## Files Overview

| File/Dir | Purpose |
|----------|---------|
| `install.ps1` | Windows one-click installer |
| `exam_solver.py` | Simple Python interface (recommended for OpenCode) |
| `exam_solver_pipeline.py` | Full pipeline (vision → code generation → test) |
| `Modelfile` | Ollama model configuration |
| `convert_to_gguf.py` | **If GGUF not provided**: Converts training weights to GGUF |
| `validate_setup.py` | Validation and health check |
| `tests/test_exam_solver.py` | Test suite |
| `models/` | GGUF model files (created by installer) |
| `data/` | Training data, dataset, and `gguf_model.zip` |
| `outputs/` | Generated solutions |

---

## Manual Model Conversion (Alternative)

If `data/gguf_model.zip` doesn't contain a ready-to-use GGUF file, you need to convert the trained weights yourself.

**⚠️ Requires:** ~20GB RAM, 2+ hours CPU time (or GPU for faster conversion)

### Prerequisites

```bash
pip install torch transformers sentencepiece
git clone https://github.com/ggerganov/llama.cpp.git llama.cpp_repo
```

### Conversion Steps

```bash
# 1. Extract source files
python -c "
import zipfile
z = zipfile.ZipFile('data/gguf_model.zip')
z.extractall('models/convert_source')
"

# 2. Convert to F16 (14GB, takes 10-30 min)
python llama.cpp_repo/convert_hf_to_gguf.py models/convert_source \
  --outfile models/exam_r1_7b_q4_F16.gguf \
  --outtype f16

# 3. Quantize to Q4_K_M (4GB, takes 5-15 min)
# Use llama-quantize from llama.cpp_repo/
llama.cpp_repo/quantize.exe models/exam_r1_7b_q4_F16.gguf \
  models/exam_r1_7b_q4.gguf q4_k_m

# 4. Verify
ls -lh models/exam_r1_7b_q4.gguf
# Should show ~4GB
```

**Or use the automated script:**
```bash
python convert_to_gguf.py
```

After conversion, run `ollama create exam-r1-solver -f Modelfile` to register the model.

---

## Common Issues & Solutions

### Ollama not running
```powershell
# Start Ollama
ollama serve

# In another terminal, test
ollama list
```

### Model not found after conversion
```powershell
# Verify model file exists
dir models\exam_r1_7b_q4.gguf

# If missing, run conversion
python convert_to_gguf.py

# Recreate Ollama model
ollama create exam-r1-solver -f Modelfile
```

### Import errors (Python)
```powershell
pip install requests python-dotenv Pillow
```

### Out of memory on 8GB machine
- Close all other applications
- Reduce `--num-ctx` in Modelfile from 4096 to 2048
- If still failing, use `--outtype q4_0` instead of Q4_K_M (lower quality but smaller)

### GPU not used
- This is CPU-only by design. To enable GPU, edit install.ps1 to use CUDA build of llama.cpp
- Requires: NVIDIA GPU with 4GB+ VRAM, CUDA 12.x drivers

---

## Advanced Usage

### Using with OpenCode

```bash
# Start Ollama first
ollama serve

# From OpenCode terminal or /bash:
python exam_solver.py "Your question here"
```

**Example questions:**
- "Write a Python function to compute word frequency distribution using NLTK"
- "Given a text, remove stopwords and stem the remaining tokens"
- "Implement POS tagging with NLTK's averaged_perceptron_tagger"

### Batch Processing

Create a text file with questions (`questions.txt`):
```text
Write a function to tokenize text using NLTK's word_tokenize
Compute the conditional frequency distribution of bigrams
Implement a stemmer using NLTK's PorterStemmer
```

Run:
```powershell
python exam_solver.py --file questions.txt --output solutions.py
```

### Integration with IDE

Add to your `.vscode/tasks.json`:
```json
{
  "label": "Solve NLP Question",
  "type": "shell",
  "command": "python",
  "args": ["${workspaceFolder}/exam_solver.py", "${input:question}"]
}
```

---

## Performance Benchmarks (Expected)

| Model | Size | RAM Usage | Inference Time (first token) | Quality |
|-------|------|-----------|----------------------------|---------|
| Q4_K_M (this) | ~4 GB | ~6 GB | ~10-30s | Excellent |
| Q4_0 | ~3.5 GB | ~5 GB | ~8-25s | Good |
| Q8_0 | ~8 GB | ~12 GB | ~15-40s | Very Good |

On 8GB RAM machine with modern CPU (Intel i5+ / AMD Ryzen 5+):
- First token: 10-30 seconds
- Full response (200 tokens): 30-90 seconds
- Can run alongside VS Code and browser

---

## Technical Details

### Model Training
- **Base**: DeepSeek-R1-Distill-Qwen-7B
- **Fine-tuning**: LoRA (rank 32) on 208 NLP exam problems
- **Data**: 5 real exams + 150 synthetic problems
- **Framework**: Unsloth + Transformers
- **Training**: 2× T4 GPUs, 3 epochs

### Quantization
- **Method**: GGUF Q4_K_M (4-bit normal, 6-bit super-block)
- **Tool**: llama.cpp
- **Target**: CPU inference (AVX2)
- **Context**: 4096 tokens

### Prompt Format
```
<｜User｜>{question}<｜Assistant｜><｜think｜>
{reasoning}
<｜/think｜>
```

---

## Validation & Testing

Run the validation script:
```powershell
python validate_setup.py
```

All checks should show `[OK]`:
- ✅ Project structure
- ✅ Model files
- ✅ Ollama installation
- ✅ Python environment
- ✅ Dependencies

Run the test suite:
```powershell
pytest tests/ -v
```

---

## What to Transfer to Target Machine

Copy these files/folders to the Windows machine:

```
NLP/
├── install.ps1
├── exam_solver.py
├── exam_solver_pipeline.py
├── Modelfile
├── validate_setup.py
├── requirements.txt
├── convert_to_gguf.py  (optional, if GGUF not pre-converted)
├── data/
│   ├── gguf_model.zip      # OR: models/exam_r1_7b_q4.gguf if already converted
│   └── ... (other data files not needed for inference)
├── models/                 # Will be created by installer
├── outputs/               # Will be created automatically
└── tests/                 # Optional
```

**Minimum transfer size:** 12 GB (including GGUF model)
**Without GGUF:** 200 MB (conversion script + data)

---

## Uninstallation

```powershell
# Remove Ollama model
ollama rm exam-r1-solver

# Remove project files
Remove-Item -Recurse -Force D:\GitHub\projects\NLP

# Optionally uninstall Ollama from Settings → Apps
```

---

## Support & Debugging

### Logs

- **Ollama logs**: Check Windows Event Viewer or `%LOCALAPPDATA%\ollama\`
- **Python errors**: Run with `-v` flag for verbose output
- **Conversion log**: Check `convert.log` if using `convert_to_gguf.py`

### Reset Everything
```powershell
# Stop Ollama
ollama stop

# Delete model
ollama rm exam-r1-solver
Remove-Item -Recurse -Force models

# Reinstall
.\install.ps1
```

### Check System Info
```powershell
# Memory
Get-WmiObject -Class Win32_OperatingSystem | Select-Object @{Name="RAM(GB)"; Expression={[math]::Round($_.TotalVisibleMemorySize / 1MB, 1)}}

# CPU
Get-WmiObject -Class Win32_Processor | Select-Object -First 1 Name
```

---

## License & Credits

- **Model**: DeepSeek-R1-Distill-Qwen-7B (Apache 2.0)
- **Fine-tuned on**: Custom NLP exam dataset
- **Inference**: Ollama + llama.cpp
- **Target course**: NLP301c (University level)

---

## Checklist for User

Before running on the target machine, verify:

- [ ] Ollama is installed (`ollama --version` works)
- [ ] `data/gguf_model.zip` exists (12 GB)
- [ ] Python 3.9+ installed
- [ ] Ran `.\install.ps1` as Administrator
- [ ] Vision model `qwen2.5vl:3b` pulled (`ollama list` shows it)
- [ ] `models/exam_r1_7b_q4.gguf` exists (~4 GB)
- [ ] `ollama create exam-r1-solver -f Modelfile` succeeded
- [ ] `ollama list` shows `exam-r1-solver`
- [ ] `python exam_solver.py --interactive` works

If all checked, you're ready to solve NLP exams offline!

---

**Last Updated**: 2026-03-27  
**Version**: 1.0