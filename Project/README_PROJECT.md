# DL MCQ Solver — Deep Learning Multiple Choice Question Solver

## Architecture
Single-model pipeline using **Qwen3.5-27B-FP8** via vLLM with self-consistency verification.

### Pipeline Stages
0. Image Preprocessing (P→RGB, upscale, enhance, tile)
1. Structured OCR Extraction → JSON
2. Program of Thought (code verification for computational questions)
3. Elimination-Based Reasoning with formula injection
4. Multi-Angle Voting (3 prompt framings)
5. Self-Consistency Verification (same model, different system prompt)
6. Reflexion Self-Critique (borderline answers only)
7. Confidence Gate → Output Guard → submission.csv

## Setup
```bash
bash setup.bash
conda activate gnr_project_env
python inference.py --test_dir <absolute_path_to_test_dir>
```

### Requirements
- Linux, L40S GPU (48GB VRAM), CUDA 12.6, 16GB RAM
- Conda, Python 3.11

### What setup.bash does
1. Clones the repository
2. Creates `gnr_project_env` conda environment
3. Installs dependencies (PyTorch 2.4.0, vLLM, Transformers)
4. Downloads Qwen3.5-27B-FP8 weights (~27GB)

## Output
`submission.csv` in the current directory:
```csv
image_name,option
image_1,3
image_2,5
```
Options 1-4 = answer, 5 = skip.

## Project Structure
```
inference.py          # Main entry point
setup.bash            # Automated setup (submitted in zip)
src/
├── __init__.py
├── preprocessing.py  # Stage 0
├── extraction.py     # Stage 1
├── pot_solver.py     # Stage 2
├── elimination.py    # Stage 3
├── voting.py         # Stages 4-5
├── reflexion.py      # Stage 6
├── confidence.py     # Confidence gate
├── formulas.py       # DL formula lookup
├── output_guard.py   # Output hardening + CSV
├── timer.py          # Time-adaptive strategy
└── model_manager.py  # vLLM server lifecycle
```
