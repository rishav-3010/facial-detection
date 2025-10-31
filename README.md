Facial Detection / Recognition — Quick start

What this repo contains
- `dl project_github_file.ipynb` — main notebook with functions to collect webcam data, train a MobileNetV2-based classifier, generate embeddings, visualize them, and run real-time recognition.
- `dataset/` — folders per person (example: `Chinmay/`, `Rohan/`, `Sarvesh/`).
- `face_model.h5` — (optional) saved Keras model.
- `embeddings.pkl` — (optional) saved embeddings used for recognition.
- `venv/` — a Python virtual environment (if present).

Quick setup (Windows PowerShell)
1. Open PowerShell in the repo folder (c:\Users\Rishav\Downloads\facial-detection).

2. Activate the included virtual environment (if you want to use it):

   # PowerShell
   .\venv\Scripts\Activate.ps1

   # If the above is blocked by ExecutionPolicy, use the cmd activate or set the policy temporarily:
   # .\venv\Scripts\activate

3. Install dependencies (recommended):

   pip install -r requirements.txt

4. Run the project (recommended: use Jupyter / VS Code Notebook UI):
   - Start Jupyter: `jupyter notebook` or `jupyter lab`, open `dl project_github_file.ipynb`, then run cells in order.
   - Or open the notebook in VS Code (Notebook view) and run cells interactively.

How to run the pipeline from the notebook
- The notebook includes a `main()` that prompts for a step (1-4):
  1 — Collect dataset from webcam (will add images to `dataset/<label>`)
  2 — Train model (saves `face_model.h5`)
  3 — Generate & visualize embeddings (saves `embeddings.pkl`)
  4 — Run real-time recognition (loads `face_model.h5` + `embeddings.pkl`)

If you already have `face_model.h5` and `embeddings.pkl` in the repo root, run option 4 to skip training and start recognition.

Troubleshooting & notes
- Webcam not detected: try changing `cv2.VideoCapture(0)` to another index (1, 2) in the notebook.
- If you get a TF / GPU mismatch error, try installing a CPU-only `tensorflow` compatible with your Python and GPU drivers (or use the included `venv` which already has packages).
- If `Activate.ps1` is blocked: run PowerShell as Admin or use the cmd `.\venv\Scripts\activate.bat`.

Next improvements (suggested)
- Add a `run.py` CLI that wraps `main()` so you can run `python run.py --step 4`.
- Add a `requirements-dev.txt` / pinned versions for reproducibility.
- Add a short `LICENSE` and automated tests for non-GUI functions (embedding generation, model shape checks).

Contact / help
If any step fails, paste the error message and I'll guide you through fixes (dependency versions, Python version mismatches, or camera issues).