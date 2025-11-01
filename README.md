# Facial Detection / Recognition — Quick Start

A real-time facial recognition system built with TensorFlow and OpenCV using transfer learning with MobileNetV2.

## What this repo contains

- `dl project_github_file.ipynb` — main notebook with functions to collect webcam data, train a MobileNetV2-based classifier, generate embeddings, visualize them, and run real-time recognition
- `camera_test.py` — quick camera connectivity test
- `dataset/` — folders per person (example: `Chinmay/`, `Rohan/`, `Sarvesh/`)
- `face_model.h5` — (optional) saved Keras classification model
- `embedding_model.h5` — (optional) saved feature extraction model
- `embeddings.pkl` — (optional) saved embeddings used for recognition
- `venv/` — Python virtual environment (if present)
- `requirements.txt` — Python dependencies

## Quick Setup (Windows PowerShell)

1. **Open PowerShell** in the repo folder (`c:\Users\Rishav\Downloads\facial-detection`)

2. **Activate the virtual environment** (if you want to use it):
   ```powershell
   # PowerShell
   .\venv\Scripts\Activate.ps1
   
   # If blocked by ExecutionPolicy, use cmd activate or set policy temporarily:
   # .\venv\Scripts\activate
   ```

3. **Install dependencies** (recommended):
   ```bash
   pip install -r requirements.txt
   ```

4. **Test your camera** (optional but recommended):
   ```bash
   python camera_test.py
   ```

5. **Run the project** (recommended: use Jupyter / VS Code Notebook UI):
   - Start Jupyter: `jupyter notebook` or `jupyter lab`, open `dl project_github_file.ipynb`, run cells in order
   - Or open the notebook in VS Code (Notebook view) and run cells interactively

## How to Run the Pipeline

The notebook includes a `main()` function that prompts for a step (1-4):

### Option 1 — Collect dataset from webcam
- Captures face images from your webcam
- Creates labeled folders in `dataset/<label>/`
- **Tip**: Capture 100-300 images per person with varied poses and lighting
- Press **'q'** to stop capturing

### Option 2 — Train model
- Trains on collected data with augmentation
- Saves `face_model.h5` and `embedding_model.h5`
- Generates and saves `embeddings.pkl`
- Shows training curves and confusion matrix

### Option 3 — Generate & visualize embeddings
- Creates face embeddings using trained model
- Shows PCA visualization of embeddings
- Saves `embeddings.pkl`

### Option 4 — Run real-time recognition
- Loads `face_model.h5`, `embedding_model.h5`, and `embeddings.pkl`
- Performs real-time face recognition from webcam
- Green box = recognized face, Red box = unknown
- Press **'q'** to quit

**Quick tip**: If you already have `face_model.h5`, `embedding_model.h5`, and `embeddings.pkl` in the repo root, jump straight to option 4 to skip training.

## Key Features

- **Real-time Recognition**: Detects and recognizes faces from webcam feed
- **Transfer Learning**: MobileNetV2 pre-trained on ImageNet
- **Data Augmentation**: Rotation (±15°), zoom (±20%), horizontal flip
- **Similarity Matching**: Cosine distance with k-NN (threshold: 0.5)
- **Multi-person Support**: Recognizes multiple individuals simultaneously
- **Unknown Detection**: Identifies faces not in training dataset

## Model Architecture

- **Input**: 256×256 RGB images
- **Base**: MobileNetV2 (frozen, ImageNet weights)
- **Feature Layer**: GlobalAveragePooling2D → 81,920-dim embeddings
- **Classification Head**: 128-unit Dense → Softmax
- **Optimizer**: Adam (lr=0.0001)
- **Training**: 5 epochs, 80/20 train/val split

## Troubleshooting & Notes

### Webcam not detected
```bash
python camera_test.py
# If failed, try changing cv2.VideoCapture(0) to VideoCapture(1) in the notebook
```

### ExecutionPolicy blocks `Activate.ps1`
- Run PowerShell as Admin, or
- Use cmd: `.\venv\Scripts\activate.bat`, or
- Temporarily set policy: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`

### TensorFlow / GPU mismatch error
- Try CPU-only TensorFlow: `pip install tensorflow==2.15.0 --force-reinstall`
- Or use the included `venv` which has compatible packages
- Check Python version (3.8-3.11 recommended for TF 2.15)

### Model/embeddings mismatch
- Dataset changed? Re-run option 2 to retrain
- Ensure `dataset/` structure matches when loading saved models

### Poor recognition accuracy
- Collect more diverse training images (different angles, lighting, expressions)
- Lower threshold for stricter matching (default: 0.5)
- Increase training epochs (change in `train_model()`)
- Ensure good lighting during capture and recognition

## Technical Details

### Face Detection & Preprocessing
- Haar Cascade classifier for face detection
- 20% margin around detected faces
- Resize to 256×256 with MobileNetV2 preprocessing

### Recognition Process
1. Detect face with Haar Cascade
2. Extract 81,920-dim embedding via MobileNetV2
3. Find nearest neighbor in stored embeddings (cosine distance)
4. If distance < 0.5 → Valid (show name), else → Unknown

### File Formats
- Models: HDF5 (`.h5`)
- Embeddings: Pickle (`.pkl`)
- Images: JPEG (`.jpg`)

## Next Improvements (Suggested)

- [ ] Add `run.py` CLI: `python run.py --step 4`
- [ ] Add `requirements-dev.txt` with pinned versions
- [ ] Add LICENSE file
- [ ] Automated tests for embedding generation and model checks
- [ ] Face alignment preprocessing
- [ ] Video file input support
- [ ] Export to TensorFlow Lite for mobile
- [ ] Anti-spoofing / liveness detection

## Requirements

```
tensorflow==2.15.0
opencv-python
numpy
scikit-learn
matplotlib
seaborn
tqdm
h5py
jupyter
```

## Contact / Help

If any step fails, paste the error message for troubleshooting guidance (dependency versions, Python version mismatches, or camera issues).

---

**Note**: This system is for educational/research purposes. Always respect privacy and obtain consent before collecting facial data.
