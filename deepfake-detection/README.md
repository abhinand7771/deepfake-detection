# Explainable Deepfake Detection System

A modular Python project for face-based deepfake detection using EfficientNet-B0, Grad-CAM explanations, a Streamlit dashboard, and evaluation scripts. The goal is to estimate whether a facial image or sampled video frame looks likely real or likely manipulated, while clearly explaining that the output is a model estimate and not forensic proof.

## Project overview

This project includes:
- Image classification pipeline for face crops using EfficientNet-B0
- Grad-CAM heatmaps and overlays for explainability
- Dataset validation and preprocessing utilities
- Training and evaluation workflows
- Video frame sampling and summary reporting
- Streamlit UI for demonstration and reporting
- Pytest coverage for preprocessing, inference, and Grad-CAM behavior

Important: The system does not prove an image is fake. It produces a model estimate. Highlighted regions indicate where the model was influenced, not necessarily where manipulation artifacts exist.

## Project structure

- `app.py` – Streamlit dashboard
- `train.py` – model training pipeline
- `predict.py` – reusable prediction API
- `preprocess.py` – image validation, face detection, preprocessing
- `gradcam.py` – Grad-CAM implementation and overlays
- `evaluate.py` – evaluation metrics and ROC plots
- `config.py` – global configuration and GPU detection
- `requirements.txt` – dependencies
- `dataset/README.md` – dataset preparation instructions
- `tests/` – automated tests
- `models/` – trained model checkpoints
- `outputs/` – evaluation outputs and generated visualizations

## Windows setup

Use PowerShell in VS Code.

1. Open PowerShell in the project folder:
   ```powershell
   cd "C:\Users\<YourUser>\OneDrive\Desktop\deepfake detector\deepfake-detection"
   ```

2. Create and activate a virtual environment:
   ```powershell
   py -3.11 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

3. Upgrade pip:
   ```powershell
   python -m pip install --upgrade pip setuptools wheel
   ```

4. Install dependencies:
   ```powershell
   python -m pip install -r requirements.txt
   ```

## NVIDIA GPU and CUDA detection

The project supports CUDA when available and automatically detects GPU support at runtime.

Check GPU availability in Python:
```powershell
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.device_count()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

If CUDA is unavailable, the code falls back to CPU automatically. This project does not require a full local CUDA toolkit installation; it uses the PyTorch package that matches the installed environment.

## CPU fallback

If there is no GPU, the model will run on CPU. This is slower but functional. The application warns about performance and uses CPU automatically.

## Dataset access and preparation

The system expects a labeled dataset with real and fake directories:

```text
dataset/
  real/
  fake/
```

Do not scrape arbitrary faces or download restricted datasets without permission.

The application supports access to datasets such as FaceForensics++ when you are authorized to use them under their licensing and access requirements.

To prepare data:
1. Obtain a legal dataset with proper authorization.
2. Place real face images in `dataset/real/`.
3. Place manipulated face images in `dataset/fake/`.
4. Ensure the files are readable and not corrupted.
5. Do not mix frames from the same source video across train/test splits.

If no dataset is present, the app still starts and displays a helpful message.

## Training

Before training, make sure the dataset is prepared. Use the script:

```powershell
python train.py --dataset-dir dataset --batch-size 32 --epochs 12 --learning-rate 1e-4 --device cpu
```

For GPU:
```powershell
python train.py --dataset-dir dataset --batch-size 32 --epochs 12 --learning-rate 1e-4 --device cuda
```

Outputs:
- `models/best_model.pth`
- `outputs/training_metrics.json`

The project uses binary cross-entropy with logits, EfficientNet-B0, early stopping, and validation-based model selection.

## Image inference

For a single image path:
```powershell
python -c "from predict import predict_image_from_path; print(predict_image_from_path('path/to/image.jpg'))"
```

The prediction API requires a valid model checkpoint and returns:
- predicted class
- probability score
- raw model score
- device info
- status
- face boxes

If no face is detected, the system returns a structured no-face message instead of guessing.

## Video analysis

Video analysis uses frame sampling and face detection on sampled frames. It can handle MP4 and other standard video formats. A single odd-looking frame is not proof that the whole video is fake.

Example usage:
```powershell
python -c "import cv2; print('ready')"
```

The actual video analysis workflow is implemented in the `app.py` Streamlit interface and in the reusable project modules for frame processing.

## Streamlit app

Start the dashboard:
```powershell
streamlit run app.py
```

Then open the local URL in your browser, usually `http://localhost:8501`.

## Evaluation

After a model is trained, run:
```powershell
python evaluate.py
```

This saves evaluation metrics such as accuracy, precision, recall, F1-score, ROC-AUC, confusion matrix, and ROC curve in `outputs/`.

## Tests

Run pytest:
```powershell
python -m pytest -q
```

The project includes tests for:
- image preprocessing
- invalid file handling
- prediction structure
- CPU inference with a small test model
- Grad-CAM size and numerical validity

## Troubleshooting

- Missing module errors: reinstall dependencies with `python -m pip install -r requirements.txt`.
- CUDA unsupported: the app will fallback to CPU automatically.
- No dataset found: the app starts but training/inference cannot run until a valid dataset is supplied.
- Missing model checkpoint: the app warns clearly and does not fabricate predictions.
- Corrupted or unreadable image/video: the validation layer raises explicit errors.

## Responsible use and limitations

This system is intended for research, educational use, and demonstration. It is not a forensic certification tool.

Known limitations:
- model performance depends on dataset quality and diversity
- compression and lighting distortions affect predictions
- unseen manipulation methods can reduce generalization
- Grad-CAM highlights model saliency, not proof of modification

## Licensing notes

This project is a lightweight demonstration and educational implementation. Use any dataset only if you have the required rights and comply with its license. Some datasets and source media may be subject to restrictions. The code is intended for authorized, lawful research and educational use.
