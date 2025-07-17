# Broken Bone Detection from X-ray Images

This project uses a deep learning model (ResNet50) to classify X-ray images as either **Normal** or **Abnormal**. It is trained on the Stanford MURA dataset.

## Project Summary

- **Dataset:** MURA from Stanford AI
- **Model:** ResNet50 with transfer learning (fine-tuned on MURA)
- **Accuracy:** 93% (train), 78% (test)
- **Tools:** PyTorch, Google Colab, Matplotlib, scikit-learn, PIL

## How to Use

1. Open the notebook in `notebook/`
2. Make sure GPU is enabled
3. Run all cells to preprocess data, train, evaluate
4. Use `predict_image()` to test your own X-ray image
5. Example usage:
   ```python
   predict_image("/content/MURA/MURA-v1.1/valid/XR_SHOULDER/patient123/study1_positive/image2.png", model, device)


## Project Structure
This is how the project is organized:

 ``` BrokenBone-Xray-Detection/ │ ├── models/ # Trained model weights (.pth) │ └── resnet50_epoch15.pth │ ├── notebook/ # Colab or Jupyter notebook │ └── final_model.ipynb │ ├── utils/ # Python helper functions │ └── predict_image.py │ ├── examples/ # Example images for testing (optional) │ └── test1.png │ ├── requirements.txt # Python dependencies └── README.md # Project documentation ``` </code></pre>

Feel free to fork, test, or extend this project!

