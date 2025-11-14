# Car Make Recognition (ResNet18 + PCA + KNN)

This project builds a complete machine learning pipeline to classify car makes using:

- **ResNet18 (pretrained) → feature extraction**
- **PCA → dimensionality reduction (512 → 128)**
- **KNN → final classifier (cosine + distance weighted)**
- **Gradio → interactive image upload & prediction**

No fine-tuning.  
No training deep networks.  
Just clean transfer learning + classical ML.

---

## 🚀 Pipeline Overview

1. **Load dataset from Drive**  
2. **Extract embeddings using ResNet18**  
3. **Reduce dimensionality with PCA (128 components)**  
4. **Train KNN classifier (k=3, cosine)**  
5. **Evaluate on train/test split**  
6. **Visualize clusters with PCA & t-SNE**  
7. **Predict on new images**  
8. **Launch Gradio app**

---

## 📊 Results

### Confusion Matrix
![Confusion Matrix](images/confusion_matrix.png)

### PCA 2D Embedding
![PCA 2D Plot](images/pca_2d.png)

### t-SNE
![t-SNE Plot](images/tsne.png)

---

## 🧪 Example Prediction

```python
test_img = "Cars Dataset/test/Audi/1000.jpg"
print("Prediction:", predict_image(test_img))
```

Output:

```
Prediction: Audi
```

---

## 🎛️ Gradio Interface

Run locally:

```bash
python ui/gradio_app.py
```

Upload any car image → model predicts the make.

---

## 📂 Project Structure

```
car-make-recognition/
│
├── car_make_recognition.ipynb        # Clean final notebook
├── requirements.txt
│
├── models/
│   ├── car_pca_model.pkl
│   ├── car_knn_model.pkl
│   └── label_encoder.pkl
│
├── utils/
│   └──
|   predict.py
│
├── ui/
│   └── gradio_app.py
│
└── images/
    ├── confusion_matrix.png
    ├── pca_2d.png
    ├── tsne.png
    └── sample_prediction.png
    |__ sample_prediction_1.png

```

---

## 🧱 Dependencies

Install:

```bash
pip install -r requirements.txt
```

---

## ⚠ Notes

- Dataset is **not** included in the repo (too large).  
- Models are included so the notebook can run without recomputing features.  
- Everything is tested on Google Colab.

---

## 👤 Author

**Safwan Shaikh**  
Computer Science | Machine Learning | Computer Vision

---

If you like the project, consider ⭐ the repo.
