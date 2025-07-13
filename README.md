# 🧠 Handwritten Digit Recognition (MNIST) with CNN in PyTorch

This project demonstrates a Convolutional Neural Network (CNN) built with **PyTorch** to classify handwritten digits from the **MNIST dataset**.

The code is cleanly separated into:
- `main.py` → for training and saving the model
- `inference.py` → for loading the trained model and making predictions

---

## 🧪 Sample Result

✅ Achieves over **98% accuracy** on the MNIST test set  
📷 Displays predicted digit with image using `matplotlib`  
🧠 Trained model is saved to `digit_model.pth`  

---

## 📁 Project Structure

```
digit-recognition-pytorch/
├── main.py            # Trains the CNN and saves model
├── inference.py       # Loads model, predicts & visualizes digits
├── .gitignore         # Excludes venv, cache, IDE files
├── digit_model.pth    # (Generated after training)
└── README.md
```

---

## 🚀 Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/digit-recognition-pytorch.git
cd digit-recognition-pytorch
```

### 2. Create and Activate Virtual Environment (Optional)

```bash
python -m venv venv
source venv/bin/activate    # Mac/Linux
venv\Scripts\activate       # Windows
```

### 3. Install Requirements

```bash
pip install torch torchvision matplotlib
```

---

## 🏋️‍♀️ Train the Model (`main.py`)

```bash
python main.py
```

- Trains the CNN for 10 epochs
- Evaluates accuracy after each epoch
- Saves the trained model to `digit_model.pth`

---

## 🔎 Run Inference (`inference.py`)

```bash
python inference.py
```

- Loads `digit_model.pth`
- Predicts **10 random test images**
- Displays each digit with predicted and actual label

---

## 🏷️ Versioning

- `v1.0`: First stable release with training + inference separation and model saving/loading.

---

## 📌 To Do (Future Ideas)

- [ ] Add confusion matrix
- [ ] Train on GPU with CUDA (if available)
- [ ] Deploy with Gradio or Streamlit
- [ ] Save accuracy/loss plots

---

## 🤝 Credits

- PyTorch
- Torchvision
- MNIST Dataset by Yann LeCun

---

## 📄 License

MIT License — feel free to use and modify.

