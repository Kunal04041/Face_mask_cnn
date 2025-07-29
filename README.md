#  Face Mask Detection using CNN

This project uses Convolutional Neural Networks (CNNs) to detect whether a person is wearing a face mask or not in real-time using a webcam or images. The model is trained on labeled face images with and without masks.

---

##  Objective

To build a deep learning model that accurately detects the presence or absence of face masks in images or video streams, useful in health monitoring and public safety systems.

---

##  Dataset


- **Classes**: 
  - With Mask   
  - Without Mask 
- **Total Images**: ~10,000 images (balanced)

---

##  Model Architecture

- CNN architecture (custom or pre-trained like MobileNetV2)
- Layers include:
  - Convolution + ReLU
  - Max Pooling
  - Dropout
  - Fully Connected Dense layers
- Final layer: Softmax (2 classes)

---

## 🛠 Technologies Used

- Python
- TensorFlow / Keras
- OpenCV
- NumPy, Matplotlib

---



