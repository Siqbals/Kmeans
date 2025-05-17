# KMeans Clustering on MNIST Dataset

This repository implements the **K-Means clustering algorithm** from scratch and applies it to the **MNIST handwritten digit dataset** using `NumPy` and `PyTorch`'s `torchvision` library.

The project includes:
- A modular KMeans implementation with configurable hyperparameters.
- A wrapper for clustering the MNIST dataset.
- Support for saving/loading clustering results as JSON files.
- Utility for computing distortion and visualizing clustering performance.
- Unit tests to validate core algorithm components.

---


---

## 📊 Features

- Custom implementation of KMeans clustering.
- One-hot encoded membership matrix.
- Distortion calculation as optimization objective (minimizing J = ||X - ZM||²).
- Repeated clustering with best result selection.
- JSON serialization and deserialization of clustering outputs.
- Unit tests covering centroid updates, membership assignment, and distortion computation.

---

## 📈 Example: MNIST Clustering

The script automatically downloads the MNIST dataset and flattens each 28×28 image into a 784-dimensional vector before applying K-Means.

Example command:

```bash
python Kmeans.py --k 10 --max-iterations 100 --epsilon 0.001 --repeats 5 --filename clustering.json


