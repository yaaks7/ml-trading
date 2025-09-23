# 🤖 Machine Learning Models Guide

> Guide to the ML models used in the trading prediction system: Random Forest and Multi-Layer Perceptron (MLP).

## 📋 Overview

The system implements two machine learning approaches:
- **Random Forest**: Ensemble tree-based method
- **MLP (Multi-Layer Perceptron)**: Neural network approach

Both models are designed for **binary classification** to predict market direction (up/down).

---

## 🌳 Random Forest

### **Concept**
Random Forest is an ensemble method that combines multiple decision trees to make predictions. Each tree is trained on a different subset of the data and features.

### **How It Works**
1. **Bootstrap Sampling**: Create multiple datasets by sampling with replacement
2. **Feature Randomness**: Each tree uses only a random subset of features
3. **Tree Training**: Train decision trees on each bootstrapped dataset
4. **Voting**: Final prediction is the majority vote of all trees

### **Key Parameters**
```python
RandomForestClassifier(
    n_estimators=200,    # Number of trees (more = better but slower)
    max_depth=10,        # Maximum tree depth (controls overfitting)
    random_state=42      # For reproducible results
)
```

---

## 🧠 Multi-Layer Perceptron (MLP)

### **Concept**
MLP is a neural network with multiple layers of neurons. It learns complex non-linear relationships through weighted connections and activation functions.

### **Architecture**
```
Input Layer → Hidden Layer(s) → Output Layer
[Features] → [100 neurons] → [2 classes: Up/Down]
```

### **How It Works**
1. **Forward Pass**: Input features flow through network layers
2. **Activation**: Each neuron applies an activation function (ReLU/tanh)
3. **Output**: Final layer produces class probabilities
4. **Backpropagation**: Errors are propagated back to update weights
5. **Iteration**: Process repeats until convergence

### **Key Parameters**
```python
MLPClassifier(
    hidden_layer_sizes=(100,),  # Network architecture
    max_iter=1000,             # Training iterations
    random_state=42            # For reproducible results
)
```

---

## 🔍 Model Comparison

| Aspect | Random Forest | MLP |
|--------|---------------|-----|
| **Interpretability** | Medium (feature importance) | Low (black box) |
| **Training Speed** | Fast | Slow |
| **Prediction Speed** | Medium | Fast |
| **Overfitting Risk** | Low | High |
| **Feature Engineering** | Less critical | More important |
| **Memory Usage** | High | Medium |
| **Hyperparameter Tuning** | Simple | Complex |

---
