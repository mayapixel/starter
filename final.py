import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from tensorflow.keras import layers
from keras.models import Sequential
from keras.layers import Dense, Dropout, Input
from keras.initializers import HeNormal, Zeros, RandomNormal, GlorotUniform
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.preprocessing import LabelEncoder, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from xgboost import XGBClassifier
import torch
import torch.nn as nn
import torch.optim as optim

print("\n===== Dataset Loading =====")
data = pd.read_csv("lab_dataset.csv")   # Or .read_excel("lab_dataset.xlsx")

features = data.drop("label", axis=1).values
target = data["label"].values

# Encode labels if categorical
if target.dtype == 'object' or target.dtype == 'str':
    encoder = LabelEncoder()
    labels = encoder.fit_transform(target)
else:
    labels = target.values if hasattr(target, "values") else target
class_count = len(np.unique(labels))

# Binary vs multi-class
if class_count == 2:
    act_func = "sigmoid"
    loss_type = "binary_crossentropy"
    out_neurons = 1
    final_labels = labels
else:
    act_func = "softmax"
    loss_type = "sparse_categorical_crossentropy"
    out_neurons = class_count
    final_labels = labels

feat_train, feat_test, lab_train, lab_test = train_test_split(
    features, final_labels, test_size=0.2, random_state=42
)
print(f"Dataset loaded: {features.shape[0]} samples, {features.shape[1]} features, {class_count} classes")

print("\n===== XOR with MLP =====")
xor_inputs = np.array([[0,0],[0,1],[1,0],[1,1]])
xor_outputs = np.array([[0],[1],[1],[0]])

xor_net = Sequential([
    Input(shape=(2,)),
    Dense(4, activation='relu'),
    Dense(1, activation='sigmoid')
])
xor_net.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
xor_net.fit(xor_inputs, xor_outputs, epochs=200, verbose=0)
xor_preds = xor_net.predict(xor_inputs).round().flatten()
print("XOR Predictions:", xor_preds)
print("Expected:", xor_outputs.flatten())

print("\n===== Gradient Boosting (XGBoost) =====")
gb_model = XGBClassifier(eval_metric='logloss')
gb_model.fit(feat_train, lab_train)
gb_preds = gb_model.predict(feat_test)
gb_acc = accuracy_score(lab_test, gb_preds)
print("XGBoost Test Accuracy:", gb_acc)

print("\n===== Computational Graph =====")
a = tf.constant(3.0)
b = tf.constant(2.0)
c = tf.constant(5.0)
func_val = a * b + c
print("f(a,b,c) = 3*2+5 =", func_val.numpy())

print("\n===== Neural Network in PyTorch =====")
torch_X = torch.tensor(feat_train, dtype=torch.float32)
torch_y = torch.tensor(lab_train, dtype=torch.long)

class TorchNet(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(TorchNet, self).__init__()
        self.hidden = nn.Linear(in_dim, 16)
        self.output = nn.Linear(16, out_dim if out_dim > 2 else 2)
    def forward(self, x):
        x = torch.relu(self.hidden(x))
        return self.output(x)

torch_model = TorchNet(features.shape[1], class_count)
loss_func = nn.CrossEntropyLoss()
opt = optim.Adam(torch_model.parameters(), lr=0.01)

for epoch in range(100):
    opt.zero_grad()
    out = torch_model(torch_X)
    l = loss_func(out, torch_y)
    l.backward()
    opt.step()
print("Final Training Loss:", l.item())

print("\n===== Optimizers + Initialization =====")
nn_model = Sequential([
    Input(shape=(features.shape[1],)),
    Dense(16, activation='relu', kernel_initializer=HeNormal()),
    Dense(out_neurons, activation=act_func)
])
nn_model.compile(optimizer='adam', loss=loss_type, metrics=['accuracy'])
train_hist = nn_model.fit(feat_train, lab_train, epochs=5, batch_size=16,
                          validation_data=(feat_test, lab_test), verbose=0)
val_loss, val_acc = nn_model.evaluate(feat_test, lab_test, verbose=0)
print(f"Test Accuracy with He Initialization + Adam Optimizer: {val_acc:.4f}")

print("\n===== Overfitting Handling =====")
drop_model = Sequential([
    Input(shape=(features.shape[1],)),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(out_neurons, activation=act_func)
])
drop_model.compile(optimizer='adam', loss=loss_type, metrics=['accuracy'])
stopper = EarlyStopping(monitor='val_loss', patience=3)
drop_hist = drop_model.fit(feat_train, lab_train, validation_data=(feat_test, lab_test),
                           epochs=20, callbacks=[stopper], verbose=0)
print("Final Training Accuracy:", drop_hist.history['accuracy'][-1])
print("Final Validation Accuracy:", drop_hist.history['val_accuracy'][-1])

print("\n===== Bias vs Variance =====")
x_vals = np.linspace(0, 1, 30).reshape(-1, 1)
y_vals = np.sin(2 * np.pi * x_vals).ravel() + np.random.normal(0, 0.1, x_vals.shape[0])

poly_small = PolynomialFeatures(degree=2)
X_small = poly_small.fit_transform(x_vals)
pred_small = LinearRegression().fit(X_small, y_vals).predict(X_small)
bias_err = mean_squared_error(y_vals, pred_small)

poly_big = PolynomialFeatures(degree=15)
X_big = poly_big.fit_transform(x_vals)
pred_big = LinearRegression().fit(X_big, y_vals).predict(X_big)
var_err = mean_squared_error(y_vals, pred_big)

print(f"Bias Error (Underfitting): {bias_err:.4f}")
print(f"Variance Error (Overfitting): {var_err:.4f}")

plt.figure(figsize=(8,4))
plt.scatter(x_vals, y_vals, color="black", label="Data")
plt.plot(x_vals, pred_small, label="Low-degree fit (Bias)")
plt.plot(x_vals, pred_big, label="High-degree fit (Variance)")
plt.legend()
plt.title("Bias vs Variance")
plt.show()

print("\n===== Optimizer Comparison =====")
opt_list = ["SGD", "Adam", "RMSprop", "Adagrad"]
opt_results = {}
for o in opt_list:
    temp_model = Sequential([
        Input(shape=(features.shape[1],)),
        Dense(16, activation='relu'),
        Dense(out_neurons, activation=act_func)
    ])
    temp_model.compile(optimizer=o, loss=loss_type, metrics=['accuracy'])
    h = temp_model.fit(feat_train, lab_train, validation_data=(feat_test, lab_test),
                       epochs=20, verbose=0)
    opt_results[o] = h
    print(f"{o:8s} -> Final Val Accuracy: {h.history['val_accuracy'][-1]:.4f}")
plt.figure(figsize=(8,5))
for o, h in opt_results.items():
    plt.plot(h.history['val_accuracy'], label=o)
plt.title("Optimizer Comparison (Validation Accuracy)")
plt.xlabel("Epochs")
plt.ylabel("Validation Accuracy")
plt.legend()
plt.show()

print("\n===== Initialization Comparison =====")
init_options = {"Zeros": Zeros(), "RandomNormal": RandomNormal(),
                "GlorotUniform": GlorotUniform(), "HeNormal": HeNormal()}
init_results = {}
for init_name, init_func in init_options.items():
    temp_model = Sequential([
        Input(shape=(features.shape[1],)),
        Dense(16, activation='relu', kernel_initializer=init_func),
        Dense(out_neurons, activation=act_func)
    ])
    temp_model.compile(optimizer="adam", loss=loss_type, metrics=['accuracy'])
    h = temp_model.fit(feat_train, lab_train, validation_data=(feat_test, lab_test),
                       epochs=20, verbose=0)
    init_results[init_name] = h
    print(f"{init_name:12s} -> Final Val Accuracy: {h.history['val_accuracy'][-1]:.4f}")

plt.figure(figsize=(8,5))
for init_name, h in init_results.items():
    plt.plot(h.history['val_accuracy'], label=init_name)
plt.title("Initialization Comparison (Validation Accuracy)")
plt.xlabel("Epochs")
plt.ylabel("Validation Accuracy")
plt.legend()
plt.show()
