import numpy as np
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
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from xgboost import XGBClassifier
import torch
import torch.nn as nn
import torch.optim as optim

print("\n===== Synthetic Dataset Creation =====")
# Generate dummy dataset (1000 samples, 10 features, 3 classes)
np.random.seed(42)
X_data = np.random.rand(1000, 10)
y_data = np.random.randint(0, 3, 1000)

n_classes = len(np.unique(y_data))
activation_out = "softmax" if n_classes > 2 else "sigmoid"
loss_used = "sparse_categorical_crossentropy" if n_classes > 2 else "binary_crossentropy"
output_nodes = n_classes if n_classes > 2 else 1

X_train, X_test, y_train, y_test = train_test_split(
    X_data, y_data, test_size=0.2, random_state=42
)
print(f"Synthetic dataset: {X_data.shape[0]} samples, {X_data.shape[1]} features, {n_classes} classes")

print("\n===== XOR with MLP =====")
xor_X = np.array([[0,0],[0,1],[1,0],[1,1]])
xor_y = np.array([[0],[1],[1],[0]])

xor_model = Sequential([
    Input(shape=(2,)),
    Dense(4, activation='relu'),
    Dense(1, activation='sigmoid')
])
xor_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
xor_model.fit(xor_X, xor_y, epochs=200, verbose=0)
xor_pred = xor_model.predict(xor_X).round().flatten()
print("XOR Predictions:", xor_pred)
print("Expected:", xor_y.flatten())

print("\n===== Gradient Boosting (XGBoost) =====")
gb = XGBClassifier(eval_metric='logloss')
gb.fit(X_train, y_train)
gb_pred = gb.predict(X_test)
print("XGBoost Test Accuracy:", accuracy_score(y_test, gb_pred))

print("\n===== Computational Graph =====")
a = tf.constant(3.0)
b = tf.constant(2.0)
c = tf.constant(5.0)
f = a * b + c
print("f(a,b,c) = 3*2+5 =", f.numpy())

print("\n===== Neural Network in PyTorch =====")
torch_X = torch.tensor(X_train, dtype=torch.float32)
torch_y = torch.tensor(y_train, dtype=torch.long)

class SimpleNet(nn.Module):
    def __init__(self, inp, outp):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(inp, 16)
        self.fc2 = nn.Linear(16, outp if outp > 2 else 2)
    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))

net = SimpleNet(X_data.shape[1], n_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.01)

for ep in range(50):
    optimizer.zero_grad()
    out = net(torch_X)
    loss = criterion(out, torch_y)
    loss.backward()
    optimizer.step()
print("Final Training Loss:", loss.item())

print("\n===== Optimizers + Initialization =====")
model = Sequential([
    Input(shape=(X_data.shape[1],)),
    Dense(16, activation='relu', kernel_initializer=HeNormal()),
    Dense(output_nodes, activation=activation_out)
])
model.compile(optimizer='adam', loss=loss_used, metrics=['accuracy'])
model.fit(X_train, y_train, epochs=5, batch_size=16, validation_data=(X_test, y_test), verbose=0)
print("Validation Accuracy:", model.evaluate(X_test, y_test, verbose=0)[1])

print("\n===== Overfitting Handling =====")
drop_model = Sequential([
    Input(shape=(X_data.shape[1],)),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(output_nodes, activation=activation_out)
])
drop_model.compile(optimizer='adam', loss=loss_used, metrics=['accuracy'])
early_stop = EarlyStopping(monitor='val_loss', patience=3)
drop_hist = drop_model.fit(X_train, y_train, validation_data=(X_test, y_test),
                           epochs=15, callbacks=[early_stop], verbose=0)
print("Train Acc:", drop_hist.history['accuracy'][-1])
print("Val Acc:", drop_hist.history['val_accuracy'][-1])

print("\n===== Bias vs Variance =====")
x_lin = np.linspace(0, 1, 30).reshape(-1, 1)
y_lin = np.sin(2 * np.pi * x_lin).ravel() + np.random.normal(0, 0.1, 30)

poly2 = PolynomialFeatures(degree=2)
pred2 = LinearRegression().fit(poly2.fit_transform(x_lin), y_lin).predict(poly2.fit_transform(x_lin))
bias_err = mean_squared_error(y_lin, pred2)

poly15 = PolynomialFeatures(degree=15)
pred15 = LinearRegression().fit(poly15.fit_transform(x_lin), y_lin).predict(poly15.fit_transform(x_lin))
var_err = mean_squared_error(y_lin, pred15)

print("Bias Error (deg=2):", bias_err)
print("Variance Error (deg=15):", var_err)

plt.scatter(x_lin, y_lin, color="black")
plt.plot(x_lin, pred2, label="Low-degree")
plt.plot(x_lin, pred15, label="High-degree")
plt.legend()
plt.title("Bias vs Variance")
plt.show()

print("\n===== Optimizer Comparison =====")
opts = ["SGD", "Adam", "RMSprop", "Adagrad"]
for o in opts:
    temp = Sequential([
        Input(shape=(X_data.shape[1],)),
        Dense(16, activation='relu'),
        Dense(output_nodes, activation=activation_out)
    ])
    temp.compile(optimizer=o, loss=loss_used, metrics=['accuracy'])
    hist = temp.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, verbose=0)
    print(f"{o} -> Final Val Acc:", hist.history['val_accuracy'][-1])

print("\n===== Initialization Comparison =====")
inits = {"Zeros": Zeros(), "RandomNormal": RandomNormal(),
         "GlorotUniform": GlorotUniform(), "HeNormal": HeNormal()}
for name, init in inits.items():
    temp = Sequential([
        Input(shape=(X_data.shape[1],)),
        Dense(16, activation='relu', kernel_initializer=init),
        Dense(output_nodes, activation=activation_out)
    ])
    temp.compile(optimizer="adam", loss=loss_used, metrics=['accuracy'])
    hist = temp.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, verbose=0)
    print(f"{name:12s} -> Final Val Acc:", hist.history['val_accuracy'][-1])
