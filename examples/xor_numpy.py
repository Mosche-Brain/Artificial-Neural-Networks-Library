import numpy as np


# The same architecture, parameters and data as xor.cpp.
X = np.array(
    [[0.0, 0.0],
     [0.0, 1.0],
     [1.0, 0.0],
     [1.0, 1.0]],
    dtype=np.float32,
)
Y = np.array([[0.0], [1.0], [1.0], [0.0]], dtype=np.float32)

W1 = np.array(
    [[-0.37002477, 0.99407870],
     [0.41585749, -0.61816370]],
    dtype=np.float32,
)
b1 = np.zeros((2, 1), dtype=np.float32)
W2 = np.array([[0.67214078, -1.45004952]], dtype=np.float32)
b2 = np.zeros((1, 1), dtype=np.float32)


def forward(x):
    z1 = W1 @ x + b1
    hidden = np.tanh(z1)
    z2 = W2 @ hidden + b2
    prediction = 1.0 / (1.0 + np.exp(-z2))
    return z1, hidden, z2, prediction


def binary_cross_entropy(prediction, target):
    p = np.clip(prediction, 1e-7, 1.0 - 1e-7)
    return float(-(target * np.log(p) + (1.0 - target) * np.log(1.0 - p))[0, 0])


learning_rate = np.float32(1.0)
epochs = 100
gradient_trace = []

for epoch in range(epochs):
    epoch_gradients = np.zeros((2, 2), dtype=np.float32)
    for sample, target in zip(X, Y):
        x = sample[:, None]
        y = target.reshape(1, 1)

        _, hidden, _, prediction = forward(x)
        prediction = np.clip(prediction, 1e-7, 1.0 - 1e-7)

        d_prediction = (1.0 - y) / (1.0 - prediction) - y / prediction
        d_z2 = d_prediction * prediction * (1.0 - prediction)
        d_W2 = d_z2 @ hidden.T
        d_b2 = d_z2

        d_hidden = W2.T @ d_z2
        d_z1 = d_hidden * (1.0 - hidden * hidden)
        d_W1 = d_z1 @ x.T
        d_b1 = d_z1

        epoch_gradients[0] += np.mean(np.abs(d_W1)), np.mean(np.abs(d_b1))
        epoch_gradients[1] += np.mean(np.abs(d_W2)), np.mean(np.abs(d_b2))

        W1 -= learning_rate * d_W1
        b1 -= learning_rate * d_b1
        W2 -= learning_rate * d_W2
        b2 -= learning_rate * d_b2

    gradient_trace.append(epoch_gradients / len(X))

with open("plots/numpy_gradient_trace.txt", "w", encoding="utf-8") as trace:
    for epoch, gradients in enumerate(gradient_trace):
        for layer, (weights, biases) in enumerate(gradients, start=1):
            trace.write(f"{epoch} {layer} {weights:.9g} {biases:.9g}\n")

predictions = forward(X.T)[3].ravel()
loss = np.mean([binary_cross_entropy(np.array([[p]], dtype=np.float32), y)
                for p, y in zip(predictions, Y)])

print("NumPy XOR after training:")
for sample, prediction in zip(X, predictions):
    print(f"[{sample[0]:.0f},{sample[1]:.0f}] -> {prediction:.8f}")
print(f"loss: {loss:.8f}")
