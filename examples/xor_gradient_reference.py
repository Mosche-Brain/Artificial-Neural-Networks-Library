import numpy as np
from pathlib import Path


def load_dump(path):
    lines = Path(path).read_text().splitlines()
    return np.array([[float(value) for value in line.split()] for line in lines[1:]], dtype=np.float32)


# Same parameters and first sample as xor.cpp.
W1 = np.array([[1.0, -1.0], [-1.0, 2.0]], dtype=np.float32)
b1 = np.array([[-0.5], [0.5]], dtype=np.float32)
W2 = np.array([[1.0, -1.0]], dtype=np.float32)
b2 = np.array([[0.0]], dtype=np.float32)
x = np.array([[0.0], [0.0]], dtype=np.float32)
y = np.array([[0.0]], dtype=np.float32)

z1 = W1 @ x + b1
h = np.tanh(z1)
z2 = W2 @ h + b2
p = 1.0 / (1.0 + np.exp(-z2))

dloss_dp = (1.0 - y) / (1.0 - p) - y / p
dp_dz2 = p * (1.0 - p)
dz2 = dloss_dp * dp_dz2
dW2 = dz2 @ h.T
db2 = dz2
dh = W2.T @ dz2
dz1 = dh * (1.0 - h * h)
dW1 = dz1 @ x.T
db1 = dz1
dx = W1.T @ dz1

reference = {
    "delta_output": dloss_dp,
    "activation_derivative": dp_dz2,
    "preactivation_gradient": dz2,
    "weights_gradient": dW2,
    "biases_gradient": db2,
}

for name, expected in reference.items():
    path = Path(f"plots/dense_backward_0_{name}.txt")
    if not path.exists():
        print(f"{name}: reference={expected.tolist()} (C++ dump missing)")
        continue

    actual = load_dump(path)
    difference = np.abs(actual - expected)
    print(
        f"{name}: max_abs_diff={difference.max():.8g}, "
        f"actual={actual.tolist()}, expected={expected.tolist()}"
    )

print(f"hidden dW1={dW1.tolist()}")
print(f"hidden db1={db1.tolist()}")
print(f"input dx={dx.tolist()}")
