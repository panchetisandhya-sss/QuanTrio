"""QRC forecasting pipeline for option pricing.

This script implements a simple Quantum Reservoir Computing (QRC)
feature extractor using PennyLane and trains a classical Ridge regressor
to forecast option prices (call/put). It builds lag features from the
target columns and performs iterative forecasting for a configurable
number of future steps.

Usage example:
  python QML.py --input train.xlsx --targets call_price,put_price --train-rows 490 --forecast-steps 6
"""

import argparse
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

try:
    import pennylane as qml
except Exception:
    qml = None


def detect_targets(df, targets_arg=None):
    if targets_arg:
        cols = [c.strip() for c in targets_arg.split(",")]
        for c in cols:
            if c not in df.columns:
                raise ValueError(f"Target column '{c}' not found in data")
        return cols

    lowered = [c.lower() for c in df.columns]
    call_idx = next((i for i, c in enumerate(lowered) if "call" in c), None)
    put_idx = next((i for i, c in enumerate(lowered) if "put" in c), None)
    if call_idx is not None and put_idx is not None:
        return [df.columns[call_idx], df.columns[put_idx]]

    # Fallback: use last one or two numeric columns
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric:
        raise ValueError("No numeric columns found to use as targets")
    if len(numeric) >= 2:
        return numeric[-2:]
    return [numeric[-1]]


def make_lag_dataset(targets_values, lags):
    # targets_values: (T, n_targets)
    T, n = targets_values.shape
    X, Y = [], []
    for t in range(lags, T):
        window = targets_values[t - lags:t].flatten()
        X.append(window)
        Y.append(targets_values[t])
    return np.array(X), np.array(Y)


def build_qrc(n_qubits, n_layers, input_dim, seed=42):
    if qml is None:
        raise RuntimeError("PennyLane is not installed. See requirements.txt")

    dev = qml.device("default.qubit", wires=n_qubits)

    rng = np.random.default_rng(seed)
    # Fixed reservoir parameters
    reservoir_params = rng.normal(scale=0.5, size=(n_layers, n_qubits, 3))

    @qml.qnode(dev)
    def circuit(x):
        # Angle embedding: tile or truncate x to n_qubits
        x_enc = np.array(x)
        if x_enc.size < n_qubits:
            x_pad = np.zeros(n_qubits)
            x_pad[: x_enc.size] = x_enc
            x_enc = x_pad
        elif x_enc.size > n_qubits:
            # reduce by averaging blocks
            x_enc = x_enc[: n_qubits]

        qml.templates.AngleEmbedding(x_enc, wires=range(n_qubits))

        for l in range(n_layers):
            for q in range(n_qubits):
                a, b, c = reservoir_params[l, q]
                qml.RX(a, wires=q)
                qml.RY(b, wires=q)
                qml.RZ(c, wires=q)
            for q in range(n_qubits - 1):
                qml.CNOT(wires=[q, q + 1])

        return [qml.expval(qml.PauliZ(w)) for w in range(n_qubits)]

    return circuit


def extract_qrc_features(circuit, X):
    feats = [circuit(x) for x in X]
    return np.array(feats)


def forecast(args):
    df = pd.read_excel(args.input, engine="openpyxl")

    targets = detect_targets(df, args.targets)
    T = len(df)
    if T < args.lags + 1:
        raise ValueError("Not enough rows for the chosen lag size")

    target_values = df[targets].values.astype(float)

    # Build lag dataset
    X_all, Y_all = make_lag_dataset(target_values, args.lags)

    # Map samples to original time index: sample t corresponds to original index t+lags
    sample_indices = np.arange(args.lags, args.lags + X_all.shape[0])

    # Determine training cutoff: use samples whose target time index <= train_rows-1
    cutoff_idx = args.train_rows - 1
    train_mask = sample_indices <= cutoff_idx
    if not np.any(train_mask):
        raise ValueError("No training samples: check --train-rows and --lags")

    X_train = X_all[train_mask]
    Y_train = Y_all[train_mask]

    # Validate backend-specific constraints
    if args.backend == "simulator":
        max_modes = 20
        max_photons = 10
    else:  # quandela
        max_modes = 24
        max_photons = 12

    if args.modes < 1 or args.modes > max_modes:
        raise ValueError(f"--modes must be between 1 and {max_modes} for backend {args.backend}")
    if args.photons < 1 or args.photons > max_photons:
        raise ValueError(f"--photons must be between 1 and {max_photons} for backend {args.backend}")

    if args.backend == "quandela" and (args.amplitude_encoding or args.state_injection):
        raise ValueError("Quandela QPU does not support amplitude encoding or state injection")

    # Build QRC
    n_qubits = min(args.modes, X_train.shape[1])
    circuit = build_qrc(n_qubits=n_qubits, n_layers=3, input_dim=X_train.shape[1])

    # Extract features (may be slow depending on device)
    Q_train = extract_qrc_features(circuit, X_train)

    scaler = StandardScaler()
    Q_train_scaled = scaler.fit_transform(Q_train)

    model = Ridge(alpha=1.0)
    model.fit(Q_train_scaled, Y_train)

    # Iterative forecasting: start from last available values at index train_rows-1
    last_index = args.train_rows - 1
    if last_index + 1 < args.lags:
        raise ValueError("train_rows too small relative to lags")

    # Build initial lag buffer using the last 'lags' values ending at last_index
    start_buffer = target_values[last_index - args.lags + 1 : last_index + 1].tolist()
    buffer = [list(r) for r in start_buffer]

    preds = []
    for step in range(args.forecast_steps):
        x_in = np.array(buffer[-args.lags:]).flatten()
        q_feat = circuit(x_in)
        q_feat_scaled = scaler.transform([q_feat])
        y_pred = model.predict(q_feat_scaled)[0]
        preds.append(y_pred)
        buffer.append(list(y_pred))

    preds = np.array(preds)
    out_df = pd.DataFrame(preds, columns=targets)
    out_df.index = np.arange(T, T + args.forecast_steps)
    out_df.index.name = "time_index"

    out_df.to_csv(args.output, index=True)
    print(f"Saved {args.forecast_steps} step predictions to {args.output}")


def parse_args():
    p = argparse.ArgumentParser(description="QRC forecasting for option pricing")
    p.add_argument("--input", default="train.xlsx", help="Input Excel file (xlsx)")
    p.add_argument("--targets", default=None, help="Comma-separated target columns (optional)")
    p.add_argument("--lags", type=int, default=5, help="Number of lag steps for features")
    p.add_argument("--train-rows", type=int, default=490, help="Number of rows to use for training")
    p.add_argument("--forecast-steps", type=int, default=6, help="How many future steps to predict")
    p.add_argument("--output", default="predictions.csv", help="Output CSV for predictions")
    p.add_argument("--backend", choices=["simulator", "quandela"], default="simulator",
                   help="Quantum backend to use: 'simulator' or 'quandela' (provider)")
    p.add_argument("--modes", type=int, default=8, help="Number of modes (photonic/qubit mapping)")
    p.add_argument("--photons", type=int, default=1, help="(Photonic) number of photons (informational)")
    p.add_argument("--amplitude-encoding", action="store_true",
                   help="Enable amplitude encoding (not supported on some QPUs)")
    p.add_argument("--state-injection", action="store_true",
                   help="Enable state injection (not supported on some QPUs)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    forecast(args)
