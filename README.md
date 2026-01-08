# my_micrograd

A minimal scalar reverse-mode autodiff engine (micrograd-style) built from scratch in Python.

## What’s implemented
- `Value` scalar autograd engine with reverse-mode backprop through a DAG
- Ops: add, mul, pow, tanh, exp, log, div (+ neg/sub and reflected ops)
- Tiny neural net stack: `Neuron`, `Layer`, `MLP`
- Stable softmax + negative log likelihood (NLL) loss
- Unit tests + numerical gradient checking (central difference)

## Install (editable)
```bash
python -m pip install -e .

Inspired by Andrej Karpathy's micrograd project!
