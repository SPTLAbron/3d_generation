# Generative 3D Trophy Models

Exploring learned representations for controllable generation of
procedurally generated 3D trophy geometry.

## Project

This project investigates whether generative models can learn meaningful
latent representations of parameterized 3D shapes.

The initial dataset consists of procedurally generated trophy-like geometry
with controlled variations in properties such as:

- ball size
- ball position
- support geometry
- base dimensions
- overall proportions

The goal is to study whether these geometric properties emerge as
interpretable directions in a learned latent space.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt