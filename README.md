# SHINE: SHear INference Environment

<div align="center">
  <img src="assets/logo.png" alt="SHINE Logo" width="1024"/>

  **A JAX-powered framework for probabilistic shear estimation in weak gravitational lensing**

  [![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
  [![JAX](https://img.shields.io/badge/JAX-latest-green.svg)](https://github.com/google/jax)
  [![License](https://img.shields.io/badge/license-MIT-orange.svg)](LICENSE)
</div>

---

## 🌟 Overview

SHINE (SHear INference Environment) is a modern, high-performance framework for probabilistic shear estimation in weak gravitational lensing studies. Built on JAX, it leverages automatic differentiation and just-in-time compilation to deliver fast, scalable inference for cosmological applications.

## ✨ Key Features

- 🚀 **JAX-powered**: Automatic differentiation and JIT compilation for optimal performance
- 📊 **Probabilistic Inference**: Full posterior distributions for shear estimates
- 🔧 **Modular Design**: Flexible architecture for easy extension and customization
- 🎯 **GPU Acceleration**: Seamless GPU support for large-scale analyses
- 📈 **Scalable**: Efficient handling of large imaging surveys

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/CosmoStat/SHINE.git
cd SHINE

# Install dependencies (recommended: use a virtual environment)
pip install -e .
```

## 🚀 Quick Start

```python
import shine
import jax.numpy as jnp

# Some TBD magic happens

# Get shear estimates
g1, g2 = posterior.mean()
```

## 🤝 Contributing

We welcome contributions! This project is in early development, and we're excited to collaborate with the community.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🏗️ Status

⚠️ **Early Development**: This project is under active development. APIs may change.

---

<div align="center">
  Developed by <a href="https://www.cosmostat.org/">CosmoStat Lab</a>
</div>