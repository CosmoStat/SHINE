# SHINE: SHear INference Environment
<!-- ALL-CONTRIBUTORS-BADGE:START - Do not remove or modify this section -->
[![All Contributors](https://img.shields.io/badge/all_contributors-2-orange.svg?style=flat-square)](#contributors-)
<!-- ALL-CONTRIBUTORS-BADGE:END -->

<div align="center">
  <img src="assets/logo.png" alt="SHINE Logo" width="1024"/>

  **A JAX-powered framework for probabilistic shear estimation in weak gravitational lensing**

  [![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
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

### Run inference from a config file

SHINE is driven by YAML configuration files. Any parameter specified as a
distribution (e.g. `type: Normal`) becomes a latent variable; everything else
is fixed. To run the full pipeline (data generation → model building → MCMC):

```bash
python -m shine.main --config configs/test_run.yaml
```

Results (posterior samples in NetCDF format) are saved to the `results/`
directory by default. Override with `--output`:

```bash
python -m shine.main --config configs/test_run.yaml --output my_output/
```

### Pedagogical example

For a step-by-step walkthrough that builds the config inline and plots
diagnostics, see `examples/shear_inference.py`:

```bash
python examples/shear_inference.py
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
## Contributors ✨

Thanks goes to these wonderful people ([emoji key](https://allcontributors.org/docs/en/emoji-key)):

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tbody>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://centofantieze.github.io"><img src="https://avatars.githubusercontent.com/u/42658822?v=4?s=100" width="100px;" alt="Ezequiel Centofanti"/><br /><sub><b>Ezequiel Centofanti</b></sub></a><br /><a href="#ideas-CentofantiEze" title="Ideas, Planning, & Feedback">🤔</a> <a href="#projectManagement-CentofantiEze" title="Project Management">📆</a></td>
      <td align="center" valign="top" width="14.28%"><a href="http://sfarrens.github.io"><img src="https://avatars.githubusercontent.com/u/6851839?v=4?s=100" width="100px;" alt="Samuel Farrens"/><br /><sub><b>Samuel Farrens</b></sub></a><br /><a href="#ideas-sfarrens" title="Ideas, Planning, & Feedback">🤔</a> <a href="#projectManagement-sfarrens" title="Project Management">📆</a></td>
    </tr>
  </tbody>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->

This project follows the [all-contributors](https://github.com/all-contributors/all-contributors) specification. Contributions of any kind welcome!