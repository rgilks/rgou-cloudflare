# Royal Game of Ur

[![CI/CD](https://github.com/rgilks/rgou-cloudflare/actions/workflows/deploy.yml/badge.svg)](https://github.com/rgilks/rgou-cloudflare/actions/workflows/deploy.yml)

<img src="/docs/screenshot.png" alt="rgou Screenshot" style="max-width:408px; max-height:712px; width:100%; height:auto;" />

<div align="center">
  <a href='https://ko-fi.com/N4N31DPNUS' target='_blank'><img height='36' style='border:0px;height:36px;' src='https://storage.ko-fi.com/cdn/kofi2.png?v=6' border='0' alt='Buy Me a Coffee at ko-fi.com' /></a>
  <hr />
</div>

A modern web implementation of the ancient Royal Game of Ur (2500 BCE) with dual AI opponents, offline support, and beautiful animations. Built with Next.js, TypeScript, Rust, and WebAssembly.

## 🎮 Play Now

**[Play Online](https://rgou.tre.systems/)** - Works in any modern browser, no installation required.

## ✨ Features

- **Dual AI System**: Classic expectiminimax + Neural network
- **Browser-Native**: All AI runs locally via WebAssembly
- **Offline Support**: PWA with full offline gameplay
- **Modern UI**: Responsive design with animations and sound effects

## 🚀 Quick Start

### Prerequisites

- **Node.js 20+** ([Download](https://nodejs.org/)) - Required for Next.js 15
- **Rust & Cargo** ([Install](https://www.rust-lang.org/tools/install)) - For WebAssembly compilation
- **wasm-pack**: `cargo install wasm-pack --version 0.12.1 --locked` - For WASM builds

**Note**: This project was developed on an M1 Mac. While it should work on other platforms, some optimizations (especially for ML training) are specifically tuned for Apple Silicon.

### Development Setup

```bash
# Clone the repository
git clone https://github.com/rgilks/rgou-cloudflare.git
cd rgou-cloudflare

# Install dependencies
npm install

# Set up local database
npm run migrate:local

# Start development server
npm run dev
```

The game will open at http://localhost:3000

### First Run Notes

- The first run may take longer as it builds WebAssembly assets
- If you encounter WASM build issues, run: `npm run build:wasm-assets`
- For ML training, you'll need Rust and Apple Silicon Mac (see [Training System](./docs/training-system.md))

### Common Setup Issues

- **WASM Build Failures**: Ensure wasm-pack version 0.12.1 is installed
- **Database Issues**: Run `npm run migrate:local` to set up local SQLite
- **Dependency Issues**: Try `npm run nuke` to reset the environment

See [Troubleshooting Guide](./docs/troubleshooting.md) for detailed solutions.

## 🤖 AI System

The project features two distinct AI opponents:

- **Classic AI**: Expectiminimax algorithm with alpha-beta pruning
- **ML AI**: Neural network trained through self-play
  - **v2 Model**: **44% win rate vs Classic AI** - **Best Performance** ✅
  - **Fast Model**: 36% win rate vs Classic AI - **Competitive**
  - **v4 Model**: 32% win rate vs Classic AI - **Needs Improvement** ⚠️
  - **Hybrid Model**: 30% win rate vs Classic AI - **Needs Improvement** ⚠️

Both AIs run entirely in the browser via WebAssembly. See [AI System](./docs/ai-system.md) for details.

## 🧠 Machine Learning

Train and improve the neural network AI with two training systems:

### 🚀 PyTorch Training (Recommended)

Fast GPU-accelerated training using PyTorch with Rust data generation:

```bash
# Install PyTorch dependencies
pip install -r requirements.txt

# Quick test
./ml/scripts/train-pytorch.sh 100 10 0.001 32 3 test_weights.json

# Standard training
./ml/scripts/train-pytorch.sh 1000 50 0.001 32 3 ml_ai_weights_v6.json

# Production training
./ml/scripts/train-pytorch.sh 2000 75 0.001 32 4 ml_ai_weights_v7.json

# Convert PyTorch weights for Rust use
python3 ml/scripts/load_pytorch_weights.py ml/weights/ml_ai_weights_v7.json --test
```

**PyTorch Features:**
- **🎮 GPU Acceleration**: Automatic CUDA/MPS detection and utilization
- **🦀 Rust Data Generation**: Fast parallel game simulation using all CPU cores
- **⚡ Optimized Training**: PyTorch's highly optimized neural network operations
- **📊 Advanced Features**: Dropout, Adam optimizer, early stopping
- **🔄 Seamless Integration**: Weights automatically compatible with Rust system
- **📁 Organized Storage**: Training data stored in `~/Desktop/rgou-training-data/`

### 🦀 Rust Training (Legacy)

Pure Rust training system for maximum compatibility:

```bash
# Quick test
npm run train:rust:quick

# Standard training
npm run train:rust

# Production training
npm run train:rust:production

# Custom training
cd worker/rust_ai_core && cargo run --bin train --release --features training -- train 2000 75 0.001 32 4 custom_weights.json
```

**Rust Features:**
- **🦀 Pure Rust**: No external dependencies
- **⚡ CPU Training**: Efficient neural network training with custom implementation
- **🍎 Apple Silicon Optimization**: Uses 8 performance cores on M1/M2/M3
- **📊 Comprehensive Logging**: Detailed progress tracking and performance metrics

See [Training System](./docs/training-system.md) for complete training guide.

## 🧪 Testing

```bash
# Run all tests
npm run check

# Quick tests (10 games)
npm run test:rust:quick

# Comprehensive tests (100 games)
npm run test:rust:slow

# End-to-end tests
npm run test:e2e
```

See [Testing Strategy](./docs/testing-strategy.md) for detailed testing information.

## 🏗️ Architecture

- **Frontend**: Next.js with React, TypeScript, Tailwind CSS
- **AI Engine**: Rust compiled to WebAssembly (client-side only)
- **Database**: Cloudflare D1 with Drizzle ORM
- **Deployment**: Cloudflare Pages with GitHub Actions

The project evolved from hybrid client/server AI to pure client-side execution for optimal performance and offline capability. See [Architecture Overview](./docs/architecture-overview.md) for detailed system design.

## 📚 Documentation

### Core System

- **[Architecture Overview](./docs/architecture-overview.md)** - System design and components
- **[AI System](./docs/ai-system.md)** - Classic expectiminimax AI and ML AI implementation
- **[ML System Overview](./docs/ml-system-overview.md)** - Complete ML system guide with PyTorch and Rust training
- **[Training System](./docs/training-system.md)** - Detailed training system documentation
- **[Game Rules and Strategy](./docs/game-rules-strategy.md)** - Game rules and strategic concepts

### Development

- **[Testing Strategy](./docs/testing-strategy.md)** - Testing approach and methodology
- **[Troubleshooting Guide](./docs/troubleshooting.md)** - Common issues and solutions
- **[TODO](./docs/TODO.md)** - Consolidated task list and improvements

### Infrastructure

- **[Cloudflare Worker Infrastructure](./docs/cloudflare-worker-infrastructure.md)** - Preserved server-side infrastructure

### Historical

- **[AI Development History](./docs/ai-development-history.md)** - Historical experiments and findings

## 🔧 Troubleshooting

**Common Issues:**

- **WASM Build**: Run `npm run build:wasm-assets` before `npm run check`
- **ML Training**: Ensure GPU available, use `--reuse-games` for faster iteration
- **Deployment**: Pin exact dependency versions for Cloudflare compatibility

See [Troubleshooting Guide](./docs/troubleshooting.md) for detailed solutions.

## 📄 License

Open source. See [LICENSE](LICENSE).
