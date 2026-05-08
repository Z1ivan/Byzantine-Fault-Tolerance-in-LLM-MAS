# Byzantine Fault Tolerance in LLM-Based Multi-Agent Systems (CP-WBFT)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![AAAI 2026](https://img.shields.io/badge/AAAI-2026-red.svg)](https://aaai.org/conference/aaai/aaai-26/)

Official implementation of the paper:

> **"Rethinking the Reliability of Multi-agent System: A Perspective from Byzantine Fault Tolerance"**  
> *AAAI Conference on Artificial Intelligence (AAAI 2026)*

This repository provides a clean, academic implementation of the **CP-WBFT** (Confidence-Probe Weighted Byzantine Fault Tolerance) framework. CP-WBFT achieves superior performance under extreme Byzantine conditions (85.7% fault rate: 6 out of 7 nodes malicious, you can freely modify the number of nodes), including:

- 🔬 **Pilot experiments** comparing traditional vs LLM-based agents under Byzantine faults
- 🎯 **PCP (Prompt-level Confidence Probe)** for API-based LLMs
- 🧠 **HCP (Hidden-level Confidence Probe)** for local LLMs
- 📊 Experiments on **GSM8K**, **XSTest/Safe**, and **CommonsenseQA** (10 questions each)
- 🔧 **Pretrained confidence probes** (no training required)

## 🏗️ Architecture Overview

CP-WBFT is a **protocol layer** deliberately decoupled from the content it reasons about. The framework sits between your input task and the underlying models:

```
┌─────────────────────────────────────────────────┐
│              Your Input / Task                  │
│   (math problems, safety questions, MCQ, etc.)  │
├─────────────────────────────────────────────────┤
│           Network Topology Layer                │
│   (complete, star, chain, tree, random...)      │
├─────────────────────────────────────────────────┤
│         Byzantine Consensus Protocol            │
│   (majority vote  →  CP-WBFT weighted vote)     │
├─────────────────────────────────────────────────┤
│           Confidence Probe Layer                │
│   (PCP via prompt  /  HCP via hidden states)    │
├─────────────────────────────────────────────────┤
│              Agent / Model Layer                │
│   (traditional lookup / API LLM / local LLaMA) │
└─────────────────────────────────────────────────┘
```

The protocol does not care what the question is — it only needs to know what answer each agent gave and how confident they were. The three included datasets are chosen to stress-test this generality across different task types:

| Dataset | Task type | Why it tests the protocol |
|---|---|---|
| **GSM8K** | Math — single numeric answer | Objectively verifiable correct/wrong |
| **XSTest/Safe** | Safety classification | Binary judgment, not factual recall |
| **CommonsenseQA** | Multiple-choice reasoning | Structured answer space (A/B/C/D) |

Because the consensus layer is content-agnostic, the same framework can in principle be applied to any multi-agent decision task where robustness against compromised nodes matters: code review, medical diagnosis agreement, fraud detection, document classification, and more.

---

## 📑 Table of Contents

- [Quick Start](#-quick-start)
- [Installation](#-installation)
- [Usage](#-usage)
  - [Pilot Experiments](#1-pilot-experiments)
  - [Prompt Probe (PCP)](#2-prompt-probe-pcp)
  - [Decoder Probe (HCP)](#3-decoder-probe-hcp)
- [Datasets](#-datasets)
- [Pretrained Models](#-pretrained-models)
- [Project Structure](#-project-structure)
- [Results & Visualization](#-results--visualization)
- [Citation](#-citation)
- [License](#-license)

---

## 🚀 Quick Start

### 5-Minute Demo (No GPU Required)

```bash
# Clone the repository
git clone https://github.com/Z1ivan/Byzantine-Fault-Tolerance-in-LLM-MAS.git
cd Byzantine-Fault-Tolerance-in-LLM-MAS

# Install dependencies
pip install numpy scipy pandas scikit-learn matplotlib networkx python-dotenv openai

# Run a single question (quick smoke-test, no API needed)
python methods/unified_entry.py pilot --dataset-type gsm8k --topology complete --agents 7 --malicious 6 --agent-type traditional --mode single --rounds 1

# Run all 10 questions
python methods/unified_entry.py pilot --dataset-type gsm8k --topology complete --agents 7 --malicious 6 --agent-type traditional --mode all --rounds 1

# English log output (optional)
python methods/unified_entry.py pilot --dataset-type gsm8k --topology complete --agents 7 --malicious 6 --agent-type traditional --mode single --rounds 1 --lang en

# Check results in: results/pilot/gsm8k/traditional/
```

> **Note**: `--mode` accepts `single` (one question) or `all` (all 10 questions). The value `test` is not valid.

**Expected Output**: Consensus accuracy, topology visualization, and detailed analysis reports.

---

## 📦 Installation

### Requirements

- **Python**: 3.8+ (tested on 3.9-3.11)
- **OS**: Linux / macOS / Windows
- **GPU**: Optional (required only for HCP/Decoder Probe experiments)

### Option 1: Minimal Installation (Pilot + PCP)

```bash
pip install numpy scipy pandas scikit-learn matplotlib networkx
pip install python-dotenv pydantic openai
```

### Option 2: Full Installation (All Experiments)

```bash
pip install -r requirements.txt
```

For GPU support (HCP experiments):

```bash
# CUDA 11.8+
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# CPU only
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### Windows: UTF-8 Encoding

On Windows, Python defaults to `cp1252` encoding which causes a `charmap codec` error when the visualization code reads files containing Chinese characters. Fix this by setting `PYTHONUTF8=1`, which tells Python to use UTF-8 everywhere.

**Option A — Git Bash profile (applies to your terminal sessions):**

```bash
echo 'export PYTHONUTF8=1' >> ~/.bashrc
source ~/.bashrc
```

**Option B — Windows user environment variable (applies system-wide):**

```powershell
[System.Environment]::SetEnvironmentVariable('PYTHONUTF8', '1', 'User')
```

Then reopen your terminal. Without this, experiments still run and results are saved correctly, but the visualization PNG files will not be generated.

### English log output

All log messages are in Chinese by default. Pass `--lang en` to switch to English:

```bash
python methods/unified_entry.py pilot --dataset-type gsm8k --topology complete \
  --agents 7 --malicious 6 --agent-type traditional --mode single --rounds 1 --lang en
```

### Configuration

Create a `.env` file from the template:

```bash
cp .env.example .env
```

#### Option A — OpenAI

```bash
API_KEY="your_openai_api_key_here"
API_BASE_URL="https://api.openai.com/v1"
```

#### Option B — OpenRouter

[OpenRouter](https://openrouter.ai) provides a single API key that routes to 400+ models (GPT-5.5, Claude Opus 4.7, Gemini, Llama, DeepSeek, and more) with automatic fallback across providers. Get your key at [openrouter.ai/keys](https://openrouter.ai/keys).

```bash
OPENROUTER_API_KEY="your_openrouter_api_key_here"
API_BASE_URL="https://openrouter.ai/api/v1"

# Optional — for attribution in OpenRouter's usage dashboard
OPENROUTER_SITE_URL="https://your-site.com"
OPENROUTER_SITE_NAME="CP-WBFT"
```

Then use any OpenRouter model name as `--strong-model` or `--weak-model`:

```bash
# Example: Claude as honest agent, GPT-4o-mini as malicious agent
python methods/unified_entry.py prompt_probe --dataset-type gsm8k --topology complete --agents 7 --malicious 6 --strong-model anthropic/claude-opus-4.7 --weak-model openai/gpt-4o-mini --lang en

# Example: Gemini vs DeepSeek
python methods/unified_entry.py prompt_probe --dataset-type gsm8k --topology complete --agents 7 --malicious 6 --strong-model google/gemini-3.1-pro-preview --weak-model deepseek/deepseek-chat --lang en
```

Browse available models at [openrouter.ai/models](https://openrouter.ai/models).

---

## 📖 Usage

### 1. Pilot Experiments

Compare traditional agents vs LLM-based agents under Byzantine faults.

#### Traditional Agents (No API Required)

```bash
python methods/unified_entry.py pilot \
  --dataset-type gsm8k \
  --topology complete \
  --agents 7 \
  --malicious 6 \
  --agent-type traditional
```

#### LLM Agents

```bash
python methods/unified_entry.py pilot \
  --dataset-type gsm8k \
  --topology complete \
  --agents 7 \
  --malicious 6 \
  --agent-type llm \
  --strong-model gpt-4o-mini \
  --weak-model gpt-3.5-turbo
```

#### All Supported Topologies

```bash
# Complete graph (fully connected)
--topology complete

# Tree topology
--topology tree

# Star topology
--topology star

# Chain topology
--topology chain

# Random graph
--topology random

# Layered graph
--topology layered_graph
```

---

### 2. Prompt Probe (PCP)

Confidence-weighted consensus using prompt-level confidence extraction.

#### GSM8K Dataset

```bash
python methods/unified_entry.py prompt_probe \
  --dataset-type gsm8k \
  --topology complete \
  --agents 7 \
  --malicious 6 \
  --strong-model gpt-4o-mini \
  --weak-model gpt-3.5-turbo
```

#### XSTest/Safe Dataset

```bash
python methods/unified_entry.py prompt_probe \
  --dataset-type safe \
  --topology tree \
  --agents 7 \
  --malicious 6
```

#### CommonsenseQA Dataset

```bash
python methods/unified_entry.py prompt_probe \
  --dataset-type commonsense \
  --topology complete \
  --agents 7 \
  --malicious 6
```

---

### 3. Decoder Probe (HCP)

Hidden-layer confidence probe using local LLaMA models.

**Prerequisites**: Download LLaMA models and create symbolic links:

```bash
# Link to your LLaMA model directories
ln -s /path/to/llama3-8b models/LLama-3-8B-Instruct
ln -s /path/to/llama3.1-8b models/LLama-3.1-8B-Instruct
```

#### Run HCP Experiment

```bash
python methods/unified_entry.py decoder \
  --dataset-type gsm8k \
  --topology complete \
  --agents 7 \
  --malicious 6 \
  --llama3-model-path models/LLama-3-8B-Instruct \
  --probe-path lcd_models/gsm8k/3_pooled_layer16_pca256_logistic/lcd_model.pkl
```

**Pretrained probes** are provided in `lcd_models/` for all datasets.

---

### Command-Line Arguments

| Argument | Description | Default | Required |
|----------|-------------|---------|----------|
| `--dataset-type` | Dataset to use (`gsm8k`, `safe`, `commonsense`) | - | ✅ |
| `--topology` | Network topology (`complete`, `tree`, `star`, `chain`, `random`, `layered_graph`) | `complete` | ❌ |
| `--agents` | Total number of agents | `7` | ❌ |
| `--malicious` | Number of malicious agents | `6` | ❌ |
| `--agent-type` | Agent type (`traditional`, `llm`) | `llm` | ❌ |
| `--strong-model` | Model for normal agents | `gpt-4o-mini` | ❌ |
| `--weak-model` | Model for malicious agents | `gpt-3.5-turbo` | ❌ |
| `--rounds` | Number of consensus rounds | `1` | ❌ |
| `--mode` | Execution mode (`test`, `all`) | `test` | ❌ |

---

## 📊 Datasets

All datasets are preprocessed and stored in `data/byzantine/`:

| Dataset | Questions | Task | Source |
|---------|-----------|------|--------|
| **GSM8K** | 10 | Math reasoning | Grade school math problems |
| **XSTest/Safe** | 10 | Safety assessment | Safety evaluation benchmark |
| **CommonsenseQA** | 10 | Commonsense reasoning | Multiple-choice QA |

### Dataset Format

```json
{
  "question_id": "unique_id",
  "question": "Question text",
  "answer": "Correct answer",
  "options": ["A", "B", "C", "D"]  // for CommonsenseQA
}
```

---

## 🧠 Pretrained Models

Pretrained confidence probes are located in `lcd_models/`:

```
lcd_models/
├── gsm8k/
│   ├── 3_pooled_layer16_pca256_logistic/lcd_model.pkl
│   └── 3.1_pooled_layer12_pca256_logistic/lcd_model.pkl
├── safe/
│   ├── 3_pooled_layer17_pca256_logistic/lcd_model.pkl
│   └── 3.1_pooled_layer12_pca256_logistic/lcd_model.pkl
└── commonsense/
    ├── 3_pooled_layer14_pca256_mlp/lcd_model.pkl
    └── 3.1_query_layer14_pca256_mlp/lcd_model.pkl
```

**Model Naming Convention**:
- `3` / `3.1`: LLaMA-3 / LLaMA-3.1
- `pooled` / `query`: Hidden state extraction method
- `layer{N}`: Which transformer layer
- `pca256`: PCA dimensionality reduction to 256
- `logistic` / `mlp`: Probe classifier type

Each model includes `metrics.json` with training accuracy and performance metrics.

---

## 📂 Project Structure

```
Byzantine-Fault-Tolerance-in-LLM-MAS/
├── config/                  # Configuration management
│   ├── base_config.py
│   ├── prompt_probe_config.py
│   └── decoder_probe_config.py
├── core/                    # Core components
│   ├── agents/              # Agent implementations
│   │   ├── traditional_agent.py
│   │   ├── llm_agent.py
│   │   └── decoder_agent.py
│   ├── consensus/           # Consensus algorithms
│   │   └── consensus_algorithms.py  # CP-WBFT implementation
│   ├── confidence/          # Confidence extraction
│   │   └── lcd_confidence_extractor.py
│   ├── topologies/          # Network topologies
│   └── visualization/       # Result visualization
├── methods/
│   └── unified_entry.py     # Main experiment entry point
├── data/byzantine/          # Preprocessed datasets
├── lcd_models/              # Pretrained confidence probes
├── results/                 # Experiment outputs
└── tools/                   # Utility scripts
    ├── hidden_states_extract.py  # Hidden state extraction
    └── pareto_analyzer.py        # Pareto optimization
```

---

## 📈 Results & Visualization

### Output Directory Structure

After running an experiment, results are saved in:

```
results/{method}/{dataset}/{agent_type}/{topology}_{agents}_{malicious}/
├── {experiment_id}.json                    # Raw experiment results
├── {experiment_id}.summary.json            # Evaluation metrics summary
├── {prefix}_report.txt                     # Detailed analysis report
├── {prefix}_topology.png                   # Network topology visualization
├── {prefix}_core_metrics.png               # Core metrics charts
└── {prefix}_attack_effect.png              # Byzantine attack effect analysis
```

### Key Metrics

- **Consensus Accuracy**: Final consensus correctness rate
- **Node Accuracy**: Individual agent accuracy (initial vs final)
- **Answer Change Rate**: Percentage of agents changing answers
- **Byzantine Tolerance**: Performance degradation under malicious agents

### Visualization Files Explained

1. **`{prefix}_topology.png`**: Network structure with agent states
   - 🟢 Green circles: Normal LLM agents
   - 🔴 Red circles: Malicious LLM agents  
   - 🟩 Green squares: Normal traditional agents
   - 🟥 Red squares: Malicious traditional agents

2. **`{prefix}_core_metrics.png`**: Multi-panel metrics dashboard
   - Consensus accuracy trends
   - Initial vs final agent accuracy
   - Answer change rates
   - Round-by-round performance

3. **`{prefix}_attack_effect.png`**: Byzantine attack analysis
   - Attack success rate
   - Position sensitivity
   - Fault tolerance breakdown

4. **`{prefix}_report.txt`**: Human-readable analysis
   - Question-by-question results
   - Statistical summaries
   - Performance metrics
   - Topology-specific insights

---

## 🔬 Reproduce Paper Results

### Table 1: Pilot Experiments (Traditional vs LLM)

```bash
# Traditional agents
python methods/unified_entry.py pilot --dataset-type gsm8k --topology complete --agents 7 --malicious 6 --agent-type traditional

# LLM agents
python methods/unified_entry.py pilot --dataset-type gsm8k --topology complete --agents 7 --malicious 6 --agent-type llm
```

### Table 2: CP-WBFT Performance (PCP)

```bash
# Run for all 6 topologies
for topology in complete tree star chain random layered_graph; do
  python methods/unified_entry.py prompt_probe --dataset-type gsm8k --topology $topology --agents 7 --malicious 6
done
```

### Table 3: HCP vs PCP Comparison

```bash
# PCP
python methods/unified_entry.py prompt_probe --dataset-type gsm8k --topology complete --agents 7 --malicious 6

# HCP
python methods/unified_entry.py decoder --dataset-type gsm8k --topology complete --agents 7 --malicious 6
```

---

## 🛠️ Advanced Usage

### Extract Hidden States (Step 1: Data Preparation)

Extract hidden states from LLaMA models for probe training:

```bash
python tools/hidden_states_extract.py \
  --model-path models/LLama-3-8B-Instruct \
  --dataset-path data/byzantine/gsm8k/gsm8k_questions.json \
  --output-dir data/hidden_states/gsm8k/llama3
```

**Output Structure**:
```
data/hidden_states/gsm8k/llama3/
├── train/
│   ├── train_query_hidden_states.npy    # (N, L, D)
│   ├── train_pooled_hidden_states.npy   # (N, L, D)
│   └── train_data_with_labels.json      # Labels
└── test/
    └── ...
```

### Train Confidence Probes (Step 2: Probe Training)

Train custom confidence probes using the unified trainer:

#### Train Single Probe

```bash
# Train GSM8K probe for LLaMA-3.1
python core/unified_probe_trainer.py \
  --dataset gsm8k_llama31 \
  --probe-type pooled \
  --method logistic \
  --layer 12 \
  --pca-dim 256 \
  --epochs 100 \
  --save-model
```

#### Batch Test Multiple Configurations

```bash
# Find optimal layer and configuration
python core/unified_probe_trainer.py \
  --dataset gsm8k_llama3 \
  --test-layers 8 12 16 20 24 \
  --test-types pooled query answer \
  --test-dims 128 256 512 \
  --save-model
```

#### Train MLP Probe (for CommonsenseQA)

```bash
python core/unified_probe_trainer.py \
  --dataset commonsense_llama31 \
  --method mlp \
  --mlp-hidden-dims 128 64 \
  --dropout 0.1 \
  --layer 14 \
  --save-model
```

**Supported Datasets**:
- `gsm8k_llama3` / `gsm8k_llama31`
- `safe_llama3` / `safe_llama31`
- `commonsense_llama3` / `commonsense_llama31`

**Probe Types**:
- `pooled`: Averaged hidden states (most common)
- `query`: Query token states
- `answer`: Answer token states

**Output**:
```
lcd_outputs/gsm8k/3.1_pooled_layer12_pca256_logistic/
├── lcd_model.pkl       # Trained probe + PCA + Scaler
├── metrics.json        # Performance metrics
└── mlp_model.pth       # (if method=mlp)
```

### Pareto Analysis (Step 3: Optimal Configuration Selection)

```bash
python tools/pareto_analyzer.py \
  --probe-dir lcd_models/gsm8k \
  --output pareto_results.json
```

---

## 📄 Citation

If you find this work useful, please cite our paper (we will update the reference information later~):

**Paper**: [Paper](https://ojs.aaai.org/index.php/AAAI/article/view/40806)

```bibtex
@inproceedings{zheng2026rethinking,
  title={Rethinking the reliability of multi-agent system: A perspective from byzantine fault tolerance},
  author={Zheng, Lifan and Chen, Jiawei and Yin, Qinghong and Zhang, Jingyuan and Zeng, Xinyi and Tian, Yu},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={40},
  number={41},
  pages={35012--35020},
  year={2026}
}
```

---

## 🤝 Contributing

We welcome contributions and feedback from the community! 

This is an academic research project, and we acknowledge that there may be areas for improvement. We sincerely appreciate:
- 🐛 Bug reports and fixes
- 💡 Suggestions for improvements
- 📖 Documentation enhancements
- 🔬 Extension to new datasets or methods

Please feel free to:
- Open an issue for questions or discussions
- Submit a Pull Request with improvements
- Share your experiences and results

**Note**: This is my first work, and I am a researcher continuously learning and improving. I hope everyone can give me more kindness, patience, and help so that I can grow better. 🙏

---

## 📧 Contact

For questions or issues, please:
- Open an issue on GitHub
- Contact the authors (see paper for details)

---

## 🙏 Acknowledgments

This work builds upon research in Byzantine fault tolerance and large language models. We thank the community for their foundational contributions.

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
