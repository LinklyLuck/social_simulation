# 🌐 SocialSimBench V3

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B.svg)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**An Interactive LLM-Powered Multi-Agent Social Media Simulation + Benchmark Builder (ACL 2026 Demo Track)**

SocialSimBench is a platform for simulating social media dynamics using **LLM-powered agents**. It supports multiple platform modes (Twitter, Reddit, Weibo), multiple network topologies, and a **benchmark task pool** for studying information diffusion, misinformation/rumor correction, stance evolution, collaboration, and polarization. It also supports **dataset-grounded runs** via **Kaggle import** and local file upload, and exports **reproducible episode artifacts** (config + logs + metrics + plots).

<p align="center">
  <img src="docs/images/demo.gif" alt="SocialSimBench Demo" width="800"/>
</p>

---

## 🚀 2–4 Minute Demo Flow (what reviewers/attendees do)

1. **Select data source**: Generated / Kaggle / Upload  
2. **Pick platform mode**: Twitter / Reddit / Weibo / Generic  
3. **Pick a task** (5 tasks) + tune parameters  
4. **Run simulation** (T rounds)  
5. **Inspect**: cascade graph + stance trajectories + polarization curve + metrics  
6. **Export**: episode bundle (for reproducibility + benchmarking)

---

## ✨ Features

### 🤖 Multi-Agent Simulation Engine
- **LLM-Powered Agents**: agents generate replies/decisions using pluggable LLM backends
- **Trust Networks**: receiver-to-sender trust weights modulate propagation
- **Stance Evolution**: 5-level stance system with dynamic updates (normalized for metrics)
- **Platform Modes**:
  - Twitter: rapid spread, shallower cascades  
  - Reddit: deeper conversational threads  
  - Weibo: higher virality patterns  
  - Generic: neutral baseline

### 📊 Benchmark Task Pool
| Task | Description | Key Metrics |
|------|-------------|-------------|
| **Information Diffusion** | Track how content spreads through network | Coverage, Cascade Depth, R₀ proxy |
| **Rumor Detection** | Simulate misinformation spread and correction | Detection Rate/F1, Time-to-Correction |
| **Stance Evolution** | Model opinion change over time | Stance Changes, Distribution Entropy |
| **Multi-Role Collaboration** | Heterogeneous agents solving problems | Coordination Efficiency, Information Coverage |
| **Group Polarization** | Measure opinion clustering | Polarization Index, Echo Chamber Intensity |

### 🔌 Pluggable LLM Backends
- **OpenAI**: GPT-4o, GPT-4o-mini, GPT-3.5-turbo
- **DeepSeek**: DeepSeek-chat, DeepSeek-coder
- **Anthropic**: Claude-3.5-sonnet, Claude-3-haiku
- **Mock**: Template-based responses for testing (no API needed)

### 📁 Flexible Data Sources
- **Kaggle Integration**: Import real datasets via API (stored in `datalake/`)
- **Synthetic Generation**: Barabási-Albert, Watts-Strogatz, Erdős-Rényi networks
- **File Upload**: CSV/JSON network and content files

---

## 📌 Core Formulas (Metrics + Dynamics)

> Note: GitHub may not render LaTeX on all pages by default; the equations are included for clarity.

**Share probability** (message from agent \(i\) to neighbor \(j\)):

\[
P_j(\text{share}\mid i\to j)=\mathrm{clip}_{[0,1]}\Big(p_{base}\cdot\beta_{platform}\cdot(\alpha\cdot\tau_{j,i}+(1-\alpha)\cdot\tau_0)\Big)
\]

where \(\tau_{j,i}\) is receiver-to-sender trust, and \(\tau_0=0.5\) is a baseline noise term.

**Coverage**
\[
\text{Coverage}=\frac{|I|}{|N|}
\]
where \(I\) is the set of agents that have received the target message at least once.

**Cascade depth**
\[
d(m)=
\begin{cases}
0 & \text{if } parent(m)=\varnothing \\
d(parent(m))+1 & \text{otherwise}
\end{cases}
\]

**Polarization (size-weighted between-group variance)** (stance normalized to \([-1,1]\)):
\[
\text{Polarization}=\sqrt{\sum_{c\in C}\frac{n_c}{N}\,(\bar{s}_c-\bar{s})^2}
\]

---

## 🚀 Quick Start

### Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/socialsimbench.git
cd socialsimbench

# Install dependencies
pip install -r requirements.txt

# Optional: Install LLM backends
pip install openai          # For OpenAI/DeepSeek
pip install anthropic       # For Anthropic
pip install kaggle          # For Kaggle dataset import
```

### Run the Application
```bash
cd social_simulation
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`.

---

## 📖 Usage Guide

### Basic Workflow
1. **Configure Network**
   - Select data source (Generated / Kaggle / Upload)
   - Choose network topology and size
   - Set platform mode (Twitter / Reddit / Weibo / Generic)

2. **Configure LLM**
   - Select provider (Mock for testing, or real API)
   - Enter API key
   - Choose model

3. **Select Task**
   - Pick from 5 benchmark tasks
   - Configure task-specific parameters (e.g., correction delay for rumor)

4. **Run + Analyze**
   - Run simulation for T rounds
   - Inspect cascade/stance/polarization plots
   - Export episode artifacts (config/logs/metrics)

### Example: Running a Polarization Study
```python
from simulation.manager import SimulationManager
from tasks.tasks import GroupPolarizationTask

manager = SimulationManager()
manager.setup(
    n_agents=30,
    network_type='watts_strogatz',
    platform_mode='twitter',
    trust_weight=0.7,
    seed=42
)

task = GroupPolarizationTask(manager)
task.setup(topic='Climate Policy', n_communities=2)
task.run(n_rounds=10, infection_probability=0.5)

results = task.evaluate()
print(f"Polarization Index: {results['final_polarization']:.3f}")
```

---

## 🧱 Reproducibility: Episode Artifacts
Each run exports an **episode bundle** (recommended for benchmarking / ablations):
- `episode_config.json` (seed + all params)
- `events.jsonl` (message log with `origin_id`, `parent_id`, `depth`, round)
- `metrics.json` (task metrics + per-round traces)
- `plots/` (charts + optional interactive graph html)

This supports ablations across LLM backends, platform modes, and intervention schedules without changing evaluation code.

---

## 🏗️ Architecture
```text
socialsimbench/
├── social_simulation/
│   ├── app.py                    # Streamlit UI
│   ├── config.py                 # Configuration and enums
│   ├── agents/
│   │   ├── agent.py              # Agent + stance updates
│   │   └── llm_service.py        # Pluggable LLM backends
│   ├── simulation/
│   │   ├── manager.py            # Simulation orchestration
│   │   └── network.py            # Topology generation + communities
│   ├── tasks/
│   │   └── tasks.py              # 5 benchmark tasks
│   ├── data/
│   │   └── loader.py             # Kaggle & file import -> datalake
│   ├── visualization/
│   │   └── visualizer.py         # Charts + cascade graphs
│   └── evaluation/
│       └── metrics.py            # Metrics (coverage/depth/polarization)
└── docs/
    └── images/                   # Demo assets
```

---

## 📊 Supported Datasets

### Kaggle Datasets (Recommended)
| Dataset | Type | Use Case |
|---------|------|----------|
| `kazanova/sentiment140` | Content | Sentiment analysis, opinion spread |
| `clmentbisaillon/fake-and-real-news-dataset` | Content | Rumor detection |
| `ashwinpathak/facebook-social-network` | Network | Social network topology |
| `mathurinache/twitter-edge-nodes` | Network | Twitter network structure |

### Custom Datasets
Upload CSV files with:
- **Network**: `source,target` columns (or `from,to`, `node1,node2`)
- **Content**: `text` column (or `content`, `tweet`, `message`)

---

## ⚙️ Configuration

### Environment Variables (Optional)
```bash
export OPENAI_API_KEY="sk-..."
export DEEPSEEK_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-..."
export KAGGLE_USERNAME="your_username"
export KAGGLE_KEY="your_key"
```

### Platform Mode Parameters
| Mode | Share Boost | Description |
|------|-------------|-------------|
| Twitter | 1.2× | Rapid, shallow cascades |
| Reddit | 0.8× | Deep conversation threads |
| Weibo | 1.3× | Viral spread patterns |
| Generic | 1.0× | Neutral baseline |

---

## 🔐 Kaggle Notes (Licensing)
SocialSimBench **does not redistribute** Kaggle datasets. Users must download datasets via Kaggle API and comply with each dataset’s license/terms.

---

## 🔧 Extending SocialSimBench

### Adding a New Task
```python
from tasks.tasks import BaseTask

class MyCustomTask(BaseTask):
    def setup(self, **kwargs):
        pass

    def run(self, n_rounds, **kwargs):
        for round_num in range(n_rounds):
            self.manager.run_round(round_num)

    def evaluate(self):
        return {"my_metric": 0.0}
```

### Adding a New LLM Backend
```python
from agents.llm_service import LLMClient, LLMResponse

class MyLLMClient(LLMClient):
    def generate(self, prompt, system_prompt="", **kwargs):
        response = my_api_call(prompt)
        return LLMResponse(content=response, model="my-model", tokens_used=100)
```

---

## 📝 Citation
```bibtex
@misc{socialsimbench2026demo,
  title={SocialSimBench V3: An Interactive LLM-Powered Multi-Agent Social Media Simulation + Benchmark Builder},
  author={Anonymous},
  year={2026},
  note={ACL 2026 System Demonstrations submission}
}
```

---

## ⚠️ Limitations & Ethics
- LLM agents may generate biased or unsafe content; use safe prompts / demo-safe topics.
- Simulation outcomes are not predictions of real-world behavior.
- Respect dataset licenses (especially for Kaggle-imported content).


---

<p align="center">
  Thanks for Using Social Simulation Research
</p>
