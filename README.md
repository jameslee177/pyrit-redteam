# PyRIT Red Team Scripts (Single-turn + Multi-turn)

This repo contains two interactive Python scripts that run AI red teaming tests using **Microsoft PyRIT** (Python Risk Identification Tool for generative AI).

- **Single-turn script**: quick, one-shot style red teaming
- **Multi-turn script**: Crescendo, TAP (Tree of Attacks with Pruning), Violent Durian strategy, etc.

> ⚠️ Use responsibly. Only test systems you own or have explicit permission to test.

---

## 1) Prerequisites

- Ubuntu 20.04+ (or equivalent Linux)
- Python **3.10+** (3.11/3.12 also OK in many environments)
- `git`
- API access to an LLM endpoint (OpenAI / Azure OpenAI / Ollama / custom OpenAI-compatible)

Install OS packages:

```bash
sudo apt update
sudo apt install -y git python3 python3-venv python3-pip
