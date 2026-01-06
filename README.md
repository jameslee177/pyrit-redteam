# PyRIT Red Team Codes (Single-turn + Multi-turn)

This repo contains two interactive Python codes that run AI red teaming tests using **Microsoft PyRIT** (Python Risk Identification Tool for generative AI).

- **Single-turn code**: quick, one-shot style red teaming
- **Multi-turn code**: Crescendo, TAP (Tree of Attacks with Pruning), Violent Durian strategy, etc.

> ⚠️ Use responsibly. Only test systems you own or have explicit permission to test.

---

## 1) Prerequisites

- Ubuntu 24.04+ (tested)
- 4 x CPU / 8GB+ Memory (tested)
- Python **3.10+**
- `git`
- API access to an LLM endpoint (OpenAI / Azure OpenAI / Ollama / custom OpenAI-compatible)

Install OS packages:

```bash
sudo apt update
sudo apt install -y git python3 python3-venv python3-pip
