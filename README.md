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

```
sudo apt update
sudo apt install -y git python3 python3-venv python3-pip
```
## 2) Clone the Code repo (this git repo)
Clone the code repository, which contains two Python codes—single_turn_universal.py and multi_turn_universal.py—using the commands below.
```
git clone https://github.com/jameslee177/pyrit-redteam.git
cd pyrit-redteam
```

## 3) Clone the official PyRIT repo
It is still necessary to clone the official repository for testing, even if PyRIT has already been installed via pip, because the code relies on the official templates provided in the PyRIT repository.
```
git clone https://github.com/Azure/PyRIT.git
```

## 4) Install PyRIT
```
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install pyrit
```
Confirm installation:
```
pyrit_shell
```
```
╔══════════════════════════════════════════════════════════════════════════════════════════════╗
║                                                                                              ║
║                       ██████╗ ██╗   ██╗██████╗ ██╗████████╗                                  ║
║                       ██╔══██╗╚██╗ ██╔╝██╔══██╗██║╚══██╔══╝                                  ║
║                       ██████╔╝ ╚████╔╝ ██████╔╝██║   ██║                                     ║
║                       ██╔═══╝   ╚██╔╝  ██╔══██╗██║   ██║                                     ║
║                       ██║        ██║   ██║  ██║██║   ██║                                     ║
║                       ╚═╝        ╚═╝   ╚═╝  ╚═╝╚═╝   ╚═╝                                     ║
║                                                                                              ║
║                          Python Risk Identification Tool                                     ║
║                                Interactive Shell                                             ║
║                                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════════════════════╣
║                                                                                              ║
║  Commands:                                                                                   ║
║    • list-scenarios        - See all available scenarios                                     ║
║    • list-initializers     - See all available initializers                                  ║
║    • run <scenario> [opts] - Execute a security scenario                                     ║
║    • scenario-history      - View your session history                                       ║
║    • print-scenario [N]    - Display detailed results                                        ║
║    • help [command]        - Get help on any command                                         ║
║    • exit                  - Quit the shell                                                  ║
║                                                                                              ║
║  Quick Start:                                                                                ║
║    pyrit> list-scenarios                                                                     ║
║    pyrit> run foundry_scenario --initializers openai_objective_target load_default_datasets  ║
║                                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════════════════════╝

pyrit> 
pyrit> 
pyrit> exit

Goodbye!
```








