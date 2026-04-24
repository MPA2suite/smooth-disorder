python3.12 -m venv .venv_bne
source ./.venv_bne/bin/activate

pip install --upgrade pip
pip install -e ".[jupyter,dev]"
