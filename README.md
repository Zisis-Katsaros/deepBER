# deepBER

## Setup

This project is best run inside a virtual environment so its Python packages stay isolated from your system installation.
Inside the project root:

### Windows

1. Create and activate a virtual environment:

	```powershell
	py -3.12 -m venv .venv
	.\.venv\Scripts\Activate.ps1
	```

2. Upgrade `pip` and install the project dependencies:

	```powershell
	python -m pip install --upgrade pip
	pip install -r requirements.txt
	```

3. Run the project entry point:

	```powershell
	python main.py
	```

### Notes

- Use a Python version supported by the PyTorch build you install. In practice, Python 3.11 or 3.12 is usually the safest choice for this project.
- `torch` installation depends on your Python version and whether you want CPU or GPU support. If the default install fails, follow the official PyTorch installation instructions for your platform.
