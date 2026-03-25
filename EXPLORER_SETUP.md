# Explorer setup reminders

## Environments

CPU:
- Path: /projects/ComputationalPhilosophyLab/TwitterDataAnalysis/envs/twitter-ai
- Kernel: Python (twitter-ai)

GPU:
- Path: /projects/ComputationalPhilosophyLab/TwitterDataAnalysis/envs/twitter-ai-gpu
- Kernel: Python (twitter-ai-gpu)

## OOD settings

GPU sessions:
- Partition: gpu-interactive
- GPU type: t4
- CUDA module: cuda/12.1.1

## Always check this

### Correct Python
import sys
print(sys.executable)

### GPU available
import torch
print(torch.cuda.is_available())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))

## Notebook setup pattern

Replace Colab setup with:

import sys
from pathlib import Path

REPO_ROOT = Path("/projects/ComputationalPhilosophyLab/TwitterDataAnalysis/code/twitter_ai")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config_paths import *

## Key warnings

- Kernel name can lie, always check sys.executable
- conda activate can lie on Explorer, use full python path if needed
- Check for file writes before running notebooks
