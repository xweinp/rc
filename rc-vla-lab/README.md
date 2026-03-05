# VLA Lab: Vision-Language-Action Models with π0 and Libero

This lab provides a comprehensive introduction to Vision-Language-Action (VLA) models using Physical Intelligence's π0 model and the Libero robotics benchmark.

## Overview

In this lab, you will:
- Learn about VLA model architecture and capabilities
- Run π0 model on Libero benchmark tasks with MuJoCo simulation
- Experiment with different language prompts and observe robot behavior
- Fine-tune the model for specific manipulation tasks

## Installation

### 1. Create virtual environment and install dependencies

```bash
uv sync
```



### 4. Install OpenPI and LIBERO Python packages

```bash
uv sync
```

`openpi` and `libero` are declared in `pyproject.toml`.

### 5. Download LIBERO source with curl

Create the local `LIBERO/` directory used by the notebook setup:

```bash
curl -sL https://github.com/Lifelong-Robot-Learning/LIBERO/archive/refs/heads/master.tar.gz | tar -xz
mv LIBERO-master LIBERO
```

### 6. Download π0 Model Checkpoint (Optional)

To use the actual π0 model, download the checkpoint:

```bash
# Install gsutil if needed (managed by uv)
uv sync  # gsutil is declared in pyproject.toml

# Download checkpoint
mkdir -p checkpoints
.venv/bin/gsutil -m cp -r gs://openpi-assets/checkpoints/pi05_libero/ ./checkpoints/

# Or download manually from:
# https://console.cloud.google.com/storage/browser/openpi-assets/checkpoints/pi05_libero
```

**Note:** The lab includes mock implementations that work without the actual checkpoint, so you can start learning immediately.

## Launching the Lab

### Start Jupyter Notebook

```bash
# Make sure virtual environment is activated
source .venv/bin/activate

# Launch Jupyter
jupyter notebook lab_2.ipynb
```

Your browser should open automatically with the notebook. If not, copy the URL from the terminal.

### Using the Notebook

The notebook is structured as follows:

1. **Part 1-2: Setup and Background** - Install dependencies and learn about VLA models
2. **Part 3-4: Environment Setup** - Initialize Libero and load π0 model
3. **Part 5: Interactive Simulation** - Run tasks with different prompts using the interactive widget
4. **Part 6: Prompt Experiments** - Compare model behavior with prompt variations
5. **Part 7: Fine-tuning** - Collect data and fine-tune for specific tasks
6. **Part 8: Summary** - Review key takeaways

Execute cells sequentially by pressing `Shift+Enter`.

## Lab Structure

```
rc-vla-lab/
├── README.md              # This file
├── pyproject.toml        # Project dependencies
├── lab_2.ipynb           # Main lab notebook
├── openpi/               # OpenPI repository (cloned)
├── checkpoints/          # Model checkpoints (downloaded)
│   └── pi05_libero/
└── .venv/                # Virtual environment
```

## Environment Configuration

### MuJoCo Rendering

The lab uses MuJoCo for physics simulation. Configure rendering mode based on your system:

```python
# In the notebook, cell for environment setup:

# For headless systems (default):
os.environ['MUJOCO_GL'] = 'osmesa'

# For systems with GPU and display:
os.environ['MUJOCO_GL'] = 'egl'  # or 'glfw'
```

### GPU Configuration

If you have a CUDA-capable GPU:

```bash
# Verify CUDA installation
nvidia-smi

# Check PyTorch CUDA
python -c "import torch; print(torch.cuda.is_available())"
```

If CUDA is not available, the lab will run on CPU (slower but functional).

## Troubleshooting

### ImportError: No module named 'libero'

Install dependencies in the active environment:
```bash
uv sync
```

### MuJoCo rendering issues

Try different rendering backends:
```python
os.environ['MUJOCO_GL'] = 'osmesa'  # CPU rendering
# or
os.environ['MUJOCO_GL'] = 'egl'     # GPU rendering
```

### CUDA out of memory

Reduce batch size in the fine-tuning section:
```python
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=8,  # Reduce from 16
    ...
)
```

### Jupyter widgets not displaying

```bash
pip install ipywidgets
jupyter nbextension enable --py widgetsnbextension
```

## Key Features

### Interactive Task Execution

Use the dropdown menu and custom prompt field to:
- Select from 10 pre-defined Libero tasks
- Enter custom language instructions
- Run episodes and visualize robot behavior
- Save videos of executions

### Prompt Engineering

Experiment with different formulations:
```python
base_prompt = "pick up the black bowl and put it in the basket"
variations = [
    "grasp the black bowl and place it in the basket",
    "put the black bowl in the basket",
    "move the black bowl to the basket"
]
```

### Fine-tuning Pipeline

Complete workflow for task-specific adaptation:
1. Collect expert demonstrations
2. Create PyTorch dataset
3. Run training loop with progress tracking
4. Evaluate fine-tuned model
5. Compare performance before/after

## Learning Objectives

By completing this lab, you will:

1. Understand how VLA models combine vision, language, and action
2. Gain hands-on experience with robotic simulation
3. Learn to experiment with language-conditioned policies
4. Master fine-tuning techniques for robot learning
5. Analyze model behavior through visualization

## Additional Resources

- **π0 Model**: [Physical Intelligence Website](https://www.physicalintelligence.company)
- **Libero Benchmark**: [Official Project Page](https://libero-project.github.io/)
- **OpenPI Repository**: [GitHub](https://github.com/Physical-Intelligence/openpi)
- **MuJoCo Documentation**: [mujoco.org](https://mujoco.org/)

## Tasks and Experiments

### Suggested Experiments

1. **Prompt Sensitivity Analysis**
   - Compare success rates with synonym variations
   - Test formal vs. informal language
   - Analyze the impact of instruction length

2. **Multi-task Fine-tuning**
   - Collect data from multiple related tasks
   - Train a single model on diverse demonstrations
   - Evaluate zero-shot transfer to new tasks

3. **Behavioral Analysis**
   - Visualize action trajectories
   - Compare gripper control strategies
   - Analyze failure modes

## Performance Benchmarks

Expected performance on Libero tasks:

| Model | Success Rate | Notes |
|-------|-------------|-------|
| π0.5 (pre-trained) | ~97% | Zero-shot on Libero benchmark |
| π0.5 (fine-tuned) | >98% | After task-specific fine-tuning |

## Citation

If you use this lab for research or education, please cite:

```bibtex
@misc{pi0-vla-lab,
  title={Vision-Language-Action Lab with π0 and Libero},
  year={2024},
  howpublished={\url{https://github.com/your-repo}}
}
```

## License

This lab is provided for educational purposes. Please refer to individual component licenses:
- OpenPI: See [openpi/LICENSE](https://github.com/Physical-Intelligence/openpi/blob/main/LICENSE)
- Libero: See [Libero project](https://libero-project.github.io/)

## Support

For issues or questions:
1. Check the Troubleshooting section above
2. Review OpenPI documentation
3. Open an issue on the repository

## Contributing

Contributions are welcome! Areas for improvement:
- Additional visualization tools
- More task variations
- Advanced fine-tuning techniques
- Real robot deployment guides

---

**Happy Learning! 🤖**
