# A3 - Transformers Assignment | ViT

This project focuses on understanding the Vision Transformer and using provided weights to fine tune on an existing given task

```
Project Root
├── Code Along/
│   ├── A3 - ViT.ipynb
│   ├── A3 - ViT.py
│   ├── extra resource/
│   │   └── ... (other files)
│   └── img/
├── finetune_vit_complete.py
├── finetune_vit_complete+50Epochs.py
├── logs/
│   ├── cURL_dataset_w_NVIDIA-SMI.txt
│   └── training_logs.txt
├── Model Checkpoints/
│   ├── best_model_epoch_100.pth
│   ├── best_model_epoch_50.pth
│   └── Ep.7.pth
├── results_20250208_114438/
│   └── test_report.txt
├── results_20250208_144934/
│   └── test_report.txt
├── runs/
│   └── Dec31_11-57-50_puffervit_b16_sport_dataset/
│       └── ... (other files/folders)
├── sports-dataset/
└── st125462_Arya_Shah_A3_Transformers_ViT_Report.ipynb
```

# Key Components:

1. Code Along
Contains interactive notebooks (*.ipynb) and associated Python scripts for step-by-step guidance along with any extra resources and images. It’s  used for demonstration or development.

2. finetune_vit_complete.py / finetune_vit_complete+50Epochs.py
These scripts  contain code to fine-tune a Vision Transformer (ViT) model. The variant with "+50Epochs" might have configurations for extended training.

3. logs Folder
Contains log files like training logs and output text files (e.g., from cURL or system outputs). This assists with monitoring and debugging training runs.

4. Model Checkpoints Folder
Stores saved states (checkpoint files such as *.pth) of the model at different epochs. These files are useful for loading the model for further training or evaluation.

5. results_... Folders
Each of these directories (named with timestamps) hold evaluation reports, like test reports, which summarize performance metrics from experiments.

6. runs Folder
Contains subdirectories for individual training runs (e.g., with naming based on date-time and model settings). Each run folder may hold details like logs, configuration files, and other related experiment artifacts.

7. sports-dataset Folder
This folder  contains the dataset used for training or inference, possibly including images or annotations for a sports-related task.

# Results & Observations

## Performance Patterns

**Perfect Classification (100% Accuracy)**
The model achieved perfect accuracy in several sports that have distinctive visual characteristics:
- Balance beam
- Boxing
- Curling
- Giant slalom
- Ice climbing
- Ice yachting
- Jai alai
- Polo
- Swimming
- Water polo

**Zero Performance (0% Accuracy)**
Interestingly, the model completely failed to classify certain sports:
- Frisbee
- Gaga
- Parallel bar
- Trapeze
- Ultimate

## Training Insights

**Learning Progression**
- The model showed rapid initial improvement, jumping from 6.51% accuracy in epoch 1 to 42% by epoch 14
- The learning rate decreased from 0.0001 to 9.87e-8 over 50 epochs, showing fine-tuning of weights

**Performance Plateaus**
- The model hit several performance plateaus, particularly between epochs 35-50
- The best validation accuracy (55.00%) was achieved in epoch 42

## Interesting Correlations

**Sport Type Patterns**
- Individual sports generally had better classification accuracy than team sports
- Indoor sports with fixed environments (like swimming, boxing) showed higher accuracy than outdoor sports with variable conditions

**Visual Distinctiveness**
- Sports with unique equipment or settings (like ice yachting, curling) were classified more accurately
- Sports with similar visual elements (like different types of racing) showed more classification confusion

|Epoch 1-50|Epoch 51-100|
|----------|------------|
|![1]()|![2]()|
|![3]()|![4]()|
|<a href="add">Test Report</a>|<a href="add">Test Report</a>|

Model Checkpoints can be found <a href="https://1024terabox.com/s/1FY-YmkfRRXlc74u7GQtTNg">here (Terabox Link)</a>

## Usage

Install dependencies if not already:

```bash
pip install -r requirements.txt
```

```bash
python fine_tune_vit_complete.py
```
