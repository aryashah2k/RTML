# A1 - CNNs Assignment

This section of the Repository consists of the solution for the A1 - CNNs Assignemnt.

The Original Assignment Question and Starter Notebooik for reference can be found <a href="https://github.com/aryashah2k/RTML/tree/main/A1/A1_Question">here</a>

# Solution

The solution is designed as follows:

|Solution|Code|
|--------|----|
|alexnet.py|<a href="https://github.com/aryashah2k/RTML/blob/main/A1/alexnet.py">click here</a>|
|googlenet.py|<a href="https://github.com/aryashah2k/RTML/blob/main/A1/googlenet.py">click here</a>|
|model_comparison.py(equivalent of main/train.py)|<a href="https://github.com/aryashah2k/RTML/blob/main/A1/model_comparison.py">click here</a>|
|EXTRA: alexnet_multigpu.py|<a href="https://github.com/aryashah2k/RTML/blob/main/A1/alexnet_multigpu.py">click here</a>|

# Reports

|Solution|Link|
|--------|----|
|Jupyter Notebook Solution|<a href="https://github.com/aryashah2k/RTML/blob/main/A1/st125462_Arya_Shah_A1_Pytorch_AlexNet_GoogleNet_Report.ipynb">click here</a>|
|Report PDF|<a href="https://github.com/aryashah2k/RTML/blob/main/A1/st125462_Arya_Shah_Pytorch-AlexNet-GoogleNet_Final_Report.pdf">click here</a>|
|Model checkpoints and Weights|<a href="https://1024terabox.com/s/1gRb23goPwNOBo0Ry_y_low">click here</a>|
|Training Logs|<a href=https://github.com/aryashah2k/RTML/tree/main/A1/Logs>click here</a>|

# Results Summary

| Model | Parameters | Train Loss | Train Acc | Val Loss | Val Acc | Test Loss | Test Acc | Best Val Acc | Training Time | Avg Batch Time |
|-------|------------|------------|-----------|----------|---------|-----------|-----------|--------------|---------------|----------------|
| AlexNet with LRN | 57,044,810 | 0.3515 | 87.84% | 0.4513 | 84.46% | 0.4743 | 83.78% | 84.46% | 17.1 min | 37.6 ms |
| AlexNet without LRN | 57,044,810 | 0.2718 | 90.49% | 0.4816 | 83.74% | 0.4816 | 84.56% | 85.06% | 17.5 min | 41.0 ms |
| Custom GoogLeNet | 10,635,134 | 0.2215 | 97.47% | 0.4001 | 88.24% | 0.4276 | 88.32% | 88.48% | 33.4 min | 83.4 ms |
| Pretrained AlexNet | 57,044,810 | 0.1078 | 96.26% | 0.2323 | 92.12% | 0.2587 | 91.60% | 92.80% | 16.1 min | 40.5 ms |
| Pretrained GoogLeNet | 5,610,154 | 0.0179 | 99.68% | 0.1529 | 95.46% | 0.1621 | 94.87% | 95.74% | 26.9 min | 69.9 ms |

![Plot](https://github.com/aryashah2k/RTML/blob/main/A1/Plots%20%26%20Results/model_comparison.png)

# Setup

1. Clone the Repository

2. Install the required libraries

```python
pip install -r requirements.txt
```

3. Run the model_comparison.py file

```python
python model_comparison.py
```

4. In order to run alexnet_multigpu.py, make sure you have >2 GPUs

```python
python alexnet_multigpu.py
```










