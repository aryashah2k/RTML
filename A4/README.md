# A4 - Masked Auto Encoders Assignment

This project focuses on understanding the Masked Auto Encoder from scratch and using it for downstream tasks such as classification as well as comparing and analyzing the results on various banchmark datasets such as MNIST and CIFAR10

### Directory Structure:

```markdown
.
├── A4_MAE.py                                             # Main MAE implementation
├── cifar10_experiments.py                                # CIFAR-10 experiments
├── Code Along/
│   ├── A4 - MAE.ipynb                                   # Tutorial notebook
│   └── img/                                             # Tutorial images
├── data/                                                # Dataset directory
│   └── README.md
├── logs/                                                # Training logs
│   ├── cifar10_experiments_logs.txt
│   ├── mae_mnist_training_logs.txt
│   └── mnist_classifier_logs.txt
├── mae_mnist_training.py                                # MNIST MAE training
├── mnist_classifier.py                                  # MNIST classifier
├── Models_Checkpoints/                                 # Saved models
│   └── README.md
├── plots/                                              # Generated plots
│   ├── cifar10_classifier_patch4/
│   ├── cifar10_classifier_patch8/
│   └── mae_cifar10_patch4_mask075/
├── README.md                                           # Project documentation
├── saved/                                              # Additional saved files
│   └── README.md
├── st125462_Arya_Shah_A4_Masked_Auto_Encoders_Report.ipynb   # Project report
└── st125462_Arya_Shah_A4_Masked_Auto_Encoders_Report.pdf     # PDF version of report
```

# Summmary And Visualizations

1. Experimenting with different patch sizes and mask ratios

## Experimental Configurations
- Config 1: patch_size=2, mask_ratio=0.75 (baseline)

25th Epoch Results:
![epoch_24_batch_400.png](https://github.com/aryashah2k/RTML/blob/main/A4/plots/mae_mnist_patch2_mask075/reconstructions/epoch_24_batch_400.png)

Loss Curve:
![loss_curve.png](https://github.com/aryashah2k/RTML/blob/main/A4/plots/mae_mnist_patch2_mask075/loss_curve.png)

- Config 2: patch_size=4, mask_ratio=0.75 (larger patches)

25th Epoch Results:
![epoch_24_batch_400.png](https://github.com/aryashah2k/RTML/blob/main/A4/plots/mae_mnist_patch4_mask075/reconstructions/epoch_24_batch_400.png)

Loss Curve:
![loss_curve.png](https://github.com/aryashah2k/RTML/blob/main/A4/plots/mae_mnist_patch4_mask075/loss_curve.png)

- Config 3: patch_size=2, mask_ratio=0.85 (more masking)

25th Epoch Results:
![epoch_24_batch_400.png](https://github.com/aryashah2k/RTML/blob/main/A4/plots/mae_mnist_patch2_mask085/reconstructions/epoch_24_batch_400.png)

Loss Curve:
![loss_curve.png](https://github.com/aryashah2k/RTML/blob/main/A4/plots/mae_mnist_patch2_mask085/loss_curve.png)

- Config 4: patch_size=4, mask_ratio=0.85 (larger patches + more masking)

25th Epoch Results:
![epoch_24_batch_400.png](https://github.com/aryashah2k/RTML/blob/main/A4/plots/mae_mnist_patch4_mask085/reconstructions/epoch_24_batch_400.png)

Loss Curve:
![loss_curve.png](https://github.com/aryashah2k/RTML/blob/main/A4/plots/mae_mnist_patch4_mask085/loss_curve.png)

## Key Observations

1. **Final Loss Values**:
- Config 1: ~0.0421 (lowest)
- Config 2: ~0.0764 (moderate)
- Config 3: ~0.0752 (moderate)
- Config 4: ~0.1265 (highest)

2. **Convergence Speed**:
- Smaller patch size (2) configurations converged more slowly but achieved better final results
- Larger patch size (4) configurations showed faster initial convergence but higher final loss

3. **Impact of Parameters**:
- Patch Size: Larger patches (4x4) led to faster training but higher reconstruction error
- Mask Ratio: Higher mask ratio (0.85) resulted in higher reconstruction loss compared to 0.75

## Conclusions

1. **Optimal Configuration**: The baseline configuration (patch_size=2, mask_ratio=0.75) performed best overall, achieving the lowest final loss of 0.0421.

2. **Patch Size Effect**: 
- Smaller patches (2x2) allow for more detailed reconstruction
- Larger patches (4x4) trade reconstruction quality for computational efficiency

3. **Masking Ratio Impact**:
- Lower masking ratio (0.75) generally performed better than higher ratio (0.85)
- Higher masking makes the reconstruction task more challenging

4. **Trade-offs**:
- There's a clear trade-off between reconstruction quality and computational efficiency
- Choose smaller patches and lower mask ratio for better reconstruction
- Choose larger patches if training speed is prioritized over reconstruction quality

---------------------

2. Using MAE in downstream tasks

## Experimental Setup
- Task: MNIST Classification using pretrained MAE encoders
- Configurations tested:
  1. patch_size=2, mask_ratio=0.75
  2. patch_size=4, mask_ratio=0.75
  3. patch_size=2, mask_ratio=0.85
  4. patch_size=4, mask_ratio=0.85

## Results Analysis

### Final Test Accuracies
1. patch2_mask0.75: 99.30%
2. patch4_mask0.75: 99.24%
3. patch2_mask0.85: 99.43% (Best)
4. patch4_mask0.85: 99.06%

### Final Plots

1. For Patch 2

- Training Curve

![training_curves.png](https://github.com/aryashah2k/RTML/blob/main/A4/plots/mnist_classifier_patch2/training_curves.png)

- Confusion Matrix

![confusion_matrix.png](https://github.com/aryashah2k/RTML/blob/main/A4/plots/mnist_classifier_patch2/confusion_matrix.png)

3. For Patch 4

- Training Curve

![training_curves.png](https://github.com/aryashah2k/RTML/blob/main/A4/plots/mnist_classifier_patch4/training_curves.png)

- Confusion Matrix

![confusion_matrix.png](https://github.com/aryashah2k/RTML/blob/main/A4/plots/mnist_classifier_patch4/confusion_matrix.png)

### Key Observations

1. **Overall Performance**:
- All configurations achieved excellent accuracy (>99%)
- The differences between configurations are relatively small (<0.4%)

2. **Patch Size Impact**:
- Smaller patch size (2x2) consistently outperformed larger patch size (4x4)
- patch_size=2 models achieved better final accuracy in both masking ratios

3. **Masking Ratio Effect**:
- For patch_size=2, higher masking ratio (0.85) performed better
- For patch_size=4, lower masking ratio (0.75) performed better

4. **Training Characteristics**:
- Fast convergence in early epochs
- Consistent improvement in both training and test accuracy
- Good generalization with small gaps between training and test performance

## Conclusions

1. **Best Configuration**: patch_size=2 with mask_ratio=0.85 achieved the highest accuracy (99.43%)

2. **Architectural Insights**:
- Smaller patches retain more fine-grained information useful for classification
- Higher masking ratio during pretraining can lead to more robust feature learning

3. **Transfer Learning Effectiveness**:
- Successfully demonstrated that MAE pretraining can be effectively used for downstream classification
- All configurations achieved high accuracy, showing the robustness of the approach

4. **Practical Recommendations**:
- For MNIST classification, use smaller patch sizes (2x2)
- Higher masking ratios during pretraining can be beneficial when using small patches
- The choice between configurations might depend on computational constraints, as smaller patches require more processing

-----------------

3. Experiment with CIFAR10

## Plots & Visualizations:

1. Patch 4 Mask 0.75

- 25th Epoch

![epoch_49_batch_200.png](https://github.com/aryashah2k/RTML/blob/main/A4/plots/mae_cifar10_patch4_mask075/reconstructions/epoch_49_batch_200.png)

- Loss Curve

![loss_curve.png](https://github.com/aryashah2k/RTML/blob/main/A4/plots/mae_cifar10_patch4_mask075/loss_curve.png)

2. Patch 4 Mask 0.85

- 25th Epoch

![epoch_49_batch_200.png](https://github.com/aryashah2k/RTML/blob/main/A4/plots/mae_cifar10_patch4_mask085/reconstructions/epoch_49_batch_200.png)

- Loss Curve

![loss_curve.png](https://github.com/aryashah2k/RTML/blob/main/A4/plots/mae_cifar10_patch4_mask085/loss_curve.png)

3. Patch 8 Mask 0.75

- 25th Epoch
![epoch_49_batch_200.png](https://github.com/aryashah2k/RTML/blob/main/A4/plots/mae_cifar10_patch8_mask075/reconstructions/epoch_49_batch_200.png)  

- Loss Curve
![loss_curve.png](https://github.com/aryashah2k/RTML/blob/main/A4/plots/mae_cifar10_patch8_mask075/loss_curve.png)

4. Classifier Patch 4

- Confusion Matrix

![confusion_matrix.png](https://github.com/aryashah2k/RTML/blob/main/A4/plots/cifar10_classifier_patch4/confusion_matrix.png)

- Training Curve
![training_curves.png](https://github.com/aryashah2k/RTML/blob/main/A4/plots/cifar10_classifier_patch4/training_curves.png)

5. Classifier Patch 8

- Confusion matrix
![confusion_matrix.png](https://github.com/aryashah2k/RTML/blob/main/A4/plots/cifar10_classifier_patch8/confusion_matrix.pngg)  

- Training Curve
![training_curves.png](https://github.com/aryashah2k/RTML/blob/main/A4/plots/cifar10_classifier_patch8/training_curves.png)

## Key Differences Between MNIST and CIFAR-10 Results

**Training Performance**:
- MNIST achieved much higher accuracy (>99%) compared to CIFAR-10's peak accuracy of around 80%
- CIFAR-10 showed slower convergence and required more epochs to reach optimal performance
- MNIST training was more stable with consistent improvements, while CIFAR-10 showed more fluctuations in loss

**Patch Size Impact**:
- For MNIST, smaller patch sizes (2x2) performed better than larger ones
- For CIFAR-10, patch_size=4 achieved better results than patch_size=8, suggesting an optimal middle ground
- The impact of patch size was more pronounced in CIFAR-10 than MNIST

**Masking Ratio Effects**:
- MNIST was relatively robust to different masking ratios
- CIFAR-10 showed more sensitivity to masking ratios, with performance degrading more significantly at higher ratios

## Reasons for Differences

1. **Dataset Complexity**:
- MNIST contains simple grayscale digits
- CIFAR-10 has complex RGB images with varied objects, backgrounds, and orientations

2. **Feature Extraction**:
- MNIST features are more structured and consistent
- CIFAR-10 requires learning more sophisticated features across multiple channels

## Suggested Improvements for CIFAR-10

1. **Architecture Enhancements**:
- Increase model capacity (more layers, wider networks)
- Implement hierarchical feature learning
- Add skip connections for better gradient flow

2. **Training Optimizations**:
- Implement progressive learning strategies
- Use curriculum learning with increasing image complexity
- Employ more sophisticated data augmentation techniques

3. **Hyperparameter Refinements**:
- Experiment with smaller patch sizes (3x3, 5x5)
- Test adaptive masking ratios during training
- Optimize learning rate schedules for better convergence

4. **Data Processing**:
- Implement advanced normalization techniques
- Add color jittering and other image transformations
- Use mixup or cutmix augmentation strategies

These improvements could potentially help bridge the performance gap between MNIST and CIFAR-10 results while maintaining computational efficiency.