# A5 - Generative Adversarial Networks (GANs)

This project focuses on understanding the 

### Directory Structure:

```markdown
📂 A5/
├── 📂 Code Along/
│   ├── 📄 A5 - Generative Adversarial Networks (GANs).ipynb
│   └── 📂 img/
├── 📄 README.md
├── 📄 custom_gan.py
├── 📂 data/
│   ├── 📄 README.md
│   ├── 📂 celeba/
│   ├── 📂 cifar/
│   └── 📂 mnist/
├── 📄 face_gan_celeba.py
├── 📄 gan_trainer.py
├── 📄 gan_training.log
├── 📂 logs/
│   ├── 📄 custom_gan_training.txt
│   ├── 📄 dcgan_training.txt
│   ├── 📄 face_gan_training.txt
│   └── 📄 vanilla_gan_training.txt
├── 📂 models/
│   ├── 📄 README.md
│   ├── 📂 custom_gan/
│   ├── 📂 dcgan/
│   ├── 📂 face_gan/
│   └── 📂 vanilla_gan/
├── 📂 plots/
│   ├── 📄 README.md
│   ├── 📂 dcgan/
│   └── 📂 vanilla_gan/
├── 📂 results/
│   ├── 📄 README.md
│   ├── 📂 custom_gan/
│   ├── 📂 dcgan/
│   ├── 📂 face_gan/
│   └── 📂 vanilla_gan/
├── 📂 runs/
├── 📄 st125462_Arya_Shah_A5_GANs_Report.ipynb
└── 📄 st125462_Arya_Shah_A5_GANs_Report.pdf
```

# Task-wise Summary and Visualizations

1. Reproduce the vanilla GAN and DCGAN results on MNIST and CIFAR. Get the training and test loss for the generator and discriminator over time, plot them, and interpret them. ✅

|Discriminator Scores|Loss Plot|
|--------------------|---------|
|![discriminator_scores.png](https://github.com/aryashah2k/RTML/blob/main/A5/plots/vanilla_gan/mnist/discriminator_scores.png)|![loss_plot.png](https://github.com/aryashah2k/RTML/blob/main/A5/plots/vanilla_gan/mnist/loss_plot.png)|

## Training Dynamics

### Initial Phase (Epochs 0-5)
- **Discriminator Loss**: Started high (~1.38) and quickly decreased to around 0.2-0.5 by epoch 4
- **Generator Loss**: Initially around 0.7, then increased significantly to 3-4 range in epochs 1-3
- **D(x)**: Started at ~0.49 (random guessing) and rapidly improved to 0.95+ by epoch 4
- **D(G(z))**: Initially around 0.49, dropped to 0.03-0.08 by epochs 3-4

This indicates the discriminator quickly learned to distinguish real from fake images, becoming too strong compared to the generator in the early stages.

### Middle Phase (Epochs 6-15)
- **Discriminator Loss**: Stabilized around 0.8-1.0
- **Generator Loss**: Decreased and stabilized around 1.5-2.0
- **D(x)**: Decreased from 0.95+ to 0.6-0.8 range
- **D(G(z))**: Increased from near 0 to 0.2-0.4 range

This shows a more balanced training dynamic where the generator started catching up, producing more convincing images that occasionally fooled the discriminator.

### Late Phase (Epochs 16-29)
- **Discriminator Loss**: Remained stable around 1.0-1.1
- **Generator Loss**: Stabilized around 1.1-1.3
- **D(x)**: Consistently in the 0.6-0.7 range
- **D(G(z))**: Consistently in the 0.3-0.4 range

The relatively stable losses and discriminator outputs suggest the GAN reached an equilibrium where both networks were improving together at similar rates.

## Key Observations

1. **Nash Equilibrium**: By epoch 15, the system approached a Nash equilibrium where D(x) ≈ 0.7 and D(G(z)) ≈ 0.3, indicating a healthy balance between the generator and discriminator.

2. **Mode Collapse Avoidance**: The training logs don't show signs of mode collapse (which would manifest as very low generator loss with high discriminator loss).

3. **Label Smoothing Effect**: The implementation used label smoothing (0.9 instead of 1.0 for real labels), which likely contributed to the stability of training, preventing the discriminator from becoming overconfident.

4. **Training Efficiency**: Each epoch took approximately 18-19 seconds, with the entire training process completing in about 9 minutes and 27 seconds for 30 epochs.

## Training Stability Indicators

- The losses didn't exhibit wild oscillations after the initial phase
- D(x) and D(G(z)) values maintained a reasonable gap throughout training
- The generator loss decreased gradually without sudden drops or spikes

|Trained Image (Final Epoch)|Test Generated Image|
|---------------------------|--------------------|
|![_epoch_29_batch_500.png](https://github.com/aryashah2k/RTML/blob/main/A5/results/vanilla_gan/mnist/_epoch_29_batch_500.png)|![generated_images.png](https://github.com/aryashah2k/RTML/blob/main/A5/results/vanilla_gan/mnist/test/generated_images.png)|



# DCGAN On CIFAR

|Discriminator Scores|Loss Plot|
|--------------------|---------|
|![discriminator_scores.png](https://github.com/aryashah2k/RTML/blob/main/A5/plots/dcgan/cifar/discriminator_scores.png)|![loss_plot.png](https://github.com/aryashah2k/RTML/blob/main/A5/plots/dcgan/cifar/loss_plot.png)|

### Initial Phase (Epochs 0-2)

- **Discriminator Loss**: Started very high (1.6325) and fluctuated significantly in the first epoch (ranging from 1.4717 to 2.2391)
- **Generator Loss**: Began extremely high (7.0369) and rapidly decreased to more stable values around 1.0-3.0 by the end of epoch 0
- **D(x)**: Initially around 0.6673 (epoch 0) and stabilized to approximately 0.45 by epoch 2
- **D(G(z))**: Started high at 0.5265 and fluctuated between 0.35-0.57 during the first epochs

This indicates a period of rapid adjustment where both networks were learning the basic structure of the data distribution. The generator initially produced very poor samples (high loss), but quickly improved.

### Middle Phase (Epochs 3-6)

- **Discriminator Loss**: Stabilized in the range of 1.35-1.50
- **Generator Loss**: Consistently around 0.78-1.01
- **D(x)**: Maintained values between 0.41-0.46
- **D(G(z))**: Settled in the range of 0.42-0.46

This phase shows a more balanced training dynamic where both networks reached a relative equilibrium. The similar values of D(x) and D(G(z)) indicate the discriminator was having difficulty distinguishing between real and fake images, suggesting the generator was producing increasingly convincing samples.

### Later Phase (Epochs 7-9)

- **Discriminator Loss**: Remained stable around 1.40-1.50
- **Generator Loss**: Continued in the 0.83-0.95 range
- **D(x)**: Fluctuated between 0.42-0.50
- **D(G(z))**: Consistently around 0.43-0.49

The stability in these metrics suggests the GAN reached a healthy equilibrium. The discriminator's ability to classify real and fake images remained similar, with D(x) and D(G(z)) values close to each other, indicating the generator was producing convincing images.

## Key Observations

1. **Nash Equilibrium**: By epoch 3, the system approached a Nash equilibrium where D(x) ≈ D(G(z)) ≈ 0.45, indicating a well-balanced adversarial training process.

2. **Training Stability**: Unlike many GAN implementations that suffer from mode collapse or oscillating losses, this DCGAN implementation showed remarkable stability after the initial adjustment period.

3. **Computational Requirements**: Each epoch took approximately 83 seconds, significantly longer than the vanilla GAN on MNIST (which took about 19 seconds per epoch), reflecting the increased complexity of both the dataset and the model architecture.

4. **Alternating Training Strategy**: The implementation used an alternating training approach (training the discriminator only every other batch), which likely contributed to the training stability by preventing the discriminator from becoming too powerful.

5. **Learning Rate Balance**: The discriminator used a lower learning rate (0.0001) than the generator (0.0002), which helped maintain balance between the two networks.

|Trained Image (Final Epoch)|Test Generated Image|
|---------------------------|--------------------|
|![_epoch_29_batch_400.png](https://github.com/aryashah2k/RTML/blob/main/A5/results/dcgan/cifar/_epoch_29_batch_400.png)|![generated_images.png](https://github.com/aryashah2k/RTML/blob/main/A5/results/dcgan/cifar/test/generated_images.png)|

## Comparison with Vanilla GAN on MNIST

Compared to the vanilla GAN training on MNIST:

1. **Stability**: The DCGAN showed more consistent and stable training metrics from the early epochs, while the vanilla GAN had more pronounced fluctuations throughout training.

2. **Equilibrium Point**: The DCGAN reached an equilibrium where D(x) ≈ D(G(z)) ≈ 0.45, whereas the vanilla GAN settled at D(x) ≈ 0.65 and D(G(z)) ≈ 0.35, suggesting a different balance point.

3. **Convergence Speed**: The DCGAN appeared to reach stable performance metrics faster (by epoch 3) than the vanilla GAN (which took until around epoch 15).

-------------------------

2. Develop your own GAN to model data generated as follows: ✅

```markdown
    $$\begin{eqnarray} \theta & \sim & {\cal U}(0,2\pi) \\
                       r      & \sim & {\cal N}(0, 1) \\
                       \mathbf{x} & \leftarrow & \begin{cases} \begin{bmatrix} (10+r)\cos\theta \\ (10+r)\sin\theta + 10\end{bmatrix} & \frac{1}{2}\pi \le \theta \le \frac{3}{2}\pi \\ \begin{bmatrix} (10+r)\cos\theta \\ (10+r)\sin\theta - 10\end{bmatrix} & \mathrm{otherwise} \end{cases} \end{eqnarray} $$
```

You should create a PyTorch DataSet that generates the 2D data in the `__init__()` method, outputs a sample in the `__getitem__()` method, and returns the dataset size in the `__len__()` method. Use the vanilla GAN approach above with an appropriate structure for the generator. Can your GAN generate a convincing facsimile of a set of samples from the actual distribution?

|Name of Plot|Image|
|------------|-----|
|Comparison of Real vs Generated Data|![comparison.png](https://github.com/aryashah2k/RTML/blob/main/A5/results/custom_gan/comparison.png)|
|Dataset Samples|![dataset_samples.png](https://github.com/aryashah2k/RTML/blob/main/A5/results/custom_gan/dataset_samples.png)|
|Generated Samples|![generated_samples.png](https://github.com/aryashah2k/RTML/blob/main/A5/results/custom_gan/generated_samples.png)|
|Real Samples|![real_samples.png](https://github.com/aryashah2k/RTML/blob/main/A5/results/custom_gan/real_samples.png)|
|Loss Plot|![loss_plot.png](https://github.com/aryashah2k/RTML/blob/main/A5/results/custom_gan/loss_plot.png)|
|Score Plot|![score_plot.png](https://github.com/aryashah2k/RTML/blob/main/A5/results/custom_gan/score_plot.png)|

## Dataset Characteristics

The GAN is trained on a synthetic 2D dataset with the following distribution:
- Points lie on a circle with radius approximately 10 (with Gaussian noise)
- The distribution forms two semi-circles that are vertically offset:
  - Upper semi-circle: y = (10+r)sinθ + 10 when π/2 ≤ θ ≤ 3π/2
  - Lower semi-circle: y = (10+r)sinθ - 10 otherwise


## Architecture Overview

The implementation uses a relatively simple GAN architecture:

**Generator:**
- Input: 100-dimensional noise vector
- 3 hidden layers with LeakyReLU activations (128 → 256 → 256 units)
- Output: 2D coordinates with no activation function

**Discriminator:**
- Input: 2D coordinates
- 3 hidden layers with LeakyReLU and Dropout (128 → 256 → 128 units)
- Output: Single sigmoid unit

## Training Strategy Analysis

Several effective training techniques were implemented:

1. **Label smoothing**: Real labels set to 0.9 instead of 1.0
2. **Alternate training**: Discriminator trained only every other batch
3. **Adam optimizer**: Using β₁=0.5, β₂=0.999 (standard for GANs)
4. **Dropout in discriminator**: 30% dropout rate to prevent overfitting

## Training Dynamics Analysis

Looking at the training logs:

### Stability at Equilibrium

The most striking observation is the remarkable stability in the training metrics:
- **Discriminator Loss**: Consistently around 1.375-1.380
- **Generator Loss**: Consistently around 0.795-0.805
- **D(x)**: Stabilized at ~0.450
- **D(G(z))**: Stabilized at ~0.450

This indicates the GAN has reached a Nash equilibrium where:
1. The discriminator can no longer distinguish between real and fake samples (both scored around 0.45)
2. The generator has successfully learned to produce samples that match the real data distribution

### Absence of Mode Collapse

The stability of both losses and the near-identical D(x) and D(G(z)) values strongly suggest that mode collapse has been avoided. In mode collapse, we would typically see the generator loss decrease while discriminator loss increases, and D(G(z)) would fluctuate significantly.

### Training Efficiency

Each epoch completed in approximately 1.36 seconds, with the entire training process (200 epochs) taking under 5 minutes. This efficiency is due to:
1. The relatively simple architecture
2. The modest dataset size (10,000 samples)
3. Effective batch processing (batch size of 128)

## Insights on GAN Equilibrium

The training logs show a textbook example of a GAN reaching the theoretical equilibrium point where D(x) ≈ D(G(z)) ≈ 0.5. In practice, we see values around 0.45, which is remarkably close to the ideal 0.5.

This equilibrium indicates that:
1. The discriminator is maximally confused
2. The generator has successfully learned the data distribution
3. The training process has converged to a stable solution

-------------------------
   
3. Use the DCGAN (or an improvement to it) to build a generator for a face image set of your choice. Can you get realistic faces that are not in the training set? ✅

|Name of Plot|Plot|
|------------|----|
|Face Interpolation|![face_interpolation.png](https://github.com/aryashah2k/RTML/blob/main/A5/results/face_gan/face_interpolation.png)|
|Trained Image(Final Epoch)|![epoch_49_batch_1500.png](https://github.com/aryashah2k/RTML/blob/main/A5/results/face_gan/epoch_49_batch_1500.png)|
|Generated Faces|![generated_faces_grid.png](https://github.com/aryashah2k/RTML/blob/main/A5/results/face_gan/generated_faces_grid.png)|
|Loss Plot|![loss_plot.png](https://github.com/aryashah2k/RTML/blob/main/A5/results/face_gan/loss_plot.png)|
|Score Plot|![score_plot.png](https://github.com/aryashah2k/RTML/blob/main/A5/results/face_gan/score_plot.png)|

## Training Dynamics

### Initial Phase (Epochs 0-5)
- **Discriminator Loss**: Started high (~2.05) and fluctuated significantly
- **Generator Loss**: Initially very high (4.36-12.28), indicating poor generation quality
- **D(x)**: Began around 0.33 (poor real image classification) and improved to 0.7+
- **D(G(z))**: Started at 0.01 (excellent fake detection) and remained low (0.03-0.08)

This phase shows the discriminator quickly learning to distinguish real faces from the initially poor generator outputs.

### Middle Phase (Epochs 28-32)
- **Discriminator Loss**: Stabilized in the range of 0.5-0.9
- **Generator Loss**: Fluctuated widely between 0.6-5.2
- **D(x)**: Consistently high (0.7-0.9), showing strong real image recognition
- **D(G(z))**: Mostly low (0.05-0.15) with occasional spikes to 0.5+

This indicates the discriminator maintained dominance while the generator struggled with consistent improvement, showing signs of mode collapse and recovery cycles.

### Later Phase (Epochs 33-36)
- **Discriminator Loss**: Continued in the 0.4-1.2 range
- **Generator Loss**: Alternated between very high (3.0-4.6) and low (0.6-1.5) values
- **D(x)**: Typically 0.7-0.9 with occasional drops to 0.4-0.5
- **D(G(z))**: Mostly 0.02-0.08 with occasional spikes to 0.6+

The significant oscillations in generator loss and D(G(z)) suggest the training had not reached equilibrium, with the generator occasionally making breakthroughs before being countered by the discriminator.

## Key Observations

1. **Discriminator Dominance**: Throughout training, the discriminator consistently outperformed the generator, as evidenced by high D(x) and low D(G(z)) values. This indicates the discriminator could easily distinguish real from generated faces.

2. **Training Instability**: The generator loss showed high variance (0.6-5.2), suggesting unstable training dynamics typical of GANs on complex datasets.

3. **Mode Collapse Cycles**: Periodic spikes in D(G(z)) to values like 0.65 (epoch 36) followed by rapid drops suggest temporary mode collapse followed by recovery.

4. **Alternating Training Strategy**: The implementation used an alternating training approach (training the discriminator only every other batch), which likely prevented complete discriminator dominance.

5. **Label Smoothing Effect**: Using 0.9 instead of 1.0 for real labels helped stabilize training, though the discriminator still maintained an advantage.

## Computational Requirements

- Each epoch took approximately 95-97 seconds on an NVIDIA GeForce RTX 2080 Ti
- The dataset contained 202,599 images, processed in batches of 128
- Training was set for 50 epochs, with model checkpoints saved every 5 epochs

-------------------------
