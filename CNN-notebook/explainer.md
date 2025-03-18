## Deep Learning Project
This notebook explores different approaches to car classification using end to end Convolutional Neural Networks (CNNs).

### Notebook strcture

1. **Installations** - Setting up the required libraries and dependencies.
2. **Data Preparation** - Process the data one time so it would be used later with ease and speed (no need to run it again). Processing steps: Train test split to 70/30, converting images to grayscale, scaling images down, augmentation, save as tensors.
3. **Utilities** - Helper functions that would be used while training the models.
4. **Training Preperation**: Load Tensors. (only if training)
5. **Experiment 1 : Basic_CNN_GAP_MP** - This is a basic CNN model with Global Average Pooling and Max Pooling layers.
6. **Experiment 2 : QuickTrain** - A small model that uses depthwise separable convolutions, desinged for speed of and ease of training.
7. **Experiment 3 : BiggerIsBetter** - A bigger version of the Quick Train model.
8. **Test Best Model** - A small test envitoment to try out our best model (This enviorment is self contained, can be runed without any thing else).

### How To Run

This notebook has 3 ways to run it.

1. Test - Just run the test environment.
2. Data Preperation - Run the Installations section then the Data Preparation section (remember to remove skips).
3. Train a model yourself - Run the Installations section then the Utilities and Training Preparation section and then run model you want to train (remember to remove skips).
