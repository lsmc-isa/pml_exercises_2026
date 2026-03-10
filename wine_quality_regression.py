"""
Wine Quality Regression Model
This script trains a neural network regression model to predict wine quality.
It demonstrates key machine learning concepts with explicit variable labeling.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import urllib.request
import io

# ============================================================================
# STEP 1: LOAD AND PREPARE DATA
# ============================================================================

print("Loading Wine Quality Dataset from UCI Repository...")

# Download the white wine quality dataset (4898 examples)
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"
response = urllib.request.urlopen(url)
data = pd.read_csv(io.StringIO(response.read().decode('utf-8')), sep=';')

print(f"Dataset shape: {data.shape}")
print(f"Total examples: {len(data)}")

# ============================================================================
# STEP 2: DEFINE INPUT AND OUTPUT
# ============================================================================

# INPUT: All features except 'quality' - these are the explanatory variables
# The independent variables that will be used to make predictions
INPUT_features = data.drop('quality', axis=1).values
INPUT_feature_names = data.drop('quality', axis=1).columns.tolist()

# OUTPUT: The response variable we want to predict
# The dependent variable (target)
OUTPUT_target = data['quality'].values

print(f"\nINPUT shape (explanatory variables): {INPUT_features.shape}")
print(f"INPUT features: {INPUT_feature_names}")
print(f"OUTPUT shape (response variable): {OUTPUT_target.shape}")

# ============================================================================
# STEP 3: SPLIT INTO TRAINING, VALIDATION, AND TEST DATASETS
# ============================================================================

# TRAIN DATASET: Used to train the model and learn the patterns (70%)
# VALIDATION DATASET: Used during training to monitor performance (15%)
# TEST DATASET: Completely independent, used ONLY for final evaluation (15%)

# First split: 70% train, 30% temporary (for validation+test)
TRAIN_INPUT, TEMP_INPUT, TRAIN_OUTPUT, TEMP_OUTPUT = train_test_split(
    INPUT_features, 
    OUTPUT_target, 
    test_size=0.3,  # 30% for validation + test
    random_state=42
)

# Second split: Split the 30% into 50% validation and 50% test (15% each of total)
VALIDATION_INPUT, TEST_INPUT, VALIDATION_OUTPUT, TEST_OUTPUT = train_test_split(
    TEMP_INPUT,
    TEMP_OUTPUT,
    test_size=0.5,  # 50% of 30% = 15% of total
    random_state=42
)

print(f"\nTRAIN DATASET size: {len(TRAIN_INPUT)} examples (70%)")
print(f"VALIDATION DATASET size: {len(VALIDATION_INPUT)} examples (15%)")
print(f"TEST DATASET size: {len(TEST_INPUT)} examples (15%)")

# Standardize the features for better training
# IMPORTANT: Fit the scaler ONLY on TRAIN data to avoid data leakage
scaler = StandardScaler()
TRAIN_INPUT_scaled = scaler.fit_transform(TRAIN_INPUT)
VALIDATION_INPUT_scaled = scaler.transform(VALIDATION_INPUT)
TEST_INPUT_scaled = scaler.transform(TEST_INPUT)

# ============================================================================
# STEP 4: BUILD THE MODEL
# ============================================================================

# MODEL: A neural network with multiple layers
# The model learns to map INPUT to OUTPUT
MODEL = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(INPUT_features.shape[1],)),
    layers.Dense(32, activation='relu'),
    layers.Dense(16, activation='relu'),
    layers.Dense(1)  # Output layer for regression (single continuous value)
])

print("\nMODEL Architecture:")
MODEL.summary()

# ============================================================================
# STEP 5: DEFINE LOSS FUNCTION AND OPTIMIZER
# ============================================================================

# LOSS FUNCTION: Measures how wrong the MODEL predictions are
# We use Mean Squared Error (MSE) for regression
LOSS_FUNCTION = 'mean_squared_error'

# Optimizer: Algorithm that updates model weights to minimize LOSS_FUNCTION
optimizer = 'adam'

MODEL.compile(
    loss=LOSS_FUNCTION,
    optimizer=optimizer,
    metrics=['mae']  # Mean Absolute Error as additional metric
)

print(f"\nLOSS FUNCTION: {LOSS_FUNCTION}")

# ============================================================================
# STEP 6: TRAIN THE MODEL
# ============================================================================

print("\nTraining the MODEL...")

# BATCH: Small subset of training data processed before updating weights
# EPOCH: One complete pass through the entire TRAIN DATASET
BATCH_size = 32
NUM_EPOCHS = 100

# Train the model
# IMPORTANT: Use VALIDATION dataset during training, NOT TEST dataset
training_history = MODEL.fit(
    TRAIN_INPUT_scaled, 
    TRAIN_OUTPUT,
    epochs=NUM_EPOCHS,           # Number of complete passes through TRAIN DATASET
    batch_size=BATCH_size,        # Size of BATCH used in each update
    validation_data=(VALIDATION_INPUT_scaled, VALIDATION_OUTPUT),  # Monitor on VALIDATION data
    verbose=1
)

print("\nTraining completed!")

# ============================================================================
# STEP 7: MAKE PREDICTIONS
# ============================================================================

# PREDICT: Use the trained MODEL to generate OUTPUT values for new INPUT data

# Predictions on TRAIN DATASET
train_predictions = MODEL.predict(TRAIN_INPUT_scaled, verbose=0)

# Predictions on VALIDATION DATASET
validation_predictions = MODEL.predict(VALIDATION_INPUT_scaled, verbose=0)

# Predictions on TEST DATASET (independent data not seen during training)
test_predictions = MODEL.predict(TEST_INPUT_scaled, verbose=0)

print(f"\nPredictions generated!")
print(f"Train predictions shape: {train_predictions.shape}")
print(f"Validation predictions shape: {validation_predictions.shape}")
print(f"Test predictions shape: {test_predictions.shape}")

# ============================================================================
# STEP 8: EVALUATE THE MODEL ON ALL THREE DATASETS
# ============================================================================

# Evaluate on TRAIN dataset (should be best performance - seen during training)
train_results = MODEL.evaluate(TRAIN_INPUT_scaled, TRAIN_OUTPUT, verbose=0)

# Evaluate on VALIDATION dataset (seen during training for monitoring)
validation_results = MODEL.evaluate(VALIDATION_INPUT_scaled, VALIDATION_OUTPUT, verbose=0)

# Evaluate on TEST dataset (completely independent - NOT seen during training)
test_results = MODEL.evaluate(TEST_INPUT_scaled, TEST_OUTPUT, verbose=0)

print(f"\nTRAIN DATASET - LOSS (MSE): {train_results[0]:.4f} | MAE: {train_results[1]:.4f}")
print(f"VALIDATION DATASET - LOSS (MSE): {validation_results[0]:.4f} | MAE: {validation_results[1]:.4f}")
print(f"TEST DATASET (Independent) - LOSS (MSE): {test_results[0]:.4f} | MAE: {test_results[1]:.4f}")

# ============================================================================
# STEP 9: CREATE VISUALIZATIONS
# ============================================================================

fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# -------- PLOT 1: Actual vs Predicted Values --------
# This plot shows how well the MODEL's PREDICTIONS match the actual OUTPUT
# using the TEST DATASET which is completely independent
ax1 = axes[0]

# Plot for TEST DATASET (independent)
ax1.scatter(TEST_OUTPUT, test_predictions.flatten(), alpha=0.5, label='TEST DATASET (Independent)', s=30)

# Plot perfect prediction line
min_val = min(TEST_OUTPUT.min(), test_predictions.min())
max_val = max(TEST_OUTPUT.max(), test_predictions.max())
ax1.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')

ax1.set_xlabel('Actual OUTPUT (Quality)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Predicted OUTPUT (Quality)', fontsize=12, fontweight='bold')
ax1.set_title('Actual vs Predicted Values\n(TEST DATASET - Independent)', fontsize=14, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# -------- PLOT 2: Loss vs Epochs --------
# This plot shows how the LOSS FUNCTION decreases as we go through more EPOCHS
ax2 = axes[1]

# Training LOSS over EPOCHS
ax2.plot(training_history.history['loss'], 
         label='TRAIN DATASET LOSS', 
         linewidth=2)

# Validation LOSS over EPOCHS
ax2.plot(training_history.history['val_loss'], 
         label='VALIDATION DATASET LOSS', 
         linewidth=2)

ax2.set_xlabel('EPOCH', fontsize=12, fontweight='bold')
ax2.set_ylabel('LOSS FUNCTION (MSE)', fontsize=12, fontweight='bold')
ax2.set_title('LOSS vs EPOCHS\n(During Training)', fontsize=14, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
output_path = 'wine_quality_results.jpg'
plt.savefig(output_path, dpi=100, format='jpg')
print(f"\nPlots saved as '{output_path}'")
plt.show()

# ============================================================================
# SUMMARY OF KEY CONCEPTS
# ============================================================================

print("\n" + "="*70)
print("KEY MACHINE LEARNING CONCEPTS DEMONSTRATED")
print("="*70)
print(f"1. INPUT: {INPUT_features.shape[1]} explanatory variables ({len(INPUT_features)} samples)")
print(f"2. OUTPUT: 1 response variable (wine quality)")
print(f"3. MODEL: Neural network with 4 layers (64->32->16->1 neurons)")
print(f"4. LOSS FUNCTION: Mean Squared Error (measures prediction error)")
print(f"5. EPOCH: One complete pass through TRAIN DATASET ({NUM_EPOCHS} total)")
print(f"6. BATCH: {BATCH_size} samples processed before each weight update")
print(f"7. PREDICT: MODEL generates quality predictions for new wines")
print(f"8. TRAIN DATASET: {len(TRAIN_INPUT)} samples (70%) - used to learn patterns")
print(f"9. VALIDATION DATASET: {len(VALIDATION_INPUT)} samples (15%) - used to monitor training")
print(f"10. TEST DATASET: {len(TEST_INPUT)} samples (15%) - completely INDEPENDENT for final evaluation")
print(f"\nDATA INDEPENDENCE:")
print(f"  - TEST dataset is NOT used during training")
print(f"  - TEST dataset provides truly unbiased performance estimation")
print("="*70)
