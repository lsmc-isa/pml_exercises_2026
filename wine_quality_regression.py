"""
Wine Quality Regression Model
This script trains a neural network regression model to predict wine quality.
It demonstrates key machine learning concepts with explicit variable labeling.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
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

print("Loading Wine Quality Dataset...")

# Download the red wine quality dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
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
# STEP 3: SPLIT INTO TRAINING AND TEST DATASETS
# ============================================================================

# TRAIN DATASET: Used to train the model and learn the patterns
# TEST DATASET: Used to evaluate the model on unseen data
TRAIN_INPUT, TEST_INPUT, TRAIN_OUTPUT, TEST_OUTPUT = train_test_split(
    INPUT_features, 
    OUTPUT_target, 
    test_size=0.2,  # 20% for testing, 80% for training
    random_state=42
)

print(f"\nTRAIN DATASET size: {len(TRAIN_INPUT)} examples")
print(f"TEST DATASET size: {len(TEST_INPUT)} examples")

# Standardize the features for better training
scaler = StandardScaler()
TRAIN_INPUT_scaled = scaler.fit_transform(TRAIN_INPUT)
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
training_history = MODEL.fit(
    TRAIN_INPUT_scaled, 
    TRAIN_OUTPUT,
    epochs=NUM_EPOCHS,           # Number of complete passes through TRAIN DATASET
    batch_size=BATCH_size,        # Size of BATCH used in each update
    validation_data=(TEST_INPUT_scaled, TEST_OUTPUT),
    verbose=1
)

print("\nTraining completed!")

# ============================================================================
# STEP 7: MAKE PREDICTIONS
# ============================================================================

# PREDICT: Use the trained MODEL to generate OUTPUT values for new INPUT data

# Predictions on TRAIN DATASET
train_predictions = MODEL.predict(TRAIN_INPUT_scaled, verbose=0)

# Predictions on TEST DATASET (independent data not seen during training)
test_predictions = MODEL.predict(TEST_INPUT_scaled, verbose=0)

print(f"\nPredictions generated!")
print(f"Train predictions shape: {train_predictions.shape}")
print(f"Test predictions shape: {test_predictions.shape}")

# ============================================================================
# STEP 8: EVALUATE THE MODEL
# ============================================================================

train_loss = MODEL.evaluate(TRAIN_INPUT_scaled, TRAIN_OUTPUT, verbose=0)
test_loss = MODEL.evaluate(TEST_INPUT_scaled, TEST_OUTPUT, verbose=0)

print(f"\nTRAIN DATASET LOSS: {train_loss[0]:.4f}")
print(f"TEST DATASET LOSS: {test_loss[0]:.4f}")

# ============================================================================
# STEP 9: CREATE VISUALIZATIONS
# ============================================================================

fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# -------- PLOT 1: Actual vs Predicted Values --------
# This plot shows how well the MODEL's PREDICTIONS match the actual OUTPUT
ax1 = axes[0]

# Plot for TEST DATASET
ax1.scatter(TEST_OUTPUT, test_predictions.flatten(), alpha=0.5, label='TEST DATASET', s=30)

# Plot perfect prediction line
min_val = min(TEST_OUTPUT.min(), test_predictions.min())
max_val = max(TEST_OUTPUT.max(), test_predictions.max())
ax1.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')

ax1.set_xlabel('Actual OUTPUT (Quality)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Predicted OUTPUT (Quality)', fontsize=12, fontweight='bold')
ax1.set_title('Actual vs Predicted Values\n(TEST DATASET)', fontsize=14, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# -------- PLOT 2: Loss vs Epochs --------
# This plot shows how the LOSS FUNCTION decreases as we go through more EPOCHS
ax2 = axes[1]

# Training LOSS over EPOCHS
ax2.plot(training_history.history['loss'], 
         label=f'TRAIN DATASET LOSS', 
         linewidth=2, 
         color='blue')

# Validation (TEST DATASET) LOSS over EPOCHS
ax2.plot(training_history.history['val_loss'], 
         label=f'TEST DATASET LOSS', 
         linewidth=2, 
         color='orange')

ax2.set_xlabel('EPOCH', fontsize=12, fontweight='bold')
ax2.set_ylabel('LOSS FUNCTION (MSE)', fontsize=12, fontweight='bold')
ax2.set_title('LOSS vs EPOCHS\n(During Training)', fontsize=14, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
output_path = r'C:\Users\media\OneDrive\Ambiente de Trabalho\pml_exercises_2026\wine_quality_results.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\nPlots saved as '{output_path}'")
# plt.show()  # Commented out to avoid display issues in non-interactive environment

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
print(f"8. TRAIN DATASET: {len(TRAIN_INPUT)} samples used to train the MODEL")
print(f"9. TEST DATASET: {len(TEST_INPUT)} independent samples for evaluation")
print("="*70)
