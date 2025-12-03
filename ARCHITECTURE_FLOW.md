# SE-CNN Complete Architecture Flow

## Overview: Input → Processing → Output (with Channel Attention)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           RAW INPUT DATA                                 │
│  CSV File: 15,000 marketing campaign records                            │
│  - Ad_ID, Campaign_ID, Platform, Country                                │
│  - CPC_USD, CTR, Conversion_Rate, Impressions, Clicks                   │
│  - Spend_USD, ROI, LTV_Proxy, etc.                                      │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│                    PHASE 1: DATA PREPROCESSING                           │
├─────────────────────────────────────────────────────────────────────────┤
│ Step 1.1: DATA CLEANING                                                 │
│   • Remove empty rows                                                   │
│   • Remove duplicates (based on Ad_ID)                                  │
│   • Handle missing values                                               │
│   • Remove outliers using IQR method                                    │
│   Output: ~14,000-14,500 clean records                                  │
├─────────────────────────────────────────────────────────────────────────┤
│ Step 1.2: FEATURE ENGINEERING                                           │
│   • Calculate High_Performing_Label:                                    │
│     Label = 1 if (CTR ≥ 75th percentile AND CPC ≤ 25th percentile)    │
│     Label = 0 otherwise                                                 │
│   • Verify/Recalculate LTV_Proxy:                                       │
│     LTV_Proxy = CTR × CPC × Conversion_Rate                            │
│   • Ensure ROI, Cost_per_Lead exist                                     │
├─────────────────────────────────────────────────────────────────────────┤
│ Step 1.3: CATEGORICAL ENCODING                                          │
│   Platform    → [0, 1, 2, 3, 4]  (Google, Meta, Twitter, etc.)        │
│   Country     → [0, 1, 2, ..., N]  (unique country IDs)               │
│   Campaign_ID → [0, 1, 2, ..., M]  (unique campaign IDs)              │
├─────────────────────────────────────────────────────────────────────────┤
│ Step 1.4: NUMERICAL NORMALIZATION                                       │
│   StandardScaler (Z-score normalization):                               │
│   • CPC_USD, CTR, Conversion_Rate → mean=0, std=1                      │
│   • All numerical features standardized                                 │
├─────────────────────────────────────────────────────────────────────────┤
│ Step 1.5: DATA SPLITTING                                                │
│   • Training:   70% (~10,000 samples)                                   │
│   • Validation: 15% (~2,200 samples)                                    │
│   • Testing:    15% (~2,200 samples)                                    │
│   Stratified split maintains class balance                              │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│              PHASE 2: SE-CNN MODEL ARCHITECTURE                          │
│         (Squeeze-and-Excitation CNN with Channel Attention)             │
├─────────────────────────────────────────────────────────────────────────┤
│ INPUT LAYER                                                             │
│   ┌──────────────────────┐  ┌──────────────────────┐                  │
│   │ Numerical Features   │  │ Categorical Features │                  │
│   │ Shape: (batch, 10)   │  │ Platform, Country,   │                  │
│   │ - CPC_USD            │  │ Campaign_ID          │                  │
│   │ - CTR                │  │                      │                  │
│   │ - Conversion_Rate    │  │ Each → Embedding     │                  │
│   │ - Impressions        │  │ Dim: 8               │                  │
│   │ - Clicks             │  │                      │                  │
│   │ - Spend_USD          │  │ Output: vectors      │                  │
│   │ - ROI                │  │ Shape: (batch, 8)    │                  │
│   │ - Click_Through_Lift │  │ each                 │                  │
│   │ - Cost_per_Lead      │  │                      │                  │
│   │ - LTV_Proxy          │  │                      │                  │
│   └──────────────────────┘  └──────────────────────┘                  │
│              │                          │                               │
│              └──────────┬───────────────┘                               │
│                         ↓                                               │
│   ┌─────────────────────────────────────────────────┐                  │
│   │ CONCATENATE ALL FEATURES                        │                  │
│   │ Shape: (batch, 10 + 8×3) = (batch, 34)         │                  │
│   └─────────────────────────────────────────────────┘                  │
│                         ↓                                               │
│   ┌─────────────────────────────────────────────────┐                  │
│   │ RESHAPE FOR 1D CNN                              │                  │
│   │ Shape: (batch, 34, 1)                           │                  │
│   └─────────────────────────────────────────────────┘                  │
├─────────────────────────────────────────────────────────────────────────┤
│ CONVOLUTIONAL LAYERS WITH SE BLOCKS                                     │
│                                                                          │
│   ┌─────────────────────────────────────────────────┐                  │
│   │ Conv1D Layer 1                                  │                  │
│   │ • Filters: 32                                   │                  │
│   │ • Kernel Size: 3                                │                  │
│   │ • Activation: ReLU                              │                  │
│   │ • Padding: same                                 │                  │
│   │ • Output Shape: (batch, 34, 32)                │                  │
│   └─────────────────────────────────────────────────┘                  │
│                         ↓                                               │
│   ┌─────────────────────────────────────────────────┐                  │
│   │ Batch Normalization                             │                  │
│   │ Purpose: Stabilize training                     │                  │
│   └─────────────────────────────────────────────────┘                  │
│                         ↓                                               │
│   ┌─────────────────────────────────────────────────┐                  │
│   │ ⭐ SE BLOCK 1 (Squeeze-and-Excitation)          │                  │
│   │ ┌─────────────────────────────────────────────┐│                  │
│   │ │ STEP 1: SQUEEZE                              ││                  │
│   │ │ GlobalAveragePooling1D                        ││                  │
│   │ │ Input:  (batch, 34, 32)                      ││                  │
│   │ │ Output: (batch, 32)  [channel-wise stats]    ││                  │
│   │ └─────────────────────────────────────────────┘│                  │
│   │                         ↓                       │                  │
│   │ ┌─────────────────────────────────────────────┐│                  │
│   │ │ STEP 2: EXCITATION                           ││                  │
│   │ │ Dense(32//16 = 2, ReLU)  [reduction]        ││                  │
│   │ │ Dense(32, Sigmoid)        [attention weights]││                  │
│   │ │ Output: (batch, 32)  [0-1 per channel]      ││                  │
│   │ └─────────────────────────────────────────────┘│                  │
│   │                         ↓                       │                  │
│   │ ┌─────────────────────────────────────────────┐│                  │
│   │ │ STEP 3: SCALE                                ││                  │
│   │ │ Reshape to (batch, 1, 32)                   ││                  │
│   │ │ Multiply with original Conv1D output       ││                  │
│   │ │ Purpose: Channel-wise attention weighting    ││                  │
│   │ │ Output: (batch, 34, 32)  [attended features]││                  │
│   │ └─────────────────────────────────────────────┘│                  │
│   └─────────────────────────────────────────────────┘                  │
│                         ↓                                               │
│   ┌─────────────────────────────────────────────────┐                  │
│   │ MaxPooling1D (pool_size=2)                      │                  │
│   │ Purpose: Reduce dimensionality                  │                  │
│   │ Output Shape: (batch, 17, 32)                   │                  │
│   └─────────────────────────────────────────────────┘                  │
│                         ↓                                               │
│   ┌─────────────────────────────────────────────────┐                  │
│   │ Dropout (rate=0.2)                              │                  │
│   │ Purpose: Prevent overfitting                    │                  │
│   └─────────────────────────────────────────────────┘                  │
│                         ↓                                               │
│   ┌─────────────────────────────────────────────────┐                  │
│   │ Conv1D Layer 2                                  │                  │
│   │ • Filters: 64                                   │                  │
│   │ • Kernel Size: 3                                │                  │
│   │ • Activation: ReLU                              │                  │
│   │ • Output Shape: (batch, 17, 64)                 │                  │
│   └─────────────────────────────────────────────────┘                  │
│                         ↓                                               │
│   ┌─────────────────────────────────────────────────┐                  │
│   │ Batch Normalization                             │                  │
│   └─────────────────────────────────────────────────┘                  │
│                         ↓                                               │
│   ┌─────────────────────────────────────────────────┐                  │
│   │ ⭐ SE BLOCK 2 (Squeeze-and-Excitation)          │                  │
│   │ ┌─────────────────────────────────────────────┐│                  │
│   │ │ STEP 1: SQUEEZE                              ││                  │
│   │ │ GlobalAveragePooling1D                        ││                  │
│   │ │ Input:  (batch, 17, 64)                       ││                  │
│   │ │ Output: (batch, 64)  [channel-wise stats]    ││                  │
│   │ └─────────────────────────────────────────────┘│                  │
│   │                         ↓                       │                  │
│   │ ┌─────────────────────────────────────────────┐│                  │
│   │ │ STEP 2: EXCITATION                           ││                  │
│   │ │ Dense(64//16 = 4, ReLU)  [reduction]        ││                  │
│   │ │ Dense(64, Sigmoid)        [attention weights]││                  │
│   │ │ Output: (batch, 64)  [0-1 per channel]      ││                  │
│   │ └─────────────────────────────────────────────┘│                  │
│   │                         ↓                       │                  │
│   │ ┌─────────────────────────────────────────────┐│                  │
│   │ │ STEP 3: SCALE                                ││                  │
│   │ │ Reshape to (batch, 1, 64)                   ││                  │
│   │ │ Multiply with original Conv1D output       ││                  │
│   │ │ Purpose: Channel-wise attention weighting    ││                  │
│   │ │ Output: (batch, 17, 64)  [attended features]││                  │
│   │ └─────────────────────────────────────────────┘│                  │
│   └─────────────────────────────────────────────────┘                  │
│                         ↓                                               │
│   ┌─────────────────────────────────────────────────┐                  │
│   │ GlobalMaxPooling1D                              │                  │
│   │ Purpose: Extract most important features        │                  │
│   │ Output Shape: (batch, 64)                       │                  │
│   └─────────────────────────────────────────────────┘                  │
├─────────────────────────────────────────────────────────────────────────┤
│ SHARED DENSE LAYER                                                      │
│   ┌─────────────────────────────────────────────────┐                  │
│   │ Dense Layer (64 units, ReLU)                    │                  │
│   │ Purpose: High-level feature representation      │                  │
│   └─────────────────────────────────────────────────┘                  │
│                         ↓                                               │
│   ┌─────────────────────────────────────────────────┐                  │
│   │ Dropout (rate=0.2)                              │                  │
│   └─────────────────────────────────────────────────┘                  │
│                         ↓                                               │
│              ┌──────────┴──────────┐                                    │
│              ↓                     ↓                                    │
├─────────────────────────┬───────────────────────────────────────────────┤
│ MULTI-TASK HEADS        │                                               │
│                         │                                               │
│  ┌──────────────────────┴────────────────────────┐                     │
│  │ HEAD 1: CLASSIFICATION                        │                     │
│  │ ┌────────────────────────────────────────┐    │                     │
│  │ │ Dense (32 units, ReLU)                 │    │                     │
│  │ └────────────────────────────────────────┘    │                     │
│  │              ↓                                 │                     │
│  │ ┌────────────────────────────────────────┐    │                     │
│  │ │ Dense (1 unit, Sigmoid)                │    │                     │
│  │ │ Output: Probability [0, 1]             │    │                     │
│  │ │ Prediction: High/Low Performing        │    │                     │
│  │ └────────────────────────────────────────┘    │                     │
│  └───────────────────────────────────────────────┘                     │
│                         │                                               │
│  ┌──────────────────────┴────────────────────────┐                     │
│  │ HEAD 2: REGRESSION (LTV ESTIMATION)           │                     │
│  │ ┌────────────────────────────────────────┐    │                     │
│  │ │ Dense (32 units, ReLU)                 │    │                     │
│  │ └────────────────────────────────────────┘    │                     │
│  │              ↓                                 │                     │
│  │ ┌────────────────────────────────────────┐    │                     │
│  │ │ Dense (1 unit, Linear)                 │    │                     │
│  │ │ Output: Continuous value               │    │                     │
│  │ │ Prediction: Long-Term Value (LTV)      │    │                     │
│  │ └────────────────────────────────────────┘    │                     │
│  └───────────────────────────────────────────────┘                     │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│                    PHASE 3: MODEL TRAINING                               │
├─────────────────────────────────────────────────────────────────────────┤
│ LOSS FUNCTIONS:                                                         │
│   • Classification: Binary Cross-Entropy (weight: 0.6)                  │
│   • Regression: Mean Squared Error (weight: 0.4)                        │
│   • Total Loss = 0.6 × BCE + 0.4 × MSE                                 │
├─────────────────────────────────────────────────────────────────────────┤
│ OPTIMIZER:                                                              │
│   • Adam Optimizer                                                      │
│   • Initial Learning Rate: 0.001                                        │
│   • Learning Rate Decay: Reduces by 0.5× if no improvement             │
├─────────────────────────────────────────────────────────────────────────┤
│ TRAINING PROCESS:                                                       │
│   For each epoch (max 20):                                              │
│     1. Forward pass on training batch                                   │
│     2. SE blocks learn channel importance                               │
│     3. Calculate losses (classification + regression)                   │
│     4. Backward propagation                                             │
│     5. Update weights (including SE attention weights)                   │
│     6. Validate on validation set                                       │
│     7. Save best model (based on val_loss)                              │
│     8. Early stop if no improvement for 15 epochs                       │
├─────────────────────────────────────────────────────────────────────────┤
│ OUTPUTS:                                                                │
│   • Trained model saved to: se_cnn_model/models/se_cnn_marketing_model.h5│
│   • Training history: se_cnn_model/results/training_history.pkl         │
│   • Plots: se_cnn_model/results/training_history.png                    │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│                    PHASE 4: MODEL EVALUATION                             │
├─────────────────────────────────────────────────────────────────────────┤
│ TEST SET PREDICTIONS:                                                   │
│   For each test sample:                                                 │
│     • Input: Normalized features                                        │
│     • SE blocks apply learned channel attention                         │
│     • Output 1: Classification probability [0, 1]                     │
│     • Output 2: LTV prediction (continuous value)                       │
├─────────────────────────────────────────────────────────────────────────┤
│ CLASSIFICATION METRICS:                                                 │
│   • Accuracy = (TP + TN) / Total                                        │
│   • Precision = TP / (TP + FP)                                          │
│   • Recall = TP / (TP + FN)                                             │
│   • F1-Score = 2 × (Precision × Recall) / (Precision + Recall)         │
│   • AUC = Area Under ROC Curve                                          │
├─────────────────────────────────────────────────────────────────────────┤
│ REGRESSION METRICS:                                                     │
│   • MAE = Mean Absolute Error                                           │
│   • RMSE = Root Mean Squared Error                                      │
│   • R² = Coefficient of Determination                                   │
├─────────────────────────────────────────────────────────────────────────┤
│ VISUALIZATION OUTPUTS:                                                  │
│   • Confusion Matrix: True vs Predicted labels                          │
│   • ROC Curve: TPR vs FPR with AUC score                               │
│   • Regression Scatter: Predicted vs Actual LTV                         │
│   • Residual Plot: Prediction errors                                    │
├─────────────────────────────────────────────────────────────────────────┤
│ BASELINE COMPARISON:                                                    │
│   Compare SE-CNN with:                                                 │
│   • Logistic Regression (baseline 1)                                    │
│   • XGBoost (baseline 2)                                                │
│   • Wide & Deep Learning (baseline 3)                                 │
│   Show improvement in table and charts                                  │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│                PHASE 5: BUDGET OPTIMIZATION                              │
├─────────────────────────────────────────────────────────────────────────┤
│ PREPARE OPTIMIZATION DATA:                                              │
│   For each campaign in test set:                                        │
│     • Get predicted probability (p) from SE-CNN                         │
│     • Get predicted LTV from SE-CNN                                     │
│     • Calculate Expected Value:                                         │
│       EV = p × ROI × LTV                                                │
│     • Filter campaigns with EV > 0                                      │
├─────────────────────────────────────────────────────────────────────────┤
│ LINEAR PROGRAMMING FORMULATION:                                         │
│                                                                          │
│   Decision Variables:                                                   │
│     Budget_i = amount to allocate to campaign i                         │
│                                                                          │
│   Objective Function (MAXIMIZE):                                        │
│     Σ (Budget_i × Expected_Value_i)                                     │
│     for all campaigns i                                                 │
│                                                                          │
│   Subject to Constraints:                                               │
│     1. Total Budget:                                                    │
│        Σ Budget_i ≤ $200,000                                            │
│                                                                          │
│     2. Per-Campaign Limits:                                             │
│        $100 ≤ Budget_i ≤ $50,000  ∀i                                    │
│                                                                          │
│     3. Platform Constraints:                                            │
│        Google:   20% ≤ Google_Budget ≤ 60% of total                     │
│        Meta:     15% ≤ Meta_Budget ≤ 50% of total                      │
│        Twitter:  10% ≤ Twitter_Budget ≤ 50% of total                    │
│        LinkedIn: 10% ≤ LinkedIn_Budget ≤ 40% of total                   │
│        TikTok:   10% ≤ TikTok_Budget ≤ 20% of total                    │
├─────────────────────────────────────────────────────────────────────────┤
│ SOLVER:                                                                 │
│   • Algorithm: PuLP Linear Programming Solver                           │
│   • Method: Simplex or Interior Point                                   │
│   • Finds optimal Budget_i values that maximize total expected value    │
├─────────────────────────────────────────────────────────────────────────┤
│ ALLOCATION ANALYSIS:                                                    │
│   • Platform-wise distribution                                          │
│   • Country-wise distribution                                           │
│   • High vs Low performing campaigns                                    │
│   • Budget utilization percentage                                       │
│   • Expected ROI improvement                                            │
├─────────────────────────────────────────────────────────────────────────┤
│ VISUALIZATION OUTPUTS:                                                  │
│   • Pie chart: Platform budget distribution                             │
│   • Bar chart: Top 10 countries                                         │
│   • Scatter: Budget vs Expected Value                                   │
│   • Histogram: Budget distribution                                      │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│                      FINAL OUTPUTS                                       │
├─────────────────────────────────────────────────────────────────────────┤
│ 1. SAVED MODEL:                                                         │
│    • se_cnn_model/models/se_cnn_marketing_model.h5                      │
│    • se_cnn_model/models/scaler.pkl (feature scaler)                    │
│    • se_cnn_model/models/encoder.pkl (categorical encoders)            │
├─────────────────────────────────────────────────────────────────────────┤
│ 2. VISUALIZATIONS:                                                      │
│    • se_cnn_model/results/training_history.png                           │
│    • se_cnn_model/results/confusion_matrix.png                           │
│    • se_cnn_model/results/roc_curve.png                                  │
│    • se_cnn_model/results/regression_results.png                         │
│    • se_cnn_model/results/budget_allocation.png                          │
│    • se_cnn_model/results/baseline_comparison.png                        │
├─────────────────────────────────────────────────────────────────────────┤
│ 3. DATA FILES:                                                          │
│    • se_cnn_model/results/budget_recommendations.csv                     │
│      Columns: Ad_ID, Campaign_ID, Platform, Country,                   │
│               Predicted_Prob, Predicted_LTV, Optimal_Budget             │
│    • se_cnn_model/results/model_comparison.csv                          │
│      Comparison with LR, XGBoost, WDL                                    │
├─────────────────────────────────────────────────────────────────────────┤
│ 4. REPORTS:                                                             │
│    • se_cnn_model/results/summary_report.txt                            │
│      Complete textual summary of all results                            │
│    • se_cnn_model/results/experiment_config.json                        │
│      All hyperparameters and settings used                              │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Key Difference: Squeeze-and-Excitation (SE) Blocks

### What are SE Blocks?

SE blocks are **channel attention mechanisms** that allow the model to:
1. **Squeeze**: Compress spatial information into channel-wise statistics
2. **Excite**: Learn which channels are most important
3. **Scale**: Apply learned attention weights to emphasize important channels

### SE Block Architecture:

```
Input: (batch, timesteps, channels)
    ↓
┌─────────────────────────────────────┐
│ 1. SQUEEZE                          │
│ GlobalAveragePooling1D               │
│ Output: (batch, channels)           │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ 2. EXCITATION                       │
│ Dense(channels//16, ReLU)  [reduce] │
│ Dense(channels, Sigmoid)   [weights]│
│ Output: (batch, channels) [0-1]     │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ 3. SCALE                             │
│ Reshape to (batch, 1, channels)     │
│ Multiply with original input        │
│ Output: (batch, timesteps, channels)│
└─────────────────────────────────────┘
```

### Why SE Blocks Help:

1. **Channel Attention**: Automatically learns which feature channels are most informative
2. **Adaptive Feature Recalibration**: Emphasizes important channels, suppresses less useful ones
3. **Improved Performance**: Typically improves accuracy by 1-2% compared to standard CNN
4. **Minimal Overhead**: Adds only ~2-5% more parameters

### SE Block Parameters:

- **Reduction Ratio**: 16 (default)
  - Controls compression in excitation network
  - Lower ratio = more parameters, more capacity
  - Higher ratio = fewer parameters, less capacity

- **Placement**: After each Conv1D layer
  - SE Block 1: After Conv1D with 32 filters
  - SE Block 2: After Conv1D with 64 filters

---

## Purpose of Each Feature

### Numerical Features (Inputs to SE-CNN):

1. **CPC_USD** (Cost Per Click)
   - Purpose: Measures ad cost efficiency
   - Used for: Classification decision (low CPC = high performing)
   - SE Attention: Model learns to focus on CPC-related channels
   - Optimization: Lower CPC campaigns get priority

2. **CTR** (Click-Through Rate)
   - Purpose: Engagement metric (clicks/impressions)
   - Used for: Classification decision (high CTR = high performing)
   - SE Attention: Emphasizes CTR importance in feature channels
   - Optimization: Higher CTR campaigns get more budget

3. **Conversion_Rate**
   - Purpose: Success rate of turning clicks into actions
   - Used for: LTV calculation and classification
   - SE Attention: Highlights conversion-related patterns
   - Optimization: Higher conversion = higher expected value

4. **Impressions**
   - Purpose: Campaign reach indicator
   - Used for: Scale and performance context
   - SE Attention: Helps identify scale-related patterns
   - Optimization: Helps estimate campaign potential

5. **Clicks**
   - Purpose: Actual engagement count
   - Used for: CTR calculation verification
   - SE Attention: Emphasizes engagement channels
   - Optimization: Volume indicator for scaling

6. **Spend_USD**
   - Purpose: Total campaign cost
   - Used for: ROI calculation
   - SE Attention: Focuses on cost-related features
   - Optimization: Efficiency baseline

7. **ROI** (Return on Investment)
   - Purpose: Profitability measure
   - Used for: Expected value calculation
   - SE Attention: Critical feature - gets high attention weight
   - Optimization: CRITICAL - multiplied with probability

8. **Click_Through_Lift**
   - Purpose: Performance improvement over baseline
   - Used for: Campaign quality indicator
   - SE Attention: Identifies outperforming patterns
   - Optimization: Identifies outperforming campaigns

9. **Cost_per_Lead**
   - Purpose: Acquisition cost efficiency
   - Used for: Budget efficiency metric
   - SE Attention: Emphasizes efficiency channels
   - Optimization: Lower CPL = better allocation

10. **LTV_Proxy** (Long-Term Value)
    - Purpose: Estimated customer lifetime value
    - Used for: Regression target AND optimization
    - SE Attention: High attention for LTV prediction
    - Optimization: CRITICAL - determines long-term ROI

### Categorical Features (Embedded then fed to SE-CNN):

11. **Platform**
    - Purpose: Ad channel (Google, Meta, TikTok, etc.)
    - Used for: Platform-specific patterns
    - SE Attention: Learns platform-specific channel importance
    - Optimization: Platform budget constraints applied

12. **Country**
    - Purpose: Geographic market
    - Used for: Regional performance patterns
    - SE Attention: Identifies country-specific feature importance
    - Optimization: Identifies high-ROI markets

13. **Campaign_ID**
    - Purpose: Unique campaign identifier
    - Used for: Campaign-level tracking
    - SE Attention: Captures campaign-specific patterns
    - Optimization: Individual budget allocation

---

## Target Variables

### 1. High_Performing_Label (Classification Target)
**Formula:**
```
IF (CTR ≥ 75th percentile) AND (CPC ≤ 25th percentile):
    Label = 1  (High Performing)
ELSE:
    Label = 0  (Low Performing)
```

**Purpose:** Binary classification to identify successful campaigns

**Training:** Used as y_classification in SE-CNN Head 1

**Evaluation:** Accuracy, Precision, Recall, F1, AUC

**Optimization:** Predicted probability used in expected value

**SE Block Impact:** Channel attention helps focus on CTR and CPC-related features

### 2. LTV_Proxy (Regression Target)
**Formula:**
```
LTV_Proxy = CTR × CPC_USD × Conversion_Rate
```

**Purpose:** Estimate long-term customer value

**Training:** Used as y_regression in SE-CNN Head 2

**Evaluation:** MAE, RMSE, R²

**Optimization:** Predicted LTV multiplied with probability and ROI

**SE Block Impact:** Emphasizes channels related to CTR, CPC, and conversion

---

## How Everything Works Together

1. **Raw Data** → Cleaned and normalized features
2. **Features** → SE-CNN learns complex patterns with channel attention
3. **SE Blocks** → Automatically identify and emphasize important feature channels
4. **SE-CNN Predictions** → 
   - Probability: Is this campaign high-performing? (with channel attention)
   - LTV: What's the long-term value? (with channel attention)
5. **Predictions + Business Logic** → Optimization
   - Expected Value = Probability × ROI × LTV
   - Linear Programming maximizes total expected value
   - Subject to budget constraints
6. **Output** → Optimal budget for each campaign

---

## Advantages of SE-CNN over Standard CNN

1. **Channel Attention**: Automatically learns which feature channels matter most
2. **Better Feature Utilization**: Emphasizes important channels, suppresses noise
3. **Improved Generalization**: Better performance on test set
4. **Interpretability**: SE attention weights show which channels are important
5. **Minimal Overhead**: Only adds ~2-5% more parameters

---

## Model Comparison

| Aspect | Standard CNN | SE-CNN |
|--------|-------------|--------|
| **Architecture** | Conv1D → Pool → Conv1D | Conv1D → SE → Pool → Conv1D → SE |
| **Parameters** | ~50K | ~52K (+2-5%) |
| **Channel Attention** | ❌ No | ✅ Yes |
| **Feature Recalibration** | ❌ No | ✅ Adaptive |
| **Expected Improvement** | Baseline | +1-2% accuracy |
| **Training Time** | Baseline | +5-10% longer |

---

## Technical Details

### SE Block Implementation:

```python
def se_block(input_tensor, reduction_ratio=16, name_prefix=''):
    channels = input_tensor.shape[-1]  # e.g., 32 or 64
    
    # Squeeze: Global Average Pooling
    se = GlobalAveragePooling1D()(input_tensor)  # (batch, channels)
    
    # Excitation: Two Dense layers
    se = Dense(channels // reduction_ratio, activation='relu')(se)
    se = Dense(channels, activation='sigmoid')(se)  # (batch, channels)
    
    # Scale: Reshape and multiply
    se = Reshape((1, channels))(se)  # (batch, 1, channels)
    output = Multiply()([input_tensor, se])  # Element-wise multiplication
    
    return output
```

### SE Block Placement:

- **After Conv1D Layer 1** (32 filters):
  - Input: (batch, 34, 32)
  - SE processes 32 channels
  - Output: (batch, 34, 32) with channel attention

- **After Conv1D Layer 2** (64 filters):
  - Input: (batch, 17, 64)
  - SE processes 64 channels
  - Output: (batch, 17, 64) with channel attention

---

## Summary

The SE-CNN model extends the standard CNN architecture by adding **Squeeze-and-Excitation blocks** that implement channel attention. This allows the model to:

1. Automatically identify which feature channels are most important
2. Emphasize important channels while suppressing less useful ones
3. Improve prediction accuracy through adaptive feature recalibration
4. Maintain computational efficiency with minimal parameter overhead

The SE blocks are integrated seamlessly into the existing CNN architecture, placed after each convolutional layer to provide channel-wise attention throughout the feature extraction process.

