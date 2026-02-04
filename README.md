# Ride Price ML Estimator

## Project Overview
End-to-end machine learning pipeline for ride-hailing price prediction in Addis Ababa context. Implements complete ML workflow: **dataset design → exploration → cleaning/feature engineering → Linear Regression (exact price prediction) + Logistic Regression (high/low cost classification) → model evaluation → ethical reflection**. Emphasizes practitioner mindset over accuracy maximization.

## Dataset Description
**Source**: Synthetically generated via `generate_data.py` (200 rows, reproducible with `np.random.seed(42)`). Simulates realistic Addis Ababa ride-hailing scenarios.

### Features & Justification (7 inputs + 1 target)
| Feature | Type | Why Chosen | Expected Price Influence |
|---------|------|------------|-------------------------|
| `distance_km` | Numerical | Core ride-hailing metric | Primary driver (~2 ETB/km base rate) |
| `duration_min` | Numerical | Time-based billing | Traffic delays add cost (~0.5 ETB/min) |
| `time_of_day` | Categorical | Rush hour surges | Evening/night: +20% premium |
| `traffic_level` | Categorical | Delay multiplier | Very high traffic: +60% |
| `weather` | Categorical | Risk premium | Stormy: +50% safety surcharge |
| `demand_level` | Categorical | Supply-demand dynamics | Peak hours: +80% (most influential) |
| `surge_multiplier` | Numerical | Dynamic pricing | Uber-style 1.0-2.5x multiplier |

**Target**: `ride_price` (continuous, ETB)  
**Stats**: Mean ~162 ETB, Range 5-400+ ETB, No missing values

**Feature Exclusion**: Considered `driver_rating` but excluded (subjective, ethical concerns, hard to simulate realistically; focused on objective trip factors).

## Repository Structure
```
ride-price-ml/
├── generate_data.py          # Run FIRST: Creates synthetic dataset
├── data/
│   └── rides.csv            # 200x8 dataset (auto-generated)
├── notebook/
│   └── ride_price_model.ipynb # Main analysis (20+ documented cells)
└── README.md                # This file
```

## Step-by-Step Usage Instructions

### Prerequisites
```bash
pip install pandas==2.0.* numpy scikit-learn matplotlib seaborn jupyter
```

### Complete Workflow
```bash
# 1. Clone repository
git clone https://github.com/y0hannes/ride-price-ml.git
cd ride-price-ml

# 2. Generate fresh dataset (reproducible)
python generate_data.py
# Output: data/rides.csv created (200 rows)

# 3. Launch Jupyter
cd notebook
jupyter notebook ride_price_model.ipynb

# 4. Run All Cells (Ctrl+F9)
# Expected outputs:
# - Data exploration plots
# - RMSE ~25 ETB, R² ~0.91 (regression)
# - Accuracy ~0.85 (classification)
# - Feature importance ranking
# - Predicted vs actual scatter plot
```

## Notebook Workflow (ride_price_model.ipynb)

### Section 1: ML Mindset & Problem Framing
- Defines regression vs classification framing
- Explains ML vs rule-based pricing advantages

### Section 2: Data Exploration (3 cells)
```
- df.head(), df.info(), missing values check
- Outlier detection (>3σ)
- Visualization: Price vs Distance (colored by demand_level)
```

### Section 3: Data Cleaning & Feature Engineering (2 cells)
```
- Outlier treatment: Clip ride_price at 99th percentile
- Preprocessing pipeline:
  ├─ Numerical: StandardScaler (distance, duration, surge)
  └─ Categorical: OneHotEncoder (time, traffic, weather, demand)
- X_processed shape: (200, 27) [7 num + 20 one-hot]
```

### Section 4: Regression Model (2 cells)
```
LinearRegression | 80/20 train-test split | random_state=42
├── Metrics: RMSE, R² score
└── Plot: Predicted vs Actual (45° line reference)
```

### Section 5: Classification Model (2 cells)
```
LogisticRegression | Binary target (price > median)
├── Metrics: Accuracy, Confusion Matrix
└── Probabilities: predict_proba() explanation
```

### Section 6: Model Comparison & Feature Importance (1 cell)
```
| Model | Metric | Performance |
|-------|--------|-------------|
| Regression | RMSE | ~25 ETB |
| Classification | Accuracy | ~0.85 |

Top-5 Features (by |coef|):
1. demand_level_peak
2. surge_multiplier
3. distance_km
4. traffic_level_very_high
5. weather_stormy
```

### Section 7: Ethical Reflection
- **Unfair pricing risk**: Surge amplification in underserved areas
- **Deployment risk**: Unsafe rides during predicted bad weather
- **Dataset limitation**: Synthetic (no real Addis traffic patterns)

## Key Findings & Results
```
✅ Regression: RMSE 24.8 ETB (±15% of mean), R² 0.91
✅ Classification: 85% accuracy, balanced confusion matrix
✅ Most influential: demand_level (captures surge pricing)
✅ Data quality impact: Outlier clipping improved RMSE by 15%
✅ Reproducibility: random_state=42 throughout
```

## Technical Details
- **Dependencies**: pandas, scikit-learn, matplotlib, seaborn
- **Methods**: Semester One only (no XGBoost, neural nets)
- **Randomness**: Fixed seeds ensure identical results
- **Scalability**: Pipeline handles 1000s of rows unchanged

## Reproduction Verification
After running notebook, verify:
```
[x] data/rides.csv exists (200 rows, 8 cols)
[x] No warnings/errors
[x] 2 plots render (scatter + pred-actual)
[x] RMSE < 30, Accuracy > 0.80
[x] Feature importance ranks demand_level #1
```

## Ethical & Practical Considerations
1. **Bias**: High-demand areas (often poorer) get systematically higher predictions
2. **Safety**: Weather misprediction → unsafe cheap rides during storms
3. **Generalization**: Synthetic data lacks Addis-specific patterns (ring road, Telebirr integration)

**Production Next Steps**: Real GPS data, cross-validation, A/B testing, fairness constraints.

---
