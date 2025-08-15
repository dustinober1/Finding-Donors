
# Finding Donors for CharityML
*A Machine Learning Project for Predicting Income Levels*

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## Project Overview

This project applies supervised learning techniques to help CharityML (a fictitious charity organization) identify individuals most likely to donate based on census data. Using machine learning algorithms, we predict whether an individual makes more than $50,000 annually to optimize donation solicitation strategies.

## Key Results

- **Best Model**: Gradient Boosting Classifier
- **Accuracy**: 86.9% on test data
- **F-score**: 74.8% (β = 0.5)
- **Top Predictive Features**: Age, Education, Capital Gains, Hours per Week, Marital Status

## Technologies Used

- **Python 3.8+**
- **scikit-learn** - Machine learning algorithms
- **pandas** - Data manipulation and analysis
- **NumPy** - Numerical computing
- **matplotlib** - Data visualization
- **Jupyter Notebook** - Interactive development environment

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/Finding-Donors.git
cd Finding-Donors
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Launch Jupyter Notebook:
```bash
jupyter notebook notebooks/finding_donors.ipynb
```

## Dataset

The project uses a modified census dataset with approximately 45,000 data points and 13 features. This dataset is derived from the 1994 U.S. Census and was originally published in Ron Kohavi's paper *"Scaling Up the Accuracy of Naive-Bayes Classifiers: a Decision-Tree Hybrid"*.

- **Source**: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Census+Income)
- **Target**: Predict if income exceeds $50K annually
- **Class Distribution**: 24.8% earn >$50K, 75.2% earn ≤$50K

### Key Features
- **Demographic**: Age, sex, race, native country
- **Education**: Education level, years of education
- **Employment**: Work class, occupation, hours per week
- **Financial**: Capital gains/losses
- **Personal**: Marital status, relationship status

## Methodology

### 1. Data Preprocessing
- **Log transformation** of skewed features (capital-gain, capital-loss)
- **MinMax normalization** of numerical features
- **One-hot encoding** of categorical variables
- **Train/test split**: 80/20

### 2. Model Comparison
Evaluated three supervised learning algorithms:
- **AdaBoost Classifier**
- **Random Forest Classifier**  
- **Gradient Boosting Classifier** (selected)

### 3. Model Optimization
- **Grid Search** hyperparameter tuning
- **F-score optimization** (β = 0.5) to emphasize precision
- **Feature importance analysis**

## Results Summary

| Metric | Naive Predictor | Unoptimized Model | **Optimized Model** |
|--------|----------------|-------------------|-------------------|
| Accuracy | 24.8% | 86.3% | **86.9%** |
| F-score | 29.2% | 74.0% | **74.8%** |

### Feature Importance
The top 5 most predictive features account for over 50% of the model's decision-making:
1. **Age** - Strong correlation with earning potential
2. **Education Level** - Higher education → higher income
3. **Capital Gains** - Investment income indicator
4. **Hours per Week** - Work commitment level
5. **Marital Status** - Household income dynamics

## Project Structure
```
Finding-Donors/
├── data/
│   └── census.csv             # Census dataset
├── notebooks/
│   └── finding_donors.ipynb  # Main analysis notebook
├── src/
│   └── visuals.py            # Visualization utilities
├── requirements.txt          # Dependencies
├── README.md                # Project documentation
├── project_description.md   # Original assignment details
└── .gitignore              # Git ignore file
```

## Learning Outcomes

This project demonstrates:
- **Data preprocessing** techniques for real-world datasets
- **Model selection** and evaluation methodologies
- **Hyperparameter tuning** using grid search
- **Feature engineering** and importance analysis
- **Performance metrics** selection for imbalanced datasets

## Future Improvements

- Experiment with ensemble methods combining multiple algorithms
- Implement advanced feature selection techniques
- Explore deep learning approaches for comparison
- Add cross-validation for more robust model evaluation

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Udacity Machine Learning Nanodegree Program
- Ron Kohavi for the original dataset and research
- UCI Machine Learning Repository for hosting the dataset
