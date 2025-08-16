# Notebooks Directory Structure

This directory contains all Jupyter notebooks organized by project phase:

## 01_data_preparation
- Data loading and initial exploration
- Data cleaning and preprocessing

## 02_feature_engineering  
- Feature selection experiments
- Feature creation and transformation
- `feature_selection_01.ipynb` - Initial feature selection analysis
- `feature_selection_02.ipynb` - Advanced feature selection techniques

## 03_basic_models
- Basic machine learning model implementations
- `linear_regression_01.ipynb` - Linear regression baseline
- `ridge_regression_01.ipynb` - Ridge regression with regularization
- `ridge_regression_02.ipynb` - Advanced ridge regression analysis
- `random_forest_basic.ipynb` - Basic random forest implementation

## 04_hyperparameter_tuning
- Advanced hyperparameter optimization techniques
- `random_forest_grid_search.ipynb` - Basic grid search approach
- `random_forest_random_search.ipynb` - Random search optimization
- `random_forest_random+grid.ipynb` - Combined random and grid search
- `complete_resumable_grid_search.ipynb` - Resumable grid search implementation
- `ultimate_grid_search_strategy.ipynb` - Comprehensive grid search with 9,600 combinations

## 05_distributed_computing
- Distributed computing implementations
- `databricks_distributed_grid_search.ipynb` - Databricks Spark implementation for large-scale hyperparameter tuning

## File Path References
All notebooks now reference data and model files using relative paths:
- Data files: `../../data/processed/`
- Model files: `../../models/`
- Results: `../../results/`