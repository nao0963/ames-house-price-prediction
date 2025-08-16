# Models Directory

This directory contains trained machine learning models:

## Files
- `best_ridge_model.pkl` - Best performing ridge regression model
- `efficient_grid_search_rf_model.pkl` - Random forest model from efficient grid search
- `optimized_random_forest_model.pkl` - Optimized random forest model
- `quick_tuned_rf_model.pkl` - Quick tuned random forest model
- `quick_tuned_rf_model_fixed.pkl` - Fixed version of quick tuned model

## Usage
These models are saved using pickle format and can be loaded for predictions or further analysis. Notebooks reference these using the relative path `../../models/` from notebook directories.