# Processed Data Directory

This directory contains processed and feature-engineered datasets:

## Files
- `df_selected_01.pkl` - First feature selection result
- `df_selected_02.pkl` - Second feature selection iteration  
- `df_selected_03.pkl` - Third feature selection iteration
- `df_selected_04.pkl` - Fourth feature selection iteration
- `df_selected_05.pkl` - Final feature selection result (pickle format)
- `df_selected_05.parquet` - Final feature selection result (parquet format)

## Usage
These files are referenced by notebooks using the relative path `../../data/processed/` from notebook directories.