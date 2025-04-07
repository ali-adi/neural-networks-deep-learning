#!/usr/bin/env python3

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob
import json
from collections import defaultdict

def load_experiment_results():
    """Load all experiment results from the results directory."""
    results = []
    
    # Walk through all result directories
    for feature_type in ['MFCC', 'LOGMEL', 'HUBERT', 'FUSION']:
        for dataset in ['EMODB', 'RAVDESS']:
            result_dir = f'results/{feature_type}/{dataset}'
            if not os.path.exists(result_dir):
                continue
                
            # Find all Excel files
            excel_files = glob(f'{result_dir}/*.xlsx')
            for excel_file in excel_files:
                try:
                    # Extract parameters from filename
                    params = extract_params_from_filename(excel_file)
                    
                    # Read the Excel file
                    df = pd.read_excel(excel_file)
                    
                    # Get the best validation accuracy
                    best_val_acc = df['val_accuracy'].max()
                    
                    # Store results
                    results.append({
                        'feature_type': feature_type,
                        'dataset': dataset,
                        'best_val_acc': best_val_acc,
                        'params': params,
                        'file': excel_file
                    })
                except Exception as e:
                    print(f"Error processing {excel_file}: {e}")
    
    return pd.DataFrame(results)

def extract_params_from_filename(filename):
    """Extract parameters from the Excel filename."""
    # This function should be customized based on how your filenames are structured
    # Example: if filename contains parameters like "lr0.0001_batch32_..."
    params = {}
    basename = os.path.basename(filename)
    
    # Add your parameter extraction logic here
    # This is just an example
    if 'lr' in basename:
        params['learning_rate'] = float(basename.split('lr')[1].split('_')[0])
    if 'batch' in basename:
        params['batch_size'] = int(basename.split('batch')[1].split('_')[0])
    # Add more parameter extraction as needed
    
    return params

def analyze_parameter_importance(results_df):
    """Analyze the importance of each parameter on model performance."""
    param_importance = defaultdict(list)
    
    # Group by each parameter and calculate mean performance
    for param in results_df['params'].iloc[0].keys():
        grouped = results_df.groupby(f'params.{param}')['best_val_acc'].mean()
        param_importance[param] = grouped.to_dict()
    
    return param_importance

def find_optimal_parameters(results_df):
    """Find the best parameter combination."""
    # Sort by validation accuracy
    best_results = results_df.sort_values('best_val_acc', ascending=False)
    
    # Get the top 5 configurations
    top_5 = best_results.head(5)
    
    return top_5

def plot_parameter_importance(param_importance):
    """Plot the importance of each parameter."""
    plt.figure(figsize=(15, 10))
    
    for i, (param, values) in enumerate(param_importance.items(), 1):
        plt.subplot(3, 3, i)
        plt.plot(list(values.keys()), list(values.values()), 'o-')
        plt.title(f'Impact of {param}')
        plt.xlabel(param)
        plt.ylabel('Mean Validation Accuracy')
        plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('parameter_importance.png')
    plt.close()

def main():
    print("Loading experiment results...")
    results_df = load_experiment_results()
    
    print("\nAnalyzing parameter importance...")
    param_importance = analyze_parameter_importance(results_df)
    
    print("\nFinding optimal parameters...")
    optimal_params = find_optimal_parameters(results_df)
    
    print("\nTop 5 configurations:")
    for _, row in optimal_params.iterrows():
        print(f"\nFeature Type: {row['feature_type']}")
        print(f"Dataset: {row['dataset']}")
        print(f"Validation Accuracy: {row['best_val_acc']:.4f}")
        print("Parameters:")
        for param, value in row['params'].items():
            print(f"  {param}: {value}")
    
    print("\nPlotting parameter importance...")
    plot_parameter_importance(param_importance)
    
    # Save results to JSON
    results = {
        'optimal_parameters': optimal_params.to_dict('records'),
        'parameter_importance': param_importance
    }
    
    with open('analysis_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\nAnalysis complete! Results saved to analysis_results.json and parameter_importance.png")

if __name__ == '__main__':
    main() 