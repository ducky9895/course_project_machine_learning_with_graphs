#!/usr/bin/env python
"""
Evaluate all models on all available datasets and generate comprehensive comparison.
"""

import os
import sys
import subprocess
import pandas as pd
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def run_evaluation(dataset, checkpoint_dir='chkpts', output_dir='results/accuracy'):
    """Run evaluation for a single dataset."""
    print(f"\n{'='*60}")
    print(f"Evaluating on dataset: {dataset.upper()}")
    print(f"{'='*60}")
    
    output_file = os.path.join(output_dir, f'{dataset}_comparison.csv')
    os.makedirs(output_dir, exist_ok=True)
    
    cmd = [
        sys.executable, 'scripts/eval_predictive_accuracy.py',
        '--dataset', dataset,
        '--checkpoint_dir', checkpoint_dir,
        '--output', output_file
    ]
    
    # Add dataset-specific arguments
    if dataset == 'synthetic':
        cmd.extend(['--n_test', '2000'])
    elif dataset == 'ba2motif':
        # BA2Motif uses its own test split
        pass
    elif dataset == 'mutag':
        # Mutag uses its own test split
        pass
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.path.dirname(os.path.dirname(__file__)))
        if result.returncode == 0:
            print(f"✓ Successfully evaluated {dataset}")
            return output_file
        else:
            print(f"✗ Error evaluating {dataset}:")
            print(result.stderr)
            return None
    except Exception as e:
        print(f"✗ Exception evaluating {dataset}: {e}")
        return None


def combine_results(results_dir='results/accuracy', output_file='results/accuracy/all_datasets_comparison.csv'):
    """Combine results from all datasets into a single table."""
    print(f"\n{'='*60}")
    print("Combining results from all datasets...")
    print(f"{'='*60}")
    
    all_results = []
    datasets = ['synthetic', 'ba2motif', 'mutag']
    
    for dataset in datasets:
        csv_file = os.path.join(results_dir, f'{dataset}_comparison.csv')
        if os.path.exists(csv_file):
            try:
                df = pd.read_csv(csv_file)
                # Check if Dataset column already exists
                if 'Dataset' not in df.columns:
                    df['Dataset'] = dataset
                # Ensure Model column exists (might be index)
                if 'Model' not in df.columns and df.index.name == 'Model':
                    df.reset_index(inplace=True)
                elif 'Model' not in df.columns:
                    # Try to infer from index
                    if df.index.name:
                        df.reset_index(inplace=True)
                        df.rename(columns={df.index.name: 'Model'}, inplace=True)
                    else:
                        df['Model'] = df.index
                        df.reset_index(drop=True, inplace=True)
                all_results.append(df)
                print(f"  ✓ Loaded {dataset}: {len(df)} models")
            except Exception as e:
                print(f"  ✗ Failed to load {dataset}: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"  ⚠ {dataset} results not found: {csv_file}")
    
    if not all_results:
        print("No results to combine!")
        return
    
    # Combine all results
    combined = pd.concat(all_results, ignore_index=True)
    
    # Reorder columns
    cols = ['Dataset', 'Model', 'Accuracy', 'F1-Macro', 'F1-Micro', 'Precision', 'Recall']
    available_cols = [c for c in cols if c in combined.columns]
    combined = combined[available_cols]
    
    # Save combined results
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    combined.to_csv(output_file, index=False)
    print(f"\n✓ Combined results saved to: {output_file}")
    
    # Print summary table
    print(f"\n{'='*80}")
    print("Summary: Accuracy by Dataset and Model")
    print(f"{'='*80}")
    
    if 'Dataset' in combined.columns and 'Model' in combined.columns and 'Accuracy' in combined.columns:
        pivot = combined.pivot_table(
            index='Model', 
            columns='Dataset', 
            values='Accuracy', 
            aggfunc='first'
        )
        print(pivot.to_string())
        print(f"\n{'='*80}")
    
    return output_file


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate all models on all datasets')
    parser.add_argument('--checkpoint_dir', type=str, default='chkpts',
                       help='Directory with trained models')
    parser.add_argument('--output_dir', type=str, default='results/accuracy',
                       help='Output directory for results')
    parser.add_argument('--datasets', type=str, nargs='+',
                       default=['synthetic', 'ba2motif'],
                       choices=['synthetic', 'ba2motif', 'mutag', 'all'],
                       help='Datasets to evaluate (or "all")')
    parser.add_argument('--skip_eval', action='store_true',
                       help='Skip evaluation, only combine existing results')
    
    args = parser.parse_args()
    
    if 'all' in args.datasets:
        datasets = ['synthetic', 'ba2motif', 'mutag']
    else:
        datasets = args.datasets
    
    if not args.skip_eval:
        # Run evaluation for each dataset
        results_files = []
        for dataset in datasets:
            result_file = run_evaluation(dataset, args.checkpoint_dir, args.output_dir)
            if result_file:
                results_files.append(result_file)
    else:
        print("Skipping evaluation (using existing results)...")
    
    # Combine results
    combine_results(args.output_dir, os.path.join(args.output_dir, 'all_datasets_comparison.csv'))


if __name__ == '__main__':
    main()
