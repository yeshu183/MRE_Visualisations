"""
Analyze sampling comparison results from grid search.
"""
import pandas as pd
from pathlib import Path

# Load results
output_dir = Path(__file__).parent / 'outputs' / 'sampling_comparison'
df = pd.read_csv(output_dir / 'sampling_comparison_results.csv')

print("="*80)
print("SAMPLING STRATEGY COMPARISON - KEY FINDINGS")
print("="*80)

# Filter to heterogeneous mu only (the interesting case)
df_hetero = df[df['mu_type'] == 'heterogeneous'].copy()

# Group by sampling config and neurons, get mean metrics
summary = df_hetero.groupby(['sampling_config', 'neurons']).agg({
    'r2': 'mean',
    'blob_r2': 'mean',
    'background_r2': 'mean',
    'mse': 'mean',
    'blob_mse': 'mean'
}).reset_index()

print("\n" + "="*80)
print("HETEROGENEOUS MU RESULTS (mu = 3-10 kPa)")
print("="*80)

# Sort by blob R² (most important metric)
summary_sorted = summary.sort_values('blob_r2', ascending=False)

print("\nRanked by Blob R² (Higher is Better):")
print("-"*80)
for idx, row in summary_sorted.iterrows():
    if pd.notna(row['blob_r2']):
        print(f"{row['sampling_config']:30s} neurons={row['neurons']:4d}: "
              f"Blob R²={row['blob_r2']:6.4f}, Overall R²={row['r2']:6.4f}")

# Compare with/without replacement
print("\n" + "="*80)
print("REPLACEMENT vs NO-REPLACEMENT COMPARISON")
print("="*80)

for base_config in ['adaptive_5_25_70', 'adaptive_10_20_70', 'adaptive_20_10_70']:
    repl = summary[summary['sampling_config'] == f'{base_config}_repl']
    noRepl = summary[summary['sampling_config'] == f'{base_config}_noRepl']

    if len(repl) > 0 and len(noRepl) > 0:
        print(f"\n{base_config}:")
        for n in [100, 500, 1000]:
            repl_row = repl[repl['neurons'] == n]
            noRepl_row = noRepl[noRepl['neurons'] == n]

            if len(repl_row) > 0 and len(noRepl_row) > 0:
                repl_blob_r2 = repl_row.iloc[0]['blob_r2']
                noRepl_blob_r2 = noRepl_row.iloc[0]['blob_r2']

                if pd.notna(repl_blob_r2) and pd.notna(noRepl_blob_r2):
                    diff = noRepl_blob_r2 - repl_blob_r2
                    winner = "NO-REPL" if diff > 0 else "REPL" if diff < 0 else "TIE"
                    print(f"  neurons={n}: REPL={repl_blob_r2:.4f}, NO-REPL={noRepl_blob_r2:.4f}, "
                          f"Diff={diff:+.4f} => {winner}")

# Compare adaptive vs uniform
print("\n" + "="*80)
print("ADAPTIVE vs UNIFORM COMPARISON")
print("="*80)

uniform = summary[summary['sampling_config'] == 'uniform']
adaptive_configs = summary[summary['sampling_config'] != 'uniform']

print("\nUniform baseline:")
for idx, row in uniform.iterrows():
    if pd.notna(row['blob_r2']):
        print(f"  neurons={row['neurons']:4d}: Blob R²={row['blob_r2']:6.4f}, Overall R²={row['r2']:6.4f}")

print("\nBest adaptive configs (Blob R² > uniform):")
for n in [100, 500, 1000]:
    uniform_row = uniform[uniform['neurons'] == n]
    if len(uniform_row) == 0:
        continue

    uniform_blob_r2 = uniform_row.iloc[0]['blob_r2']

    if pd.notna(uniform_blob_r2):
        # Find adaptive configs that beat uniform
        adaptive_n = adaptive_configs[adaptive_configs['neurons'] == n]
        better = adaptive_n[adaptive_n['blob_r2'] > uniform_blob_r2]

        if len(better) > 0:
            print(f"\n  neurons={n} (uniform blob R²={uniform_blob_r2:.4f}):")
            for idx, row in better.sort_values('blob_r2', ascending=False).iterrows():
                improvement = row['blob_r2'] - uniform_blob_r2
                print(f"    {row['sampling_config']:30s}: Blob R²={row['blob_r2']:.4f} (+{improvement:.4f})")
        else:
            print(f"\n  neurons={n}: No adaptive config beats uniform (blob R²={uniform_blob_r2:.4f})")

# Overall winner
print("\n" + "="*80)
print("OVERALL WINNER")
print("="*80)

best_overall = summary_sorted.iloc[0]
print(f"\nBest configuration:")
print(f"  Config: {best_overall['sampling_config']}")
print(f"  Neurons: {best_overall['neurons']}")
print(f"  Blob R²: {best_overall['blob_r2']:.4f}")
print(f"  Overall R²: {best_overall['r2']:.4f}")
print(f"  Background R²: {best_overall['background_r2']:.4f}")

# Compare with uniform at same neuron count
uniform_same_n = uniform[uniform['neurons'] == best_overall['neurons']]
if len(uniform_same_n) > 0 and pd.notna(uniform_same_n.iloc[0]['blob_r2']):
    uniform_blob_r2 = uniform_same_n.iloc[0]['blob_r2']
    improvement = best_overall['blob_r2'] - uniform_blob_r2
    print(f"\nComparison with uniform (neurons={best_overall['neurons']}):")
    print(f"  Uniform Blob R²: {uniform_blob_r2:.4f}")
    print(f"  Improvement: {improvement:+.4f} ({100*improvement/uniform_blob_r2:+.2f}%)")

print("\n" + "="*80)
print("CONCLUSION")
print("="*80)

# Check if any adaptive config beats uniform
uniform_best = uniform['blob_r2'].max()
adaptive_best = adaptive_configs['blob_r2'].max()

if pd.notna(adaptive_best) and pd.notna(uniform_best):
    if adaptive_best > uniform_best:
        print("\n[+] ADAPTIVE SAMPLING WINS!")
        print(f"  Best adaptive: {adaptive_best:.4f}")
        print(f"  Best uniform: {uniform_best:.4f}")
        print(f"  Improvement: {adaptive_best - uniform_best:+.4f}")
    elif adaptive_best < uniform_best:
        print("\n[-] UNIFORM SAMPLING WINS")
        print(f"  Best uniform: {uniform_best:.4f}")
        print(f"  Best adaptive: {adaptive_best:.4f}")
        print(f"  Uniform is better by: {uniform_best - adaptive_best:.4f}")
    else:
        print("\n[=] TIE between adaptive and uniform")

print("\n" + "="*80)
