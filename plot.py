import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("comprehensive_perturb_metrics.csv")
sns.set(style="whitegrid", palette="Paired")

try:
    os.mkdir('figures')
except:
    pass

methods = ['ED', 'ER', 'RR', 'RS']
map = {
    'ED': 'Edge Deletion',
    'ER': 'Edge Rewiring',
    'RR': 'Relation Replacement',
    'RS': 'Relation Swaping'
}

metrics = ['ATS', 'SC2D', 'SD2']
for method in methods:
    method_df = df[df['Method'] == method]
    if method_df.empty:
        print(f"No data available for method: {method}")
        continue
    
    plt.figure(figsize=(4.5, 4))

    for metric in metrics:
        plt.plot(
            method_df['Perturbation_Level'],
            method_df[metric],
            marker='o',
            label=metric,
            linewidth=3
        )
    
    plt.title(f'{map[method]}', fontsize=20)
    plt.xlabel('Perturbation Level', fontsize=20)
    plt.ylabel('Metrics', fontsize=20)
    
    plt.xlim(-0.025, 1.025)
    plt.ylim(-0.025, 1.025)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    plt.legend(fontsize=14, loc='lower left', fancybox=True, framealpha=0.05)
    plt.grid(True, which='both', linestyle='--', linewidth=1.5)

    plt.tight_layout()
    plt.savefig(f'figures/{method}_perturb_metrics.png', dpi=300, bbox_inches='tight')
    plt.show()