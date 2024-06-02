
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


import sys
import os


# Define the file path
file_path = "./outputs/experiments/exp01/policy_gen/fastaa_hyperopt/policies_top_10.csv"
required_count = 66


def transforms_names():
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
    from autoda_plugins.fastaa import op_creator
    t_names = [t[0].__name__ for t in op_creator.default_transform_creators()]

    for i, n in enumerate(t_names):
        parts = n.split('_')
        t_names[i] = ''.join(part.capitalize() for part in parts)

    return t_names


# Read the CSV file into a pandas DataFrame
df = pd.read_csv(file_path)

# Count occurrences of policy values
policy_counts = df.filter(regex=r'^config/policy_\d+_\d+$').stack().value_counts()
t_names = transforms_names()
print(policy_counts)
t_data = {
    'transformation': t_names,
    'count': []
}

for i in range(len(policy_counts)):
    t_data['count'].append(policy_counts[i])


df = pd.DataFrame(t_data)

sns.set(style="whitegrid")


colors = ['#179C7D' if freq >= required_count else '#A6BBC8' for freq in df['count']]


plt.figure(figsize=(8, 4))
barplot = sns.barplot(x='transformation', y='count', data=df, palette=colors)

plt.axhline(required_count, color='#BB0056', linestyle='--', label='threshold')

plt.grid(True, which='both', linestyle='--', linewidth=0.7)
sns.despine(offset=0)

# barplot.set_title('Häufigkeitsanalyse der von FastAA ausgewählten Bildtransformationen')
barplot.set_xlabel('')
barplot.set_ylabel('frequency')

plt.xticks(rotation=45, ha='center')
plt.legend(title='', loc='upper right')
plt.tight_layout()
plt.savefig('./plots/exp02_count.svg')
plt.show()
