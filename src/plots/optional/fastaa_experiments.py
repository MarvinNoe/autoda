from scipy import stats
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

from utils import read_map_values

matplotlib.rcParams['font.size'] = 12

base_dir = './outputs/experiments/exp04/val'

data = read_map_values(base_dir)
df = pd.DataFrame(data)

print(df)

df['Experiment'] = df['Experiment'].replace(
    'fastaa_gc10_transforms_validation_best_1_op', 'FastAA 1 op best')
df['Experiment'] = df['Experiment'].replace(
    'fastaa_gc10_transforms_validation_100_1_op', 'FastAA 1 op 100')
df['Experiment'] = df['Experiment'].replace(
    'fastaa_gc10_transforms_validation_1_op', 'FastAA 1 op')
df['Experiment'] = df['Experiment'].replace(
    'fastaa_validation', 'FastAA')
df['Experiment'] = df['Experiment'].replace(
    'trivial_augment_validation', 'TrivialAugment')

# exit()
# calculate mean and standard deviation
df_summary = df.groupby('Experiment').agg(
    Mean=('mAP', 'mean'),
    Std=('mAP', 'std'),
    Min=('mAP', 'min'),
    Max=('mAP', 'max')
).reset_index()

print(df_summary)


# Boxplot erstellen mit verbesserten visuellen Aspekten
plt.figure(figsize=(8, 6))
sns.set(style="whitegrid")  # Stil setzen

# Definieren einer benutzerdefinierten Farbpalette
custom_palette = ['#179C7D', '#A6BBC8', '#669DB2', '#39C1CD', '#B2D235']  # Beispiel-Farben


# Definieren der gewünschten Reihenfolge der Kategorien
order = ['TrivialAugment', 'FastAA 1 op best', 'FastAA 1 op 100', 'FastAA 1 op', 'FastAA']

sns.boxplot(x='Experiment', y='mAP', data=df, palette=custom_palette, width=0.3, order=order)

plt.xlabel('')
plt.ylabel('mAP@0.5-0.95')

# Gitternetzlinien für bessere Lesbarkeit
plt.grid(True, which='both', linestyle='--', linewidth=0.7)

# Rahmenlinien ausblenden
sns.despine(offset=0)

plt.xticks(rotation=45, ha='center')
plt.subplots_adjust(bottom=0.25)

# Plot als PGF-Datei speichern
plt.savefig('./plots/exp_fastaa.svg')
plt.show()
