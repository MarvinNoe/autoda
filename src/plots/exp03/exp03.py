from scipy import stats
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

from utils import read_map_values

matplotlib.rcParams['font.size'] = 12

base_dir = './outputs/experiments/exp03/val'

data = read_map_values(base_dir)
df = pd.DataFrame(data)

df['Experiment'] = df['Experiment'].replace('baseline_validation_double', 'Baseline')
df['Experiment'] = df['Experiment'].replace('fastaa_validation', 'FastAA')
df['Experiment'] = df['Experiment'].replace('gc10_augment_validation_double', 'GC10-DET Augment')
df['Experiment'] = df['Experiment'].replace('smart_augment_validation', 'SmartAugment')
df['Experiment'] = df['Experiment'].replace('trivial_augment_validation', 'TrivialAugment')

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
order = ['Baseline', 'FastAA', 'SmartAugment', 'TrivialAugment', 'GC10-DET Augment']

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
plt.savefig('./plots/exp03.svg')
plt.show()
