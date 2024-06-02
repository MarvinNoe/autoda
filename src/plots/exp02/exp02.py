from scipy import stats
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

from utils import read_map_values

matplotlib.rcParams['font.size'] = 12

base_dir_exp1 = './outputs/experiments/exp01/val'
base_dir_exp2 = './outputs/experiments/exp02/val'

data_exp2 = read_map_values(base_dir_exp2)
df_exp2 = pd.DataFrame(data_exp2)

df_exp2['Experiment'] = df_exp2['Experiment'].replace('baseline_validation', 'Baseline')
df_exp2['Experiment'] = df_exp2['Experiment'].replace('fastaa_validation', 'FastAA')
df_exp2['Experiment'] = df_exp2['Experiment'].replace('gc10_augment_validation', 'GC10-DET Augment')
df_exp2['Experiment'] = df_exp2['Experiment'].replace('smart_augment_validation', 'SmartAugment')
df_exp2['Experiment'] = df_exp2['Experiment'].replace(
    'trivial_augment_validation', 'TrivialAugment')

# exit()
# calculate mean and standard deviation
df_exp2_summary = df_exp2.groupby('Experiment').agg(
    Mean=('mAP', 'mean'),
    Std=('mAP', 'std'),
    Min=('mAP', 'min'),
    Max=('mAP', 'max')
).reset_index()

# Boxplot erstellen mit verbesserten visuellen Aspekten
plt.figure(figsize=(8, 6))
sns.set(style="whitegrid")  # Stil setzen

# Definieren einer benutzerdefinierten Farbpalette
custom_palette = ['#179C7D', '#A6BBC8', '#669DB2', '#39C1CD', '#B2D235']  # Beispiel-Farben


# Definieren der gewünschten Reihenfolge der Kategorien
order = ['Baseline', 'FastAA', 'SmartAugment', 'TrivialAugment', 'GC10-DET Augment']

sns.boxplot(x='Experiment', y='mAP', data=df_exp2, palette=custom_palette, width=0.3, order=order)

plt.xlabel('')
plt.ylabel('mAP@0.5-0.95')

# Gitternetzlinien für bessere Lesbarkeit
plt.grid(True, which='both', linestyle='--', linewidth=0.7)

# Rahmenlinien ausblenden
sns.despine(offset=0)

plt.xticks(rotation=45, ha='center')
plt.subplots_adjust(bottom=0.25)

# Plot als PGF-Datei speichern
plt.savefig('./plots/exp02.svg')
plt.show()


data_exp1 = read_map_values(base_dir_exp1)
df_exp1 = pd.DataFrame(data_exp1)

df_exp1['Experiment'] = df_exp1['Experiment'].replace('baseline_validation', 'Baseline')
df_exp1['Experiment'] = df_exp1['Experiment'].replace('fastaa_validation', 'FastAA')
df_exp1['Experiment'] = df_exp1['Experiment'].replace('gc10_augment_validation', 'GC10-DET Augment')
df_exp1['Experiment'] = df_exp1['Experiment'].replace('smart_augment_validation', 'SmartAugment')
df_exp1['Experiment'] = df_exp1['Experiment'].replace(
    'trivial_augment_validation', 'TrivialAugment')

# exit()
# calculate mean and standard deviation
df_exp1_summary = df_exp1.groupby('Experiment').agg(
    Mean=('mAP', 'mean'),
    Std=('mAP', 'std'),
    Min=('mAP', 'min'),
    Max=('mAP', 'max')
).reset_index()


print(df_exp1_summary)
print(df_exp2_summary)

df_comparison = df_exp1_summary[['Experiment']].copy()

df_comparison['Mean Change (Total)'] = df_exp2_summary['Mean'] - df_exp1_summary['Mean']
df_comparison['Mean Change (%)'] = (
    (df_exp2_summary['Mean'] - df_exp1_summary['Mean']) / df_exp1_summary['Mean']) * 100

df_comparison['Std Change (Total)'] = df_exp2_summary['Std'] - df_exp1_summary['Std']
df_comparison['Std Change (%)'] = (
    (df_exp2_summary['Std'] - df_exp1_summary['Std']) / df_exp1_summary['Std']) * 100

# Berechnung der totalen Veränderung


print(df_comparison)
