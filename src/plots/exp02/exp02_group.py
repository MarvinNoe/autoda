from scipy import stats
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

from utils import read_map_values

matplotlib.rcParams['font.size'] = 12

base_dir_exp1 = './outputs/experiments/exp01/val'
base_dir_exp2 = './outputs/experiments/exp02/val'

# Exp1
data_exp1 = read_map_values(base_dir_exp1)
df_exp1 = pd.DataFrame(data_exp1)

df_exp1['Experiment'] = df_exp1['Experiment'].replace('baseline_validation', 'Baseline')
df_exp1['Experiment'] = df_exp1['Experiment'].replace('fastaa_validation', 'FastAA')
df_exp1['Experiment'] = df_exp1['Experiment'].replace('gc10_augment_validation', 'GC10-DET Augment')
df_exp1['Experiment'] = df_exp1['Experiment'].replace('smart_augment_validation', 'SmartAugment')
df_exp1['Experiment'] = df_exp1['Experiment'].replace(
    'trivial_augment_validation', 'TrivialAugment')

df_exp1['Group'] = 'Experiment I'
# exit()
# calculate mean and standard deviation
df_exp1_summary = df_exp1.groupby('Experiment').agg(
    Mean=('mAP', 'mean'),
    Std=('mAP', 'std'),
    Min=('mAP', 'min'),
    Max=('mAP', 'max')
).reset_index()

# print(df_exp1)

# Exp2
data_exp2 = read_map_values(base_dir_exp2)
df_exp2 = pd.DataFrame(data_exp2)


df_exp2['Experiment'] = df_exp2['Experiment'].replace('fastaa_validation', 'FastAA')
df_exp2['Experiment'] = df_exp2['Experiment'].replace('smart_augment_validation', 'SmartAugment')
df_exp2['Experiment'] = df_exp2['Experiment'].replace(
    'trivial_augment_validation', 'TrivialAugment')

df_exp2['Group'] = 'Experiment II'

# print(df_exp2)

df_groups = pd.concat([df_exp1, df_exp2], ignore_index=True)

# calculate mean and standard deviation
df_exp2_summary = df_exp2.groupby('Experiment').agg(
    Mean=('mAP', 'mean'),
    Std=('mAP', 'std'),
    Min=('mAP', 'min'),
    Max=('mAP', 'max')
).reset_index()


print(df_exp1_summary)
print(df_exp2_summary)
# Boxplot erstellen mit verbesserten visuellen Aspekten
plt.figure(figsize=(8, 6))
sns.set(style="whitegrid")  # Stil setzen

# Definieren einer benutzerdefinierten Farbpalette
custom_palette = ['#A6BBC8', '#179C7D', '#669DB2', '#39C1CD', '#B2D235']  # Beispiel-Farben


# Definieren der gewünschten Reihenfolge der Kategorien
order = ['Baseline', 'FastAA', 'SmartAugment', 'TrivialAugment', 'GC10-DET Augment']


sns.boxplot(x='Experiment', y='mAP', data=df_groups, hue='Group',
            palette=custom_palette, width=0.5, order=order)

plt.xlabel('')
plt.ylabel('mAP@0.5-0.95')

# Gitternetzlinien für bessere Lesbarkeit
plt.grid(True, which='both', linestyle='--', linewidth=0.7)

# Rahmenlinien ausblenden
sns.despine(offset=0)

plt.xticks(rotation=45, ha='center')
plt.subplots_adjust(bottom=0.25)
# Legendentitel entfernen
plt.legend(title='')

# Plot als PGF-Datei speichern
plt.savefig('./plots/exp02.svg')
plt.show()
