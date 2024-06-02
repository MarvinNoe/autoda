import matplotlib.pyplot as plt
import seaborn as sns

# Set Seaborn style
sns.set(style="whitegrid")

plt.figure(figsize=(5, 3))
# Transformierte Datenpunkte (verschoben entlang einer Achse)
transformed_points = [(1.5, 2.5), (2, 3.5), (1.5, 4.5), (1.7, 3.5), (2.5, 4.5), (2.3, 3)]
# Beispielhafte Datenpunkte (Originaldaten)
original_points = [(1, 2), (1.5, 3), (1, 4), (1.2, 3), (2, 4), (1.8, 2.5)]

start = original_points[1]
end = transformed_points[1]
plt.plot([start[0], end[0]], [start[1], end[1]], '--',
         linewidth=1.5, color='#BB0056', label='transformation')

# Linien zeichnen, um die Transformation zu zeigen
# for orig, trans in zip(original_points, transformed_points):
#    plt.plot([orig[0], trans[0]], [orig[1], trans[1]], '--', linewidth=1.5, color='#BB0056')

# Plotten der Originaldaten
for point in original_points:
    plt.scatter(point[0], point[1], s=100, color='#179C7D', zorder=5, label='original')

# Plotten der transformierten Daten
for point in transformed_points:
    plt.scatter(point[0], point[1], s=100, color='#A6BBC8', zorder=5, label='augmented')

# Achsenbeschriftungen entfernen
plt.xticks([])
plt.yticks([])

# Grid und Despine anpassen
plt.grid(True)
sns.despine(offset=0)

# Sicherstellen, dass das Grid angezeigt wird
ax = plt.gca()
ax.set_axisbelow(True)
ax.axis('off')

handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys(), loc='lower right')

plt.tight_layout()

# Diagramm anzeigen
plt.savefig('./plots/exp03_data_ta.svg')
plt.show()
