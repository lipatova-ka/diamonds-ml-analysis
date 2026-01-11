import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

df = pd.read_csv("diamonds_clean_final.csv")

df['log_price'] = np.log(df['price'])

#Fluor
plt.figure(figsize=(8, 5))
fluor_order = ['SB', 'S', 'MB', 'M', 'F', 'SLT', 'NEG', 'NN']
sns.boxplot(
    data=df,
    x='stone.flour.name',
    y='log_price',
    order=fluor_order,
    showfliers=True
)

plt.xlabel('Fluorescence')
plt.ylabel('log10(Price)')
plt.title('Распределение логарифма цены по уровню флуоресценции')
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()



#Symmetry
plt.figure(figsize=(8, 5))
symmetry_order = ['ID', 'EX', 'VG', 'GD']
sns.boxplot(
    data=df,
    x='stone.symmetry.name',
    y='log_price',
    order=symmetry_order,
    showfliers=True
)

plt.xlabel('Symmetry')
plt.ylabel('log10(Price)')
plt.title('Распределение логарифма цены по категориям симметрии')
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()



#Polish
plt.figure(figsize=(8, 5))
polish_order = ['ID', 'EX', 'VG', 'GD']
sns.boxplot(
    data=df,
    x='stone.polish.name',
    y='log_price',
    order=polish_order,
    showfliers=True
)

plt.xlabel('Polish')
plt.ylabel('log10(Price)')
plt.title('Распределение логарифма цены по качеству полировки')
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()



#Связь Fluor и Color
ct = pd.crosstab(
    df['stone.color.name'],
    df['stone.flour.name'],
    normalize='index'
)

plt.figure(figsize=(10, 6))

sns.heatmap(
    ct,
    cmap='Blues',
    annot=True,
    fmt='.2f'
)

plt.xlabel('Fluorescence')
plt.ylabel('Color')
plt.title('Распределение уровней флуоресценции по цветовым категориям')
plt.tight_layout()
plt.show()

import seaborn as sns
import matplotlib.pyplot as plt

g = sns.catplot(
    data=df,
    x='stone.flour.name',
    y='log_price',
    col='stone.color.name',
    col_wrap=4,
    kind='box',
    height=3,
    aspect=1,
    showfliers=False
)

g.set_axis_labels('Fluorescence', 'log10(Price)')
g.set_titles('Color = {col_name}')
g.fig.suptitle(
    'Влияние флуоресценции на цену в разрезе цветовых категорий',
    y=1.03
)

for ax in g.axes.flatten():
    ax.tick_params(axis='x', rotation=45)

plt.show()
