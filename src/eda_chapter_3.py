import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("diamonds_clean_final.csv")

print(df.shape)
print(df.columns)
print(df.head())

#Строим матрицу корреляции Пирсона для основных
# количественных характеристик бриллиантов: Carat, Price per Carat, Depth (%), Table (%) и Ratio.
cols = [
    'stone.carat',
    'enriched.pricePerCarat',
    'stone.depth',
    'stone.tableSize',
    'enriched.ratio'
]
df_corr = df[cols].corr()
# Новые названия чтобы было понятно и красиво читалось
rename_dict = {
    'stone.carat': 'Carat',
    'enriched.pricePerCarat': 'Price per Carat',
    'stone.depth': 'Depth (%)',
    'stone.tableSize': 'Table (%)',
    'enriched.ratio': 'ratio'

}
# Переименование строк и столбцов
df_corr = df_corr.rename(index=rename_dict, columns=rename_dict)

plt.figure(figsize=(8, 6))
sns.heatmap(df_corr, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Корреляция основных числовых характеристик')
plt.show()


#Диаграмма рассеяния, отражающая зависимость цены бриллианта от его массы (карат)
sns.scatterplot(data=df, x='stone.carat', y='price', alpha=0.3)
plt.xlabel('Carat')
plt.ylabel('Price, $')
plt.title("Зависимость цены от массы бриллианта (приближ.)")
plt.show()



#Графики Boxplot: логарифм цены по категориям Color / Clarity / Cut
import numpy as np

df['log_price'] = np.log(df['price'])

plt.figure(figsize=(8, 6))
color_order = ['D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M']
sns.boxplot(
    data=df,
    x='stone.color.name',
    y='log_price',
    order=color_order
)
plt.xlabel('Color')
plt.ylabel('log(Price)')
plt.title('Распределение логарифма цены бриллиантов по категориям цвета')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


plt.figure(figsize=(8, 6))
clarity_order = ['FL', 'IF', 'VVS1', 'VVS2', 'VS1', 'VS2', 'SI1', 'SI2', 'I1']
sns.boxplot(
    data=df,
    x='stone.clarity.name',
    y='log_price',
    order=clarity_order
)
plt.xlabel('Clarity (от лучшей к худшей)')
plt.ylabel('log(Price)')
plt.title('Распределение логарифма цены бриллиантов по категориям чистоты')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


plt.figure(figsize=(8, 6))
cut_order = ['Ideal', 'Excellent', 'Very Good', 'Good']
sns.boxplot(
    data=df,
    x='stone.cut.name',
    y='log_price',
    order=cut_order
)
plt.xlabel('Cut (от лучшей к худшей)')
plt.ylabel('log(Price)')
plt.title('Распределение логарифма цены бриллиантов по качеству огранки')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

#посчитаем межквартильные размахи
df_iqr = df[['stone.color.name', 'log_price']].dropna()
iqr_color = (
    df_iqr
    .groupby('stone.color.name')['log_price']
    .agg(
        Q1=lambda x: x.quantile(0.25),
        Q3=lambda x: x.quantile(0.75)
    )
)
iqr_color['IQR'] = iqr_color['Q3'] - iqr_color['Q1']
print(iqr_color)

df_iqr = df[['stone.clarity.name', 'log_price']].dropna()
iqr_color = (
    df_iqr
    .groupby('stone.clarity.name')['log_price']
    .agg(
        Q1=lambda x: x.quantile(0.25),
        Q3=lambda x: x.quantile(0.75)
    )
)
iqr_color['IQR'] = iqr_color['Q3'] - iqr_color['Q1']
print(iqr_color)

df_iqr = df[['stone.cut.name', 'log_price']].dropna()
iqr_color = (
    df_iqr
    .groupby('stone.cut.name')['log_price']
    .agg(
        Q1=lambda x: x.quantile(0.25),
        Q3=lambda x: x.quantile(0.75)
    )
)
iqr_color['IQR'] = iqr_color['Q3'] - iqr_color['Q1']
print(iqr_color)



#Гистограмма + KDE для Price и log(Price)
#копия DataFrame для предотвращения побочных эффектов при добавлении производных признаков
df = df.copy()
df['log_price'] = np.log(df['price'].clip(lower=1))

plt.figure(figsize=(12, 5))

# Price
plt.subplot(1, 2, 1)
sns.histplot(df['price'], bins=80, kde=True)
plt.xlabel('Price')
plt.ylabel('Frequency (частота)')
plt.title('Распределение цены бриллиантов')

# log(Price)
plt.subplot(1, 2, 2)
sns.histplot(df['log_price'], bins=80, kde=True)
plt.xlabel('log(Price)')
plt.ylabel('Frequency (частота)')
plt.title('Распределение логарифма цены бриллиантов')

plt.tight_layout()
plt.show()


#Гистограмма + KDE распределения Carat
plt.figure(figsize=(8, 5))
sns.histplot(
    df['stone.carat'],
    bins=80,
    kde=True
)

plt.xlabel('Carat')
plt.ylabel('Frequency (частота)')
plt.title('Распределение массы бриллиантов (Carat)')
plt.tight_layout()
plt.show()



#PricePerCarat vs Carat
plt.figure(figsize=(8, 6))

# Scatter
sns.scatterplot(
    data=df,
    x='stone.carat',
    y='enriched.pricePerCarat',
    alpha=0.3,
    edgecolor=None
)

# LOWESS линия поверх scatter
sns.regplot(
    data=df,
    x='stone.carat',
    y='enriched.pricePerCarat',
    scatter=False,     # чтобы не дублировать точки
    lowess=True,
    color='red'
)

plt.xscale('log')
plt.xlabel('Carat (log scale)')
plt.ylabel('Price per Carat')
plt.title('Зависимость цены за карат от массы бриллианта')
plt.tight_layout()
plt.show()



#Price vs Carat + Color / Clarity / Cut (hue)
df = df.copy()
df['log_price'] = np.log10(df['price'].clip(lower=1))

fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)  # шире, чтобы влезла легенда

# 1) Color
color_order = ['D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M']
palette_color = sns.color_palette("tab10", n_colors=len(color_order))

sns.scatterplot(
    data=df, x='stone.carat', y='log_price',
    hue='stone.color.name', hue_order=color_order, palette=palette_color,
    alpha=0.5, s=15, linewidth=0, ax=axes[0]
)
axes[0].set_xscale('log')
axes[0].set_title('Цвет (Color)')
axes[0].set_ylabel('log10(Price)')
axes[0].legend(title='Color', bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)

# 2) Clarity
clarity_order = ['FL', 'IF', 'VVS1', 'VVS2', 'VS1', 'VS2', 'SI1', 'SI2', 'I1']
palette_clarity = sns.color_palette("tab10", n_colors=len(clarity_order))

sns.scatterplot(
    data=df, x='stone.carat', y='log_price',
    hue='stone.clarity.name', hue_order=clarity_order, palette=palette_clarity,
    alpha=0.5, s=15, linewidth=0, ax=axes[1]
)
axes[1].set_xscale('log')
axes[1].set_title('Чистота (Clarity)')
axes[1].set_ylabel('log10(Price)')
axes[1].legend(title='Clarity', bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)

# 3) Cut
cut_order = ['Ideal', 'Excellent', 'Very Good', 'Good']
palette_cut = sns.color_palette("Set2", n_colors=len(cut_order))

sns.scatterplot(
    data=df, x='stone.carat', y='log_price',
    hue='stone.cut.name', hue_order=cut_order, palette=palette_cut,
    alpha=0.5, s=15, linewidth=0, ax=axes[2]
)
axes[2].set_xscale('log')
axes[2].set_title('Качество огранки (Cut)')
axes[2].set_xlabel('Carat (log scale)')
axes[2].set_ylabel('log10(Price)')
axes[2].legend(title='Cut', bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)

plt.suptitle('Зависимость логарифма цены от массы с дифференциацией по качественным характеристикам', fontsize=12)

# оставляем место справа под легенды
plt.tight_layout(rect=[0, 0, 0.80, 0.97])
plt.show()



df = df.copy()
df['log_price'] = np.log(df['price'].clip(lower=1))

order = (
    df.groupby('stone.shape.name')['log_price']
    .median()
    .sort_values(ascending=False)
    .index
)

plt.figure(figsize=(12, 6))

sns.boxplot(
    data=df,
    x='stone.shape.name',
    y='log_price',
    order=order,
    showfliers=True
)

plt.xlabel('Форма огранки (Shape)')
plt.ylabel('log(Price)')
plt.title('Распределение логарифма цены по формам огранки')
plt.xticks(rotation=35, ha='right')
plt.tight_layout()
plt.show()

