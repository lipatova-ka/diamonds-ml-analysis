#Работа с новым полученным файлом diamonds_balanced_50k.csv со всеми фоомами огранки
import pandas as pd
# Настройки отображения
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.width', 200)

# Чтение файла CSV
df = pd.read_csv('diamonds_balanced_50k.csv', sep=None, engine='python', encoding='utf-8')

# Просмотр первых строк
print("\nПервые строки файла:")
print(df.head(10))

# Информация о данных
print("\nИнформация о структуре данных:")
print(df.info())

# Названия столбцов
print("\nНазвания столбцов:")
print(df.columns.tolist())


#UPP - платформенная (внутренняя) система ценового ориентирования
#Проводим небольшую корректировку столбцов из-за особенностей выгрузки из корпоративной БД
# Переименовываем enriched.uppInner -> enriched.UPP
df = df.rename(columns={
    'enriched.uppInner': 'enriched.uppPrice'
})
print("\nСтолбец enriched.uppInner переименован в enriched.UPP")

#удаление лишних столбцов
cols_to_drop = [
    'enriched.uppMin',
    'enriched.uppOuter'
]

# Удаляем только те столбцы, которые реально есть в датасете
df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

print("\nУдалены столбцы:", cols_to_drop)

#сохранение
df.to_csv('diamonds_balanced_50k_1.csv', index=False, encoding='utf-8')

print("\nФайл 'diamonds_balanced_50k_1.csv' успешно сохранён.")
print("Итоговое количество столбцов:", df.shape[1])



#True Hearts - собственная характерисика качества огранки, используемая на конткретной платформе Jamesallen
#таких значений всего несколько, соответственно для анализа эти данные неинформативны
df = pd.read_csv("diamonds_balanced_50k_1.csv", encoding="utf-8", low_memory=False)

# Подсчёт строк с 'True Hearts' в stone.shape.name
true_hearts_mask = df['stone.cut.name'] == 'True Hearts'
count_true_hearts = true_hearts_mask.sum()

print(f"\nКоличество строк с 'True Hearts' в stone.cut.name: {count_true_hearts}")

# Удаление строк с 'True Hearts'
df = df[~true_hearts_mask].copy()

print(f"После удаления строк осталось: {len(df)}")

# Анализ столбца enriched.isTrueHearts
value_counts = df['enriched.isTrueHearts'].value_counts(dropna=False)

print("\nРаспределение значений в enriched.isTrueHearts:")
print(value_counts)

# Удаление столбца enriched.isTrueHearts
df = df.drop(columns=['enriched.isTrueHearts'])

print("Столбец 'enriched.isTrueHearts' удалён.")

# Сохранение датасета
df.to_csv(
    "diamonds_balanced_50k_final.csv",
    index=False,
    encoding="utf-8"
)
print("\nФайл 'diamonds_balanced_50k_final.csv' сохранён")
print(f"Итоговое количество строк: {len(df)}")
print(f"Итоговое количество столбцов: {df.shape[1]}")




df = pd.read_csv('diamonds_balanced_50k_final.csv', sep=None, engine='python', encoding='utf-8')
# Подсчёт пропусков по каждому столбцу
missing = df.isna().sum().sort_values(ascending=False)

# Переводим в проценты
missing_percent = (missing / len(df)) * 100

# Объединяем в таблицу
missing_table = pd.DataFrame({
    'Пропусков': missing,
    'Процент_пропусков': missing_percent.round(2)
})

# Фильтруем только столбцы, где есть хотя бы 1 пропуск
missing_table = missing_table[missing_table['Пропусков'] > 0]

# Сортируем по количеству пропусков
missing_table = missing_table.sort_values(by='Процент_пропусков', ascending=False)

# Выводим ТОП-30 столбцов с наибольшим количеством пропусков
print(missing_table.head(30))

import matplotlib.pyplot as plt

# Столбцы с более чем 25% пропусков
missing_table_25 = missing_table[missing_table['Процент_пропусков'] > 25]

plt.figure(figsize=(12, 10))
plt.barh(missing_table_25.index[:30], missing_table_25['Процент_пропусков'][:30])
plt.xlabel('% пропусков')
plt.ylabel('Столбцы')
plt.title('Топ-30 признаков с наибольшим числом пропусков')
plt.gca().invert_yaxis()
plt.show()

print("\nСтолбцы с >25% пропусков:")
print(missing_table_25.index.tolist())

# Подсчёт количества пропусков
missing = df.isna().sum()
missing_percent = (missing / len(df)) * 100

# Определяем столбцы с более чем 65% пропусков
cols_to_drop = missing_percent[missing_percent > 65].index.tolist()

print(f"Найдено {len(cols_to_drop)} столбцов с >65% пропусков:")
print(cols_to_drop)

df_clean = df.drop(columns=cols_to_drop)

print(f"\nПосле очистки осталось {df_clean.shape[1]} столбцов из {df.shape[1]}.")

# Сохраняем очищенный датафрейм
df_clean.to_csv('diamonds_clean_new.csv', index=False, encoding='utf-8')

print("\nФайл 'diamonds_clean_new.csv' успешно сохранён.")




df = pd.read_csv("diamonds_clean_new.csv", encoding="utf-8")

# Список для хранения константных столбцов
constant_columns = []

# Проверяем каждый столбец
for col in df.columns:
    # Количество уникальных значений (исключая NaN)
    unique_values = df[col].dropna().unique()

    # Если уникальное значение одно или их нет вовсе
    if len(unique_values) <= 1:
        value = unique_values[0] if len(unique_values) == 1 else None
        constant_columns.append((col, value))

# Вывод результатов
print(f"Найдено {len(constant_columns)} столбцов с одинаковыми значениями:\n")
for col, val in constant_columns:
    print(f"{col} → значение: {val}")

# Удаляем найденные столбцы
cols_to_drop = [col for col, _ in constant_columns]
df_reduced = df.drop(columns=cols_to_drop)

print(f"\nПосле удаления осталось {df_reduced.shape[1]} столбцов из {df.shape[1]}.")

cols_to_drop_manual = [
    "productID",
    "itemID",
    "enriched.itemID",
    "sku",
    "url",
    "status",
    "shippingDate",
    "isFirmShipping",
    "enrichedDate",
    "createDate",
    "enriched.datePriced",
    "enriched.isFirmShipping"
]

# Удаляем только те столбцы, которые реально существуют
cols_to_drop_manual = [c for c in cols_to_drop_manual if c in df_reduced.columns]

df_final = df_reduced.drop(columns=cols_to_drop_manual)

print("Удалены столбцы:")
print(cols_to_drop_manual)

print("Итоговый размер датасета:")
print(df_final.shape)
print(f"\nПосле очистки осталось {df_final.shape[1]} столбцов из {df.shape[1]}.")

# Сохраняем новый файл
df_final.to_csv("diamonds_clean_new_2.csv", index=False, encoding="utf-8")

print("\nФайл 'diamonds_clean_new_2.csv' успешно сохранён.")




#Проверка совпадений значений в столбцах с ценой
df = pd.read_csv("diamonds_clean_new_2.csv", encoding="utf-8")

# Убираем строки с пропусками в этих столбцах
df_compare = df[['price', 'enriched.price']].dropna()

# Проверяем, где значения совпадают
same_values = df_compare['price'] == df_compare['enriched.price']

# Считаем количество и процент совпадений
same_count = same_values.sum()
total_count = len(df_compare)
same_percent = same_count / total_count * 100

print(f"Всего строк для сравнения: {total_count}")
print(f"Совпадающих значений: {same_count}")
print(f"Процент совпадений: {same_percent:.2f}%")

df_clean = df.drop(columns=['enriched.price'])
print("Столбец 'enriched.price' успешно удалён.")
print(f"\nПосле очистки осталось {df_clean.shape[1]} столбцов из {df.shape[1]}.")

# Сохраняем новый файл
df_clean.to_csv("diamonds_clean_new_3.csv", index=False, encoding="utf-8")

print("\nФайл 'diamonds_clean_new_3.csv' успешно сохранён.")




df = pd.read_csv("diamonds_clean_new_3.csv", encoding="utf-8")
# Заменяем столбец _id на простые номера строк, начиная с 1
df["_id"] = range(1, len(df) + 1)

# Сохраняем обновлённый файл
df.to_csv("diamonds_clean_final.csv", index=False, encoding="utf-8")
print("\nФайл 'diamonds_clean_final.csv' успешно сохранён.")

df.to_excel(
    "primer.xlsx",
    index=False,
    engine="openpyxl"
)
print("Файл primer.xlsx успешно сохранён")




df = pd.read_csv("diamonds_clean_final.csv", encoding="utf-8")
if 'stone.shape.name' in df.columns:
    # Уникальные значения
    unique_shapes = df['stone.shape.name'].dropna().unique()
    print("Уникальные значения в столбце 'stone.shape.name':")
    print(unique_shapes)

    # Количество каждого значения
    print("\nЧастота встречаемости:")
    print(df['stone.shape.name'].value_counts())