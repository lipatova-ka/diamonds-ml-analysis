import pandas as pd
import numpy as np

# Проведя анализ, вычислили поля, которые попали в выборку, по которым было много пропусков
# попытаемся это изменить
SRC_FILE = 'cadas-pas2.items.csv'
DST_FILE = 'items_balanced_50k.csv'
TARGET_N = 50_000
shape_col = 'stone.shape.name'
prio_cols = ['stone.flour.fullName', 'stone.cut.name', 'stone.cut.id']  # поля-приоритеты

df = pd.read_csv(SRC_FILE, sep=None, engine='python', encoding='utf-8')

# Нормализация формы
df[shape_col] = df[shape_col].astype(str).str.strip().str.lower()
df = df[df[shape_col].notna() & (df[shape_col] != '')]

# Классы и квоты
counts = df[shape_col].value_counts()
classes = counts.index.tolist()
m = len(classes)

if m == 0:
    raise ValueError("Не найдено значений в столбце формы огранки.")

base_quota = TARGET_N // m
remaining = TARGET_N % m

# базовый план: минимум из базовой квоты и фактического объёма класса
sample_plan = {c: int(min(base_quota, counts[c])) for c in classes}
current_total = sum(sample_plan.values())
to_fill = TARGET_N - current_total

# остаточную квоту распределяем по классам, где есть запас
capacity = {c: int(counts[c] - sample_plan[c]) for c in classes}
pool = [c for c in classes if capacity[c] > 0]
# сортируем по убыванию запаса, чтобы добирать из самых крупных
pool.sort(key=lambda c: capacity[c], reverse=True)

i = 0
while to_fill > 0 and pool:
    c = pool[i % len(pool)]
    if capacity[c] > 0:
        sample_plan[c] += 1
        capacity[c]   -= 1
        to_fill       -= 1
        if capacity[c] == 0:
            pool.pop(i % len(pool))
        else:
            i += 1
    else:
        pool.pop(i % len(pool))

print("План отбора по формам (целевые количества):")
print(pd.Series(sample_plan).sort_values(ascending=False))

# приоритетные строки: сначала берём записи с заполненными prio_cols
def is_filled_block(block: pd.DataFrame, cols):
    b = block.copy()
    for c in cols:
        if c not in b.columns:
            # если колонки нет — считаем её пустой
            b[c] = np.nan
        # нормализуем строки
        if b[c].dtype == object:
            b[c] = b[c].astype(str).str.strip()
        b[c] = b[c].replace({'': np.nan})
    mask = b[cols].notna().all(axis=1)
    return mask

parts = []
rng_state = 42

for cls, n_take in sample_plan.items():
    if n_take <= 0:
        continue

    block = df[df[shape_col] == cls]

    # приоритет: все три поля заполнены
    filled_mask = is_filled_block(block, prio_cols)
    preferred = block[filled_mask]
    fallback  = block[~filled_mask]

    take_pref = min(len(preferred), n_take)
    take_rest = n_take - take_pref

    chosen = []
    if take_pref > 0:
        chosen.append(preferred.sample(n=take_pref, random_state=rng_state))
    if take_rest > 0 and len(fallback) > 0:
        take_rest = min(take_rest, len(fallback))
        chosen.append(fallback.sample(n=take_rest, random_state=rng_state))

    if chosen:
        parts.append(pd.concat(chosen, axis=0))
    # если класса оказалось меньше плана — просто берём всё что есть (учтено в квоте и capacity)

balanced = pd.concat(parts, axis=0, ignore_index=True)
balanced = balanced.sample(frac=1.0, random_state=rng_state).reset_index(drop=True)

print("\nИтоговый размер выборки:", balanced.shape)

# Проверки распределения и заполненности полей-приоритетов
print("\nРаспределение форм в итоговой выборке:")
print(balanced[shape_col].value_counts())

print("\nДоля заполненности полей-приоритетов в итоговой выборке:")
for c in prio_cols:
    if c not in balanced.columns:
        print(f"{c}: колонки нет в данных")
    else:
        filled_ratio = balanced[c].notna().mean() * 100
        print(f"{c}: {filled_ratio:.2f}%")

# Сохранение
balanced.to_csv(DST_FILE, index=False, encoding='utf-8')
print(f"\nСохранено в 'items_balanced_50k.csv'")