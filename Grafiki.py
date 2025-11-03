import json
import pandas as pd
import matplotlib.pyplot as plt
import os
import ast
import numpy as np
import seaborn as sns

# === 1. Настройки ===
folder_path = "C:/Users/Admin/Desktop/Server_TEST/LOG"

# === 2. Поиск последнего .txt файла ===
txt_files = [f for f in os.listdir(folder_path) if f.endswith(".txt")]
if not txt_files:
    raise FileNotFoundError("В папке нет .txt файлов")

latest_file = max(txt_files, key=lambda f: os.path.getmtime(os.path.join(folder_path, f)))
latest_path = os.path.join(folder_path, latest_file)
print(f"Используется файл: {latest_file}")

# === 3. Чтение содержимого ===
with open(latest_path, "r", encoding="utf-8") as f:
    content = f.read()

try:
    DATA = ast.literal_eval(content)
    if not isinstance(DATA, dict):
        raise ValueError("Ожидался dict в корне файла.")
except Exception as e:
    raise ValueError(f"Не удалось преобразовать текст в словарь: {e}")

# === 4. Разбор JSON внутри DATA ===
all_subjects = []
for key, value in DATA.items():
    try:
        subjects = json.loads(value)
        if isinstance(subjects, list):
            all_subjects.extend(subjects)
        else:
            print(f"Блок {key} не список, пропускаем.")
    except Exception as e:
        print(f"Ошибка при загрузке блока {key}: {e}")

# === 5. Преобразуем данные ===
rows = []
for subj in all_subjects:
    name = subj.get("subject_name", "Неизвестно")
    try:
        avg = float(subj.get("avg_five", 0) or 0)
    except ValueError:
        avg = 0

    marks = []
    for period in subj.get("periods", []):
        for mark in period.get("marks", []):
            for val in mark.get("values", []):
                if val.get("original"):
                    marks.append(str(val["original"]))

    rows.append({
        "subject": name,
        "avg": avg,
        "marks": marks
    })
if not rows:
    raise ValueError("Не удалось извлечь ни одного предмета")

df = pd.DataFrame(rows)

# === 6. Подсчёт количества оценок ===
for grade in ["5", "4", "3", "2"]:
    df[grade] = df["marks"].apply(lambda x: x.count(grade) if isinstance(x, list) else 0)

# Агрегируем данные по предметам, чтобы избежать дубликатов
df = df.groupby('subject').agg({
    'avg': 'mean',  # Среднее по avg, если отличаются
    '5': 'sum',
    '4': 'sum',
    '3': 'sum',
    '2': 'sum'
}).reset_index()


# Убираем указанные предметы
df = df[~df['subject'].isin(['Индивидуальный проект', 'Иностранный язык'])]

def Merge(subject,*List):
    global df
    merged_df = df[df['subject'].isin(List)].copy()

    if not merged_df.empty:
        merged_row = merged_df[['5', '4', '3', '2']].sum()
        total_marks_merged = merged_row.sum()
        if total_marks_merged > 0:
            merged_row['weighted_avg'] = (merged_row['5'] * 5 + merged_row['4'] * 4 + merged_row['3'] * 3 + merged_row['2'] * 2) / total_marks_merged
        else:
            merged_row['weighted_avg'] = np.nan
        merged_row['avg'] = merged_df['avg'].mean()  # Среднее по оригинальным avg
        merged_row['subject'] = subject
        merged_row = pd.DataFrame([merged_row])[['subject', 'avg', '5', '4', '3', '2', 'weighted_avg']]

        df = df[~df['subject'].isin(List)]
        df = pd.concat([df, merged_row], ignore_index=True)

    
Merge('Математика', 'Алгебра и начала математического анализа', 'Алгебра', 'Математика', 'Геометрия', 'Вероятность и статистика')
Merge('Физика', 'Технологии современного производства', 'Инженерный практикум', 'Физика')





# --- Круговая диаграмма ---
total_counts = {
    "5": int(df["5"].sum()),
    "4": int(df["4"].sum()),
    "3": int(df["3"].sum()),
    "2": int(df["2"].sum())
}

# Цвета и подписи
colors = {
    "5": "#4CAF50",   # зелёный
    "4": "#9C27B0",   # фиолетовый
    "3": "#FF9800",   # оранжевый
    "2": "#F44336"    # красный
}

# Создаём круговую диаграмму
plt.figure(figsize=(7, 7))
plt.pie(
    total_counts.values(),
    labels=[f"{k} ({v})" for k, v in total_counts.items()],
    autopct='%1.1f%%',
    startangle=90,
    colors=[colors[k] for k in total_counts.keys()],
    textprops={'fontsize': 12, 'color': 'black'}
)

plt.title("Распределение всех оценок", fontsize=14, fontweight="bold")
plt.axis("equal")
plt.tight_layout()
plt.show()

subjects = df["subject"]

# Для отступов между группами предметов
group_width = 2          # ширина всей группы (включая все 4 столбца)
bar_width = group_width / 4  # ширина одного столбца
gap = 1                # промежуток между группами

# Создаём позиции по X с отступами
x = np.arange(len(subjects)) * (group_width + gap)

plt.figure(figsize=(max(14, len(subjects) * 0.6), 10))

# Смещения для 4 типов оценок внутри группы
offsets = np.linspace(-group_width/2 + bar_width/2, group_width/2 - bar_width/2, 4)

# Построение графиков
bars_5 = plt.bar(x + offsets[0], df["5"], bar_width, label="5", color=colors["5"])
bars_4 = plt.bar(x + offsets[1], df["4"], bar_width, label="4", color=colors["4"])
bars_3 = plt.bar(x + offsets[2], df["3"], bar_width, label="3", color=colors["3"])
bars_2 = plt.bar(x + offsets[3], df["2"], bar_width, label="2", color=colors["2"])

# Функция для подписей
def add_labels(bars):
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            plt.text(
                bar.get_x() + bar.get_width()/2,
                height + 0.2,
                f"{int(height)}",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold"
            )

add_labels(bars_5)
add_labels(bars_4)
add_labels(bars_3)
add_labels(bars_2)

# Настройки
plt.xticks(x, subjects, rotation=90)
plt.ylabel("Количество оценок")
plt.title("Количество оценок по предметам", fontsize=14, fontweight="bold")
plt.legend(title="Оценки")

plt.xlim(min(x) - 1, max(x) + 1)

plt.tight_layout()
plt.show()






# Классификация предметов на категории
phys_math_sci = ['Физика', 'Химия', 'Биология', 'Информатика', 'Алгебра', 'Геометрия', 'Алгебра и начала математического анализа', 'Математика', 'Технологии современного производства', 'Инженерный практикум']  # Добавьте больше, если нужно
humanities = ['Русский язык', 'Литература', 'История', 'Обществознание', 'География', 'Химия', 'Иностранный (английский) язык']  # Добавьте больше
df['category'] = 'Другие'
df.loc[df['subject'].isin(phys_math_sci), 'category'] = 'Физико-Математические'
df.loc[df['subject'].isin(humanities), 'category'] = 'Гуманитарные'

# Расчёт дополнительных метрик
df['total_marks'] = df['5'] + df['4'] + df['3'] + df['2']
df['weighted_avg'] = (df['5'] * 5 + df['4'] * 4 + df['3'] * 3 + df['2'] * 2) / df['total_marks'].replace(0, np.nan)
df['good_percent'] = ((df['5'] + df['4']) / df['total_marks'].replace(0, np.nan)) * 100

# Общее распределение (таблица 1)
total_5 = total_counts["5"]
total_4 = total_counts["4"]
total_3 = total_counts["3"]
total_2 = total_counts["2"]
total_all = total_5 + total_4 + total_3 + total_2

if total_all > 0:

    dist_df = pd.DataFrame({
        'Градация': ['5', '4', '3', '2', 'Итого'],
        'Количество': [total_5, total_4, total_3, total_2, total_all],
        'Процент от общего': [f"{(v / total_all * 100):.1f}%" if v != total_all else '100%' for v in [total_5, total_4, total_3, total_2, total_all]]
    })


    overall_weighted_avg = (total_5 * 5 + total_4 * 4 + total_3 * 3 + total_2 * 2) / total_all
    overall_good_percent = ((total_5 + total_4) / total_all) * 100
    overall_avg = df['avg'].mean()


# По предметам (таблица 2)
if not df.empty:
    subj_df = df[['subject', '5', '4', '3', '2', 'avg', 'weighted_avg', 'total_marks']].copy()
    subj_df.columns = ['Предмет', '5', '4', '3', '2', 'Средний балл (из данных)', 'Взвешенная средняя', 'Всего оценок']

    # Выявление сильных/слабых
    strong_subjects = df[df['weighted_avg'] >= 4.0]['subject'].tolist()
    weak_subjects = df[df['weighted_avg'] < 4.0]['subject'].tolist()

# По категориям (таблица 3)
category_df = df.groupby('category').agg({
    '5': 'sum', '4': 'sum', '3': 'sum', '2': 'sum', 'avg': 'mean'
}).reset_index()
category_df['total_marks'] = category_df['5'] + category_df['4'] + category_df['3'] + category_df['2']
category_df['weighted_avg'] = (category_df['5'] * 5 + category_df['4'] * 4 + category_df['3'] * 3 + category_df['2'] * 2) / category_df['total_marks'].replace(0, np.nan)
category_df['good_percent'] = ((category_df['5'] + category_df['4']) / category_df['total_marks'].replace(0, np.nan)) * 100

if not category_df.empty:
    print("\n#### 3. Разделение на физико-математические и гуманитарные предметы")
    cat_df = category_df[['category', '5', '4', '3', '2', 'weighted_avg', 'good_percent', 'total_marks']].copy()
    cat_df.columns = ['Категория', '5', '4', '3', '2', 'Средний балл', 'Процент хороших (4 и 5)', 'Всего оценок']
    cat_df['Процент хороших (4 и 5)'] = cat_df['Процент хороших (4 и 5)'].apply(lambda x: f"{x:.1f}%" if not pd.isna(x) else 'N/A')
    cat_df['Средний балл'] = cat_df['Средний балл'].apply(lambda x: f"{x:.2f}" if not pd.isna(x) else 'N/A')
    print(cat_df.to_string(index=False))

# === Разделение на физико-математические и гуманитарные предметы (график) ===
category_df = df.groupby('category').agg({
    '5': 'sum', '4': 'sum', '3': 'sum', '2': 'sum', 'avg': 'mean'
}).reset_index()

category_df['total_marks'] = category_df['5'] + category_df['4'] + category_df['3'] + category_df['2']
category_df['weighted_avg'] = (
    (category_df['5'] * 5 + category_df['4'] * 4 + category_df['3'] * 3 + category_df['2'] * 2)
    / category_df['total_marks'].replace(0, np.nan)
)
category_df['good_percent'] = (
    ((category_df['5'] + category_df['4']) / category_df['total_marks'].replace(0, np.nan)) * 100
)

if not category_df.empty:
    import matplotlib.pyplot as plt
    import numpy as np

    categories = category_df['category']
    x = np.arange(len(categories))
    width = 0.35

    fig, ax1 = plt.subplots(figsize=(8, 6))

    # --- Средний балл ---
    bars1 = ax1.bar(x - width/2, category_df['weighted_avg'], width, color="#4CAF50", label="Средний балл (взвеш.)")

    # --- Процент хороших ---
    ax2 = ax1.twinx()
    bars2 = ax2.bar(x + width/2, category_df['good_percent'], width, color="#2196F3", alpha=0.7, label="Процент хороших (4 и 5)")

    # --- Подписи значений ---
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, height + 0.05, f"{height:.2f}", ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, height + 0.05, f"{height:.1f}%", ha='center', va='bottom', fontsize=9, color='blue')

    # --- Настройки осей ---
    ax1.set_ylabel("Средний балл", color="#4CAF50", fontsize=11)
    ax2.set_ylabel("Процент хороших (4 и 5)", color="#2196F3", fontsize=11)
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories, rotation=0)
    ax1.set_title("Сравнение категорий предметов по качеству успеваемости", fontsize=13, fontweight="bold")

    # --- Объединённая легенда ---
    lines, labels = [], []
    for ax in [ax1, ax2]:
        line, label = ax.get_legend_handles_labels()
        lines += line
        labels += label
    ax1.legend(lines, labels, loc="upper left")

    plt.tight_layout()
    plt.show()





# === Вычисление корреляции между предметами ===
if len(df) > 1:
    corr_subjects = df.set_index('subject')[['5', '4', '3', '2']].T.corr()

    # === Тепловая карта корреляций ===
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_subjects, annot=True, cmap='RdYlGn', center=0, linewidths=0.5)
    plt.title("Корреляция между предметами (по распределению оценок)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()

    # === Топ 5 лучших и худших корреляций ===
    upper_tri = corr_subjects.where(np.triu(np.ones(corr_subjects.shape), k=1).astype(bool))
    stacked = upper_tri.stack().rename_axis(['Предмет 1', 'Предмет 2']).reset_index(name='Корреляция')

    if not stacked.empty:
        # Топ 5 лучших
        top_best = stacked.sort_values(by='Корреляция', ascending=False).head(5)
        # Топ 5 худших
        top_worst = stacked.sort_values(by='Корреляция', ascending=True).head(5)

        # --- График: топ 5 положительных ---
        plt.figure(figsize=(12, 4))
        plt.barh(
            [f"{a} — {b}" for a, b in zip(top_best['Предмет 1'], top_best['Предмет 2'])],
            top_best['Корреляция'],
            color="#4CAF50"
        )
        plt.xlabel("Коэффициент корреляции")
        plt.title("Топ 5 положительных корреляций между предметами")
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()

        # --- График: топ 5 отрицательных ---
        plt.figure(figsize=(12, 4))
        plt.barh(
            [f"{a} — {b}" for a, b in zip(top_worst['Предмет 1'], top_worst['Предмет 2'])],
            top_worst['Корреляция'],
            color="#F44336"
        )
        plt.xlabel("Коэффициент корреляции")
        plt.title("Топ 5 отрицательных корреляций между предметами")
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()
    else:
        print("Нет данных для построения топ корреляций.")
else:
    print("Недостаточно предметов для вычисления корреляции.")