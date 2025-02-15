---
created: 2025.01.27
description: Моя таблица для Pandas
author:
  - s1gle
tags:
  - Pandas
  - Python
---
> [!example] [[README|General Base]] >> [[1_Soft/README|Base #1]] >> [[2_Hard/README|Base #2]] >> [[3_Python/README|Base #3]]


##### Начало
```python
import pandas as pd
# импорт из csv файла
df = pd.read_csv('file.csv', index_col='name_col')

# Создание  датафрейма вручную
df = pd.DataFrame([[1,'Bob', 'Builder'],
                  [2,'Sally', 'Baker'],
                  [3,'Scott', 'Candle Stick Maker']], 
columns=['id','name', 'occupation'])
```

Команда | Описание 
:---|---
`df.head(n)` | n записей из начала или конца
`df.tail(n)` | n записей из конца
`len(df)` | количество строк
`df.shape`  | количество строк и столбцов
`df.columns` | список столбцов
`df.info() `| список столбцов с указанием типов и количества непустых значений
`df.dtypes` | типы столбцов
`df.describe()` | характеристики распределения числовых значений (min, max, mean, ...)
`df.describe(include='all')` | характеристики всех столбцов
`df.nunique()`  | Число уникальных значений
`len(df['name_col'].unique())` | количество уникальных значений в столбце
`df.name_col.unique()` |  возвращает array с позициями уникальных значений
`df.type.value_counts()` | Подсчёт количества значений
`df_copy = df.copy(deep=True)` | копирование датафрейма
`df[:n].to_csv('saved_df.csv', index=False)` | cохранение первых n строк 

##### Доступ к данным

Команда | Описание 
:---|---
`df['name_col']#.head(3)` | Использование меток
`df.name_col`| тоже, что и предыдущее, если столбец существует
`df[0:2]` | Срез индексов строк
`df[['name_col_1','name_col_2']].head(n)` | выводит только указанные столбцы
`x = df['name_col'].tolist()`|Создание списка на основе значений столбца. Извлекает значения столбцов в переменные
`df.index.tolist()`|Получение списка значений из индекса
`df.columns.tolist()`|Получение списка значений столбцов

##### Фильтрация данных

Команда | Описание 
:---|---
`df.loc[['name_index_1','name_index_2']]`|Получение строк с нужными индексными значениями
`df.iloc[0:3]`|Получение строк по числовым индексам
`df.loc[1:4, 'name_col_1':'name_col_2']`| адресация метками
`df.iloc[1:4, 0:4]` | адресация целыми числами
`df[df['name_col_1'].isin(['name_col_2', 'name_col_3'])]`|Получение строк по заданным значениям столбцов
`df[df['name_col'] > 8]`|Фильтрация по значению. можно выбирать строки, соответствующие заданному условию.
`df.name_col == 'female'`| проверяем значение столбца, получаем массив из булевых значений (**True** or **False**)
`(df.name_col == 'female')[:4]`| проверяем значение столбца на заданном срезе строк
`df[df.name_col == 'female']#.head(3)` | Используем булев массив как маску для фильтра
`df[df.name_col == 'female'].Name.count()`| Несколько условий
`df[(df.name_col == 'female')&(df.Age >= 50)].Name.count()`|Несколько условий. Используются битовые операции **&**, **|**

##### Группировка данных

Команда | Описание 
:---|---
`df.groupby('name_col').count()`|подсчёт количество записей с различными значениями в столбцах
`df.groupby('name_col').mean()`| группировка по столбцу
`df.groupby(['name_col_1','name_col_2']).min()`| группировка по двум столбцам (**.max()**, **.sum()**, **.mean()** и т.д.)

##### Создание сводной таблицы

Для того чтобы извлечь из датафрейма некие данные, нет ничего лучше, чем сводная таблица. Обратите внимание на то, что здесь я серьёзно отфильтровал датафрейм, что ускорило создание сводной таблицы.

```python
tmp_df = df1.copy()
tmp_df.sort_values('name_col_1', ascending=True, inplace=True)
tmp_df = tmp_df[tmp_df.name_col_1 < 10] 
tmp_df = tmp_df[tmp_df.name_col_2 < 30]
tmp_df = tmp_df[tmp_df.name_col_3 != -1]
pd.pivot_table(tmp_df, values='name_col_3', index=['name_col_1'], columns=['name_col_2'], aggfunc=np.sum, fill_value=0)
```

##### Пропуски значений NaN

Команда | Описание 
:---|---
`df.isnull()` | возвращает **True**, если значение **NaN**
`df.isnull().sum() ` | сумма пропущенных значений по каждому столбцу
`dropna()` | позволяет удалить как строки с пропусками, так и столбцы. Можно удалять начиная с определенного порога непустых значений
`df.dropna().head(n)` | Без сохранения в data inplace=False
`fillna(0)`  | Замена пропусков **NaN** на **0**
`df.fillna(df.mean())` | Заполение средним значением, для числовых полей
`df.fillna(method='pad')` | Заполнение предыдущим значением

_Запись в ячейки,_ содержащие значение **NaN**, какого-то другого значения
Здесь записm значения  **0** в ячейки, содержащие значение **NaN**.  В этом примере создаём такую же сводную таблицу, как и ранее, но без использования `fill_value=0`.  Затем используем функцию `fillna(0)` для замены значений **NaN** на **0**.

```python
pivot = pd.pivot_table(tmp_df, values='rating', index=['user_id'], columns=['anime_id'], aggfunc=np.sum)
pivot.fillna(0)
```


##### Сортировка

Команда | Описание 
:---|---
`df.sort_values('name_col', ascending=False)`|Для сортировки датафреймов по значениям столбцов
`ages_sorted = df[df.name_col == 1].Age.sort_values(ascending=False)` \n `ages_sorted[:10]`| сортировка по возрасту
`df.sort_values(by='Age').head(3)` | сортировка по возрасту, отображение всей таблицы

##### Добавление и удаление данных

Команда | Описание 
:---|---
`df['train set'] = True`|Присоединение к датафрейму нового столбца с заданным значением
`df[['name_col_1','name_col_2']]`|Создание нового датафрейма из подмножества столбцов
`df.drop(['name_col_1', 'name_col_2', 'name_col_3'], axis=1).head()`|Удаление заданных столбцов
`df.append(df.sum(axis=0), ignore_index=True)`|Добавление строки с суммой значений из других строк
`df.sum(axis=1)`| позволяет суммировать значения в столбцах
`df.mean(axis=0)`|расчёт средних значений|

##### Комбинирование датафреймов

Команда | Описание 
:---|---
`pd.concat([df1, df2], ignore_index=True)`|Конкатенация двух датафреймов. Применимо для двух датафреймов с одинаковыми столбцами.
`df.merge(df1, left_on=’name_col’, right_on=’name_col’, suffixes=(‘_left’, ‘_right’))`| Слияние датафреймов. Применимо для двух датафреймов по некоему столбцу


##### Графические возможности

```Python
[In]:
df[['Age','Pclass','Fare']].hist()

[Out]:
array([[<Axes: title={'center': 'Age'}>,
        <Axes: title={'center': 'Pclass'}>],
       [<Axes: title={'center': 'Fare'}>, <Axes: >]], dtype=object)
```

![[Pasted image 20250208174339.png|400]]


