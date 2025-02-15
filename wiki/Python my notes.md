---
created: 2025.01.27
author:
  - s1gle
tags:
  - Pandas
  - Python
---

# Работа с датой и временем

```python
import datetime as DT

date_str = input("Введите дату выпуска (dd/mm/yyyy)\n")
vypusk = DT.datetime.strptime(date_str, '%d/%m/%Y').date()
print(vypusk)
```