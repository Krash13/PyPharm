PyPharm
----------
 
1) Для установки пакета используем

```
pip install pypharm
```

2) Пример описания и рассчета модели с использованием модуля

Модель

![img.png](img.png)

![img_2.png](img_2.png)

Реализация PyPharm
```python
from pypharm import BaseCompartmentModel

model = BaseCompartmentModel([[0, 0.4586], [0.1919, 0]], [0.0309, 0], volumes=[228, 629])

res = model(90, d=5700, compartment_number=0)
```
