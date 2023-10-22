PyPharm  
----------  
  **1) Установка пакета**
  
```  
pip install pypharm  
```  
  
**2) Пример использования пакета для модели, где все параметры известны** 

Задана двухкамерная модель такого вида

```mermaid
graph LR
D((Доза D)) --> V1[Камера V1]
V1 -- k12 --> V2[Камера V2]
V2 -- k21 --> V1
V1 -- k10 --> Out(Выведение)
``` 
При этом, нам известны параметры модели 
|  V1|V2  |k12 |K21 | K10
|--|--|--|--|--|
|  228| 629 |0.4586|0.1919|0.0309

Создание и расчет модели при помощи пакета PyPharm  
```python  
from PyPharm import BaseCompartmentModel  
  
model = BaseCompartmentModel([[0, 0.4586], [0.1919, 0]], [0.0309, 0], volumes=[228, 629])  
  
res = model(90, d=5700, compartment_number=0)  
```
res - Результат работы решателя scipy solve_iv

**3) Пример использования пакета для модели, где все параметры неизвестны** 

Задана многокамерная модель такого вида

```mermaid
graph LR
Br(Мозг) --'Kbr-'--> Is[Межклетачное пространство]
Is --'Kbr+'-->Br
Is--'Kis-'-->B(Кровь)
B--'Kis+'-->Is
B--'Ke'-->Out1((Выведение))
B--'Ki+'-->I(Печень)
I--'Ki-'-->Out2((Выведение))
B--'Kh+'-->H(Сердце)
H--'Kh-'-->B
``` 
При этом, известен лишь параметр Ke=0.077

Создание и расчет модели при помощи пакета PyPharm, используя метод minimize:
```python  
from PyPharm import BaseCompartmentModel
import numpy as np
matrix = [[0, None, 0, 0, 0],
[None, 0, None, 0, 0],
[0, None, 0, None, None],
[0, 0, 0, 0, 0],
[0, 0, None, 0, 0]]
outputs = [0, 0, 0.077, None, 0]

model = BaseCompartmentModel(matrix, outputs)

model.load_optimization_data(
	teoretic_x=[0.25, 0.5, 1, 4, 8, 24],
	teoretic_y=[[0, 0, 11.2, 5.3, 5.42, 3.2], [268.5, 783.3, 154.6, 224.2, 92.6, 0], [342, 637, 466, 235, 179, 158]],
	know_compartments=[0, 3, 4],
	c0=[0, 0, 20000, 0, 0]
)

x_min = [1.5, 0.01, 0.5, 0.0001, 0.1, 0.1, 4, 3]
x_max = [2.5, 0.7, 1.5, 0.05, 0.5, 0.5, 7, 5]
x0 = np.random.uniform(x_min, x_max)
bounds = ((1.5, 2.5), (0.01, 0.7), (0.5, 1.5), (0.0001, 0.05), (0.1, 0.5), (0.1, 0.5), (4, 7), (3, 5))

model.optimize(
	bounds=bounds,
	x0=x0,
	options={'disp': True}
)

print(model.configuration_matrix)
```
Или же при помощи алгоритма взаимодействующих стран
```python
from PyPharm import BaseCompartmentModel
import numpy as np
matrix = [[0, None, 0, 0, 0],
[None, 0, None, 0, 0],
[0, None, 0, None, None],
[0, 0, 0, 0, 0],
[0, 0, None, 0, 0]]
outputs = [0, 0, 0.077, None, 0]

model = BaseCompartmentModel(matrix, outputs)

model.load_optimization_data(
	teoretic_x=[0.25, 0.5, 1, 4, 8, 24],
	teoretic_y=[[0, 0, 11.2, 5.3, 5.42, 3.2], [268.5, 783.3, 154.6, 224.2, 92.6, 0], [342, 637, 466, 235, 179, 158]],
	know_compartments=[0, 3, 4],
	c0=[0, 0, 20000, 0, 0]
)

model.optimize(
	method='country_optimization',
	Xmin=[0.5, 0.001, 0.001, 0.00001, 0.01, 0.01, 1, 1],
	Xmax=[5, 2, 2.5, 0.3, 1, 1, 10, 10],
	M=10,
	N=25,
	n=[1, 10],
	p=[0.00001, 2],
	m=[1, 8],
	k=8,
	l=3,
	ep=[0.2, 0.4],
	tmax=300,
	printing=True,
)
```