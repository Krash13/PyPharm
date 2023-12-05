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

При оптимизации, вектор неизвестных это
x = [configuration_matrix (неизвестные), outputs(неизвестные), volumes(неизвестные)]

**4) Модель MagicCompartmentModel** 

Данная модель необходима нам для тех случаев, 
когда мы не знаем как именно стоит переводить входные
единицы измерения в выходные.

В модель добавляется 2 дополнительных параметра:

* magic_coefficient - множитель преобразования входных единиц в выходные;
* exclude_compartments - список номеров камер, которые не 
подвергнутся преобразованию.

```python  
from PyPharm import MagicCompartmentModel  
  
model = MagicCompartmentModel([[0, 0.4586], [0.1919, 0]], [0.0309, 0], volumes=[228, 629], magic_coefficient=None, exclude_compartments=[2])  
  
res = model(90, d=5700, compartment_number=0)  
```

Параметр magic_coefficient может быть задан None,
в таком случае он будет подвергнут оптимизации, в таком 
случае он будет браться из последнего значения в векторе
переменных.
Если оба параметра не заданы, то модель выраздается 
в простую BaseCompartmentModel.

**5) Модель MagicCompartmentModel** 

Данная модель учитывает поправку на высвобождение
ЛВ в модель вводятся дополнительные параметры:
* v_release - Объем гепотетической камеры из которой происходит высвобождение
* release_parameters -  Параметры функции высвобождения
* release_compartment - Номер камеры в которую происходит высвобождение
* release_function - Функция высвобождения по умолчанию f(t,m,b,c) = c0 * c * t ** b / (t ** b + m)

При этом d и c0 теперь везде носят характер параметров камеры,
из которой происходит высвобождение
```python
from PyPharm import ReleaseCompartmentModel
import matplotlib.pyplot as plt

model = ReleaseCompartmentModel(
    6.01049235e+00,
    [4.56683781e-03, 1.36845756e+00, 5.61175978e-01],
    0,
    configuration_matrix=[[0, 1.18292665e+01], [3.02373800e-01, 0]],
    outputs=[5.00000000e+00, 0],
    volumes=[1.98530383e+01, 3.81007392e+02],
    numba_option=True
)
teoretic_t = [5/60, 0.25, 0.5, 1, 2, 4, 24, 48]
teoretic_c = [[3558.19,	508.49,	230.95,	52.05,	44.97,	36.52,	17.89,	10.36]]
d = 5 * 0.02 * 1000000
res = model(48, d=d)
plt.plot(teoretic_t, teoretic_c[0], 'r*')
plt.plot(res.t, res.y[0])
plt.grid()
plt.show()
```
Параметры release_parameters и v_release могут подвергаться оптимизации
в таком случае, искомое нужно просто задать как None. Тогда вектор неизвестных это
x = [configuration_matrix (неизвестные), outputs(неизвестные), volumes(неизвестные), release_parameters(неизвестные), v_release]
