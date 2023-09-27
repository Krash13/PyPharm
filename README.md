PyPharm
----------
 
1) To install the package we use

```
pip install pypharm
```

2) Example of description and calculation of a model using the module

Model

![img.png](img.png)

![img_2.png](img_2.png)

Implementation of PyPharm
```python
from PyPharm import BaseCompartmentModel

model = BaseCompartmentModel([[0, 0.4586], [0.1919, 0]], [0.0309, 0], volumes=[228, 629])

res = model(90, d=5700, compartment_number=0)
```