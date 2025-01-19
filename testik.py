from PyPharm import MagicCompartmentModel
matrix = [[0, 3.08434104, 0, 0, 0],
          [0.0704116, 0, 1.156042, 0, 0],
          [0, 0.25287354, 0, 0.20776556, 0.64342174],
          [0, 0, 0, 0, 0],
          [0, 0, 3.03659391, 0, 0]]
outputs = [0, 0, 0.077, 4.48411185, 0]
model = MagicCompartmentModel(matrix, outputs, volumes=[49.92285762 , 49.41672868, 47.561541, 13.6659207, 49.90933737], magic_coefficient=50., exclude_compartments=[2], numba_option=True)
teoretic_y = [[268.5, 783.3, 154.6, 224.2, 92.6, 0], [342, 637, 466, 235, 179, 158], [0, 0, 11.2, 5.3, 5.42, 3.2]]
teoretic_x = [0.25, 0.5, 1, 4, 8, 24]
model()
model.load_optimization_data(
    teoretic_x=teoretic_x,
    teoretic_y=teoretic_y,
    know_compartments=[3, 4, 0],
    c0=[0, 0, 320, 0, 0]
)
print(model.get_kinetic_params(compartment_number=2, d=20 * 0.2 * 1000, t_max=48))
model.plot_model(d=20 * 0.2 * 1000, compartment_number=2, t_max=max(teoretic_x), compartment_numbers=[3, 4, 0], compartment_names={3: "Печень", 4: "Сердце", 0: "Мозг"})