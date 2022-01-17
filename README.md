# OS-RST
Oberseminar am Institut für Regelungs- und Steuerungstheorie zum Thema: Differentiation verrauschter Messsignale

Simulationsrahmenparameter:
point_counter = 500, step_size = 2e-3

noise_std_dev = 0.1 ... 0.5 in 0.1 Schritten

sine    = Input_Function(Input_Enum.SINE, [1, 0.5, 0, 0], sampling_period = step_size, point_counter=point_counter)

polynome = Input_Function(Input_Enum.POLYNOM, [100,-150,50,0], sampling_period = step_size, point_counter=point_counter)
