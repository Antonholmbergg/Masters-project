import matplotlib.pyplot as plt
from scipy.stats import qmc
import pandas as pd
import numpy as np
from NuRadioMC.SignalProp import analyticraytracing as ray
from NuRadioMC.utilities.medium import southpole_2015


sampler = qmc.Sobol(d=3, seed=42)
samples = sampler.random_base2(m=21)


l_bounds = [-2000, -2700, -200]
u_bounds = [-1, -1, -1]
samples_scaled = qmc.scale(samples, l_bounds, u_bounds)

#samples = np.random.random_sample((2**17,3))
#samples_scaled = np.zeros((samples.shape))
#for i in range(samples.shape[1]):
    #samples_scaled[:, i] = (l_bounds[i] - u_bounds[i]) * samples[:, i] + u_bounds[i]

source_pos = []
antenna_pos = []

for i in range(samples.shape[0]):
    source_pos.append([samples_scaled[i, 0], 0, samples_scaled[i, 1]])
    antenna_pos.append([0, 0, samples_scaled[i, 2]])


ice = southpole_2015()
ray = ray.ray_tracing(ice)

classification_data = []
raytrace_data = []

for pos in zip(source_pos, antenna_pos):
    ray.set_start_and_end_point(pos[0], pos[1])
    ray.find_solutions()

    classification_data.append({'source_pos_r':pos[0][0], 'source_pos_z':pos[0][2], 'antenna_pos_z':pos[1][2], 'n_sol':ray.get_number_of_solutions()})
    solutions = ray.get_results()

    for iS in range(ray.get_number_of_solutions()):
        solution = {'type':ray.get_solution_type(iS),
                    'source_pos_r':pos[0][0],
                    'source_pos_z':pos[0][2],
                    'antenna_pos_z':pos[1][2],
                    'travel_time':ray.get_travel_time(iS),
                    'path_length':ray.get_path_length(iS),
                    'launch_vec_r':ray.get_launch_vector(iS)[0],
                    'launch_vec_z':ray.get_launch_vector(iS)[2],
                    'recieve_vec_r':ray.get_receive_vector(iS)[0],
                    'recieve_vec_z':ray.get_receive_vector(iS)[2]}

        raytrace_data.append(solution)

    ray.reset_solutions()
    

df = pd.DataFrame(raytrace_data)

df['launch_angle'] = np.degrees(np.arctan2(df['launch_vec_r'].to_numpy(), df['launch_vec_z'].to_numpy()))
df['recieve_angle'] = np.degrees(np.arctan2(df['recieve_vec_r'].to_numpy(), df['recieve_vec_z'].to_numpy()))
#df.to_csv('/mnt/md0/aholmberg/data/raytrace_samples_angle.csv', index=False)

df.to_csv('/mnt/md0/aholmberg/data/raytrace_samples_sobol_21.csv', index=False)
df2 = pd.DataFrame(classification_data)
df2.to_csv('/mnt/md0/aholmberg/data/ana_ray_class_sobol_21.csv', index=False)
"""
type: int
    * 1: 'direct'
    * 2: 'refracted'
    * 3: 'reflected
"""