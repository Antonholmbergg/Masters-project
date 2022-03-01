from NuRadioMC.SignalGen import askaryan as ask
from NuRadioReco.utilities import units
from scipy.stats import qmc
import numpy as np

m = 14
sampler = qmc.Sobol(d=2, seed=42)
samples = sampler.random_base2(m=m)


l_bounds = [np.log(1e15), 56-15]
u_bounds = [np.log(1e19), 56+15]
samples_scaled = qmc.scale(samples, l_bounds, u_bounds)
samples_scaled[:,0] = np.exp(samples_scaled[:,0])

N = 128
dt = 1 * units.GHz
"""
trace = ask.get_time_trace(
    energy=1*units.EeV,
    theta=56*units.deg,
    N=N,
    dt=dt,
    shower_type="had",
    n_index=1.78,
    R=1*units.km,
    model="ARZ2020"
    )
"""

traces = np.zeros((2**m, N+2))
traces[:, :2] = samples_scaled

for i in range(2**m):
    trace = ask.get_time_trace(
    energy=samples_scaled[i,0]*units.eV,
    theta=samples_scaled[i,1]*units.deg,
    N=N,
    dt=dt,
    shower_type="had",
    n_index=1.78,
    R=1*units.km,
    model="ARZ2020"
    )
    traces[i, 2:] = trace

print(np.sum(traces == 0))
print(traces.shape[0]*traces.shape[1])

np.savetxt('/mnt/md0/aholmberg/data/efield.csv', traces, delimiter=',', header='first two columns are energy(0) and angle(1)')