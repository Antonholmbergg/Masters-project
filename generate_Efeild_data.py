from NuRadioMC.SignalGen import askaryan as ask
from NuRadioReco.utilities import units, fft
from scipy.stats import qmc
import numpy as np

m = 18
sampler = qmc.Sobol(d=2, seed=42)
samples = sampler.random_base2(m=m)


l_bounds = [np.log(1e15), 55.82 - 19.99]
u_bounds = [np.log(1e19), 55.82 + 19.99]
samples_scaled = qmc.scale(samples, l_bounds, u_bounds)
samples_scaled[:,0] = np.exp(samples_scaled[:,0])

N = 896
dt = 1e-10 * units.second
sr = 1/dt

ff = np.fft.rfftfreq(N, dt)

traces = np.zeros(((2**m)*10, N+3))
#traces[:, :2] = samples_scaled

for i in range(2**m):
    energy = samples_scaled[i,0]*units.eV
    theta = samples_scaled[i,1]*units.deg
    for iN in range(10):
        trace = ask.get_time_trace(
            energy=energy,
            theta=theta,
            N=N,
            dt=dt,
            shower_type="had",
            n_index=1.78,
            R=1*units.km,
            model="ARZ2020",
            iN=iN
        )
        traces[i*10 + iN,0] = energy
        traces[i*10 + iN,1] = theta
        traces[i*10 + iN, 2] = iN
        traces[i*10 + iN, 3:] = trace

print(np.sum(traces == 0))
print(traces.shape[0]*traces.shape[1])

#np.savetxt('/mnt/md0/aholmberg/data/efield_14.csv', traces, delimiter=',', header='first two columns are energy(0) and angle(1)')
np.save('/mnt/md0/aholmberg/data/signal_had_18.npy', traces)

signals_filtered = np.zeros_like(traces[:,3:])
for ind in range(traces.shape[0]):
    
    signal_spectrum = fft.time2freq(traces[ind,3:], sampling_rate=sr)
    freqs = np.abs(signal_spectrum)
    i = 0
    while  i+2 < freqs.shape[0] and (not (freqs[i] > freqs[i+1] < freqs[i+2])):
        i = i + 1

    mask = ff < ff[i] #* units.MHz

    signal_spectrum_filtered = np.zeros((signal_spectrum.shape), dtype=np.complex)
    signal_spectrum_filtered[mask] = signal_spectrum[mask]
    signal_filtered = fft.freq2time(signal_spectrum_filtered, sampling_rate=sr)
    signals_filtered[ind,:] = signal_filtered

np.save('/mnt/md0/aholmberg/data/signal_had_18_filtered.npy', signals_filtered)