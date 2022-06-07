"""For generating the training data for the signal gen model.
"""
from NuRadioMC.SignalGen import askaryan as ask
from NuRadioReco.utilities import units, fft
import numpy as np

# 2^m is the number of training points generated
m = 13

# Maximum bounds
# l_bounds = [np.log(1e15), 55.82 - 19.99]
# u_bounds = [np.log(1e19), 55.82 + 19.99]

# Set bounds for the angles and energy
l_bounds = [np.log(1e15), 55.82 - 2.5]
u_bounds = [np.log(1e19), 55.82 + 2.5]
#samples_scaled = qmc.scale(samples, l_bounds, u_bounds)
#samples_scaled[:, 0] = np.exp(samples_scaled[:, 0])

# Sample random uniform and scale to right values
samples = np.random.random_sample((2**m, 2))
samples_scaled = np.zeros((samples.shape))
for i in range(samples.shape[1]):
    samples_scaled[:, i] = (l_bounds[i] - u_bounds[i]) * samples[:, i] + u_bounds[i]

samples_scaled[:, 0] = np.exp(samples_scaled[:, 0])

# Define timebins and sampling rate
N = 896
dt = 1e-10 * units.second
sr = 1/dt

ff = np.fft.rfftfreq(N, dt)

traces = np.zeros(((2**m)*10, N+3))
# Calculate the time traces of the signals
for i in range(2**m):
    energy = samples_scaled[i, 0]*units.eV
    theta = samples_scaled[i, 1]*units.deg
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
        traces[i*10 + iN, 0] = energy
        traces[i*10 + iN, 1] = theta
        traces[i*10 + iN, 2] = iN
        traces[i*10 + iN, 3:] = trace

# Saves the unfiltered date and conditions. This is unnecesary and should be changed.
# Unfiltered signals are never used so only save the conditions instead.
np.save(f'/mnt/md0/aholmberg/data/signal_had_{m}_5deg.npy', traces)

# Filter signals by fourier transforming and setting frequencies above
# the second peak (in frequency space) to zero. Some times cuts too hard so consider changing.
signals_filtered = np.zeros_like(traces[:,3:])
for ind in range(traces.shape[0]):

    signal_spectrum = fft.time2freq(traces[ind,3:], sampling_rate=sr)
    freqs = np.abs(signal_spectrum)
    i = 0
    while i+2 < freqs.shape[0] and (not (freqs[i] > freqs[i+1] < freqs[i+2])):
        i = i + 1

    mask = ff < ff[i]

    signal_spectrum_filtered = np.zeros((signal_spectrum.shape), dtype=np.complex)
    signal_spectrum_filtered[mask] = signal_spectrum[mask]
    signal_filtered = fft.freq2time(signal_spectrum_filtered, sampling_rate=sr)
    signals_filtered[ind, :] = signal_filtered

# Save the filtered signals
np.save(f'/mnt/md0/aholmberg/data/signal_had_{m}_filtered_5deg.npy', signals_filtered)