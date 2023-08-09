"""
Tensorflow-DDSP adaptation of https://github.com/adrianbarahona/noisebandnet/tree/master/noisebandnet
"""

import math
import numpy as np
import tensorflow as tf
from scipy import signal
from ddsp import core, processors

tfkl = tf.keras.layers


class FilterBank(tfkl.Layer):
    """Tensorflow adaptation of https://github.com/adrianbarahona/noisebandnet/blob/master/noisebandnet/filterbank.py
    Filterbank class that builds a filterbank with linearly and logarithmically
    distributed filters.
    Args:
        - n_filters_linear (int): number of linearly distributed filters.
        - n_filters_log (int): number of logarithmically distributed filters.
        - linear_min_f (float): low pass filter cutoff frequency.
        - linear_max_f_cutoff_fs (float): portion of the spectrum that is
        linearly distributed in a fraction of the sampling rate.
        - sample_rate (int): number of audio samples per second.
        - attenuation (float): FIR filter attenuation used in the Kaiser window (in dB).
    """
    def __init__(self, n_filters_linear=1024, n_filters_log=1024,
                 linear_min_f=20, linear_max_f_cutoff_fs=4, attenuation=50,
                 sample_rate=16000, **kwargs):
        super().__init__(**kwargs)
        
        frequency_bands = self.get_frequency_bands(
            n_filters_linear=n_filters_linear,
            n_filters_log=n_filters_log,
            linear_min_f=linear_min_f,
            linear_max_f_cutoff_fs=linear_max_f_cutoff_fs,
            sample_rate=sample_rate)
        self.band_centers = self.get_band_centers(frequency_bands=frequency_bands,
                                                  sample_rate=sample_rate)
        self.filters = self.build_filterbank(frequency_bands=frequency_bands,
                                             sample_rate=sample_rate,
                                             attenuation=attenuation)
        self.max_filter_len = max(len(array) for array in self.filters)

    def get_linear_bands(self, n_filters_linear, linear_min_f, linear_max_f_cutoff_fs, sample_rate):
        linear_max_f = (sample_rate / 2) / linear_max_f_cutoff_fs
        linear_bands = np.linspace(linear_min_f, linear_max_f, n_filters_linear)    
        linear_bands = np.vstack((linear_bands[:-1], linear_bands[1:])).T
        return linear_bands

    def get_log_bands(self, n_filters_log, linear_max_f_cutoff_fs, sample_rate):
        linear_max_f = (sample_rate / 2) / linear_max_f_cutoff_fs
        log_bands = np.geomspace(start=linear_max_f, stop=sample_rate/2,
                                 num=n_filters_log, endpoint=False)
        log_bands = np.vstack((log_bands[:-1], log_bands[1:])).T
        return log_bands

    def get_frequency_bands(self, n_filters_linear, n_filters_log, linear_min_f, linear_max_f_cutoff_fs,  sample_rate):
        linear_bands = self.get_linear_bands(n_filters_linear=n_filters_linear,
                                             linear_min_f=linear_min_f,
                                             linear_max_f_cutoff_fs=linear_max_f_cutoff_fs,
                                             sample_rate=sample_rate)
        if linear_max_f_cutoff_fs==1:
            return linear_center_f
        log_bands = self.get_log_bands(n_filters_log=n_filters_log,
                                       linear_max_f_cutoff_fs=linear_max_f_cutoff_fs,
                                       sample_rate=sample_rate)
        return np.concatenate((linear_bands, log_bands))

    def get_band_centers(self, frequency_bands, sample_rate):
        mean_frequencies = np.mean(frequency_bands, axis=1)
        lower_edge = frequency_bands[0,0] / 2    
        upper_edge = ((sample_rate / 2) + frequency_bands[-1,-1]) / 2
        return np.concatenate(([lower_edge], mean_frequencies, [upper_edge]))

    def get_filter(self, cutoff, sample_rate, attenuation, pass_zero, transition_bandwidth=0.2, scale=True):
        if isinstance(cutoff, np.ndarray): #BPF
            bandwidth = abs(cutoff[1]-cutoff[0])
        elif pass_zero==True: #LPF
            bandwidth = cutoff
        elif pass_zero==False: #HPF
            bandwidth = abs((sample_rate/2)-cutoff)
        width = (bandwidth/(sample_rate/2))*transition_bandwidth
        N, beta = signal.kaiserord(ripple=attenuation, width=width)
        N = 2 * (N // 2) + 1 #make odd
        h = signal.firwin(numtaps=N, cutoff=cutoff, window=('kaiser', beta),
                          scale=scale, fs=sample_rate, pass_zero=pass_zero)
        return h
    
    def build_filterbank(self, frequency_bands, sample_rate, attenuation):
        filters = []
        for i in range(frequency_bands.shape[0]):
            #low pass filter
            if i == 0:
                h = self.get_filter(cutoff=frequency_bands[i,0],
                                    sample_rate=sample_rate,
                                    attenuation=attenuation,
                                    pass_zero=True)
                filters.append(h)
            #band pass filter
            h = self.get_filter(cutoff=frequency_bands[i],
                                sample_rate=sample_rate,
                                attenuation=attenuation,
                                pass_zero=False)
            filters.append(h)
            #high pass filter
            if i == frequency_bands.shape[0]-1:
                h = self.get_filter(cutoff=frequency_bands[i,-1],
                                    sample_rate=sample_rate,
                                    attenuation=attenuation,
                                    pass_zero=False)
                filters.append(h)
        return filters


class NoiseBandNetSynth(processors.Processor):
    """Tensorflow-DDSP adaptation of the synth_batch part of
    https://github.com/adrianbarahona/noisebandnet/blob/master/noisebandnet/model.py
    Args:
        - n_band (int): number of bands in the filterbank.
        - upsampling (int): synthesis window size (in samples). frame_rate = sample_rate / upsampling
        - filterbank_attenuation (float): FIR filter attenuation used in the Kaiser window (in dB).
        - sample_rate (int): Sampling rate
        - attenuation (float): FIR filter attenuation used in the Kaiser window (in dB)
        - min_noise_len (int): Minimum noise length (in samples) for the noise bands, must be a power of 2
        - linear_min_f (float): Low pass filter cutoff frequency
        - linear_max_f_cutoff_fs (float): Portion of the spectrum that is linearly distributed in a fraction of the sampling rate
        - normalize_noise_bands (bool): Normalize the noise bands to the abslute maximum value. Scale up the noise bands amplitude."""
    def __init__(self, upsampling=64,
                 filterbank_attenuation=50, sample_rate=16000, min_noise_len=2**4,  # 15, 
                 linear_min_f=20, linear_max_f_cutoff_fs=4, normalize_noise_bands=True,
                 scale_fn=core.exp_sigmoid, inference=False, name="noise", **kwargs):
        super().__init__(name=name, **kwargs)
        assert min_noise_len > 0 and isinstance(min_noise_len, int) and check_power_of_2(min_noise_len), "min_noise_len must be a positive integer and a power of 2"
        self.scale_fn = scale_fn
        self.upsampling = upsampling
        self.sample_rate = sample_rate
        self.linear_min_f = linear_min_f
        self.linear_max_f_cutoff_fs = linear_max_f_cutoff_fs
        self.filterbank_attenuation=filterbank_attenuation
        self.min_noise_len = min_noise_len
        self.normalize_noise_bands = normalize_noise_bands
        self.inference = inference

    def build(self, input_shape):
        self.n_band = input_shape[-1]
        fb = FilterBank(n_filters_linear=self.n_band // 2,
                        n_filters_log=self.n_band // 2,
                        linear_min_f=self.linear_min_f,
                        linear_max_f_cutoff_fs=self.linear_max_f_cutoff_fs, 
                        sample_rate=self.sample_rate,
                        attenuation=self.filterbank_attenuation)
        # Store center frequencies for reference
        self.center_frequencies = fb.band_centers
        self.noise_bands, self.noise_len = get_noise_bands(
            fb=fb, min_noise_len=self.min_noise_len,
            normalize=self.normalize_noise_bands)

    def get_controls(self, magnitudes):
        if self.scale_fn is not None:
            magnitudes = self.scale_fn(magnitudes)
        return {'amplitudes': magnitudes}

    def get_signal_inference(self, amplitudes):
        # TODO
        pass

    def get_signal(self, amplitudes):
        # Synth in noise_len frames to fit longer sequences on GPU memory
        frame_len = int(self.noise_len / self.upsampling)
        n_frames = math.ceil(amplitudes.shape[1] / frame_len)

        # Avoid overfitting to noise values
        noise_bands = tf.roll(
            self.noise_bands, axis=1,
            shift=tf.random.uniform(shape=(),
                                    minval=0,
                                    maxval=self.noise_bands.shape[1],
                                    dtype=tf.int32))

        n_samples = amplitudes.shape[1] * self.upsampling
        # Smaller amp len than noise_len
        if amplitudes.shape[1] / frame_len < 1:
            upsampled_amplitudes = core.resample(amplitudes, n_samples)
            signal = tf.reduce_sum(
                noise_bands[:, :n_samples, :] * upsampled_amplitudes,
                axis=-1)
        else:
            for i in range(n_frames):
                if i == 0:
                    upsampled_amplitudes = core.resample(
                        amplitudes[:, :frame_len, ...],
                        n_timesteps=frame_len * self.upsampling)
                    signal = tf.reduce_sum(noise_bands * upsampled_amplitudes,
                                           axis=-1)
                #last iteration
                elif i == (n_frames-1):
                    upsampled_amplitudes = core.resample(
                        amplitudes[:, i*frame_len:, :],
                        n_timesteps=frame_len * self.upsampling)
                    signal = tf.concat([signal,
                                        tf.reduce_sum(noise_bands[:, :upsampled_amplitudes.shape[1], ...] * upsampled_amplitudes,
                                                      axis=-1)],
                                       axis=1)
                else:
                    upsampled_amplitudes = core.resample(
                        amplitudes[:, i*frame_len:(i+1)*frame_len, ...],
                        n_timesteps=frame_len * self.upsampling)
                    signal = tf.concat([signal,
                                        tf.reduce_sum(noise_bands * upsampled_amplitudes,
                                                      axis=-1)],
                                       axis=1)
            # Final truncation
            signal = signal[:, :n_samples]
        return signal


def check_power_of_2(x):
    return 2 ** int(math.log(x, 2)) == x


def get_next_power_of_2(x):
    return int(math.pow(2, math.ceil(math.log(x)/math.log(2))))


def pad_filters(filters, n_samples):
    for i in range(len(filters)):
        filters[i] = np.pad(filters[i], (n_samples-len(filters[i]),0))
    return tf.convert_to_tensor(np.array(filters))


def compute_magnitude_filters(filters):
    magnitude_filters = tf.signal.rfft(filters)
    magnitude_filters = tf.math.abs(magnitude_filters)
    return tf.cast(magnitude_filters, dtype=tf.complex64)


def get_noise_bands(fb, min_noise_len, normalize):
    #build deterministic loopable noise bands
    if fb.max_filter_len > min_noise_len:
        noise_len = get_next_power_of_2(fb.max_filter_len)
    else:
        noise_len = min_noise_len
    filters = pad_filters(fb.filters, noise_len)
    magnitude_filters = compute_magnitude_filters(filters=filters)
    phase_noise = tf.random.uniform(shape=(magnitude_filters.shape[0], magnitude_filters.shape[-1]),
                                    minval=-math.pi, maxval=math.pi,
                                    dtype=tf.float32, seed=42)
    phase_noise = tf.math.exp(1j * tf.cast(phase_noise, dtype=tf.complex64))
    phase_noise = tf.concat([tf.zeros((phase_noise.shape[0], 1), dtype=tf.complex64),
                             phase_noise[:,1:-1],
                             tf.zeros((phase_noise.shape[0], 1), dtype=tf.complex64)],
                            axis=1)

    magphase = tf.multiply(magnitude_filters, phase_noise)
    noise_bands = tf.signal.irfft(magphase)
    if normalize:
        noise_bands /= tf.math.reduce_max(tf.math.abs(noise_bands))
    noise_bands = core.tf_float32(tf.expand_dims(noise_bands, 0))
    # Put time axis at 1
    noise_bands = tf.transpose(noise_bands, [0, 2, 1])
    return noise_bands, noise_len


if __name__ == '__main__':
    magnitudes = tf.random.uniform((8, 750, 64))
    synth = NoiseBandNetSynth(sample_rate=16000, min_noise_len=2**4)
    outs = synth(magnitudes)
    import pdb; pdb.set_trace()
