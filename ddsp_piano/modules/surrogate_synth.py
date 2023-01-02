import tensorflow as tf
from ddsp import core, processors
from ddsp_piano.modules.inharm_synth import get_inharmonic_freq


def complex_harmonic_synthesis(frequencies,
                               amplitudes,
                               complex_amplitudes=None,
                               harmonic_shifts=None,
                               harmonic_distribution=None,
                               upsampling=64,
                               sample_rate=16000,
                               amp_resample_method='window',
                               use_angular_cumsum=False):
    """Surrogate bank of quasi-harmonic oscillators usign complex amplitude
    for allowing frequency optimzation in the complex circle
    (see B. Hayes - Sinusoidal Frequency Estimation by Gradient Descent).

    Args:
        - frequencies (batch, n_frames, 1): frame-wise fundamental frequency
        in Hz.
        - amplitudes (batch, n_frames, 1): frame-wise oscillator real amplitude
        - complex_amplitudes (batch, n_frames, n_harmonics): frame-wise complex
        magnitudes.
        - harmoninc_shifts (batch, n_frames, n_harmonics): harmonnic frequency
        variation (Hz), zero-centered.
        - harmonic_distribution (batch, n_frames, n_harmonics): energy distri-
        bution along all partials.
        - n_samples (int): total length of output audio.
        - sample_rate (int): number of audio samples per second.
        - amp_resample_method (str): mode with which to resample envelopes.
        - use_angular_cumsum (bool): use angular cumulative sum on accumulating
        phase instead of tf.cumsum which is less accurate.
    Returns:
        - audio (batch, n_samples, 1): output audio.
    """
    frequencies = core.tf_float32(frequencies)
    amplitudes = core.tf_float32(amplitudes)

    batch_size = int(frequencies.shape[0])
    n_frames = int(frequencies.shape[1])
    n_samples = upsampling * n_frames

    if harmonic_distribution is not None:
        harmonic_distribution = core.tf_float32(harmonic_distribution)
        n_harmonics = int(harmonic_distribution.shape[-1])
    elif harmonic_shifts is not None:
        harmonic_shifts = core.tf_float32(harmonic_shifts)
        n_harmonics = int(harmonic_shifts.shape[-1])
    else:
        n_harmonics = 1

    # Create harmonic frequencies (batch, n_frames, n_harmonics)
    harmonic_frequencies = core.get_harmonic_frequencies(frequencies,
                                                         n_harmonics)
    if harmonic_shifts is not None:
        harmonic_frequencies *= (1.0 + harmonic_shifts)

    # Create harmonic amplitudes (batch, n_frames, n_harmonics)
    if harmonic_distribution is not None:
        harmonic_amplitudes = amplitudes * harmonic_distribution
    else:
        harmonic_amplitudes = amplitudes

    # Upsample to audio rate
    amplitude_envelopes = core.resample(harmonic_amplitudes, n_samples,
                                        method=amp_resample_method)
    frequency_envelopes = core.resample(harmonic_frequencies, n_samples)

    # Compute real exp decay of complex amplitudes
    if complex_amplitudes is not None:
        complex_amplitudes = core.tf_float32(complex_amplitudes)

        # TODO (lrenault): do not reset timestep count every upsampling rate as
        # this introduces discontinuities. Use new module ala NoteReleaseCell.
        complex_t = tf.range(upsampling, dtype=tf.float32)[tf.newaxis, ..., tf.newaxis]  # (1, upsampling, 1)
        complex_t = tf.tile(complex_t, [batch_size, n_frames, n_harmonics])  # (batch, n_samples, n_harmonics)
        complex_amp_env = tf.repeat(complex_amplitudes, upsampling, axis=1) # (batch, n_samples, n_harmonics)
        complex_amp_env = tf.math.pow(tf.math.sqrt(tf.math.square(complex_amp_env)),
                                      complex_t)

        # Multiply amplitudes deduced from complex with real amplitudes
        amplitude_envelopes *= complex_amp_env

    # Synthesize audio from harmonics (batch, n_samples)
    audio = core.oscillator_bank(frequency_envelopes,
                                 amplitude_envelopes,
                                 sample_rate=sample_rate,
                                 use_angular_cumsum=use_angular_cumsum)
    return audio


class SurrogateAdditive(processors.Processor):
    """Surrogate synthesizer with inharmonic sinusoidal oscillators optimizable
    using complex amplitudes.
    Args:
        - frame_rate (int): number of frame-wise controls per second.
        - sample_rate (int): audio sample rate.
        - min_frequency (int): 
    """
    def __init__(self,
                 frame_rate=250,
                 sample_rate=16000,
                 min_frequency=20,
                 use_amplitude=True,
                 scale_fn=core.exp_sigmoid,
                 normalize_below_nyquist=True,
                 inference=False,
                 name='inharmonic'):
        super(SurrogateAdditive, self).__init__(name=name)
        self.frame_rate = frame_rate
        self.sample_rate = sample_rate
        self.min_frequency = min_frequency
        self.use_amplitude = use_amplitude
        self.scale_fn = scale_fn
        self.normalize_below_nyquist = normalize_below_nyquist
        self.inference = inference
    
    def get_controls(self,
                     amplitudes,
                     complex_amplitudes,
                     harmonic_distribution,
                     inharm_coef,
                     f0_hz):
        """Convert network output tensors into a dictionary of synth controls.
        Args:
            - amplitudes (batch, time, 1): global real amplitude control.
            - complex_amplitudes (batch, time, n_harmonics): 
            - harmonic_distribution (batch, time, n_harmonics): per harmonic
            normalized amplitudes.
            - inharm_coef (batch, time, 1): inhamonicity coefficient.
            - f0_hz (batch, time, 1): fundamental frequency in Hz.
        Returns:
            - controls (Dict): dict of synthesizer controls.
        """
        if self.scale_fn is not None:
            amplitudes = self.scale_fn(amplitudes)
            harmonic_distribution = self.scale_fn(harmonic_distribution)
            complex_amplitudes = self.scale_fn(complex_amplitudes)

        n_harmonics = int(harmonic_distribution.shape[-1])

        inharmonic_freq, harmonic_shifts = get_inharmonic_freq(f0_hz,
                                                               inharm_coef,
                                                               n_harmonics)
        if self.normalize_below_nyquist:
            harmonic_distribution = core.remove_above_nyquist(
                inharmonic_freq,
                harmonic_distribution,
                self.sample_rate
            )
            # Set amplitude to zero if below hearable
            amplitudes *= core.tf_float32(tf.greater(f0_hz,
                                                     self.min_frequency))
        # Normalize
        if self.use_amplitude:
            harmonic_distribution /= tf.reduce_sum(harmonic_distribution,
                                                   axis=-1,
                                                   keepdims=True)
        else:
            amplitudes = tf.ones_like(amplitudes)

        return {'amplitudes': amplitudes,
                'complex_amplitudes': complex_amplitudes,
                'harmonic_distribution': harmonic_distribution,
                'harmonic_shifts': harmonic_shifts,
                'f0_hz': f0_hz}
    
    def get_signal(self,
                   amplitudes,
                   complex_amplitudes,
                   harmonic_distribution,
                   harmonic_shifts,
                   f0_hz):
        signal = complex_harmonic_synthesis(
            frequencies=f0_hz,
            amplitudes=amplitudes,
            complex_amplitudes=complex_amplitudes,
            harmonic_shifts=harmonic_shifts,
            harmonic_distribution=harmonic_distribution,
            upsampling=int(self.sample_rate / self.frame_rate),
            sample_rate=self.sample_rate,
            use_angular_cumsum=self.inference
        )
        return signal
