import gin
import tensorflow as tf
from numpy import pi
from ddsp import core
from ddsp import processors
from ddsp.synths import FilteredNoise
from priv_ddfx.effects import DelayNetwork

gin.external_configurable(DelayNetwork)


def positive_tanh(x):
    x = core.tf_float32(x)
    return 0.5 * (tf.math.tanh(x) + 1.)


@gin.register
def exp_tanh(x, max_value=2., exponent=10., gain=1., threshold=1e-7):
    # A varation of the exp_sigmoid function with tanh as it saturates faster
    y = max_value * positive_tanh(gain * x) ** tf.math.log(exponent)
    return y + threshold


def get_inharmonic_freq(f0_hz, inharm_coef, n_harmonics):
    """ Create inharmonic multiples of the fundamental frequency and provide
    deviations from pure harmonic frequencies.
    Args:
        - f0_hz (batch, :, 1): fundamental frequencies.
        - inharm_coef (batch, :, 1): inharmonicity coefficients.
        - n_harmonics (int): number of harmonics.
    Returns:
        - inharmonic_freq (batch, :, n_harmonics): oscillators
        frequencies in Hz.
        - harmonic_shifts (batch, :, n_harmonics): deviation from pure integer
        factor harmonicity.
    """
    f0_hz = core.tf_float32(f0_hz)

    # Integer ratios
    int_multiplier = tf.linspace(1.0, float(n_harmonics), int(n_harmonics))
    int_multiplier = int_multiplier[tf.newaxis, tf.newaxis, :]

    # Inharmonicity factor
    inharm_factor = tf.math.pow(int_multiplier, 2)
    inharm_factor = inharm_factor * inharm_coef + 1.
    inharm_factor = tf.math.sqrt(inharm_factor)

    # Modal frequencies
    inharmonic_freq = f0_hz * int_multiplier * inharm_factor
    # Shifts
    harmonic_shifts = inharm_factor - 1.

    return inharmonic_freq, harmonic_shifts


def cos_oscillator_bank(frequency_envelopes,
                        amplitude_envelopes,
                        sample_rate=16000,
                        sum_sinusoids=True,
                        use_angular_cumsum=False):
    frequency_envelopes = core.tf_float32(frequency_envelopes)
    amplitude_envelopes = core.tf_float32(amplitude_envelopes)

    # Don't exceed Nyquist.
    amplitude_envelopes = core.remove_above_nyquist(frequency_envelopes,
                                                    amplitude_envelopes,
                                                    sample_rate)
    # Angular frequency, Hz -> radians per sample.
    omegas = frequency_envelopes * (2.0 * pi)  # rad / sec
    omegas = omegas / float(sample_rate)  # rad / sample

    # Accumulate phase and synthesize.
    if use_angular_cumsum:
        # Avoids accumulation errors.
        phases = core.angular_cumsum(omegas)
    else:
        phases = tf.cumsum(omegas, axis=1)

    # Convert to waveforms.
    wavs = tf.cos(phases)
    audio = amplitude_envelopes * wavs  # [mb, n_samples, n_sinusoids]
    if sum_sinusoids:
        audio = tf.reduce_sum(audio, axis=-1)  # [mb, n_samples]
    return audio


def harmonic_synthesis(frequencies,
                       amplitudes,
                       harmonic_shifts=None,
                       harmonic_distribution=None,
                       n_samples=64000,
                       sample_rate=16000,
                       amp_resample_method='window',
                       sum_sinusoids=True,
                       use_angular_cumsum=False):
    frequencies = core.tf_float32(frequencies)
    amplitudes = core.tf_float32(amplitudes)

    if harmonic_distribution is not None:
        harmonic_distribution = core.tf_float32(harmonic_distribution)
        n_harmonics = int(harmonic_distribution.shape[-1])
    elif harmonic_shifts is not None:
        harmonic_shifts = core.tf_float32(harmonic_shifts)
        n_harmonics = int(harmonic_shifts.shape[-1])
    else:
        n_harmonics = 1

    # Create harmonic frequencies [batch_size, n_frames, n_harmonics].
    harmonic_frequencies = core.get_harmonic_frequencies(frequencies, n_harmonics)
    if harmonic_shifts is not None:
        harmonic_frequencies *= (1.0 + harmonic_shifts)

    # Create harmonic amplitudes [batch_size, n_frames, n_harmonics].
    if harmonic_distribution is not None:
        harmonic_amplitudes = amplitudes * harmonic_distribution
    else:
        harmonic_amplitudes = amplitudes

    # Create sample-wise envelopes.
    frequency_envelopes = core.resample(harmonic_frequencies, n_samples)  # cycles/sec
    amplitude_envelopes = core.resample(harmonic_amplitudes, n_samples,
                                        method=amp_resample_method)

    # Synthesize from harmonics [batch_size, n_samples].
    audio = cos_oscillator_bank(frequency_envelopes,
                                amplitude_envelopes,
                                sample_rate=sample_rate,
                                sum_sinusoids=sum_sinusoids,
                                use_angular_cumsum=use_angular_cumsum)
    return audio


@gin.register
class InHarmonic(processors.Processor):
    """Synthesize audio with a bank of inharmonic sinusoidal oscillators.
    Args:
        - n_samples (int): number of audio samples to generate.
        - sample_rate (int): sample per second.
        - min_frequency (int): minimum supported frequency (in Hz).
        - scale_fn (fn): scaling function for network outputs post-processing.
        - normalize_harm_distribution (bool): whether to force the sum of
        partial energy to 1.
        - normalize_below_nyquist (bool): set amplitude of frequencies abow
        Nyquist to 0.
        - inference (bool): use angular cumsum (for inference only).
    """

    def __init__(self,
                 frame_rate=250,
                 sample_rate=16000,
                 min_frequency=20,
                 scale_fn=core.exp_sigmoid,
                 normalize_harm_distribution=True,
                 normalize_below_nyquist=True,
                 inference=False,
                 name='inharmonic'):
        self.frame_rate = frame_rate
        self.sample_rate = sample_rate
        self.min_frequency = min_frequency
        self.normalize_harm_distribution = normalize_harm_distribution
        self.scale_fn = scale_fn
        self.normalize_below_nyquist = normalize_below_nyquist
        self.inference = inference
        super(InHarmonic, self).__init__(name=name)

    @property
    def upsampling(self):
        return int(self.sample_rate / self.frame_rate)

    def get_controls(self,
                     amplitudes,
                     harmonic_distribution,
                     inharm_coef,
                     f0_hz):
        """ Convert network output tensors into dict of synth controls.
        Args:
            - amplitudes (batch, time, 1): global amplitude control.
            - harmonic_distribution (batch, time, n_harmonics): per harmonic
            normalized amplitudes.
            - inharm_coef (batch, time, 1): inharmonicity coefficient.
            - f0_hz (batch, time, 1): fundamental frequency in hz.
        Returns:
            - controls (Dict): dict of synthesizer controls
        """
        # Scale network outputs to positive values
        inharm_coef = tf.math.maximum(inharm_coef, 0.)
        if self.scale_fn is not None:
            amplitudes = self.scale_fn(amplitudes)
            harmonic_distribution = self.scale_fn(harmonic_distribution)

        n_harmonics = int(harmonic_distribution.shape[-1])

        inharmonic_freq, harmonic_shifts = get_inharmonic_freq(
            f0_hz, inharm_coef, n_harmonics
        )
        # Bandlimit the harmonic distribution
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
        if self.normalize_harm_distribution:
            harmonic_distribution = core.safe_divide(
                harmonic_distribution,
                tf.reduce_sum(harmonic_distribution, axis=-1, keepdims=True)
            )

        return {'amplitudes': amplitudes,
                'harmonic_distribution': harmonic_distribution,
                'harmonic_shifts': harmonic_shifts,
                'f0_hz': f0_hz}

    def get_signal(self,
                   amplitudes,
                   harmonic_distribution,
                   harmonic_shifts,
                   f0_hz):
        """ Synthesize audio with inharmonic synthesizer from controls.
        Args:
            - amplitudes (batch, time, 1): global amplitude.
            - harmonic_distribution (batch, time, n_harmonics): harmonics
            relative amplitudes (sums to 1).
            - harmonic_shifts (batch, time, n_harmonics): harmonic shifts
            from perfect harmonic frequencies.
            - f0_hz (batch, time, 1): fundamental frequency, in Hz.
        """
        signal = harmonic_synthesis(
            frequencies=f0_hz,
            amplitudes=amplitudes,
            harmonic_shifts=harmonic_shifts,
            harmonic_distribution=harmonic_distribution,
            n_samples=self.upsampling * f0_hz.shape[1],
            sample_rate=self.sample_rate,
            use_angular_cumsum=self.inference
        )
        return signal


@gin.register
class MultiInharmonic(InHarmonic):
    """Inharmonic synthesizer with multiple F0 controls."""

    def __init__(self, name="multi_inharmonic", **kwargs):
        super(MultiInharmonic, self).__init__(name=name, **kwargs)

    def get_controls(self,
                     amplitudes,
                     harmonic_distribution,
                     inharm_coef,
                     f0_hz):
        # Get partial amplitudes and inharmonicity displacement
        controls = super(MultiInharmonic, self).get_controls(
            amplitudes,
            harmonic_distribution,
            inharm_coef,
            f0_hz[..., 0:1]
        )
        # Put back multi-f0 signal
        controls['f0_hz'] = f0_hz
        # Divide global amplitude by the number of substrings
        controls['amplitudes'] /= core.tf_float32(f0_hz.shape[-1])
        return controls

    def get_signal(self,
                   amplitudes,
                   harmonic_distribution,
                   harmonic_shifts,
                   f0_hz):
        n_substrings = f0_hz.shape[-1]
        # Audio from the first substring
        audio = super(MultiInharmonic, self).get_signal(
            amplitudes,
            harmonic_distribution,
            harmonic_shifts,
            f0_hz[..., 0:1]
        )
        # Add other substrings signals
        for substring in range(1, n_substrings):
            audio += super(MultiInharmonic, self).get_signal(
                amplitudes,
                harmonic_distribution,
                harmonic_shifts,
                f0_hz[..., substring: substring + 1]
            )
        return audio


@gin.register
class DynamicSizeFilteredNoise(FilteredNoise):
    """White noise filtering synthesis with arbitrary output length controls."""
    def __init__(self, frame_rate=250, sample_rate=16000, **kwargs):
        super().__init__(**kwargs)
        self.frame_rate = frame_rate
        self.sample_rate = sample_rate

    @property
    def upsampling(self):
        return int(self.sample_rate / self.frame_rate)

    def get_signal(self, magnitudes):
        """Synthesize audio with filtered white noise.
        Args:
          magnitudes: Magnitudes tensor of shape [batch, n_frames, n_filter_banks].
            Expects float32 that is strictly positive.
        Returns:
          signal: A tensor of harmonic waves of shape [batch, n_samples, 1].
        """
        batch_size, n_frames, *axis = magnitudes.shape

        n_samples = self.upsampling * n_frames

        signal = tf.random.uniform([batch_size, n_samples],
                                   minval=-1., maxval=1.)
        return core.frequency_filter(signal, magnitudes,
                                     window_size=self.window_size)


@gin.register
class MultiAdd(processors.Processor):
    """Sum arbitrary number of signals."""
    def __init__(self, name='add'):
        super(MultiAdd, self).__init__(name=name)
    
    def get_controls(self, *signals):
        controls = {}
        for i, signal in enumerate(signals):
            controls[f"signal_{i}"] = signal
        return controls

    def get_signal(self, **signals):
        return sum(signals.values())
