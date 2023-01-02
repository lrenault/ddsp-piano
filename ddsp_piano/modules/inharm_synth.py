import tensorflow as tf

from ddsp import core
from ddsp import processors


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


class InHarmonic(processors.Processor):
    """Synthesize audio with a bank of inharmonic sinusoidal oscillators.
    Args:
        - n_samples (int): number of audio samples to generate.
        - sample_rate (int): sample per second.
        - min_frequency (int): minimum supported frequency (in Hz).
        - use_amplitude (bool): use global amplitude or enable free harmonic
        amplitudes.
        - scale_fn (fn): scaliing function for network output post-processing.
        - normalize_below_nyquist (bool): set amplitude of frequencies abow
        Nyquist to 0.
        - inference (bool): use angular cumsum (for inference only).
    """

    def __init__(self,
                 n_samples=64000,
                 sample_rate=16000,
                 min_frequency=20,
                 use_amplitude=True,
                 scale_fn=core.exp_sigmoid,
                 normalize_below_nyquist=True,
                 inference=False,
                 name='inharmonic'):
        self.n_samples = n_samples
        self.sample_rate = sample_rate
        self.min_frequency = min_frequency
        self.use_amplitude = use_amplitude
        self.scale_fn = scale_fn
        self.normalize_below_nyquist = normalize_below_nyquist
        self.inference = inference
        super(InHarmonic, self).__init__(name=name)

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
        # Scale the amplitudes
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
        if self.use_amplitude:
            harmonic_distribution = core.safe_divide(
                harmonic_distribution,
                tf.reduce_sum(harmonic_distribution, axis=-1, keepdims=True)
            )
        else:
            amplitudes = tf.ones_like(amplitudes)

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
        signal = core.harmonic_synthesis(
            frequencies=f0_hz,
            amplitudes=amplitudes,
            harmonic_shifts=harmonic_shifts,
            harmonic_distribution=harmonic_distribution,
            n_samples=self.n_samples,
            sample_rate=self.sample_rate,
            use_angular_cumsum=self.inference
        )
        return signal


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
