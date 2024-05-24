import gin
import tensorflow as tf
from ddsp import core, processors
from ddsp_piano.modules.inharm_synth import get_inharmonic_freq, cos_oscillator_bank


def exists(x):
    return x is not None


def surrogate_harmonic_synthesis(frequencies,
                                 amplitudes,
                                 decays=None,
                                 decay_time=None,
                                 harmonic_shifts=None,
                                 harmonic_distribution=None,
                                 upsampling=64,
                                 sample_rate=16000,
                                 amp_resample_method='window',
                                 use_angular_cumsum=False):
    """Surrogate bank of quasi-harmonic oscillators using decaying amplitudes
    for allowing frequency optimzation in the complex circle
    (cf B. Hayes - Sinusoidal Frequency Estimation by Gradient Descent).

    Args:
        - frequencies (batch, n_frames, 1): frame-wise fundamental frequency
        in Hz.
        - amplitudes (batch, n_frames, 1): frame-wise oscillator real amplitude
        - decays (batch, n_frames, n_harmonics): frame-wise decay factor.
        - decay_time (batch, n_frames, 1): frame wise timesteps for
        each 
        - harmoninc_shifts (batch, n_frames, n_harmonics): harmonnic frequency
        variation (Hz), zero-centered.
        - harmonic_distribution (batch, n_frames, n_harmonics): energy distri-
        bution along all partials.
        - upsampling (int): ratio between control rate and sample rate.
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

    if exists(harmonic_distribution):
        harmonic_distribution = core.tf_float32(harmonic_distribution)
        n_harmonics = int(harmonic_distribution.shape[-1])
    elif exists(harmonic_shifts):
        harmonic_shifts = core.tf_float32(harmonic_shifts)
        n_harmonics = int(harmonic_shifts.shape[-1])
    else:
        n_harmonics = 1

    # Create (in-)harmonic frequencies (batch, n_frames, n_harmonics)
    harmonic_frequencies = core.get_harmonic_frequencies(frequencies,
                                                         n_harmonics)
    if exists(harmonic_shifts):
        harmonic_frequencies *= (1.0 + harmonic_shifts)

    # Create harmonic amplitudes (batch, n_frames, n_harmonics)
    if exists(harmonic_distribution):
        harmonic_amplitudes = amplitudes * harmonic_distribution
    else:
        harmonic_amplitudes = amplitudes

    # Upsample to audio rate
    frequency_envelopes = core.resample(harmonic_frequencies, n_samples)
    amplitude_envelopes = core.resample(harmonic_amplitudes, n_samples,
                                        method=amp_resample_method)

    # Compute real exponential decay of complex amplitudes
    if (exists(decays)) and (exists(decay_time)):
        # Convert to float
        decays = core.tf_float32(decays)
        decay_time = core.tf_float32(decay_time)

        # Upsampling frame rate to sample rate by repeating
        decay_env = tf.repeat(decays, upsampling, axis=1)
        decay_time_upsampled = tf.repeat(decay_time, upsampling, axis=1) \
                               * upsampling
        # Complete the sample-wise time count by incrementing
        upsampling_range = tf.range(upsampling, dtype=tf.float32)
        upsampling_range = tf.tile(upsampling_range[tf.newaxis, ..., tf.newaxis],
                                   [batch_size, n_frames, n_harmonics])
        decay_time_upsampled += upsampling_range
        
        decay_env = tf.math.pow(tf.math.abs(decay_env),
                                decay_time_upsampled)

        # Multiply amplitudes deduced from complex with real amplitudes
        amplitude_envelopes *= decay_env

    # Synthesize audio from harmonics (batch, n_samples)
    audio = cos_oscillator_bank(frequency_envelopes,
                                amplitude_envelopes,
                                sample_rate=sample_rate,
                                use_angular_cumsum=use_angular_cumsum)
    return audio


@gin.register
class SurrogateAdditive(processors.Processor):
    """Surrogate synthesizer with inharmonic sinusoidal oscillators optimizable
    using decaying amplitudes.
    Args:
        - frame_rate (int): number of frame-wise controls per second.
        - sample_rate (int): audio sample rate.
        - min_frequency (int): 
    """
    def __init__(self,
                 frame_rate=250,
                 sample_rate=16000,
                 min_frequency=20,
                 normalize_harm_distribution=True,
                 scale_fn=core.exp_sigmoid,
                 normalize_below_nyquist=True,
                 inference=False,
                 name='inharmonic'):
        super().__init__(name=name)
        self.frame_rate = frame_rate
        self.sample_rate = sample_rate
        self.min_frequency = min_frequency
        self.normalize_harm_distribution = normalize_harm_distribution
        self.scale_fn = scale_fn
        self.normalize_below_nyquist = normalize_below_nyquist
        self.inference = inference
    
    def get_controls(self,
                     amplitudes,
                     decays,
                     decay_time,
                     harmonic_distribution,
                     inharm_coef,
                     f0_hz):
        """Convert network output tensors into a dictionary of synth controls.
        Args:
            - amplitudes (batch, time, 1): global real amplitude control.
            - decays (batch, time, n_harmonics): 
            - harmonic_distribution (batch, time, n_harmonics): per harmonic
            normalized amplitudes.
            - inharm_coef (batch, time, 1): inhamonicity coefficient.
            - f0_hz (batch, time, 1): fundamental frequency in Hz.
        Returns:
            - controls (Dict): dict of synthesizer controls.
        """
        if exists(self.scale_fn):
            amplitudes = self.scale_fn(amplitudes)
            harmonic_distribution = self.scale_fn(harmonic_distribution)

        # Set inharmonicity to positive values
        inharm_coef = tf.math.maximum(inharm_coef, 0.)
        n_harmonics = int(harmonic_distribution.shape[-1])
        inharmonic_freq, harmonic_shifts = get_inharmonic_freq(f0_hz,
                                                               inharm_coef,
                                                               n_harmonics)
        # Clip decay values to prevent explosion
        if exists(decays):
            decays = tf.math.minimum(decays, 1.)
            decays = tf.math.maximum(decays, 1e-5)
            # Clip above Nyquist
            decays = tf.where(
                tf.greater_equal(inharmonic_freq, self.sample_rate / 2.),
                tf.ones_like(decays),
                decays,
            )
        if self.normalize_below_nyquist:
            # Remove harmonics above Nyquist
            harmonic_distribution = core.remove_above_nyquist(
                inharmonic_freq,
                harmonic_distribution,
                self.sample_rate
            )
            # Set global amplitude to zero if below hearable
            amplitudes *= core.tf_float32(tf.greater(f0_hz,
                                                     self.min_frequency))
        # Normalize
        if self.normalize_harm_distribution:
            harmonic_distribution = core.safe_divide(
                harmonic_distribution,
                tf.reduce_sum(harmonic_distribution, axis=-1, keepdims=True)
            )

        return {'amplitudes': amplitudes,
                'decays': decays,
                'decay_time': decay_time,
                'harmonic_distribution': harmonic_distribution,
                'harmonic_shifts': harmonic_shifts,
                'f0_hz': f0_hz}
    
    def get_signal(self,
                   amplitudes,
                   decays,
                   decay_time,
                   harmonic_distribution,
                   harmonic_shifts,
                   f0_hz):
        signal = surrogate_harmonic_synthesis(
            frequencies=f0_hz,
            amplitudes=amplitudes,
            decays=decays,
            decay_time=decay_time,
            harmonic_shifts=harmonic_shifts,
            harmonic_distribution=harmonic_distribution,
            upsampling=int(self.sample_rate / self.frame_rate),
            sample_rate=self.sample_rate,
            use_angular_cumsum=self.inference
        )
        return signal
