import tensorflow as tf
from ddsp import core, processors
from ddsp_piano.modules.surrogate_synth import SurrogateAdditive
from ddsp_piano.modules.inharm_synth import harmonic_synthesis


def scale_mix_fn(x, exponent=20):
    y = tf.math.pow(x, exponent)
    return y


def sustained_complex_harmonic_synthesis(frequencies,
                                         amplitudes,
                                         complex_amplitudes=None,
                                         harmonic_distribution=None,
                                         harmonic_shifts=None,
                                         upsampling=64,
                                         sample_rate=16000,
                                         amp_resample_method='window',
                                         use_angular_cumsum=False):
    """ Sustained surrogate bank of quasi-harmonic oscillators. The
    complex_amplitudes control the thickness of the modulated sin waves.
    Args:
        - frequencies (batch, n_frames, 1): frame-wise fundamental frequency.
        - amplitudes (batch, n_frames, 1):
        - complex_amplitude (batch, n_frames, n_harmonincs)
        - harmonic_shifts (batch, n_frames, n_harmonics)
        - harmonic_distribution (batch, n_frames, n_harmonics)
        - upsampling (int): upsampling ratio between sample rate & sample rate
        - sample_rate (int):
        - amp_resample_method (str)
        - use_angular_cumsum (bool)
    Returns:
        - audio (batch, n_samples, 1): real output audio.
    """
    n_frames = frequencies.shape[1]
    n_samples = upsampling * n_frames

    pure_sinusoids = harmonic_synthesis(frequencies=frequencies,
                                        amplitudes=amplitudes,
                                        harmonic_shifts=harmonic_shifts,
                                        harmonic_distribution=harmonic_distribution,
                                        n_samples=n_samples,
                                        sample_rate=sample_rate,
                                        amp_resample_method=amp_resample_method,
                                        sum_sinusoids=False,
                                        use_angular_cumsum=use_angular_cumsum)
    modulation_noise = filtered_noise()

    return audio


class SustainedSurrogateSynth(SurrogateAdditive):
    """Surrogate synthesizer without exponentially decaying amplitude."""
    def __init__(self, mix_fn=scale_mix_fn, **kwargs):
        super(SustainedSurrogateSynth, self).__init__(**kwargs)
        self.mix_fn = mix_fn

    def get_controls(self, amplitudes, complex_amplitudes,
                     harmonic_distribution, inharm_coef, f0_hz):
        controls = super(SustainedSurrogateSynth, self).get_controls(
            amplitudes=amplitudes,
            complex_amplitudes=complex_amplitudes,
            complex_time=None,
            harmonic_distribution=harmonic_distribution,
            inharm_coef=inharm_coef,
            f0_hz=f0_hz
        )
        controls['complex_amplitudes'] = self.mix_fn(controls['complex_amplitudes'])
        return controls

    def get_signal(self, amplitudes, complex_amplitudes,
                   harmonic_distribution, harmonic_shifts, f0_hz):
        signal = sustained_complex_harmonic_synthesis(
            frequencies=f0_hz,
            amplitudes=amplitudes,
            complex_amplitudes=complex_amplitudes,
            harmonic_shifts=harmonic_shifts,
            harmonic_distribution=harmonic_distribution,
            upsampling=int(self.self.sample_rate / self.frame_rate),
            sample_rate=self.sample_rate,
            use_angular_cumsum=self.inference
        )
        return signal


class OnePoleFilterCell(tf.keras.layers.Layer):
    """RNN cell for first-order filtering with time-varying gain and filtering
    coefficient. Implemented as recommended in "Differentiable IIR Filters for
    Machine Learning applications", Kuznetsov et al., DAFx2019.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.state_size = 1

    @tf.function
    def call(self, inputs, previous_sample):
        # sample, gain, filter_coef = inputs
        sample = inputs[..., 0]
        gain = inputs[..., 1]
        filter_coef = inputs[..., 2]
        filtered_sample = gain * sample + filter_coef * previous_sample[0]
        return filtered_sample, [filtered_sample]


if __name__ == "__main__":
    import os; os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    from ddsp_piano.utils.io_utils import load_audio_as_signal
    from soundfile import write

    audio = load_audio_as_signal(
        "/data3/anasynth_nonbp/renault/cache/maps_debussy.wav"
    )[tf.newaxis, :16000*3, tf.newaxis]
    filt = tf.keras.layers.RNN(OnePoleFilterCell(), return_sequences=True)

    n_samples = audio.shape[1]

    gain = 1. * tf.ones((1, n_samples, 1))
    filter_coef = 0.8 * tf.ones((1, n_samples, 1))

    filtered_audio = filt(tf.stack([audio, gain, filter_coef], axis=-1))

    write("/data3/anasynth_nonbp/renault/cache/maps_debussy_filtered.wav",
          data=filtered_audio[0].numpy(),
          samplerate=16000)
