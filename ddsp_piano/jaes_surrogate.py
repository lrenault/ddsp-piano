import ddsp
import tensorflow as tf

from ddsp.training.nn import Normalize, DictLayer
from ddsp_piano.modules import PianoModel, sub_modules, inharm_synth, \
    surrogate_synth, losses
from ddsp_piano.default_model import build_model

tfkl = tf.keras.layers


def positive_tanh(x):
    x = ddsp.core.tf_float32(x)
    return 0.5 * (tf.math.tanh(x) + 1.)


def exp_tanh(x, max_value=2., exponent=10., gain=1., threshold=1e-7):
    y = max_value * positive_tanh(gain * x) ** tf.math.log(exponent)
    return y + threshold


class ParallelPolyphonicProcessorGroup(DictLayer):
    """DDSP synthesis without using processor_group and DAGs"""
    def __init__(self,
                 n_synths=16,
                 frame_rate=250,
                 sample_rate=16000,
                 duration=3,
                 reverb_duration=None,
                 inference=False,
                 scale_fn=exp_tanh,
                 **kwargs):
        super().__init__(name="processor_group", **kwargs)
        self.n_synths = n_synths
        self.n_samples = int(duration * sample_rate)
        if reverb_duration is None:
            reverb_duration = duration
        self.reverb_length = int(reverb_duration * sample_rate)

        # Init synthesizers
        self.noise_synth = ddsp.synths.FilteredNoise(name='noise',
                                                     n_samples=self.n_samples,
                                                     scale_fn=scale_fn)
        self.additive_synth = surrogate_synth.SurrogateAdditive(
            name='additive',
            frame_rate=frame_rate,
            sample_rate=sample_rate,
            inference=inference,
            scale_fn=scale_fn,
            normalize_harm_distribution=False)
        self.reverb = ddsp.effects.Reverb(trainable=False,
                                          reverb_length=self.reverb_length)

    def build(self, input_shape):
        super().build(input_shape)
        self.batch_size = int(input_shape[0] / self.n_synths)

    def sum_monophonic_signals(self, signals):
        """Sum multiple signals into a polyphonic signal (taking batch_size
        into account).
        Input:
            - signals (n_synths * batch_size, n_samples, 1): batch of mono audio
        Returns:
            - audio (batch_size, n_samples): batch of polyphonic audio.
        """
        audio = tf.reshape(signals,
                           [self.n_synths, self.batch_size, self.n_samples])
        audio = tf.reduce_sum(audio, axis=0)  # Collapse along polyphony axis
        return audio

    def unparallelize(self, control):
        old_shape = control.shape
        new_shape = tf.concat(
            [[self.n_synths, self.batch_size], old_shape[1:]],
            axis=0
        )
        return tf.reshape(control, new_shape)

    def call(self,
             amplitudes,
             complex_amplitudes,
             complex_time,
             harmonic_distribution,
             inharm_coef,
             f0_hz,
             magnitudes,
             reverb_ir,
             return_outputs_dict=True) -> ['audio_synth']:
        magnitudes = self.unparallelize(magnitudes)
        amplitudes = self.unparallelize(amplitudes)
        complex_amplitudes = self.unparallelize(complex_amplitudes)
        complex_time = self.unparallelize(complex_time)
        harmonic_distribution = self.unparallelize(harmonic_distribution)
        inharm_coef = self.unparallelize(inharm_coef)
        f0_hz = self.unparallelize(f0_hz)

        audio = self.additive_synth(amplitudes[0],
                                    complex_amplitudes[0],
                                    complex_time[0],
                                    harmonic_distribution[0],
                                    inharm_coef[0],
                                    f0_hz[0])
        audio += self.noise_synth(magnitudes[0])
        for i in tf.range(1, self.n_synths):
            audio += self.noise_synth(magnitudes[i])
            audio += self.additive_synth(amplitudes[i],
                                         complex_amplitudes[i],
                                         complex_time[i],
                                         harmonic_distribution[i],
                                         inharm_coef[i],
                                         f0_hz[i])
        audio = self.reverb(audio,
                            reverb_ir)
        return audio


def build_polyphonic_processor_group(n_synths=16,
                                     frame_rate=250,
                                     sample_rate=16000,
                                     duration=3,
                                     reverb_duration=None,
                                     inference=False,
                                     name='processor_group'):
    """ Polyphonic bank of additive + filtered noise synthesizers.
    Args:
        - n_synths (int): number of monophonic synthesizers.
        - sample_rate (int): number of samples per second.
        - duration (float): length of generated sample (in seconds).
        - reverb_length (float): length of the reverb impulse response.
        - inference (bool): synthesis for inference (slower but can handle
        longer sequences).
        - name (string): layer name.
    Returns:
        - processor_group (ProcessorGroup): polyphonic DDSP processor group.
    """
    n_samples = int(duration * sample_rate)
    if reverb_duration is None:
        reverb_duration = duration
    reverb_length = int(reverb_duration * sample_rate)

    # Init synthesizers
    noise = ddsp.synths.FilteredNoise(name='noise', n_samples=n_samples,
                                      scale_fn=exp_tanh)
    additive = surrogate_synth.SurrogateAdditive(name='additive',
                                                 frame_rate=frame_rate,
                                                 sample_rate=sample_rate,
                                                 inference=inference,
                                                 scale_fn=exp_tanh,
                                                 normalize_harm_distribution=False)
    # DAG constructor
    dag = []
    dag.append((noise, ['magnitudes_0']))
    dag.append((additive, ['amplitudes_0',
                           'complex_amplitudes_0',
                           'complex_time_0',
                           'harmonic_distribution_0',
                           'inharm_coef_0',
                           'f0_hz_0']))
    dag.append((ddsp.processors.Add(name='add_0'),
                ['noise/signal',
                 'additive/signal']))
    # Construct synth polyphony
    for i in range(1, n_synths):
        # Synthesize monophonic additive component
        dag.append((additive, [f'amplitudes_{i}',
                               f'complex_amplitudes_{i}',
                               f'complex_time_{i}',
                               f'harmonic_distribution_{i}',
                               f'inharm_coef_{i}',
                               f'f0_hz_{i}']))
        # Synthesize and add filtered noise component
        dag.append((noise, [f'magnitudes_{i}']))
        dag.append((ddsp.processors.Add(name=f'sub_add_{i}'),
                    ['noise/signal',
                     'additive/signal']))
        # Add monophonic signal to the polyphonic signal
        dag.append((ddsp.processors.Add(name=f'add_{i}'),
                    [f'add_{i - 1}/signal',
                     f'sub_add_{i}/signal']))

    # Reverb module
    dag.append((ddsp.effects.Reverb(trainable=False,
                                    reverb_length=reverb_length),
                [f'add_{n_synths - 1}/signal',
                 'reverb_ir']))

    # Compile dag into a processor group
    processor_group = ddsp.processors.ProcessorGroup(dag=dag, name=name)

    return processor_group


def get_model(inference=False,
              duration=3,
              n_synths=16,
              n_substrings=1,
              n_partials=None,
              n_piano_models=1,
              piano_embedding_dim=16,
              n_noise_filter_banks=64,
              frame_rate=250,
              sample_rate=16000,
              reverb_duration=1.):
    # Self-contained sub-modules
    z_encoder = sub_modules.OneHotZEncoder(n_instruments=n_piano_models,
                                           z_dim=piano_embedding_dim,
                                           n_frames=int(duration * frame_rate))
    note_release = sub_modules.NoteRelease(frame_rate=frame_rate)
    parallelizer = sub_modules.Parallelizer(n_synths=n_synths,
                                            mono_keys=('f0_hz',
                                                       'inharm_coef',
                                                       'amplitudes',
                                                       'complex_amplitudes',
                                                       'complex_time',
                                                       'harmonic_distribution',
                                                       'magnitudes'))
    detuner = sub_modules.DeepDetuner(n_substrings=n_substrings)
    # inharm_model = sub_modules.DeepInharmonicity()
    inharm_model = sub_modules.InharmonicityNetwork()
    # inharm_model = sub_modules.ParametricTuning()
    surrogate_module = sub_modules.SurrogateModule()
    harmonic_masking = sub_modules.PartialMasking(n_partials=n_partials)
    reverb_model = sub_modules.MultiInstrumentReverb(
        n_instruments=n_piano_models,
        reverb_length=int(reverb_duration * sample_rate)
    )
    # Neural modules
    context_network = sub_modules.ContextNetwork(
        name='context_net',
        normalize_pitch=True,
        layers=[tfkl.Dense(16, activation=tf.nn.leaky_relu),  # 32
                tfkl.GRU(16, return_sequences=True),  # 64
                Normalize('layer'),
                ]
    )
    monophonic_network = sub_modules.MonophonicNetwork(
        name='mono_net',
        layers=[tfkl.Dense(128, activation=tf.nn.leaky_relu),
                Normalize('layer'),
                tfkl.GRU(128, return_sequences=True),
                tfkl.Dense(128, activation=tf.nn.leaky_relu),
                ],
        output_splits=(('amplitudes', 1),
                       ('harmonic_distribution', 96),
                       ('magnitudes', 4))
    )
    processor_group = build_polyphonic_processor_group(
        n_synths=n_synths,
        frame_rate=frame_rate,
        sample_rate=sample_rate,
        duration=duration,
        reverb_duration=reverb_duration,
        inference=inference
    )
    # Full piano model definition
    model = PianoModel(
        z_encoder=z_encoder,
        note_release=note_release,
        context_network=context_network,
        parallelizer=parallelizer,
        monophonic_network=monophonic_network, 
        # monophonic_network=sub_modules.MonophonicDeepNetwork(ch=128, rnn_channels=128),
        detuner=detuner,
        inharm_model=inharm_model,
        surrogate_module=surrogate_module,
        # harmonic_masking=harmonic_masking,
        reverb_model=reverb_model,
        processor_group=processor_group,
        # ddsp_synths=ParallelPolyphonicProcessorGroup(),
        losses=[losses.SpectralLoss(loss_type='L1',
                                    mag_weight=1,
                                    logmag_weight=1,
                                    name='audio_stft_loss'),
                losses.ReverbRegularizer(name='reverb_regularizer'),
                losses.InharmonicityLoss(name='inharmonicity_regularizer')]
    )
    return model


if __name__ == '__main__':
    import os; os.environ['CUDA_VISIBLE_DEVICES'] = "-1"
    model = build_model(get_model(), first_phase=True)
    import pdb; pdb.set_trace()
