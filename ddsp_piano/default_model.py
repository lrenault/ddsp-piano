import ddsp
import tensorflow as tf

from ddsp.training.nn import Normalize
from ddsp_piano.data_pipeline import get_dummy_data
from ddsp_piano.modules import PianoModel, sub_modules, losses
from ddsp_piano.modules.inharm_synth import MultiInharmonic

tfkl = tf.keras.layers


def build_polyphonic_processor_group(n_synths=16,
                                     n_substrings=2,
                                     n_piano_models=10,
                                     sample_rate=16000,
                                     duration=3,
                                     reverb_duration=None,
                                     inference=False,
                                     name='processor_group'):
    """ Polyphonic bank of additive + filtered noise synthesizers.
    Args:
        - n_synths (int): number of monophonic synthesizers.
        - n_substrings (int): number of strings per note.
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
    noise = ddsp.synths.FilteredNoise(name='noise', n_samples=n_samples)
    additive = MultiInharmonic(name='additive',
                               n_samples=n_samples,
                               sample_rate=sample_rate,
                               inference=inference)
    # DAG constructor
    dag = []
    dag.append((noise, ['magnitudes_0']))
    dag.append((additive, ['amplitudes_0',
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
              n_substrings=2,
              n_piano_models=10,
              piano_embedding_dim=16,
              n_noise_filter_banks=64,
              frame_rate=250,
              sample_rate=16000,
              reverb_duration=1.5):
    # Self-contained sub-modules
    z_encoder = sub_modules.OneHotZEncoder(n_instruments=n_piano_models,
                                           z_dim=piano_embedding_dim,
                                           n_frames=int(duration * frame_rate))
    note_release = sub_modules.NoteRelease(frame_rate=frame_rate)
    parallelizer = sub_modules.Parallelizer(n_synths=n_synths)
    inharm_model = sub_modules.InharmonicityNetwork()
    detuner = sub_modules.Detuner(n_substrings=n_substrings)
    reverb_model = sub_modules.MultiInstrumentReverb(
        n_instruments=n_piano_models,
        reverb_length=int(reverb_duration * sample_rate)
    )
    # Neural modules
    context_network = sub_modules.ContextNetwork(
        name='context_net',
        layers=[tfkl.Dense(32, activation=tf.nn.leaky_relu),
                tfkl.GRU(64, return_sequences=True),
                Normalize('layer')]
    )
    monophonic_network = sub_modules.MonophonicNetwork(
        name='mono_net',
        layers=[tfkl.Dense(128, activation=tf.nn.leaky_relu),
                tfkl.GRU(192, return_sequences=True),
                tfkl.Dense(192, activation=tf.nn.leaky_relu),
                Normalize('layer')]
    )

    processor_group = build_polyphonic_processor_group(
        n_synths=n_synths,
        n_substrings=n_substrings,
        n_piano_models=n_piano_models,
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
        inharm_model=inharm_model,
        detuner=detuner,
        reverb_model=reverb_model,
        processor_group=processor_group,
        losses=[losses.SpectralLoss(loss_type='L1',
                                    mag_weight=1,
                                    logmag_weight=1,
                                    name='audio_stft_loss'),
                losses.ReverbRegularizer(name='reverb_regularizer')]
    )
    return model


def build_model(model, batch_size=6, first_phase=False, **kwargs):
    """Finish model building by forwarding a batch sample.
    Args:
        - model (PianoModel): declared model to build.
        - batch_size (int): number of samples to handle simulatenously.
        - first_phase (bool): apply first phase training strategy.
    Returns:
        - model (PianoModel): built model.
    """

    # Toggle submodules trainability
    model.alternate_training(first_phase=first_phase)

    # Model building by forwarding a batch sample
    batch = get_dummy_data(batch_size=batch_size, **kwargs)
    _ = model(batch, training=True, return_losses=True)
    # Print model summary
    model.summary()

    return model


if __name__ == '__main__':
    model = build_model(get_model())
