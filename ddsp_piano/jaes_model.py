import ddsp
import tensorflow as tf

from ddsp.training.nn import Normalize
from ddsp_piano.modules import PianoModel, sub_modules, losses
from ddsp_piano.default_model import build_model, \
    build_polyphonic_processor_group

tfkl = tf.keras.layers


def get_model(inference=False,
              duration=3,
              n_synths=16,
              n_substrings=1,
              n_piano_models=1,
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
    inharm_model = sub_modules.ParametricTuning()
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
        reverb_model=reverb_model,
        processor_group=processor_group,
        losses=[losses.SpectralLoss(loss_type='L1',
                                    mag_weight=1,
                                    logmag_weight=1,
                                    name='audio_stft_loss'),
                losses.ReverbRegularizer(name='reverb_regularizer'),
                losses.LoudnessLoss(target_key=f"add_{n_synths - 1}",
                                    synth_key="reverb",
                                    name="reverb_loudness")
                ]
    )
    return model


if __name__ == '__main__':
    model = build_model(get_model())
