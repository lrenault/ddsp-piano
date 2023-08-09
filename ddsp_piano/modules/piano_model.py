import tensorflow as tf
import ddsp

from ddsp.training.models.model import Model


class PianoModel(Model):
    """DDSP model for piano synthesis from MIDI conditioning.
    Args:
        - z_encoder (nn.DictLayer): one-hot piano model embeddings.
        - note_release (nn.DictLayer): extend active pitch conditioning.
        - context_network (nn.DictLayer): context vector computation model from
        global inputs.
        - parallelizer (nn.DictLayer): layer managing polyphony and batch axis
        merge and unmerge.
        - monophonic_network (nn.DictLayer): monophonic string model as
        neural network.
        - surrogate_module (nn.DictLayer): compute complex amp modulus and time
        for optimizng with the surrogate synthesis.
        - inharm_model (nn.DictLayer): inharmonicity model over tessitura.
        - detuner (nn.DictLayer): tuning model for pitch to absolute f0
        frequency.
        - reverb_model (nn.DictLayer): recording environment impulse responses.
        - processor_group (ddsp.processors.ProcessorGroup): group of
        differentiable processors generating audio from controls.
        - losses (ddsp.losses.Loss): list of loss functions.
    """

    def __init__(self,
                 z_encoder=None,
                 note_release=None,
                 context_network=None,
                 parallelizer=None,
                 monophonic_network=None,
                 surrogate_module=None,
                 inharm_model=None,
                 detuner=None,
                 harmonic_masking=None,
                 background_noise_model=None,
                 reverb_model=None,
                 processor_group=None,
                 ddsp_synths=None,
                 losses=None,
                 **kwargs):
        super(PianoModel, self).__init__(**kwargs)
        self.z_encoder = z_encoder
        self.note_release = note_release
        self.context_network = context_network
        self.parallelizer = parallelizer
        self.monophonic_network = monophonic_network
        self.surrogate_module = surrogate_module
        self.inharm_model = inharm_model
        self.detuner = detuner
        self.harmonic_masking = harmonic_masking
        self.background_noise_model = background_noise_model
        self.reverb_model = reverb_model
        self.processor_group = processor_group
        self.ddsp_synths = ddsp_synths

        self.loss_objs = ddsp.core.make_iterable(losses)

    @property
    def n_synths(self):
        return self.parallelizer.n_synths

    @property
    def sample_rate(self):
        return self.processor_group.processors[0].sample_rate

    def _update_losses_dict(self, loss_objs, *args, **kwargs):
        super(PianoModel, self)._update_losses_dict(loss_objs,
                                                    *args,
                                                    **kwargs)
        self._losses_dict['regularization_loss'] = tf.reduce_sum(self.losses)

    def alternate_training(self, first_phase=True):
        """Toggle trainability of submodules for the 1st or 2nd training phase.
        Args:
            - first_phase (bool): whether using the 1st phase training strategy
        """
        # Modules involved with partial frequency computing are frozen during
        # the first training phase strategy.
        for module in [self.inharm_model,
                       self.detuner,
                       self.surrogate_module]:
            if module is not None:
                module.trainable = not first_phase

        self.z_encoder.alternate_training(first_phase)

        # Modules not involved in freq computing have inversed trainability
        for module in [self.note_release,
                       self.context_network,
                       self.background_noise_model,
                       self.monophonic_network,
                       self.reverb_model]:
            if module is not None:
                module.trainable = first_phase

        # Compute multiple note signals only when learning detuner weights
        if self.detuner is not None:
            self.detuner.use_detune = not first_phase

    def all_trainable(self, trainable=True):
        for module in [self.inharm_model,
                       self.detuner,
                       self.surrogate_module,
                       self.note_release,
                       self.context_network,
                       self.monophonic_network,
                       self.reverb_model]:
            if module is not None:
                module.trainable = trainable

    def compute_global_features(self, features, training):
        """Call all modules computing global features."""
        for sub_module in [self.z_encoder,
                           self.context_network,
                           self.background_noise_model,
                           self.reverb_model]:
            if sub_module is not None:
                features.update(sub_module(features, training=training))

        return features

    def compute_monophonic_features(self, features, training):
        """Call all modules computing monophonic features."""
        for sub_module in [self.note_release,
                           self.inharm_model,
                           self.detuner,
                           self.monophonic_network,
                           self.surrogate_module,
                           self.harmonic_masking]:
            if sub_module is not None:
                features.update(sub_module(features, training=training))

        return features

    def get_audio_from_outputs(self, outputs):
        """Extract audio output tensor from outputs dict of call()."""
        return outputs['audio_synth']

    def call(self, features, training=False):
        # Compute global features
        features = self.compute_global_features(features, training=training)

        # Merge batch axis with polyphony axis for parallelized computation
        features = self.parallelizer(features, parallelize=True)

        # Compute monophonic features
        features = self.compute_monophonic_features(features, training=training)

        if self.ddsp_synths is None:
            # Disentangle polyphony and batch axis
            features = self.parallelizer(features, parallelize=False)

            # Processor group call
            pg_out = self.processor_group(features, return_outputs_dict=True)

            # Parse outputs
            outputs = pg_out['controls']
            outputs['audio_synth'] = pg_out['signal']

            if training:
                self._update_losses_dict(self.loss_objs, outputs)

        else:
            features.update(self.ddsp_synths(features))
            outputs = features
            if training:
                self._update_losses_dict(self.loss_objs, features)

        return outputs
