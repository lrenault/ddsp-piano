import tensorflow as tf
import ddsp
from ddsp.training.models.model import Model


class PianoModel(Model):
    """DDSP model for piano synthesis from MIDI conditioning"""

    def __init__(self,
                 z_encoder=None,
                 note_release=None,
                 context_network=None,
                 parallelizer=None,
                 monophonic_network=None,
                 inharm_model=None,
                 detuner=None,
                 reverb_model=None,
                 processor_group=None,
                 losses=None,
                 **kwargs):
        super(PianoModel, self).__init__(**kwargs)
        self.z_encoder = z_encoder
        self.note_release = note_release
        self.context_network = context_network
        self.parallelizer = parallelizer
        self.monophonic_network = monophonic_network
        self.inharm_model = inharm_model
        self.detuner = detuner
        self.reverb_model = reverb_model
        self.processor_group = processor_group

        self.loss_objs = ddsp.core.make_iterable(losses)

    @property
    def n_synths(self):
        return self.parallelizer.n_synths

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
                       self.detuner]:
            if module is not None:
                module.trainable = not first_phase

        self.z_encoder.alternate_training(first_phase)

        # Modules not involved in freq computing have inversed trainability
        for module in [self.note_release,
                       self.context_network,
                       self.monophonic_network,
                       self.reverb_model]:
            if module is not None:
                module.trainable = first_phase

        # Compute multiple note signals only when learning detuner weights
        self.detuner.use_detune = not first_phase

    def compute_global_features(self, features, training):
        """Call all modules computing global features."""
        if self.z_encoder is not None:
            features.update(self.z_encoder(features, training=training))
        if self.context_network is not None:
            features.update(self.context_network(features, training=training))
        if self.reverb_model is not None:
            features.update(self.reverb_model(features, training=training))
        return features

    def compute_monophonic_features(self, features, training):
        """Call all modules computing monophonic features."""
        if self.note_release is not None:
            features.update(self.note_release(features, training=training))
        if self.inharm_model is not None:
            features.update(self.inharm_model(features, training=training))
        if self.detuner is not None:
            features.update(self.detuner(features, training=training))
        if self.monophonic_network is not None:
            features.update(self.monophonic_network(features, training=training))
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

        # Disentangle polyphony and batch axis
        features = self.parallelizer(features, parallelize=False)

        # Processor group call
        pg_out = self.processor_group(features, return_outputs_dict=True)

        # Parse outputs
        outputs = pg_out['controls']
        outputs['audio_synth'] = pg_out['signal']

        if training:
            self._update_losses_dict(self.loss_objs, outputs)

        return outputs
