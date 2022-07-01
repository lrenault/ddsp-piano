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
        self.monophonic_network = monophonic_network
        self.inharm_model = inharm_model
        self.detuner = detuner
        self.reverb_model = reverb_model
        self.processor_group = processor_group

        self.loss_objs = ddsp.core.make_iterable(losses)

    def _update_losses_dict(self, loss_objs, *args, **kwargs):
        super(PianoModel, self)._update_losses_dict(loss_objs,
                                                    *args,
                                                    **kwargs)
        self._losses_dict['regularization_loss'] = tf.reduce_sum(self.losses)

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
        features = self.parallelizer.parallelize(features)

        # Compute monophonic features
        features = self.compute_monophonic_features(features, training=training)

        # Disentangle polyphony and batch axis
        features = self.parallelizer.unparallelize(features)

        # Processor group call
        pg_out = self.processor_group(features, return_outputs_dict=True)

        # Parse outputs
        outputs = pg_out['controls']
        outputs['audio_synth'] = pg_out['signal']

        if training:
            self._update_losses_dict(self.loss_objs, outputs)

        return outputs
