import tensorflow as tf
from ddsp.core import tf_float32, resample, midi_to_hz
from ddsp.training import nn

tfkl = tf.keras.layers


class ContextNetwork(nn.OutputSplitsLayer):
    """Sequential model for computing the context vector from the global inputs
    - Wrapped inside a DictLayer for named inputs compatibility.
    Args:
        - layers (list(tfkl.Layer)): list of neural layers inside the
        Sequential model.
    """

    def __init__(self,
                 layers,
                 output_splits=(('context', 32),),
                 **kwargs):
        super(ContextNetwork, self).__init__(output_splits=output_splits,
                                             **kwargs)
        self.model = tf.keras.Sequential(layers=layers)

    @property
    def layers(self):
        return self.model.layers

    def collapse_last_axis(self, x, axis=-2):
        # Merge last axis of the tensor 'x', starting from axis 'axis'
        shape = tf.shape(x)
        new_shape = tf.concat(
            [shape[:axis], [tf.reduce_prod(shape[axis:])]],
            axis=0
        )
        return tf.reshape(x, new_shape)

    def compute_output(self, conditioning, pedal, z):
        """Forward pass.
        Args:
            - conditioning (batch, n_frames, n_synths, 2): active pitch and
            onset velocity conditioning.
            - pedal (batch, n_frames, 4): pedal signals inputs.
            - z (batch, n_frames, z_dim): model embedding pooled over time.
        """
        x = tf.concat([self.collapse_last_axis(conditioning), pedal, z],
                      axis=-1)
        x = self.model(x)
        return x


class MonophonicNetwork(nn.OutputSplitsLayer):
    """Sequential model for computing monophonic synthesizer controls from the
    parallelized monophonic inputs. Wrapped inside a DictLayer for named inputs
    compatibility.
    Args:
        - layers (list(tfkl.Layer)): list of neural layers inside the Sequen-
        tial model.
    """

    def __init__(self,
                 layers,
                 output_splits=(('amplitudes', 1),
                                ('harmonic_distribution', 96),
                                ('magnitudes', 64)),
                 **kwargs):
        super(MonophonicNetwork, self).__init__(output_splits=output_splits,
                                                **kwargs)
        self.model = tf.keras.Sequential(layers=layers)
        self.midi_norm = 128.

    @property
    def layers(self):
        return self.model.layers

    def compute_output(self, conditioning, extended_pitch, context):
        """Forward parallelized monophonic inputs through the model.
        Args:
            - conditioning (batch * n_synths, n_frames, 2): parallelized active
            and onset conditioning.
            - extended_pitch (batch * n_synths, n_frames, 1): parallelized
            active prolonged pitch conditioning.
            - context (batch * n_synths, n_frames, context_dim): context signal
        """
        # Normalize and concatenate inputs
        x = tf.concat([extended_pitch / self.midi_norm,
                       conditioning / [self.midi_norm, 1.],
                       context],
                      axis=-1)
        # Forward pass
        x = self.model(x)

        return x


class Parallelizer(tfkl.Layer):
    """Module for merging and unmerge the batch and polyphony axis of features.
    Args:
        - n_synths (int): size of polypohny axis.
    """

    def __init__(self, n_synths=16, **kwargs):
        super(Parallelizer, self).__init__(**kwargs)
        self.n_synths = n_synths

    def build(self, input_shape):
        self.batch_size = input_shape[0]

    def put_polyphony_axis_at_first(self, x):
        """Reshape feature before calling parallelize"""
        if len(x.shape) == 3:
            # Create the polyphony axis and share value over all mono channels
            x = tf.repeat(x[tf.newaxis, ...], repeats=self.n_synths, axis=0)

        elif len(x.shape) == 4:
            # Put polyphony axis as the first dimension
            x = tf.transpose(x, [2, 0, 1, 3])

        return x

    def parallelize_feature(self, x):
        # Merge the polyphony and batch axis (which are the first two axis)
        shape = tf.shape(x)
        new_shape = tf.concat(
            [[self.n_synths * shape[1]], shape[2:]],
            axis=0
        )
        return tf.reshape(x, new_shape)

    def unparallelize_feature(self, x):
        # Disentangle batch and polyphony axis
        shape = tf.shape(x)
        new_shape = tf.concat(
            [[self.n_synths, shape[0] // self.n_synths], shape[1:]],
            axis=0
        )
        return tf.reshape(x, new_shape)

    def parallelize(self, features,
                    keys=('conditioning',
                          'context',
                          'global_inharm',
                          'global_detuning')):
        for k in keys:
            features[k] = self.put_polyphony_axis_at_first(features[k])
            features[k] = self.parallelize_feature(features[k])
        return features

    def unparallelize(self, features,
                      keys=('f0_hz',
                            'inharm_coef',
                            'amplitudes',
                            'harmonic_distribution',
                            'magnitudes')):
        """Disentangle batch and polyphony axis and distribute features as
        monophonic controls.
        Args:
            - features (dict(Tensors)): named features and controls.
            - keys (list(string)): list of feature keys to unparallelize and
            create monophonic controls.
        """
        for k in keys:
            features[k] = self.unparallelize_feature(features[k])
            for i in range(self.n_synths):
                features[k + f'_{i}'] = features[k][i]
        return features


class InharmonicityNetwork(nn.DictLayer):
    """ Compute inharmonicity coefficient corresponding to MIDI notes. """

    def __init__(self, name="inharmonicity_net", **kwargs):
        super(InharmonicityNetwork, self).__init__(name=name, **kwargs)
        self.midi_norm = 128.

    def build(self, input_shape):
        """ Initialize the MIDI note to inharmonicity coefficient network.
        Initial values are taken from results in F.Rigaud et al. "A Parametric
        Model of Piano tuning", Proc. of DAFx-2011.
        """
        # Init weights
        treble_slope = 9.26e-2
        treble_intercept = - 13.64

        bass_slope = - 8.47e-2
        bass_intercept = - 5.82

        self.model_specific_weight = self.add_weight(
            name="model_specific_weight",
            shape=(1,),
            dtype=tf.float32,
            trainable=True,
            initializer='zero'
        )
        self.slopes = self.add_weight(name="slopes",
                                      shape=(2,),
                                      dtype=tf.float32,
                                      trainable=False)
        self.slopes.assign(tf.convert_to_tensor([
            treble_slope * self.midi_norm,
            bass_slope * self.midi_norm]))

        self.offsets = self.add_weight(name="offsets",
                                       shape=(2,),
                                       dtype=tf.float32,
                                       trainable=False)
        self.offsets.assign(tf.convert_to_tensor([
            treble_intercept / (self.midi_norm * treble_slope),
            bass_intercept / (self.midi_norm * bass_slope)]))

        self.slopes_modifier = self.add_weight(name="slopes_modifier",
                                               shape=(2,),
                                               dtype=tf.float32,
                                               initializer='zero',
                                               regularizer=tf.keras.regularizers.L1(0.1),
                                               trainable=True)
        self.offsets_modifier = self.add_weight(name="offsets_modifier",
                                                shape=(2,),
                                                dtype=tf.float32,
                                                initializer='zero',
                                                regularizer=tf.keras.regularizers.L1(0.1),
                                                trainable=True)
        self.trainable = False

        super(InharmonicityNetwork, self).build(input_shape)

    def call(self, extended_pitch, global_inharm=None) -> ['inharm_coef']:
        """ Compute inharmonicity coefficient corresponding to input pitch note
        Args:
            - extended_pitch (batch, n_frames, 1): input MIDI note conditioning
            signal.
            - global_inharm (batch, 1, 1): fine-tuning from a specific piano
            model.
        Returns:
            - inharm_coef (batch, n_frames, 1): inharmonicity coefficient.
        """
        # Inharmonicity tessitura model
        reduced_notes = extended_pitch / self.midi_norm
        slopes = self.slopes + self.slopes_modifier
        offsets = self.offsets + self.offsets_modifier

        bridges_asymptotes = slopes * (reduced_notes + offsets)

        # Fine-tuning according to piano model
        if global_inharm is not None:
            # Scaling
            global_inharm *= 10.
            # Only the bass bridge is model specific
            global_inharm = tf.concat(
                [tf.zeros_like(global_inharm), global_inharm],
                axis=-1
            )
            import pdb; pdb.set_trace()
            bridges_asymptotes += self.model_specific_weight * global_inharm

        # Compute inharmonicity factor (batch, n_frames, 1)
        # beta = exp(treble_asymp) + exp(bass_asymp)
        inharm_coef = tf.reduce_sum(tf.math.exp(bridges_asymptotes),
                                    axis=-1,
                                    keepdims=True)
        return inharm_coef


class Detuner(nn.DictLayer):
    """ Compute a detuning factor for each input MIDI note.
    Args:
        - n_substrings (int): number of piano strings per note.
        - use_detune (bool): use the predicted detuning for converting MIDI
        pitch to Hz.
    """

    def __init__(self, n_substrings=2, use_detune=True, name='detuner',
                 **kwargs):
        super(Detuner, self).__init__(name=name, **kwargs)
        self.n_substrings = n_substrings
        self.use_detune = use_detune

        self.tanh = tf.keras.activations.tanh
        self.layer = tfkl.Dense(self.n_substrings,
                                kernel_initializer='zeros',
                                bias_initializer='zeros',
                                trainable=False)

    def call(self, extended_pitch, global_detuning=None) -> ['f0_hz']:
        """ Forward pass
        Args:
            - extended_pitch (batch, ..., 1): input active notes.
            - global_detuning (batch, ..., 1): global detuning from
            piano type.
        Returns:
            - detuned_factors (batch, ..., n_substrings): detuning factor
            for each substring.
         """
        if self.use_detune:
            detuning = self.tanh(self.layer(extended_pitch / 128.))

            if global_detuning is not None:
                global_detuning = self.tanh(global_detuning)
                detuning += global_detuning

            extended_pitch += detuning

        return midi_to_hz(extended_pitch)


class ParametricTuning(nn.DictLayer):
    """Parametric model for piano tuning, for note inharmonicity and detuning
    according to Rigaud et al. 'A parametric model of piano tuning' (DAFx-11)
    Params:
        - inharm_model (tfkl.Layer): parametric inharmonicity model over
        tessitura.
        - pitch_translation (float)
        - decrease_slope (float)
        - low_bass_asymptote (float)
    Input:
        - notes (batch, n_frames, 1): note conditioning (in MIDI scale).
    Outputs:
        - f0_hz (batch, n_frames, 1): (detuned) frequencies of notes (in Hz).
        - inharm_coef (batch, n_frames, 1): inharmonicity coefficient.
    """

    def __init__(self, name='parametric_tuniing', **kwargs):
        super(ParametricTuning, self).__init__(name=name, **kwargs)

        # Inharmonicity network
        self.inharm_model = InharmonicityNetwork()

        # Reference note
        self.reference_a = tf.convert_to_tensor(69., dtype=tf.float32)

        # Tuning parameters
        self.pitch_translation = 64.  # m_0
        self.decrease_slope = 24.  # alpha
        self.low_bass_asymptote = 4.51 - 1  # K
        self.erf = tf.math.tanh

    def streching_model(self, notes):  # rho
        rho = 1 - self.erf((notes - self.pitch_translation) / self.decrease_slope)
        rho *= self.low_bass_asymptote / 2
        rho += 1
        return rho

    def get_deviation_from_ET(self, notes):
        # Get distance from reference note (A4)
        reference_inharm_coef = self.inharm_model(self.reference_a)
        ratio = midi_to_hz(notes) / midi_to_hz(self.reference_a)

        # Compute deviation from equal temperament of octave A
        detuning = 1 + reference_inharm_coef * (ratio * self.streching_model(notes))**2
        detuning /= 1 + self.inharm_model(notes) * self.streching_model(notes)**2
        detuning = tf.math.sqrt(detuning)

        return detuning

    def call(self, extended_pitch) -> ['f0_hz', 'inharm_coef']:
        inharm_coef = self.inharm_model(extended_pitch)
        detuning = self.get_deviation_from_ET(extended_pitch)

        f0_hz = midi_to_hz(extended_pitch) * detuning

        return f0_hz, inharm_coef


class F0ProcessorCell(tfkl.Layer):
    """Custom RNN cell for extending MIDI note signals during a trainable
    release time.
    Args:
        - frame_rate (int): number of frames per second.
    """

    def __init__(self, frame_rate=250):
        super(F0ProcessorCell, self).__init__()
        self.frame_rate = frame_rate
        self.state_size = 2

    def build(self, input_shape):
        self.decay = tf.Variable(1., name="F0decay")

        self.trainable = False
        self.built = True

    @tf.function
    def call(self, midi_note, previous_state):
        """ Extend note
        Args:
            - midi_note (batch, 1): active MIDI vector frame.
            - previous_state (batch, 2): which note was played for how long.
        """
        previous_note = previous_state[0][..., 0:1]
        decayed_steps = previous_state[0][..., 1:2]

        note_activity = tf.greater(midi_note, 0)
        decay_end = tf.greater(decayed_steps, self.decay * self.frame_rate)

        note_activity = tf_float32(note_activity)
        decay_end = 1 - tf_float32(decay_end)

        midi_note = note_activity * midi_note + (1 - note_activity) * decay_end * previous_note
        decayed_steps = (1 - note_activity) * decay_end * (decayed_steps + 1)

        updated_state = tf.concat([midi_note, decayed_steps], axis=-1)

        return midi_note, [updated_state]


class NoteRelease(nn.DictLayer):
    """NoteRelease dict layer for extending the active pitch conditioning.
    Based on the custom RNN F0ProcessorCell"""

    def __init__(self, frame_rate=250, name='note_release', **kwargs):
        super(NoteRelease, self).__init__(name=name, **kwargs)
        self.layer = tfkl.RNN(F0ProcessorCell(frame_rate=frame_rate))

    def call(self, conditioning) -> ['extended_pitch']:
        active_pitch = conditioning[..., 0:1]
        extended_pitch = self.layer(active_pitch)

        return extended_pitch


class OneHotZEncoder(nn.DictLayer):
    """ Transforms one-hot instrument model into a Z embedding.
    Args:
        - n_instruments (int): number of instrument to be supported.
        - z_dim (int): dimension of z embedding.
        - n_frames (int): pool embedding value over this number of time frames.
    """

    def __init__(self, n_instruments=16, z_dim=16, n_frames=None, **kwargs):
        super(OneHotZEncoder, self).__init__(**kwargs)
        self.n_instruments = n_instruments
        self.z_dim = z_dim
        self.n_frames = n_frames

    def build(self, input_shape):
        super(OneHotZEncoder, self).build(input_shape)
        self.embedding = tfkl.Embedding(input_dim=self.n_instruments,
                                        output_dim=self.z_dim,
                                        input_length=1)
        self.inharm_embedding = tfkl.Embedding(input_dim=self.n_instruments,
                                               output_dim=1,
                                               input_length=1)
        self.detune_embedding = tfkl.Embedding(input_dim=self.n_instruments,
                                               output_dim=1,
                                               input_length=1)

    def call(self, piano_model) -> ['z', 'global_inharm', 'global_detuning']:
        # Compute Z embedding from instrument id
        z = self.embedding(piano_model)
        global_inharm = self.inharm_embedding(piano_model)
        global_detuning = self.detune_embedding(piano_model)

        # Add time axis
        if len(z.shape) == 2:
            z = z[:, tf.newaxis, :]
            global_inharm = global_inharm[:, tf.newaxis, :]
            global_detuning = global_detuning[:, tf.newaxis, :]

        if self.n_frames is not None:
            # Expand time dim
            z = resample(z, self.n_frames)
            global_inharm = resample(global_inharm, self.n_frames)
            global_detuning = resample(global_detuning, self.n_frames)

        return z, global_inharm, global_detuning


class MultiInstrumentReverb(nn.DictLayer):
    """Reverb with learnable impulse response compatible with a multi-
    environment setting.
    Args:
        - inference (bool): training or inference setting.
        - n_instruments (int): number of instrument reverbs to model.
        - reverb_length (int): number of samples for each impulse response.
    """

    def __init__(self,
                 n_instruments=16,
                 reverb_length=32000,
                 inference=False,
                 **kwargs):
        super(MultiInstrumentReverb, self).__init__(**kwargs)
        self.reverb_length = reverb_length
        self.n_instruments = n_instruments
        self.inference = inference

    def build(self, input_shape):
        self.reverb_dict = tf.keras.Sequential(layers=[
            tf.keras.Input(batch_size=input_shape[0], shape=(1, )),
            tfkl.Embedding(self.n_instruments,
                           self.reverb_length,
                           embeddings_initializer=tf.random_normal_initializer(
                               mean=0,
                               stddev=1e-6)
                           )])
        super(MultiInstrumentReverb, self).build(input_shape)

    def exponential_decay_mask(self, ir, decay_exponent=4., decay_start=16000):
        """ Apply exponential decay mask on impulse responde as in MIDI-ddsp
        Args:
            - ir (batch, n_samples): raw impulse response.
        Returns:
            - ir (batch, n_samples): decayed impulse response.
        """
        time = tf.linspace(0.0, 1.0, self.reverb_length - decay_start)
        mask = tf.exp(- decay_exponent * time)
        mask = tf.concat([tf.ones(decay_start), mask], 0)
        return ir * mask[tf.newaxis, ...]

    def call(self, piano_model) -> ['reverb_ir']:
        """Get reverb IR from instrument id"""
        ir = self.reverb_dict(piano_model)

        if len(ir.shape) == 3:
            ir = ir[:, 0]

        # Apply decay mask
        if self.inference:
            ir = self.exponential_decay_mask(ir)

        return ir

