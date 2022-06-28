import tensorflow as tf
from ddsp.core import tf_float32, resample, midi_to_hz
from ddsp.training import nn

tfkl = tf.keras.layers


class InharmonicityNetwork(tfkl.Layer):
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

    def call(self, notes, global_inharm=None):
        """ Compute inharmonicity coefficient corresponding to input pitch note
        Args:
            - notes (batch, n_frames, 1): input MIDI note conditioning signal
            - global_inharm (batch, 1, 1): fine-tune from piano type
        Returns:
            - inharm_coef (batch, n_frames, 1): inharmonicity coefficient
        """
        # Inharmonicity tessitura model
        reduced_notes = notes / self.midi_norm
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
            bridges_asymptotes += self.model_specific_weight * global_inharm

        # Compute inharmonicity factor (batch, n_frames, 1)
        # beta = exp(treble_asymp) + exp(bass_asymp)
        inharm_coef = tf.reduce_sum(tf.math.exp(bridges_asymptotes),
                                    axis=-1,
                                    keepdims=True)
        return inharm_coef


class Detuner(tfkl.Layer):
    """ Compute a detuning factor for each input MIDI note.
    Args:
        - n_substrings (int): number of piano strings per note.
    """

    def __init__(self, n_substrings=2, **kwargs):
        super(Detuner, self).__init__(**kwargs)
        self.n_substrings = n_substrings

    def build(self, input_shape):
        self.tanh = tf.keras.activations.tanh
        self.net = tfkl.Dense(self.n_substrings,
                              kernel_initializer='zeros',
                              bias_initializer='zeros',
                              trainable=False)
        self.net.build(input_shape)
        super(Detuner, self).build(input_shape)

    def centered_sigmoid(self, x, max_value=1.):
        return 2 * max_value * (tf.math.sigmoid(x) - 0.5)

    def call(self, notes, global_detuning):
        """ Forward pass
        Args:
            - notes (batch, ..., 1): input active notes.
            - global_detuning (batch, ..., 1): global detuning from
            piano type.
        Returns:
            - detuned_factors (batch, ..., n_substrings): detuning factor
            for each substring.
         """
        detuned_factors = self.tanh(self.net(notes / 128.))
        global_detuning = self.tanh(global_detuning)

        return detuned_factors + global_detuning


class ParametricTuning(tfkl.Layer):
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

    def __init__(self, **kwargs):
        super(ParametricTuning, self).__init__(**kwargs)
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

    def call(self, notes):
        inharm_coef = self.inharm_model(notes)
        detuning = self.get_deviation_from_ET(notes)

        f0_hz = midi_to_hz(notes) * detuning

        return f0_hz, inharm_coef


class F0ProcessorCell(tfkl.Layer):
    """Custom RNN cell for extending MIDI note signals during a trainable
    release time.
    Params:
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


class OneHotZEncoder(nn.DictLayer):
    """ Transforms one-hot instrument model into a Z embedding.
    Args:
        - instrument_id_key (str): input name of instrument id in the input
        dictionary.
        - n_instruments (int): number of instrument to be supported.
        - z_dim (int): dimension of z embedding.
        - output_splits (list(key, int)): additional model-specific feature to
        compute (key and dimension).
        - n_frames (int): pool embedding value over this number of time frames.
    """

    def __init__(self,
                 instrument_id_key='piano_model',
                 n_instruments=16,
                 z_dim=16,
                 output_splits=(('global_inharm', 1),
                                ('global_detuning', 1)),
                 n_frames=None,
                 **kwargs):
        self.n_frames = n_frames
        self.n_instruments = n_instruments
        self.z_dim = z_dim

        output_splits += (('z', z_dim),)
        self.output_splits = output_splits
        self.output_dim = sum([v[1] for v in self.output_splits])

        super(OneHotZEncoder, self).__init__(
            input_keys=[instrument_id_key, ],
            output_keys=[v[0] for v in self.output_splits],
            **kwargs
        )

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

    def call(self, instrument_id):
        # Compute Z embedding from instrument id
        z = self.embedding(instrument_id)
        global_inharm = self.inharm_embedding(instrument_id)
        global_detuning = self.detune_embedding(instrument_id)

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

        # Name outputs
        return {"z": z,
                "global_inharm": global_inharm,
                "global_detuning": global_detuning}


class MultiInstrumentReverb(nn.DictLayer):
    """Reverb with learnable impulse response compatible with a multi-
    environment setting.
    Args:
        - inference (bool): training or inference setting.
        - n_instruments (int): number of instrument reverbs to model.
        - reverb_length (int): number of samples for each impulse response.
    """

    def __init__(self,
                 instrument_id_key='piano_model',
                 n_instruments=16,
                 reverb_length=32000,
                 inference=False,
                 **kwargs):
        super(MultiInstrumentReverb, self).__init__(
            input_keys=[instrument_id_key, ],
            output_keys=['reverb_ir', ],
            **kwargs
        )
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

    def call(self, instrument_id):
        """Get reverb IR from instrument id"""
        ir = self.reverb_dict(instrument_id)

        if len(ir.shape) == 3:
            ir = ir[:, 0]

        # Apply decay mask
        if self.inference:
            ir = self.exponential_decay_mask(ir)

        return ir
