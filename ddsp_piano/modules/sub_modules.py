import gin
import tensorflow as tf
from functools import partial
from numpy import array as np_array
from ddsp.core import tf_float32, resample, midi_to_hz, exp_sigmoid
from ddsp.training import nn
from ddsp_piano.modules.fdn_reverb import FeedbackDelayNetwork

tfkl = tf.keras.layers


# -----------------------------------------------------------------------------
# Global models
# -----------------------------------------------------------------------------


@gin.register
class ContextNetwork(nn.OutputSplitsLayer):
    """Sequential model for computing the context vector from the global inputs
    - Wrapped inside a DictLayer for named inputs compatibility.
    Args:
        - layers (list(tfkl.Layer)): list of neural layers inside the
        Sequential model.
        - output_split (list(key, dim)): output keys and dims.
        - normalize_pitch (bool): whether to scale input pitch to range [0,1].
    """
    def __init__(self,
                 layers,
                 output_splits=(('context', 32),),
                 normalize_pitch=False,
                 **kwargs):
        super().__init__(output_splits=output_splits, **kwargs)
        self.model = tf.keras.Sequential(layers=layers)
        self.normalize_pitch = normalize_pitch
        self.midi_norm = 128.

    @property
    def layers(self):
        return self.model.layers + [self.dense_out]

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
        if self.normalize_pitch:
            conditioning = conditioning / [self.midi_norm, 1.]

        x = tf.concat([self.collapse_last_axis(conditioning), pedal, z],
                      axis=-1)
        x = self.model(x)
        return x


@gin.register
class SimpleContextNet(nn.OutputSplitsLayer):
    """Context network that does not take the note conditioning into account.
    Uses FiLM layer to incorporate instrument embedding. Can be used at
    inference with a n_synths different from training.
    Args:
        - layers (list(tfkl.Layer)): model layers.
        - output_splits (list(key, dim)): outputs keys and dims.
    """
    def __init__(self, layers, output_splits=(('context', 32),), **kwargs):
        super().__init__(output_splits=output_splits, **kwargs)
        self.model = tf.keras.Sequential(layers=layers)

    def compute_output(self, pedal, z=None):
        """Forward pass.
        Args:
            - pedal (batch, n_frames, 4): pedal conditioning.
            - z (batch, 1, z_dim): instrument embedding.
        """
        context = self.model(pedal)

        # Apply instrument embedding as a FiLM layer
        if z is not None:
            film_coef, film_bias = tf.split(z, 2, axis=-1)
            context = context * film_coef + film_bias

        return context


@gin.register
class FiLMContextNetwork(nn.DictLayer):
    """Compute a context signal emcompasing all contextual information for
    future monophonic processing, namely polyphony, pedals signals and piano/
    speaker model.
    Uses a FiLM layer for including the computed piano embedding.
    Args:
        - n_instruments (int): number of different instruments in the dataset.
        - layer_dim (int): hidden layers dimension.
        - context_dim (int): output context signal dimension.
    """
    def __init__(self,
                 n_instruments=10,
                 layer_dim=64,
                 context_dim=32,
                 name="context_net",
                 **kwargs):
        super().__init__(name=name, **kwargs)
        self.midi_norm = 128.

        self.conditioning_head  = nn.FcStack(32, layers=2)
        self.pedal_head         = nn.FcStack(16, layers=2)
        self.piano_id_head      = nn.get_embedding(n_instruments, 32)
        self.main_model         = tf.keras.Sequential(layers=[
            tfkl.Dense(layer_dim, activation=tf.nn.leaky_relu),
            tfkl.GRU(layer_dim, return_sequences=True),
            tfkl.Dense(layer_dim),
            tfkl.LayerNormalization(),
            tfkl.Activation(tf.nn.leaky_relu),
        ])
        self.film_input_reshape = tfkl.Dense(layer_dim * 2)
        self.output_layer       = nn.FcStack(context_dim, layers=2)

    def collapse_last_axis(self, x, axis=-2):
        """Merge last axis of the tensor 'x', starting from axis 'axis'.
        For example, x[b,t,n,2] -> x[b,t,n*2] for axis=-2.
        """
        shape = tf.shape(x)
        new_shape = tf.concat(
            [shape[:axis], [tf.reduce_prod(shape[axis:])]],
            axis=0
        )
        return tf.reshape(x, new_shape)

    def apply_film(self, features, piano_feat):
        """Modify intermediate features with piano model embedding
        through a FiLM layer.
        Args:
            - features (batch, n_frames, n_features): input features.
            - piano_id (batch, 1, n_features): piano model embedding.
        """
        # Apply FiLM layer
        piano_feat = self.film_input_reshape(piano_feat)
        film_coef, film_bias = tf.split(piano_feat, 2, axis=-1)
        features = features * film_coef + film_bias
        return features

    def call(self, conditioning, pedal, piano_model) -> ['context']:
        """Forward pass.
        Args:
            - conditioning (batch, n_frames, n_synths, 2): active pitch and
            onset velocity conditioning.
            - pedal (batch, n_frames, 4): pedal signals input.
            - piano_model (batch,): 
        """
        # Normalize pitch conditioning and reshape to [b,n_frames,n_synth*2]
        conditioning = conditioning / [self.midi_norm, 1.]
        conditioning = self.collapse_last_axis(conditioning)

        # Compute input-wise features
        conditioning_feat = self.conditioning_head(conditioning)
        pedal_feat = self.pedal_head(pedal)
        piano_feat = self.piano_id_head(piano_model)
        
        # Concatenate conditioning and pedal features, then apply main model
        features = tf.concat([conditioning_feat, pedal_feat], axis=-1)
        features = self.main_model(features)

        # Apply piano embedding through FiLM layer
        features = self.apply_film(features, piano_feat)
        # Output
        context = self.output_layer(features)

        return context


@gin.register
class OneHotZEncoder(nn.DictLayer):
    """ Transforms one-hot encoded instrument model into a Z embedding and
    model-specific detuning and inharmonicity coefficient.
    Args:
        - n_instruments (int): number of instrument to be supported.
        - z_dim (int): dimension of z embedding.
        - duration (int): pool embedding value over this duration.
        - frame_rate (int): number of controls per second.
    """
    def __init__(self, n_instruments=16, z_dim=16, duration=None, frame_rate=250,
                 **kwargs):
        super().__init__(**kwargs)
        self.n_instruments = n_instruments
        self.z_dim = z_dim
        self.duration = duration
        self.frame_rate = frame_rate

        self.embedding = tfkl.Embedding(input_dim=self.n_instruments,
                                        output_dim=self.z_dim,
                                        input_length=1,
                                        name='instrument_embedding')
        self.inharm_embedding = tfkl.Embedding(input_dim=self.n_instruments,
                                               output_dim=1,
                                               input_length=1,
                                               name='instr_specific_inharm')
        self.detune_embedding = tfkl.Embedding(input_dim=self.n_instruments,
                                               output_dim=1,
                                               input_length=1,
                                               name='instr_specific_detuning')

    @property
    def n_frames(self):
        return int(self.duration * self.frame_rate) if self.duration else 1

    def alternate_training(self, first_phase=True):
        """Toggle trainability of models according to the training phase.
        (Modules involved with partial frequency computing are frozen during
        the first training phase).
        Args:
            - first_phase (bool): whether training with the 1st phase strategy
            or not.
        """
        self.embedding.trainable = first_phase
        self.inharm_embedding.trainable = not first_phase
        self.detune_embedding.trainable = not first_phase

    def call(self, piano_model) -> ['z', 'global_inharm', 'global_detuning']:
        # Fix piano_model to 0 for single piano modeling
        if self.n_instruments == 1:
            piano_model = tf.zeros_like(piano_model, dtype=tf.int32)

        # Compute Z embedding from instrument id
        z = self.embedding(piano_model)
        global_inharm = self.inharm_embedding(piano_model)
        global_detuning = self.detune_embedding(piano_model)

        # Add time axis
        if len(z.shape) == 2:
            z = z[:, tf.newaxis, :]
            global_inharm = global_inharm[:, tf.newaxis, :]
            global_detuning = global_detuning[:, tf.newaxis, :]

        # Pool over time dimension
        z = resample(z, self.n_frames)
        global_inharm = resample(global_inharm, self.n_frames)
        global_detuning = resample(global_detuning, self.n_frames)

        return z, global_inharm, global_detuning


@gin.register
class BackgroundNoiseFilter(nn.DictLayer):
    """Background noise modeler, learning a constant noise filter accross
    each recording environment.
    Args:
        - n_instruments (int): number of instrument to be supported.
        - n_filters (int): number of noise filter bands.
        - duration (float): segment duration (in s).
        - frame_rate (int): number of controls per second.
        - denoise (bool): whether to output the learnt noise or replace it
        by silence.
    """
    def __init__(self, n_instruments=16, n_filters=64,
                 duration=None, frame_rate=250, denoise=False, **kwargs):
        super().__init__(**kwargs)
        self.n_instruments = n_instruments
        self.n_filters     = n_filters
        self.duration      = duration
        self.frame_rate    = frame_rate
        self.denoise       = denoise

        self.embedding = tfkl.Embedding(input_dim=self.n_instruments,
                                        output_dim=self.n_filters,
                                        input_length=1,
                                        name='background_filter_coefs')

    @property
    def n_frames(self):
        return int(self.duration * self.frame_rate) if self.duration else 1

    def call(self, piano_model) -> ['background_mag']:
        """Forward pass.
        Args:
            - piano_model (b): integer of the piano model/recording environment
        """
        background_mag = self.embedding(piano_model)  # (b, n_filters)

        if len(background_mag.shape) == 2:
            background_mag = background_mag[:, tf.newaxis, :]

        # Expand time dim
        background_mag = resample(background_mag, self.n_frames)

        if self.denoise:
            background_mag = -10. * tf.ones_like(background_mag)

        return background_mag


@gin.register
class MultiInstrumentReverb(nn.DictLayer):
    """Reverb with learnable impulse response compatible with a multi-
    environment setting.
    Args:
        - n_instruments (int): number of instrument reverbs to model.
        - reverb_duration (int): duration of the learnt reverbs.
        - sample_rate (int): number of audio samples per second.
        - inference (bool): training or inference setting.
    """
    def __init__(self,
                 n_instruments=16,
                 reverb_duration=1.5,
                 sample_rate=16000,
                 inference=False,
                 **kwargs):
        super().__init__(**kwargs)
        self.reverb_duration = reverb_duration
        self.sample_rate     = sample_rate
        self.n_instruments   = n_instruments
        self.inference       = inference

    @property
    def reverb_length(self):
        return int(self.reverb_duration * self.sample_rate)

    def build(self, input_shape):
        super().build(input_shape)
        self.reverb_dict = tf.keras.Sequential(layers=[
            tf.keras.Input(batch_size=None, # input_shape[0],
                           shape=(1, )),
            tfkl.Embedding(self.n_instruments,
                           self.reverb_length,
                           embeddings_initializer=tf.random_normal_initializer(
                               mean=0,
                               stddev=1e-6))])

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
        if self.n_instruments == 1:
            piano_model = tf.zeros_like(piano_model, dtype=tf.int32)

        ir = self.reverb_dict(piano_model)

        if len(ir.shape) == 3:
            ir = ir[:, 0]

        # Apply decay mask
        if self.inference:
            ir = self.exponential_decay_mask(ir)

        return ir


@gin.register
class MultiInstrumentFeedbackDelayReverb(nn.DictLayer):
    """Feedback-delay reverb parameters in a multi-environment setting.
    Args:
        - n_instruments (int): number of instrument reverbs to model.
        - sample_rate (int): number of audio samples per second.
        - delay_lines (int): number of delay lines in the FDN.
        - early_ir_length (int): length of the early FIR.
        - regularize_early (bool): whether to regularize the early IR or not.
        - inference (bool): training or inference setting.
    """
    def __init__(self, n_instruments=10, sample_rate=16000, delay_lines=8,
                 early_ir_length=200, regularize_early=False, **kwargs):
        super().__init__(**kwargs)
        self.n_instruments = n_instruments
        self.sample_rate   = sample_rate
        self.delay_lines   = delay_lines

        general_initializer = tf.random_normal_initializer(mean=0., stddev=1e-1)
        delay_initializer = tf.random_normal_initializer(mean=400., stddev=60.)
        gains_initializer = tf.random_normal_initializer(mean=0.25, stddev=1e-1)

        self._input_gain = tfkl.Embedding(
            self.n_instruments, self.delay_lines,
            embeddings_initializer=gains_initializer
        )
        self._output_gain = tfkl.Embedding(
            self.n_instruments, self.delay_lines,
            embeddings_initializer=gains_initializer
        )
        self._gain_allpass = tfkl.Embedding(
            self.n_instruments, 4 * self.delay_lines,
            embeddings_initializer=gains_initializer
        )
        self._delays_allpass = tfkl.Embedding(
            self.n_instruments, 4 * self.delay_lines,
            embeddings_initializer=delay_initializer
        )
        self._time_rev_0_sec = tfkl.Embedding(
            self.n_instruments, 1,
            embeddings_initializer=tf.random_normal_initializer(mean=2., stddev=5e-1),
        )
        self._alpha_tone = tfkl.Embedding(
            self.n_instruments, 1,
            embeddings_initializer=general_initializer
        )
        self._early_ir = tfkl.Embedding(
            self.n_instruments, early_ir_length,
            embeddings_initializer=general_initializer,
            embeddings_regularizer=tf.keras.regularizers.L1(1e-1) if regularize_early else None
        )
        self.reverb_model = FeedbackDelayNetwork(trainable=False,
                                                 sampling_rate=self.sample_rate)

    def build(self, input_shape):
        super().build(input_shape)
        self.reverb_model.trainable = False
        self.reverb_model.build(input_shape)

    def reshape_embedding(self, embedding, splits=4):
        splitted = tf.split(embedding, splits, axis=-1)
        return tf.stack(splitted, axis=-1)

    def call(self, piano_model) -> ['reverb_ir']:
        if self.n_instruments == 1:
            piano_model = tf.zeros_like(piano_model, dtype=tf.int32)
        piano_model = piano_model[..., 0]
        controls_dict = {
            'input_gain': self._input_gain(piano_model),
            'output_gain': self._output_gain(piano_model),
            'gain_allpass': self.reshape_embedding(self._gain_allpass(piano_model)),
            'delays_allpass': self.reshape_embedding(self._delays_allpass(piano_model)),
            'time_rev_0_sec': tf.nn.relu(self._time_rev_0_sec(piano_model)),
            'alpha_tone': tf.math.sigmoid(self._alpha_tone(piano_model)),
            'early_ir': self._early_ir(piano_model),
        }
        ir = tf.vectorized_map(lambda x: self.reverb_model.get_ir(**x),
                               elems=controls_dict)
        return ir


# -----------------------------------------------------------------------------
# Monophonic amplitude models
# -----------------------------------------------------------------------------


@gin.register
class MonophonicNetwork(nn.OutputSplitsLayer):
    """Sequential model for computing monophonic synthesizer controls from the
    parallelized monophonic inputs. Wrapped inside a DictLayer for named inputs
    compatibility.
    Args:
        - layers (list(tfkl.Layer)): list of neural layers inside the internal
        Sequential model.
        - output_split (list(key, dim)): output keys and dims.
    """
    def __init__(self,
                 layers,
                 output_splits=(('amplitudes', 1),
                                ('harmonic_distribution', 96),
                                ('magnitudes', 64)),
                 **kwargs):
        super().__init__(output_splits=output_splits, **kwargs)
        self.model = tf.keras.Sequential(layers=layers)
        self.midi_norm = 128.

    @property
    def layers(self):
        return self.model.layers + [self.dense_out]

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
                       conditioning  / [self.midi_norm, 1.],
                       context],
                      axis=-1)
        # Forward pass
        x = self.model(x)

        return x


@gin.register
class MonophonicDeepNetwork(MonophonicNetwork):
    """Monophonic network using the same architecture as the original DDSP
    decoder MLP layers."""
    def __init__(self, rnn_channels=192, ch=64, layers_per_stack=3, **kwargs):
        super().__init__(layers=nn.Rnn(rnn_channels, 'gru'), **kwargs)
        # Layer creation
        stack = lambda: nn.FcStack(ch, layers_per_stack)
        # Layers
        self.input_stacks = [stack() for _ in range(3)]
        self.out_stack = stack()
    
    def compute_output(self, conditioning, extended_pitch, context):
        # Initial processing
        _extended_pitch = self.input_stacks[0](extended_pitch / self.midi_norm)
        _conditioning   = self.input_stacks[1](conditioning  / [self.midi_norm, 1.])
        _context = self.input_stacks[2](context)

        # Run RNN over the latents
        x = tf.concat([_extended_pitch, _conditioning, _context], axis=-1)
        x = self.model(x)
        x = tf.concat([_extended_pitch, _conditioning, _context, x], axis=-1)

        # Final processing
        x = self.out_stack(x)

        return x


@gin.register
class Parallelizer(tfkl.Layer):
    """Module for merging and unmerging the batch and polyphony axis of
    given features dictionary.
    Args:
        - n_synths (int): size of polyphony axis.
        - global_keys (list(string)): list of global features keys.
        - mono_keys (list(string)): list of monophonic features keys.
    """
    def __init__(self,
                 n_synths=16,
                 global_keys=('conditioning',
                              'context',
                              'global_inharm',
                              'global_detuning'),
                 mono_keys=('f0_hz',
                            'inharm_coef',
                            'amplitudes',
                            'harmonic_distribution',
                            'magnitudes'),
                 **kwargs):
        super().__init__(**kwargs)
        self.n_synths    = n_synths
        self.global_keys = global_keys
        self.mono_keys   = mono_keys

    def build(self, input_shape):
        self.batch_size = input_shape['conditioning'][0]
        super().build(input_shape)

    def put_polyphony_axis_at_first(self, x):
        """Reshape feature before calling parallelize"""
        if 2 <= len(x.shape) <= 3:
            # Create the polyphony axis and share value over all mono channels
            x = tf.repeat(x[tf.newaxis, ...], repeats=self.n_synths, axis=0)

        elif len(x.shape) == 4:
            # Put polyphony axis as the first dimension
            x = tf.transpose(x, [2, 0, 1, 3])

        return x

    def parallelize_feature(self, x):
        # Merge the polyphony and batch axis (which are the first two axis)
        shape = tf.shape(x)
        new_shape = tf.concat([[self.n_synths * self.batch_size], shape[2:]],
                              axis=0)
        return tf.reshape(x, new_shape)

    def unparallelize_feature(self, x):
        # Disentangle batch and polyphony axis
        shape = tf.shape(x)
        new_shape = tf.concat([[self.n_synths,  self.batch_size], shape[1:]],
                              axis=0)
        return tf.reshape(x, new_shape)

    def parallelize(self, features):
        for k in self.global_keys:
            features[k] = self.put_polyphony_axis_at_first(features[k])
            features[k] = self.parallelize_feature(features[k])
        return features

    def unparallelize(self, features):
        """Disentangle batch and polyphony axis and distribute features as
        monophonic controls."""
        for k in self.mono_keys:
            features[k] = self.unparallelize_feature(features[k])
            for i in range(self.n_synths):
                features[k + f'_{i}'] = features[k][i]
        return features

    def call(self, features, parallelize=True):
        if parallelize:
            return self.parallelize(features)
        else:
            return self.unparallelize(features)


# -----------------------------------------------------------------------------
# Parametric tuning models
# -----------------------------------------------------------------------------


@gin.register
class InharmonicityNetwork(nn.DictLayer):
    """ Compute inharmonicity coefficient corresponding to MIDI notes. """

    def __init__(self, name="inharmonicity_net", **kwargs):
        super().__init__(name=name, **kwargs)
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
        slopes  = self.slopes  + self.slopes_modifier
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


@gin.register
class ParametricTuning(InharmonicityNetwork):
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

    def __init__(self, name='parametric_tuning', **kwargs):
        super().__init__(name=name, **kwargs)

        # Reference note
        self.reference_a = tf.convert_to_tensor(69., dtype=tf.float32)

        # Tuning parameters
        self.pitch_translation  = 64.  # m_0
        self.decrease_slope     = 24.  # alpha
        self.low_bass_asymptote = 4.51 - 1  # K
        self.erf = tf.math.tanh

    def inharm_model(self, *args):
        return super().call(*args)

    def streching_model(self, notes):  # rho
        rho = 1 - self.erf((notes - self.pitch_translation) / self.decrease_slope)
        rho *= self.low_bass_asymptote / 2
        rho += 1
        return rho

    def get_deviation_from_ET(self, notes, global_inharm=None):
        # Get distance from reference note (A4)
        reference_inharm_coef = self.inharm_model(self.reference_a, global_inharm)
        ratio = midi_to_hz(notes) / midi_to_hz(self.reference_a)

        # Compute deviation from equal temperament of octave A
        detuning  = 1 + reference_inharm_coef * (ratio * self.streching_model(notes))**2
        detuning /= 1 + self.inharm_model(notes, global_inharm) * self.streching_model(notes)**2
        detuning  = tf.math.sqrt(detuning)

        return detuning

    def call(self, extended_pitch, global_inharm=None) -> ['f0_hz', 'inharm_coef']:
        inharm_coef = self.inharm_model(extended_pitch, global_inharm)
        detuning = self.get_deviation_from_ET(extended_pitch, global_inharm)

        f0_hz = midi_to_hz(extended_pitch) * detuning

        return f0_hz, inharm_coef


@gin.register
class JointParametricInharmTuning(nn.DictLayer):
    """Parametric models for joint parametrization of inharmonicity and
    detuning along the piano tessitura, taken from [1].

    DISCLAIMER: while the implementation is differentiable, the weights
    optimization with gradient descent through MSS loss is unfeasable [2].
    The layer is thus set as untrainable until the issue is solved.

    The hard-coded embedding values correspond to the parameter estimated on
    MAESTRO notes. If you are training on a different piano model, please
    follow the methodology of [1] to get your piano-specific parameters.

    [1] Rigaud et al. "A parametric model of piano tuning" (DAFx-11)
    [2] Turian et al. "I'm sorry for your loss: Spectrally-Based Audio
    Distances Are Bad at Pitch" (ICBINB - NIPS)

    Weights:
        - alpha_b   (n_instruments, 1): inharmonicity curve bass slope.
        - beta_b    (n_instruments, 1): inharmonicity curve bass intercept.
        - alpha_t   (n_instruments, 1): inharmonicity curve treble slope.
        - beta_t    (n_instruments, 1): inharmonicity curve treble intercept.
        - pitch_ref (n_instruments, 1): reference pitch.
        - K         (n_instruments, 1): octave tuning bass asymptote.
        - alpha     (n_instruments, 1): octave tuning decrase slope.
    """
    def __init__(self, n_instruments=10, pretrained_weights=None,
                 name="parametric_tuning", **kwargs):
        super().__init__(name=name, **kwargs)
        self.n_instruments = n_instruments
        self.pretrained_weights = pretrained_weights
        # Inharmonicity parameters
        self.alpha_b = nn.get_embedding(self.n_instruments, 1)
        self.beta_b  = nn.get_embedding(self.n_instruments, 1)
        self.alpha_t = nn.get_embedding(self.n_instruments, 1)
        self.beta_t  = nn.get_embedding(self.n_instruments, 1)

        # Detuning parameters (Railsback curve)
        self.pitch_ref = nn.get_embedding(self.n_instruments, 1)
        self.K         = nn.get_embedding(self.n_instruments, 1)
        self.alpha     = nn.get_embedding(self.n_instruments, 1)

    def build(self, input_shape):
        super().build(input_shape)
        batch_size = input_shape[0]

        if self.pretrained_weights is not None:
            def reshape_w(weights):
                w = np_array(weights)
                return [w]
            self.alpha_b.build((batch_size, 1))
            self.beta_b.build((batch_size, 1))
            self.alpha_t.build((batch_size, 1))
            self.beta_t.build((batch_size, 1))
            self.pitch_ref.build((batch_size, 1))
            self.K.build((batch_size, 1))
            self.alpha.build((batch_size, 1))

            self.alpha_b.set_weights(reshape_w(self.pretrained_weights['alpha_b']))
            self.beta_b.set_weights(reshape_w(self.pretrained_weights['beta_b']))
            self.alpha_t.set_weights(reshape_w(self.pretrained_weights['alpha_t']))
            self.beta_t.set_weights(reshape_w(self.pretrained_weights['beta_t']))
            self.pitch_ref.set_weights(reshape_w(self.pretrained_weights['pitch_ref']))
            self.K.set_weights(reshape_w(self.pretrained_weights['K']))
            self.alpha.set_weights(reshape_w(self.pretrained_weights['alpha']))

            self.trainable = False

    def reverse_scaled_tanh(self, x):
        return (1. - tf.math.tanh(x)) / 2.

    def get_inharm(self, extended_pitch, piano_model):
        bass_asymptote   = self.alpha_b(piano_model) * extended_pitch + self.beta_b(piano_model)
        treble_asymptote = self.alpha_t(piano_model) * extended_pitch + self.beta_t(piano_model)
        
        return tf.math.exp(bass_asymptote) + tf.math.exp(treble_asymptote)

    def get_deviation_from_ET(self, extended_pitch, piano_model):
        reference_pitch = self.pitch_ref(piano_model)
        # Frequency ratio between current note and the reference note
        ratio = midi_to_hz(extended_pitch) / midi_to_hz(reference_pitch)

        # Octave tuning choice (1st, 2nd,... partial tuned w.r.t. octave)
        rho = 1. + self.K(piano_model) * self.reverse_scaled_tanh(
            (extended_pitch - reference_pitch) / self.alpha(piano_model)
        )
        # Detuning
        detuning  = 1. + self.get_inharm(reference_pitch, piano_model) * (ratio * rho)**2
        detuning /= 1. + self.get_inharm(extended_pitch,  piano_model) * rho**2
        detuning  = tf.math.sqrt(detuning)
        
        return detuning

    def call(self, extended_pitch, piano_model) -> ['f0_hz', 'inharm_coef']:
        """Forward pass. Computing inharmonicity and detuning coefficients.
        Args:
            - extended_pitch (batch, n_frames, 1): input MIDI note activity
            conditioning.
            - piano_model (batch, 1): piano model identifer.
        Returns:
            - f0_hz (batch, n_frames, 1): detuned frequency control (in Hz).
            - inharm_coef (batch, n_frames, 1): inharmonicity coefficient.
        """
        inharm_coef = self.get_inharm(extended_pitch, piano_model)
        detuning    = self.get_deviation_from_ET(extended_pitch, piano_model)

        f0_hz = midi_to_hz(extended_pitch) * detuning

        return f0_hz, inharm_coef


# -----------------------------------------------------------------------------
# Deep tuning models
# -----------------------------------------------------------------------------


@gin.register
class DeepInharmonicity(nn.DictLayer):
    """Partial inharmonicity estimation with a deep MLP.
    Args:
        - n_layers (int): number of hidden layers.
        - ch (int): internal number of channels.
    """
    def __init__(self, ch=32, n_layers=4, name="inharmonicity_net", **kwargs):
        super().__init__(**kwargs)
        self.hidden_layers = nn.FcStack(ch, n_layers - 1)
        self.scale_layer = tfkl.Dense(ch, activation=partial(exp_sigmoid, max_value=1.))
        self.out_layer = tfkl.Dense(1, activation=lambda x: x / 1000.)

    def call(self, extended_pitch, global_inharm=None) -> ['inharm_coef']:
        inharm_coef = self.hidden_layers(extended_pitch / 128.)
        inharm_coef = self.scale_layer(inharm_coef)
        inharm_coef = self.out_layer(inharm_coef)

        if global_inharm is not None:
            inharm_coef += tf.nn.relu(global_inharm)

        return inharm_coef


@gin.register
class Detuner(nn.DictLayer):
    """ Compute a detuning factor for each input MIDI note.
    Args:
        - n_substrings (int): number of piano strings per note.
        - use_detune (bool): use the predicted detuning for converting MIDI
        pitch to Hz.
    """

    def __init__(self, n_substrings=2, use_detune=True, name='detuner',
                 **kwargs):
        super().__init__(name=name, **kwargs)
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


@gin.register
class DeepDetuner(nn.DictLayer):
    """ Compute a detuning factor for each input MIDI note.
    Args:
        - n_substrings (int): number of piano strings per note.
        - use_detune (bool): use the predicted detuning for converting MIDI
        pitch to Hz.
    """

    def __init__(self, n_substrings=2, use_detune=True, ch=32, n_layers=3,
                 name='detuner', **kwargs):
        super().__init__(name=name, **kwargs)
        self.n_substrings = n_substrings
        self.use_detune = use_detune

        self.hidden_layers = nn.FcStack(ch=ch, layers=n_layers)
        self.out_layer = tfkl.Dense(self.n_substrings,
                                    activation='tanh',
                                    kernel_initializer='zeros',
                                    bias_initializer='zeros')

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
            detuning = self.out_layer(self.hidden_layers(extended_pitch / 128.))

            if global_detuning is not None:
                detuning += tf.math.tanh(global_detuning)

            extended_pitch += detuning

        return midi_to_hz(extended_pitch)


# -----------------------------------------------------------------------------
# Dictionnary tuning models
# -----------------------------------------------------------------------------


@gin.register
class DictDetuner(nn.DictLayer):
    """Learn a detuning factor per pitch.
    Args:
        - n_instruments (int): number of multiple instrument model to handle.
    """
    def __init__(self, name="detuner", n_instruments=1, **kwargs):
        super().__init__(name=name, **kwargs)
        self.layer = tfkl.Embedding(128 * n_instruments,
                                    1,
                                    embeddings_initializer='zeros',
                                    name="detuner")

    def call(self, extended_pitch, piano_model=None) -> ['f0_hz']:
        """Forward pass.
        Args:
            - extended_pitch (b, n, 1): active pitch conditioning.
            - piano_model (b, 1): piano model conditioning.
        Returns:
            - f0_hz (b, n, 1): fundamental frequencies (in Hz).
        """
        # TODO: handle multiple instruments
        extended_pitch_int = tf.cast(extended_pitch[..., 0], dtype=tf.int32)
        return midi_to_hz(extended_pitch + self.layer(extended_pitch_int))


def l1_neg_reg(weight_matrix):
    """Penalize negative values."""
    return 1e2 * tf.math.reduce_sum(tf.nn.relu(-weight_matrix))


@gin.register
class DictInharmonicityModel(nn.DictLayer):
    """Learn a inharmonicity coefficient per pitch.
    Args:
        - n_instruments (int): number of multiple instrument model to handle.
    """
    def __init__(self, name="inharmonicity_net", n_instruments=1, **kwargs):
        super().__init__(name=name, **kwargs)
        self.layer = tfkl.Embedding(128 * n_instruments,
                                    1,
                                    embeddings_initializer='zeros',
                                    embeddings_regularizer=l1_neg_reg,
                                    name="inharm_coefs")

    def call(self, extended_pitch, piano_model=None) -> ['inharm_coef']:
        """Forward pass.
        Args:
            - extended_pitch (b, n, 1): active pitch conditioning.
        Returns:
            - inharm_coef (b, n, 1): inharmonicity coefficient.
        """
        # TODO: handle multiple instruments
        extended_pitch = tf.cast(extended_pitch[..., 0], dtype=tf.int32)
        return self.layer(extended_pitch)


class OnsetLinspaceCell(tfkl.Layer):
    """Custom RNN cell for counting the frames since the last note onset."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.state_size = 1

    @tf.function
    def call(self, onset_velocity, previous_state):
        """ Reset time if new note.
        Args:
            - onset_velocity (batch, 1): onset velocity MIDI vector frame.
            - previous_state (batch, 1): previous time step.
        """
        previous_time_frame = previous_state[0]

        new_note = tf_float32(tf.greater(onset_velocity, 0))
        reset_time = 1 - new_note  # 1 if sustained, 0 if new note

        surrogate_time_frame = reset_time * (previous_time_frame + 1)

        return surrogate_time_frame, [surrogate_time_frame]


@gin.register
class SurrogateModule(nn.DictLayer):
    """Predict amplitude and compute time parametrization for the
    surrogate synthesis with the complex synthesizer
    (see B.Hayes - Sinusoidal Frequency using Gradient Descent).
    TODO: support several piano models.
    Args:
        - n_harmonics (int): number of partials for each monophonic note.
    Inputs:
        - conditioning (b, n_frames, n_synths, 2): active pitch and onset
        velocity conditionings, for retrieving onset times.
        - extended_pitch (b, n_frames, n_synths, 1): extended active pitch
        conditioning.
    Returns:
        - decays (batch, n_frames, n_harmonics): per-harmonic damping, or
        complex amplitude modulus.
        - decay_time (batch, n_frames, 1): time parametrization that
        resets at each new note onset.
    """
    def __init__(self, n_harmonics=96, **kwargs):
        super().__init__(**kwargs)
        self.midi_norm = 128.
        self.n_harmonics = n_harmonics
        self.amp_model = tfkl.Embedding(128,
                                        self.n_harmonics,
                                        embeddings_initializer='ones')
        self.time_model = tfkl.RNN(OnsetLinspaceCell(), return_sequences=True)

    def call(self, conditioning, extended_pitch) -> ['decays', 'decay_time']:
        decays = self.amp_model(tf.cast(extended_pitch[..., 0],
                                                    dtype=tf.int32))
        decay_time = self.time_model(conditioning[..., 1:2])
        return decays, decay_time


# -----------------------------------------------------------------------------
# Util models
# -----------------------------------------------------------------------------


@gin.register
class F0ProcessorCell(tfkl.Layer):
    """Custom RNN cell for extending MIDI note signals during a trainable
    release time.
    Args:
        - frame_rate (int): number of frames per second.
    Variables:
        - release_duration (1): Release time (in s).
    """

    def __init__(self, frame_rate=250, **kwargs):
        super().__init__(**kwargs)
        self.release_duration = tf.Variable(1.1, name="F0decay")
        self.frame_rate = frame_rate
        self.state_size = 2

    def build(self, input_shape):
        super().build(input_shape)
        self.trainable = False

    def saturated_relu(self, x, threshold=0):
        """0 when <= threshold, 1 when > threshold, linear in between."""
        return tf.minimum(tf.nn.relu(x - threshold), 1.)

    @tf.function
    def call(self, midi_note, previous_state):
        """ Extend note
        Args:
            - midi_note (batch, 1): active MIDI vector frame.
            - previous_state (batch, 2): which note was played for how long.
        """
        previous_note = previous_state[0][..., 0:1]
        release_steps = previous_state[0][..., 1:2]

        # Detect if a note is being input
        note_activity = self.saturated_relu(midi_note, 0)

        # Check if the release end time has been reached
        release_end = self.saturated_relu(
            release_steps, self.release_duration * self.frame_rate
        )
        # Output note is: input note when note is inputted;
        # note in memory when in Release regime;
        # 0 otherwise
        midi_note = note_activity * midi_note \
                  + (1. - note_activity) * previous_note \
                  * (1. - release_end)

        # Advance the count of frames during release
        # Reset to 0 when a note is being played (Release regime not reached yet)
        # Or when the release end time was reached
        release_steps = (release_steps + 1) \
                      * (1. - note_activity) \
                      * (1. - release_end)

        updated_state = tf.concat([midi_note, release_steps], axis=-1)

        return midi_note, [updated_state]


@gin.register
class NoteRelease(nn.DictLayer):
    """Note Release layer for extending the active pitch conditioning.
    RNN wrapper around the custom F0ProcessorCell."""

    def __init__(self, frame_rate=250, name='note_release', **kwargs):
        super().__init__(name=name, **kwargs)
        self.layer = tfkl.RNN(F0ProcessorCell(frame_rate=frame_rate),
                              return_sequences=True)

    def call(self, conditioning) -> ['extended_pitch']:
        active_pitch = conditioning[..., 0:1]
        extended_pitch = self.layer(active_pitch)

        return extended_pitch


@gin.register
class PartialMasking(nn.DictLayer):
    """Set amplitudes of partials above `n_partials` to zero.
    Args:
        - n_partials (int): number of first partial amplitudes to keep.
    """
    def __init__(self, n_partials, **kwargs):
        super().__init__(**kwargs)
        self.n_partials = n_partials

    def call(self, harmonic_distribution, n_partials=None) -> ['harmonic_distribution']:
        if n_partials is None:
            return harmonic_distribution

        batch, n_frames, n_harmonics = tf.shape(harmonic_distribution)

        # Build the partial index
        partial_index = tf.range(n_harmonics)[tf.newaxis, tf.newaxis, ...]
        partial_index = tf.tile(partial_index, [batch, n_frames, 1])

        # Set higher partial amplitudes to zero
        harmonic_distribution = tf.where(
            tf.less(partial_index, n_partials),
            harmonic_distribution,
            -10. * tf.ones_like(harmonic_distribution)
        )
        return harmonic_distribution
