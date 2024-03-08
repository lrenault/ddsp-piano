import gin
import tensorflow as tf
from math import pi
from ddsp import processors
from ddsp.core import tf_float32, fft_convolve

tfkl = tf.keras.layers


def tf_complex64(x):
    """Ensure array/tensor is a complex64 tf.Tensor."""
    if isinstance(x, tf.Tensor):
        return tf.cast(x, dtype=tf.complex64)
    else:
        if x.dtype is tf.float32:
            x = tf.complex(x, tf.zeros_like(x))
        return tf.convert_to_tensor(x, tf.complex64)


@gin.register
class FeedbackDelayNetwork(processors.Processor):
    """Differentiable Feedback Delay Network reverb.
    Simplified from https://github.com/phvial/priv-ddfx/blob/main/effects.py
    (credits: Pierre-Hugo Vial, 2023, AQUA-RIUS project).
    Fixed config:
    - decorrelator: allpass
    - mixing_matrix_style: Householder
    - reverb_time_control: onepole
    - early_rev_style: fir
    - delay_lines: 8
    - VNS_input: False
    - VNS_output: False
    """
    def __init__(
        self,
        trainable=False,
        name="DelayNetwork",
        sampling_rate=16000.0,
        delay_lines=8,
        delay_values=None,
        delays_allpass=None,
        early_ir_length=200,
        early_reflections=6,
        time_control_bands=6,
        delay_trainable = False,
    ):
        """Initializes DelayNetwork object.

        Parameters
        ----------
        trainable : bool, optional
            trainable, by default False
        name : str, optional
            name, by default "DelayNetwork"
        freq_points : int, optional
            number of frequency points for frequency sampling, by default 200
        sampling_rate : float, optional
            sampling rate, by default 1.0
        delay_lines : int, optional
            number of delay lines, by default 6
        delay_values : _type_, optional
            input delay values, deprecated, by default None
        delays_allpass : _type_, optional
            input delay values in allpass filters, deprecated, by default None
        early_ir_length : int, optional
            length of IR for early reflections, by default 200
        mixing_matrix_style : str, optional
            mixing matrix, in ['Householder', 'Hadamard', 'RandomOrthogonal', 'trainable'], by default "Householder"
        reverb_time_control : str, optional
            method for reverberation time control, in ['onepole', 'eq', 'eq-ls', ...] by default "onepole"
        early_rev_style : str, optional
            early reflections model, in ['fir, ...], by default "fir"
        early_reflections : int, optional
            number of early reflections with early_rev_style!='fir, by default 6
        time_control_bands : int, optional
            number of bands for reverberation time control, by default 6
        """
        super().__init__(name=name, trainable=trainable)
        self.sampling_rate = tf_float32(sampling_rate)
        self.freq_points = int(2 * self.sampling_rate)
        self.delay_values = delay_values
        self.delays_allpass = delays_allpass
        self.early_ir_length = early_ir_length
        self.delay_lines = delay_lines
        self.early_reflections = early_reflections
        self.time_control_bands = time_control_bands
        self.delay_trainable = delay_trainable

    def __len__(self):
        return self.delay_lines
    
    def build(self, input_shape):
        """Builds DelayNetwork object."""
        if self.delay_values is None:
            if not self.trainable or not self.delay_trainable:
                self.delay_values = tf_float32([233, 311, 421, 461, 587, 613, 789, 891]) # last two values at random
                self.delay_lines = len(self.delay_values)

        # ---- builds decorrelator
        if self.delays_allpass is None:
            if not self.trainable:
                self.delays_allpass = tf_float32(
                    [
                        [131, 151, 337, 353],
                        [103, 173, 331, 373],
                        [ 89, 181, 307, 401],
                        [ 79, 197, 281, 419],
                        [ 61, 211, 257, 431],
                        [ 47, 229, 251, 443],
                        [ 81, 189, 287, 407],
                        [ 91, 203, 321, 377]
                    ]
                )
                if self.delays_allpass.shape[0] < 6:
                    self.delays_allpass = self.delays_allpass[: self.delay_lines]
        
        # ---- builds mixing matrix
        self.mixing_matrix = -1 * tf.eye(self.delay_lines) + 0.5 * tf.ones(
            [self.delay_lines, self.delay_lines]
        )
        # ---- builds trainable variables
        if self.trainable:
            general_initializer = tf.random_normal_initializer(mean=0.0, stddev=1e-1)
            delay_initializer = tf.random_normal_initializer(mean=400.0, stddev=60.0)
            gains_initializer = tf.random_normal_initializer(mean=0.25, stddev=1e-1)
            self.early_ir = self.add_weight(
                name="Early ir",
                shape=[self.early_ir_length],
                dtype=tf.float32,
                initializer=general_initializer,
            )
            self.input_gain = self.add_weight(
                name="Input gain",
                shape=[self.delay_lines],
                dtype=tf.float32,
                initializer=gains_initializer,
            )
            self.output_gain = self.add_weight(
                name="Output gain",
                shape=[self.delay_lines],
                dtype=tf.float32,
                initializer=gains_initializer,
            )
            self.time_rev_0_sec = self.add_weight(
                name="Reverberation time (s) at 0 Hz",
                shape=[],
                dtype=tf.float32,
                initializer=tf.random_normal_initializer(mean=2.0, stddev=5e-1),
                constraint=tf.keras.constraints.NonNeg(),
            )
            self._alpha_tone = self.add_weight(
                name="_alpha_tone",
                shape=[],
                dtype=tf.float32,
                initializer=general_initializer,
            )
            if self.delay_trainable:
                self.delay_values = self.add_weight(
                    name="Delay values",
                    shape=[self.delay_lines],
                    dtype=tf.float32,
                    initializer=delay_initializer,
                )
            self.delays_allpass = self.add_weight(
                name="Delay allpass",
                shape=[self.delay_lines, 4],
                dtype=tf.float32,
                initializer=delay_initializer,
            )
            self.gain_allpass = self.add_weight(
                name="Allpass filters gains",
                shape=[self.delay_lines, self.delays_allpass.shape[1]],
                dtype=tf.float32,
                initializer=gains_initializer,
            )
        super().build(input_shape)

    def get_late_ir(
        self,
        input_gain: tf.Tensor,
        output_gain: tf.Tensor,
        mixing_matrix: tf.Tensor,
        gain_allpass: tf.Tensor,
        delays_allpass: tf.Tensor,
        time_rev_0_sec: tf.Tensor,
        alpha_tone: tf.Tensor
    ):
        """Returns IR for late reverberation.

        Parameters
        ----------
        output_gain : tf.Tensor
            output gain vector
        input_gain : tf.Tensor
            input gain vector
        mixing_matrix : tf.Tensor
            mixing matrix
        gain_allpass : tf.Tensor
            gains of allpass filters
        delays_allpass : tf.Tensor
            delays of allpass filters

        Returns
        -------
        tf.Tensor
            IR for late reverberation

        Raises
        ------
        NotImplementedError
            _description_
        """
        input_gain = tf.complex(input_gain, 0.0)
        output_gain = tf.complex(output_gain, 0.0)
        mixing_matrix = tf.complex(mixing_matrix, 0.0)
        
        if len(mixing_matrix.shape) == 2:
            mixing_matrix = tf.expand_dims(mixing_matrix, axis=0)
            mixing_matrix = tf.tile(mixing_matrix, [self.freq_points // 2 + 1, 1, 1])
        if len(output_gain.shape) == 1:
            output_gain = tf.expand_dims(output_gain, axis=0)
            output_gain = tf.expand_dims(output_gain, axis=0)
            output_gain = tf.tile(output_gain, [self.freq_points // 2 + 1, 1, 1])
        if len(input_gain.shape) == 1:
            input_gain = tf.expand_dims(input_gain, axis=1)
            input_gain = tf.expand_dims(input_gain, axis=0)
            input_gain = tf.tile(input_gain, [self.freq_points // 2 + 1, 1, 1])
        eye_mat = tf.eye(
            mixing_matrix.shape[-1],
            batch_shape=[self.freq_points // 2 + 1],
            dtype=tf.complex64,
        )

        # generate normalized frequencies vector
        wk = tf_complex64(
            2
            * pi
            * tf.range(self.freq_points // 2 + 1, dtype=tf.float32)
            / self.freq_points
        )  # shape: [freq_points//2+1]

        z_d = tf.stack(
            [
                tf.exp(
                    -1j * wk * tf_complex64(tf.floor(self.delay_values[d]))
                )  # modif: ajout de tf floor
                for d in range(self.delay_lines)
            ],
            axis=1,
        )  # shape:  [freq_points//2+1, delay_lines], matrix of z^{-d}, elements of D
        # shape:  [freq_points//2+1, delay_lines, delay_lines]

        d_eta = tf_complex64(self.delay_values - tf.floor(self.delay_values))
        eta = (1 - d_eta) / (1 + d_eta)
        allpass_interp = tf.stack(
            [
                (eta[d] + tf.exp(-1j * wk)) / (1 + eta[d] * tf.exp(-1j * wk))
                for d in range(self.delay_lines)
            ],
            axis=1,
        )  # 

        # diag_delay_matrix = tf.linalg.diag(z_d)
        diag_delay_matrix = tf.linalg.diag(z_d * allpass_interp)  # 
        delay_sec = (
            tf_float32(self.delay_values)
            + tf.reduce_sum(delays_allpass, axis=-1)  # ajout tf_float32
        ) / self.sampling_rate  

        # ---- (if) Onepole LP filter reverb time control
        # shape[1, delay_lines]
        k = tf.expand_dims(tf.pow(10.0, -3 * delay_sec / time_rev_0_sec),
                           axis=0)
        kpi = tf.expand_dims(
            tf.pow(10.0, -3 * delay_sec / (alpha_tone * time_rev_0_sec)),
            axis=0
        )
        g = 2 * k * kpi / (k + kpi)
        p = (k - kpi) / (k + kpi)

        # shape[freq_points//2+1, delay_lines]
        # g_tiled = tf_complex64(tf.tile(g, [self.freq_points // 2 + 1, 1]))
        # p_tiled = tf_complex64(tf.tile(p, [self.freq_points // 2 + 1, 1]))
        g_tiled = tf_complex64(tf.repeat(g, self.freq_points // 2 + 1, axis=0))
        p_tiled = tf_complex64(tf.repeat(p, self.freq_points // 2 + 1, axis=0))
        # z_tiled = tf.stack(
        #     [tf.exp(-1j * wk) for d in range(self.delay_lines)], axis=-1
        # )  # shape[freq_points//2+1, delay_lines]
        z_tiled = tf.exp(-1j * wk)
        for _ in range(len(g_tiled.shape) - 1): z_tiled = z_tiled[..., tf.newaxis]
        self.sampled_transfers = tf.transpose(
            g_tiled / (1 - p_tiled * z_tiled + 1e-8)
        )
        filter_matrix = tf.linalg.diag(g_tiled / (1 - p_tiled * z_tiled + 1e-8))

        gain_allpass = tf_complex64(tf.expand_dims(gain_allpass, axis=0))
        # gain_allpass_tiled = tf.tile(gain_allpass, [self.freq_points // 2 + 1, 1, 1])
        gain_allpass_tiled = tf.repeat(gain_allpass, self.freq_points // 2 + 1, axis=0)
    
        delays_allpass = tf_complex64(tf.expand_dims(delays_allpass, axis=0))
        delays_allpass_tiled = tf.repeat(delays_allpass, self.freq_points // 2 + 1, axis=0)
        # delays_allpass_tiled = tf.tile(
        #     delays_allpass, [self.freq_points // 2 + 1, 1, 1]
        # )
        wk_tiled = wk# [:, tf.newaxis, tf.newaxis]
        for _ in range(len(delays_allpass_tiled.shape) - 1):
            wk_tiled = wk_tiled[..., tf.newaxis]
        # wk_tiled = tf.tile(
        #     wk_tiled, [1, self.delay_lines, delays_allpass_tiled.shape[2]]
        # )
        z_delays = tf.exp(1j * wk_tiled * delays_allpass_tiled)  
    
        # transfer function of allpass filters
        allpass_transfer = tf.reduce_prod(
            (1 + gain_allpass_tiled * z_delays) / (gain_allpass_tiled + z_delays),
            axis=-1,  # 2
        )
        self.allpass_transfer = tf.linalg.diag(allpass_transfer)
    
        # generate feedback matrix
        if len(filter_matrix.shape) == 4:
            mixing_matrix = tf.expand_dims(mixing_matrix, 1)
        feedback_matrix = tf.matmul(
            tf.matmul(filter_matrix, mixing_matrix), self.allpass_transfer
        )
        # feedback_matrix = mixing_matrix #uncomment to bypass allpass filters + time control filters
        if len(feedback_matrix.shape) == 4:
            eye_mat = tf.expand_dims(eye_mat, 1)
            diag_delay_matrix = tf.expand_dims(diag_delay_matrix, 1)
            input_gain = input_gain[tf.newaxis, :, :, tf.newaxis]
            output_gain = output_gain[tf.newaxis, :, tf.newaxis, :]
        ir = tf.signal.irfft(
            tf.squeeze(
                tf.matmul(
                    output_gain,
                    tf.matmul(
                        tf.matmul(
                            diag_delay_matrix,
                            tf.linalg.inv(
                                eye_mat - tf.matmul(feedback_matrix, diag_delay_matrix)
                            ),
                        ),
                        input_gain,
                    ),
                )
            )
        )
        return ir

    def get_ir(self,
               input_gain: tf.Tensor,
               output_gain: tf.Tensor,
               gain_allpass: tf.Tensor,
               delays_allpass: tf.Tensor,
               time_rev_0_sec: tf.Tensor,
               alpha_tone: tf.Tensor,
               early_ir: tf.Tensor,
               ):
        """Returns reverb IR."""
        late_ir = self.get_late_ir(input_gain,
                                   output_gain,
                                   self.mixing_matrix,
                                   gain_allpass,
                                   delays_allpass,
                                   time_rev_0_sec,
                                   alpha_tone)
        early_ir = tf.squeeze(early_ir)  # Squeeze ?
        if late_ir.shape[0] > early_ir.shape[0]:
            early_ir = tf.pad(early_ir,
                              [[0, late_ir.shape[0] - early_ir.shape[0]]],)
        return early_ir[: late_ir.shape[0]] + late_ir

    def get_controls(self,
                     audio_dry: tf.Tensor=None,
                     input_gain: tf.Tensor=None,
                     output_gain: tf.Tensor=None,
                     gain_allpass: tf.Tensor=None,
                     delays_allpass: tf.Tensor=None,
                     time_rev_0_sec: tf.Tensor=None,
                     alpha_tone: tf.Tensor=None,
                     early_ir: tf.Tensor=None):
        """Returns controls.

        Parameters
        ----------
        audio_dry : tf.Tensor, optional
            input audio tensor, by default None

        Returns
        -------
        dict
            controls
        """
        if self.trainable:
            controls_dict = {
                "input_gain": self.input_gain,
                "output_gain": self.output_gain,
                "gain_allpass": self.gain_allpass,
                "delays_allpass": self.delays_allpass,
                "time_rev_0_sec": self.time_rev_0_sec,
                "alpha_tone": tf.keras.activations.sigmoid(self.alpha_tone),
                "early_ir": self.early_ir,
            }
        else:
            controls_dict = {
                "input_gain": input_gain,
                "output_gain": output_gain,
                "gain_allpass": gain_allpass,
                "delays_allpass": delays_allpass,
                "time_rev_0_sec": time_rev_0_sec,
                "alpha_tone": alpha_tone,
                "early_ir": early_ir,
            }
        ir = self.get_ir(**controls_dict)

        return {'audio': audio_dry, 'ir': ir}

    def get_signal(self, audio: tf.Tensor, ir: tf.Tensor) -> tf.Tensor:
        # ir = tf.expand_dims(self.get_ir(**kwargs), axis=0)
        ir = tf.expand_dims(ir, axis=0)
        audio_out = fft_convolve(audio, ir, delay_compensation=0)
        return audio_out


if __name__ == "__main__":
    layer = FeedbackDelayNetwork(trainable=True)
    layer.build(input_shape=(1000,))
    import pdb; pdb.set_trace()