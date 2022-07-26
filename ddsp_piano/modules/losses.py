import tensorflow as tf

from ddsp.losses import Loss, SpectralLoss


class SpectralLoss(SpectralLoss):
    """Generalized multi-resolution spectral loss by retrieving the audio
    output of a specific processor in the processor group.
    Args:
        - output_key (str): name of the output to extract the audio from.
        - extract_signal (bool): extract audio from signal subkey variable
        (e.g. if the audio is outputs['additive/signal']).
    """

    def __init__(self,
                 output_key='audio_synth',
                 extract_signal=False,
                 **kwargs):
        super(SpectralLoss, self).__init__(**kwargs)
        self.output_key = output_key
        self.extract_signal = extract_signal

    def call(self, outputs, *args, **kwargs):
        synthesized_audio = outputs[self.output_key]

        if self.extract_signal:
            synthesized_audio = synthesized_audio['signal']

        return super(SpectralLoss, self).call(outputs['audio'],
                                              synthesized_audio)


class ReverbRegularizer(Loss):
    """Regularization loss on the reverb impulse response.
    Params:
        - weight (float): loss weight.
        - loss_type {'L1', 'L2'}: compute L1 or L2 regularization.
    """

    def __init__(self, weight=0.01, loss_type='L1', **kwargs):
        super(ReverbRegularizer, self).__init__(**kwargs)
        self.weight = weight
        self.magnitude_order = tf.abs if loss_type == 'L1' else tf.math.square

    def call(self, outputs, *args, **kwargs):
        loss = tf.reduce_sum(self.magnitude_order(outputs['reverb_ir']))
        loss /= outputs['reverb_ir'].shape[0]  # Divide by batch size
        return self.weight * loss
