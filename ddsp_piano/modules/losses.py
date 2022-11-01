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


class LoudnessLoss(SpectralLoss):
    """Loss comparing the loudness between two audio signals from two processor
    outputs.
    Args:
        - target_key (str): target signal processor output key.
        - synth_key (str): synthesize signal processor output key.
    """
    def __init__(self, target_key, synth_key, name='loudness_loss', **kwargs):
        super(LoudnessLoss, self).__init__(mag_weight=0.0,
                                           loudness_weight=1.0,
                                           name=name,
                                           **kwargs)
        self.target_key = target_key
        self.synth_key = synth_key
    
    def call(self, outputs, *args, **kwargs):
        target_signal = outputs[self.target_key]['signal']
        synth_signal = outputs[self.synth_key]['signal']

        return super(SpectralLoss, self).call(target_signal,
                                              synth_signal)
