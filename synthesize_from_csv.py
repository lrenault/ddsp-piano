import os
import argparse
import gin
import numpy as np
import pandas as pd
import tensorflow as tf
from absl import logging
from soundfile import write
from ddsp.training import trainers, train_util
from ddsp.training.models import get_model
from ddsp_piano.data_pipeline import get_dummy_data
from ddsp_piano.utils.io_utils import load_midi_as_conditioning, normalize_audio

osjoin = os.path.join


def process_args():
    parser = argparse.ArgumentParser(description="Synthesize audio from MIDI files using a trained model.")
    parser.add_argument('--config', '-c', type=str, help="A .gin model config",
                        default='ddsp_piano/configs/maestro-v2.gin')
    parser.add_argument('--ckpt', type=str, help="Model checkpoint to load.",
                        default='ddsp_piano/model_weights/v2/')
    parser.add_argument('--duration', '-d', type=float, default=10.0,
                        help="Maximum duration of synthesized audio.\
                              (default: %(default)s)")
    parser.add_argument('--warm_up', '-wu', type=float, default=0.5,
                        help="Warm-up duration (in s, default: %(default)s)")
    parser.add_argument('-n', '--normalize', type=str, default=None,
                        help="Normalize the output audio to the given level (dBFS).\
                              (default: %(default)s)")
    parser.add_argument('--decompose', '-dc', action='store_true',
                        help="Generate isolated piano elements audio.")
    parser.add_argument('maestro_dir', type=str,
                        help="Path to the maestro dataset directory.")
    parser.add_argument('csv_file', type=str,
                        help=".csv file containing paths to MIDI to synthesize.")
    parser.add_argument('out_dir', type=str,
                        help="Folder containing the synthesized test wav files.")
    return parser.parse_args()


def main(args):
    # Parse and override gin-config
    gin.parse_config_file(args.config)
    gin.bind_parameter('%inference', True)
    gin.bind_parameter('%duration', args.duration + args.warm_up)

    # Model contruction
    strategy = train_util.get_strategy()
    with strategy.scope():
        model = get_model()
        trainer = trainers.Trainer(model=model, strategy=strategy)
        trainer.build(get_dummy_data(batch_size=1,
                                     duration=args.duration + args.warm_up,
                                     sample_rate=model.sample_rate))
        trainer.restore(args.ckpt)

    # Load inference dataset
    df = pd.read_csv(args.csv_file)
    piano_models = np.sort(df['piano_model'].unique())

    os.makedirs(args.out_dir, exist_ok=True)
    for i, row in df.iterrows():
        # Load MIDI data
        logging.info(f"Loading file {row['mid_file']}")
        inputs = load_midi_as_conditioning(
            osjoin(args.maestro_dir, row['mid_file']),
            duration=args.duration,
            warm_up_duration=args.warm_up
        )
        # Add piano model conditioning
        piano_model = row['piano_model']
        composer    = row['canonical_composer'].split(' ')[-1]
        inputs['piano_model'] = tf.convert_to_tensor(
             [[np.where(piano_models == piano_model)[0][0]]])
        logging.info(f"Midi file loaded \
                     (with duration {inputs['duration'] - args.warm_up} s).\
                     \nNow synthesizing...")

        # Synthesize
        outs = model(inputs, training=False)
        logging.info("Synthesis done.")

        # Save audio
        write(osjoin(args.out_dir, f'{piano_model}{composer}.wav'),
              outs['audio_synth'][0, int(args.warm_up * model.sample_rate):].numpy(),
              model.sample_rate)
        if args.normalize:
            normalize_audio(
                osjoin(args.out_dir, f'{piano_model}{composer}.wav'),
                args.normalize)

        if args.decompose:
            # Unreverbed audio
            write(osjoin(args.out_dir, f'{piano_model}{composer}_unreverbed.wav'),
                  outs['add']['signal'][0, int(args.warm_up * model.sample_rate):].numpy(),
                  model.sample_rate)
                
            # Additive and residual signals
            additive_synth, substractive_synth = model.processor_group.processors[:2]

            additive_signal = additive_synth.get_signal(
                **additive_synth.get_controls(
                    outs['amplitudes_0'],
                    outs['harmonic_distribution_0'],
                    outs['inharm_coef_0'],
                    outs['f0_hz_0']))
            
            substractive_signal = substractive_synth.get_signal(
                **substractive_synth.get_controls(outs['magnitudes_0']))
            
            for synth_idx in range(1, model.n_synths):
                additive_signal += additive_synth.get_signal(
                    **additive_synth.get_controls(
                        outs[f'amplitudes_{synth_idx}'],
                        outs[f'harmonic_distribution_{synth_idx}'],
                        outs[f'inharm_coef_{synth_idx}'],
                        outs[f'f0_hz_{synth_idx}']))

                substractive_signal += substractive_synth.get_signal(
                    **substractive_synth.get_controls(outs[f'magnitudes_{synth_idx}']))
            
            write(osjoin(args.out_dir, f'{piano_model}{composer}_additive.wav'),
                  additive_signal[0, int(args.warm_up * model.sample_rate):].numpy(),
                  model.sample_rate)
            write(osjoin(args.out_dir, f'{piano_model}{composer}_substractive.wav'),
                  substractive_signal[0, int(args.warm_up * model.sample_rate):].numpy(),
                  model.sample_rate)
            
            if args.normalize:
                normalize_audio(
                    osjoin(args.out_dir, f'{piano_model}{composer}_unreverbed.wav'),
                    args.normalize)
                normalize_audio(
                    osjoin(args.out_dir, f'{piano_model}{composer}_additive.wav'),
                    args.normalize)
                normalize_audio(
                    osjoin(args.out_dir, f'{piano_model}{composer}_substractive.wav'),
                    args.normalize)


if __name__ == '__main__':
    logging.set_verbosity(logging.INFO)
    main(process_args())
