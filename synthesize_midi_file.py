import gin
import argparse
import tensorflow as tf
from absl import logging
from soundfile import write
from ddsp.training import trainers, train_util
from ddsp.training.models import get_model
from ddsp_piano.data_pipeline import get_dummy_data
from ddsp_piano.utils.io_utils import load_midi_as_conditioning, normalize_audio


def process_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, help="A .gin model config.",
                        default='ddsp_piano/configs/maestro-v2.gin')
    parser.add_argument('--ckpt', type=str, help="Model checkpoint to load.",
                        default='ddsp_piano/model_weights/v2/')
    parser.add_argument('--piano_type', type=int, default=9,
                        help="Piano model (from 0 to 9).\
                              (default: %(default)s)")
    parser.add_argument('-wu', '--warm_up', type=float, default=0.5,
                        help="Warm-up duration (in s, default: %(default)s)")
    parser.add_argument('-d','--duration',  type=float, default=None,
                        help="Maximum duration of synthesized audio.\
                              (default: %(default)s)")
    parser.add_argument('-n', '--normalize', type=float, default=None,
                        help="Normalize audio to this amount of dBFS.\
                              (default: %(default)s)")
    parser.add_argument('-u', '--unreverbed', action='store_true',
                        help="Also generates dry piano audio, without reverb.")
    parser.add_argument('midi_file', type=str,
                        help="Piano MIDI file to synthesize.")
    parser.add_argument('out_file', type=str,
                        help="Save audio as wav file.")
    return parser.parse_args()


def main(args):
    # Load MIDI data
    logging.info("Loading midi file...")
    inputs = load_midi_as_conditioning(args.midi_file,
                                       duration=args.duration,
                                       warm_up_duration=args.warm_up)
    # Add piano model conditioning
    inputs['piano_model'] = tf.convert_to_tensor([[args.piano_type]])

    logging.info(f"Midi file loaded (with duration {inputs['duration'] - args.warm_up} s).\
            \nNow building the piano synthesizer...")

    # Parse and override gin-config
    gin.parse_config_file(args.config)
    gin.bind_parameter('%inference', True)
    gin.bind_parameter('%duration', inputs['duration'])

    strategy = train_util.get_strategy()
    with strategy.scope():
        # Model contruction
        model = get_model()
        trainer = trainers.Trainer(model=model,
                                   strategy=strategy)
        trainer.build(get_dummy_data(batch_size=1,
                                     duration=inputs['duration'],
                                     sample_rate=model.sample_rate))
        # Restore model weight
        logging.info("Model built, now retrieving model weights...")
        # trainer.optimizer = tf.keras.optimizers.legacy.Adam()
        trainer.restore(args.ckpt)

    # Forward pass
    logging.info(f"Model weights loaded from {args.ckpt} \
                 \nNow synthesizing audio (this could take some time)...")
    outs = model(inputs)

    # Save final audio
    write(args.out_file,
          data=outs['audio_synth'][0, int(args.warm_up * model.sample_rate):].numpy(),
          samplerate=model.sample_rate)
    if args.normalize:
        normalize_audio(args.out_file, args.normalize)

    # Save dry audio (optional)
    if args.unreverbed:
        write(args.out_file + "_unreverbed.wav",
              data=outs['add']['signal'][0, int(args.warm_up * model.sample_rate):].numpy(),
              samplerate=model.sample_rate)
        if args.normalize:
            normalize_audio(args.out_file + "_unreverbed.wav", args.normalize)

    logging.info(f"Audio saved at {args.out_file}.")


if __name__ == "__main__":
    logging.set_verbosity(logging.INFO)
    main(process_args())
