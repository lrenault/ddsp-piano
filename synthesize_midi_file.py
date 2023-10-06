import os
import argparse
import gin
import tensorflow as tf
from soundfile import write
from ddsp.training import trainers, train_util
from ddsp.training.models import get_model
from ddsp_piano.data_pipeline import get_dummy_data
from ddsp_piano.utils.io_utils import load_midi_as_conditioning


def process_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, help="A .gin model config",
                        default='ddsp_piano/configs/default.gin')
    parser.add_argument('--ckpt', type=str, help="Model checkpoint to load.",
                        default='ddsp_piano/model_weights/ckpt-0')
    parser.add_argument('--piano_type', type=int, default=3,
                        help="Piano model (from 0 to 9).\
                              (default: %(default)s)")
    parser.add_argument('--duration', '-d', type=float, default=None,
                        help="Maximum duration of synthesized audio.\
                              (default: %(default)s)")
    parser.add_argument('--unreverbed', '-u', action='store_false',
                        help="Generate unreverbed audio.")
    parser.add_argument('midi_file', type=str,
                        help="Piano MIDI file to synthesize.")
    parser.add_argument('out_file', type=str,
                        help="Save audio as wav file.")
    return parser.parse_args()


def main(args):
    # Load MIDI data
    print("Loading midi file...")
    inputs = load_midi_as_conditioning(args.midi_file, duration=args.duration)
    # Add piano model conditioning
    inputs['piano_model'] = tf.convert_to_tensor([[args.piano_type]])

    print(f"Midi file loaded (with duration {inputs['duration']} s).\
            \nNow building the piano synthesizer...")

    # Parse and override gin-config
    gin.parse_config_file(args.config)
    gin.bind_parameter('%inference', True)
    gin.bind_parameter('%duration', inputs['duration'])

    strategy = train_util.get_strategy()
    with strategy.scope():
        # Model contruction
        model = get_model()
        trainer = trainers.Trainer(model=model, strategy=strategy)
        trainer.build(get_dummy_data(batch_size=1,
                                     duration=inputs['duration'],
                                     sample_rate=model.sample_rate))
        # Restore model weight
        print("Model built, now retrieving model weights...")
        trainer.restore(args.ckpt)

    # Forward pass
    print(f"Model weights loaded from {args.ckpt} \
           \nNow synthesizing audio (this could take some time)...")
    outputs = model(inputs)

    # Save final audio
    write(args.out_file,
          data=outputs['audio_synth'][0].numpy(),
          samplerate=model.sample_rate)

    # Save dry audio (optional)
    if args.unreverbed:
        write(args.out_file + "_unreverbed.wav",
              data=outputs['add']['signal'][0].numpy(),
              samplerate=model.sample_rate)

    print(f"Audio saved at {args.out_file}.")


if __name__ == "__main__":
    # Cannot put too long audio sequences on the GPU memory
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    main(process_args())
