import argparse
import tensorflow as tf
from soundfile import write
from ddsp.training import trainers, train_util
from ddsp_piano.default_model import get_model, build_model
from ddsp_piano.utils.io_utils import load_midi_as_conditioning


def process_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--piano_type', type=int, default=3,
                        help="Piano model. (default: %(default)s)")
    parser.add_argument('midi_file', type=str,
                        help="Piano MIDI file to synthesize.")
    parser.add_argument('out_file', type=str,
                        help="Save audio as wav file.")
    return parser.parse_args()


def main(args):
    # Load MIDI data
    print("Loading midi file...")
    inputs = load_midi_as_conditioning(args.midi_file)
    inputs['piano_model'] = tf.convert_to_tensor([[args.piano_type]])

    # Model contruction
    print(f"Midi file loaded (with duration {inputs['duration']} s).\
            \nNow building the piano synthesizer...")
    strategy = train_util.get_strategy()
    with strategy.scope():
        model = build_model(get_model(inference=True,
                                      duration=inputs['duration']),
                            batch_size=1,
                            duration=inputs['duration'],
                            )
        # Restore model weight
        print("Model built, now retrieving model weights...")
        trainer = trainers.Trainer(model, strategy=strategy)
        trainer.restore(
            '/data3/anasynth_nonbp/renault/audio_database/maestro-v3.0.0/models/admis/recoded/ckpt-0'
        )

    # Forward pass
    print("Model retrieved with default weights. \
           \nNow synthesizing audio (this could take some time)...")
    outputs = model(inputs)
    write(args.out_file,
          data=outputs['audio_synth'][0].numpy(),
          samplerate=16000)
    print(f"Audio saved at {args.out_file}.")


if __name__ == "__main__":
    main(process_args())