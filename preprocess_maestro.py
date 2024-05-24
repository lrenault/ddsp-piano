import argparse
from absl import logging
from os.path import join as osjoin
from ddsp_piano.data_pipeline import preprocess_data_into_tfrecord


def process_args():
    parser = argparse.ArgumentParser(
        description="Preprocess MAESTRO dataset into TFRecord format."
    )
    parser.add_argument('-sr', '--sample_rate', type=int, default=24000,
                        help="Sample rate for audio files. (default: %(default)s)")
    parser.add_argument('-fr', '--frame_rate', type=int, default=250,
                        help="Frame rate for conditioning. (default: %(default)s)")
    parser.add_argument('-p', '--polyphony', type=int, default=16,
                        help="Maximum polyphony for conditioning. (default: %(default)s)")
    parser.add_argument('maestro_dir', type=str)
    parser.add_argument('out_dir', type=str)
    return parser.parse_args()


def main(args):
    logging.set_verbosity(logging.INFO)

    # Preprocess validation data
    logging.info("Preprocessing validation data...")
    preprocess_data_into_tfrecord(osjoin(args.out_dir, "maestro_validation.tfrecord"),
                                  dataset_dir=args.maestro_dir,
                                  split='validation',
                                  sample_rate=args.sample_rate,
                                  frame_rate=args.frame_rate,
                                  max_polyphony=args.polyphony)
    logging.info(f"Finished. TFrecord file stored at {args.out_dir}/maestro_validation.tfrecord")

    # Preprocess training data
    logging.info("Preprocessing training data...")
    preprocess_data_into_tfrecord(osjoin(args.out_dir, "maestro_train.tfrecord"),
                                  dataset_dir=args.maestro_dir,
                                  split='train',
                                  sample_rate=args.sample_rate,
                                  frame_rate=args.frame_rate,
                                  max_polyphony=args.polyphony)
    logging.info("Finished. TFrecord file stored at {args.out_dir}/maestro_train.tfrecord")


if __name__ == '__main__':
    preprocess_data_into_tfrecord(process_args())
