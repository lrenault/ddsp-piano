import gin
import argparse
import tensorflow as tf

from tqdm import tqdm
from ddsp.training import trainers, train_util
from ddsp.training.models import get_model
from ddsp_piano.data_pipeline import single_track_dataset
from ddsp_piano.default_model import build_model, get_model


def process_args():
    # Get arguments from command line
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size', '-b', type=int, default=6,
                        help="Number of elements per batch.\
                        (default: %(default)s)")

    parser.add_argument('--steps_per_epoch', '-s', type=int, default=16,
                        help="Number of steps of gradient descent per epoch.\
                        (default: %(default)s)")

    parser.add_argument('--epochs', '-e', type=int, default=128,
                        help="Number of epochs. (default: %(default)s)")

    parser.add_argument('--config', '-c',
                        default='ddsp_piano/configs/default.gin',
                        help="A .gin configuration file.")

    parser.add_argument('midi_file', type=str, default=None,
                        help=".mid of the track to train on.")

    parser.add_argument('audio_file', type=str, default=None,
                        help=".wav file of the track to train on.")

    parser.add_argument('exp_dir', type=str,
                        help="Folder to store experiment results and logs.")

    return parser.parse_args()


def main(args):
    # Get training data
    dataset = single_track_dataset(args.midi_file,
                                   args.audio_file,
                                   batch_size=args.batch_size)
    # Build and distribute model
    gin.parse_config_file(args.config)
    strategy = train_util.get_strategy()
    with strategy.scope():
        trainer = trainers.Trainer(
            model=get_model().alternate_training(first_phase=True),
            strategy=strategy,
            learning_rate=1e-3)
        # Model building
        _ = trainer.run(
            tf.function(trainer.model.__call__),
            get_dummy_data(batch_size=args.batch_size,
                           sample_rate=model.sample_rate),
            training=True,
            return_losses=True,
        )
        trainer.model.summary()
        # Distribute the dataset
        dataset = trainer.distribute_dataset(dataset)

    # Training
    loss_keys = model._losses_dict.keys()
    lowest_loss_value = 999999.
    for epoch in range(args.epochs):
        # Fit data
        epoch_losses = {k: 0. for k in loss_keys}
        for train_step, train_batch in enumerate(tqdm(dataset,
                                                      desc=f"Epoch {epoch}",
                                                      ncols=64)):
            losses = trainer.train_step(train_batch)
            # Retrieve loss values
            for k in loss_keys:
                epoch_losses[k] += float(tf.debugging.check_numerics(
                    losses[k],
                    message=f"Nan loss at step {trainer.step}"))
        print(epoch_losses)

        # Save model checkpoint
        if losses['total_loss'] < lowest_loss_value:
            trainer.save(args.exp_dir)
            lowest_loss_value = losses['total_loss']

if __name__ == '__main__':
    main(process_args())
