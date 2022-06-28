import os
import shutil
import argparse
import tensorflow as tf

from tqdm import tqdm
from os.path import join
from datetime import datetime
from ddsp.training import trainers, train_util, summaries
from tensorflow.summary import create_file_writer, scalar

from polyphonic_model import build_model
from admis.training.data_pipeline \
    import get_training_dataset, get_validation_dataset
from admis.io_utils import lock_gpu, str2bool, collect_garbage

DATARD_PATH = os.environ['DATARD_PATH']
base_datasets_path = "audio_database/maestro-v3.0.0/"


def process_args():
    # Get arguments from command line
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size', type=int, default=6,
                        help="Number of elements per batch. (default: %(default)s)")

    parser.add_argument('--n_synths', type=int, default=16,
                        help="Polyphony capacity of the model. (default: %(default)s)")

    parser.add_argument('--config_id', type=str, default="default",
                        help="Select training config. (default: %(default)s)")

    parser.add_argument('--steps_per_epoch', type=int, default=16,
                        help="Number of steps of gradient descent per epoch. (default: %(default)s)")

    parser.add_argument('--epochs', type=int, default=128,
                        help="Number of epochs. (default: %(default)s)")

    parser.add_argument('--jz', type=str2bool, default=False,
                        help="Train on Jean-Zay. (default: %(default)s)")

    parser.add_argument('--restore', type=str, default=None,
                        help="Restore from a pre-trained model. (default: %(default)s)")

    parser.add_argument('--train_inharm', type=str2bool, default=False,
                        help="Only train inharmonicity sub-network. (default: %(default)s)")

    parser.add_argument('--exp_dir', type=str,
                        help="Path for experiments results and log export",
                        default=join(DATARD_PATH,
                                     base_datasets_path,
                                     "models/admis/"))
    return parser.parse_args()


def main(args):
    # Lock GPU(s)
    lock_gpu()

    # Initialize the datasets
    training_dataset = get_training_dataset(
        batch_size=args.batch_size,
        max_polyphony=args.n_synths,
        jz=args.jz,
        one_piano=("2009" in args.config_id)
    )
    val_dataset = get_validation_dataset(
        batch_size=args.batch_size,
        max_polyphony=args.n_synths,
        jz=args.jz,
        one_piano=("2009" in args.config_id)
    )
    # Build/Load and put the model in the available strategy scope
    strategy = train_util.get_strategy()
    with strategy.scope():
        model = build_model(batch_size=args.batch_size,
                            n_synths=args.n_synths,
                            config_id=args.config_id,
                            use_detune=args.train_inharm,
                            jz=args.jz)
        trainer = trainers.Trainer(
            model=model,
            strategy=strategy,
            learning_rate=1e-5 if args.train_inharm else 1e-3
        )
        if args.restore is not None:
            trainer.restore(args.restore)
            print(f"Restored model from {args.restore}")

        if args.train_inharm:
            # TODO: use alternate optimizer
            model.alternate_training()
            model.summary()

        training_dataset = trainer.distribute_dataset(training_dataset)
        val_dataset = trainer.distribute_dataset(val_dataset)

        train_iterator = iter(training_dataset)

    # Inits before the training loop
    today = datetime.today().strftime('%Y-%m-%d-%Hh%M')

    exp_dir = args.exp_dir if (args.restore is None) else args.restore
    exp_dir = join(exp_dir, today)

    os.makedirs(join(exp_dir, "logs"), exist_ok=True)
    os.makedirs(join(exp_dir, "last_iter"), exist_ok=True)
    os.makedirs(join(exp_dir, "best_iter"), exist_ok=True)

    summary_writer = create_file_writer(join(exp_dir, "logs"))

    # Training loop
    lowest_val_loss = 9999999.
    try:
        with summary_writer.as_default():
            for epoch in range(args.epochs):
                print("Epoch:", epoch, "\nTraining...")
                step = trainer.step  # step != epoch if resuming training

                # Fit training data
                train_loss = 0.
                regularization_loss = 0.
                reverb_loss = 0.
                spectral_loss = 0.
                for _ in tqdm(range(args.steps_per_epoch)):
                    # Train step
                    losses = trainer.train_step(train_iterator)
                    # Retrieve loss values
                    train_loss += float(losses['total_loss'])
                    regularization_loss += float(losses['regularization_loss'])
                    reverb_loss += float(losses['reverb_regularizer'])
                    spectral_loss += float(losses['audio_stft_loss'])

                # Write loss values in tensorboard
                print("Training loss:", train_loss / args.steps_per_epoch)
                scalar('train_loss',
                       train_loss / args.steps_per_epoch,
                       step=step)
                scalar('train_loss/spectral',
                       spectral_loss / args.steps_per_epoch,
                       step=step)
                scalar('train_loss/reverb',
                       reverb_loss / args.steps_per_epoch,
                       step=step)
                scalar('train_loss/regularization',
                       regularization_loss / args.steps_per_epoch,
                       step=step)

                # Save model epoch before validation
                shutil.rmtree(join(exp_dir, "last_iter"))
                trainer.save(join(exp_dir, "last_iter"))

                # Evaluate on validation data
                if trainer.step < 60000:
                    continue
                print("Validation...")
                val_outs_summary = None
                val_loss = 0.
                val_spectral = 0.
                val_reverb = 0.
                val_regularization = 0.
                for val_step, val_batch in enumerate(tqdm(val_dataset)):
                    # Validation step
                    outputs, val_losses = model(val_batch,
                                                return_losses=True,
                                                training=False)
                    # Retrieve loss values
                    val_loss += float(val_losses['total_loss'])
                    val_regularization += float(val_losses['regularization_loss'])
                    val_spectral += float(val_losses['audio_stft_loss'])
                    val_reverb += float(val_losses['reverb_regularizer'])

                    if val_step == 0:
                        val_outs_summary = outputs

                # Write loss values in tensorboard
                print("Validation loss:", val_loss / (val_step + 1))
                scalar('val_loss', val_loss / (val_step + 1),
                       step=step)
                scalar('val_loss/spectral',
                       val_spectral / (val_step + 1),
                       step=step)
                scalar('val_loss/reverb',
                       val_reverb / (val_step + 1),
                       step=step)
                scalar('val_loss/regularization',
                       val_regularization / (val_step + 1),
                       step=step)
                summaries.audio_summary(val_outs_summary['audio_synth'],
                                        step,
                                        sample_rate=16000,
                                        name='synthesized_audio')
                # Save if better epoch
                if val_spectral < lowest_val_loss:
                    lowest_val_loss = val_spectral
                    trainer.save(join(exp_dir, "best_iter"))

                # Collect garbage
                collect_garbage()

    except tf.errors.InvalidArgumentError as e:
        trainer.save(join(exp_dir, "crashed_iter"))
        print(e)

    except KeyboardInterrupt:
        trainer.save(join(exp_dir, "stopped_iter"))


if __name__ == '__main__':
    main(process_args())