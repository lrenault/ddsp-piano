import os
import shutil
import argparse
import tensorflow as tf

from tqdm import tqdm
from os.path import join
from ddsp.training import trainers, train_util, summaries
from tensorflow.summary import create_file_writer, scalar

from ddsp_piano.default_model import build_model, get_model
# from ddsp_piano.jaes_model import get_model
# from ddsp_piano.jaes_exp_tanh import get_model
# from ddsp_piano.jaes_relu import get_model
from ddsp_piano.data_pipeline \
    import get_training_dataset, get_validation_dataset
from ddsp_piano.utils.io_utils import collect_garbage
from ddsp_piano.utils.summaries import inharm_summary, detune_summary


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

    parser.add_argument('--lr', type=float, default=0.001,
                        help="Learning rate. (default: %(default)s)")

    parser.add_argument('--phase', '-p', type=int, default=1,
                        help="Training phase strategy to apply. \
                        Set to even for fine-tuning only the detuner and \
                        inharmonicity sub-modules.\
                        (default: %(default)s)")

    parser.add_argument('--restore', type=str, default=None,
                        help="Restore training step from a saved folder.\
                        (default: %(default)s)")

    parser.add_argument('--val_path', type=str, default=None,
                        help="Path to the validation data (if different from maestro_path).\
                        (default: %(default)s)")

    parser.add_argument('maestro_path', type=str,
                        help="Path to the MAESTRO dataset folder.")

    parser.add_argument('exp_dir', type=str,
                        help="Folder to store experiment results and logs.")

    return parser.parse_args()


def main(args):
    """Training loop script.
    Args:
        - batch_size (int): nb of elements per batch.
        - steps_per_epoch (int): nb of steps of gradient descent per epoch.
        - epochs (int): nb of epochs.
        - restore (path): load model and optimizer states from this folder.
        - phase (int): current training phase.
        - maestro_path (path): maestro dataset location.
        - exp_dir (path): folder to store experiment results and logs.
    """
    # TODO (lrenault): remove
    from admis.io_utils import lock_gpu; lock_gpu()

    # Format training phase strategy
    first_phase_strat = ((args.phase % 2) == 1)

    # Build/Load and put the model in the available strategy scope
    strategy = train_util.get_strategy()
    with strategy.scope():
        model = build_model(get_model(),
                            batch_size=args.batch_size,
                            first_phase=first_phase_strat)
        trainer = trainers.Trainer(model=model,
                                   strategy=strategy,
                                   learning_rate=args.lr)
        # Restore model and optimizer states
        if args.restore is not None:
            trainer.restore(args.restore)
            print(f"Restored model from {args.restore}")

    loss_keys = model._losses_dict.keys()

    # Dataset loading
    val_path = args.maestro_path if args.val_path is None else args.val_path

    training_dataset = get_training_dataset(args.maestro_path,
                                            batch_size=args.batch_size,
                                            max_polyphony=model.n_synths)
    val_dataset = get_validation_dataset(val_path,
                                         batch_size=args.batch_size,
                                         max_polyphony=model.n_synths)
    # Dataset distribution
    with strategy.scope():
        training_dataset = trainer.distribute_dataset(training_dataset)
        val_dataset = trainer.distribute_dataset(val_dataset)

        train_iterator = iter(training_dataset)

    # Inits before the training loop
    exp_dir = join(args.exp_dir, f'phase_{args.phase}')

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
                epoch_losses = {k: 0. for k in loss_keys}
                for _ in tqdm(range(args.steps_per_epoch)):
                    # Train step
                    losses = trainer.train_step(train_iterator)
                    # Retrieve loss values
                    for k in loss_keys:
                        epoch_losses[k] += float(tf.debugging.check_numerics(
                            losses[k],
                            message=f"Nan loss at step {trainer.step}"))

                # Write loss values in tensorboard
                print("Training loss:",
                      epoch_losses['total_loss'] / args.steps_per_epoch)
                for k, loss in epoch_losses.items():
                    scalar('train_loss/' + k,
                           loss / args.steps_per_epoch,
                           step=step)
                if not first_phase_strat:
                    inharm_summary(model.inharm_model, step=step)
                    detune_summary(model.inharm_model, step=step)

                # Save model epoch before validation
                shutil.rmtree(join(exp_dir, "last_iter"))
                trainer.save(join(exp_dir, "last_iter"))
                print('Last iteration model saved at',
                      join(exp_dir, "last_iter"))

                # Skip validation during early training
                if trainer.step < 60000:
                    collect_garbage()
                    continue

                # Evaluate on validation data
                print("Validation...")
                val_outs_summary = None
                epoch_val_losses = {k: 0. for k in loss_keys}
                for val_step, val_batch in enumerate(tqdm(val_dataset)):
                    # Validation step
                    outputs, val_losses = model(val_batch,
                                                return_losses=True,
                                                training=True)
                    # Retrieve loss values
                    for k in loss_keys:
                        epoch_val_losses[k] += float(val_losses[k])

                    if val_step == 0:
                        val_outs_summary = outputs

                # Write loss values in tensorboard
                print("Validation loss:",
                      epoch_val_losses['total_loss'] / (val_step + 1))
                for k, loss in epoch_val_losses.items():
                    scalar('val_loss/' + k,
                           loss / (val_step + 1),
                           step=step)
                summaries.audio_summary(val_outs_summary['audio_synth'],
                                        step,
                                        sample_rate=16000,
                                        name='synthesized_audio')
                # Save if better epoch
                if epoch_val_losses['audio_stft_loss'] < lowest_val_loss:
                    lowest_val_loss = epoch_val_losses['audio_stft_loss']
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
