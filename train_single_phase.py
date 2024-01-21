import os
import gin
import argparse
import tensorflow as tf

from tqdm import tqdm
from ddsp.training import trainers, train_util, summaries
from ddsp.training.models import get_model
from tensorflow.summary import create_file_writer, scalar

from ddsp_piano.data_pipeline \
    import get_dummy_data, get_training_dataset, get_validation_dataset
from ddsp_piano.utils.io_utils import collect_garbage
from ddsp_piano.utils.summaries import inharm_summary, detune_summary

osjoin = os.path.join


def process_args():
    # Get arguments from command line
    parser = argparse.ArgumentParser()

    parser.add_argument('--n_gpus', '-gpu', type=int, default=1,
                        help="Number of GPUs to lock. (default: %(default)s)")

    parser.add_argument('--batch_size', '-b', type=int, default=6,
                        help="Number of elements per batch.\
                        (default: %(default)s)")

    parser.add_argument('--steps_per_epoch', '-s', type=int, default=5000,
                        help="Number of steps of gradient descent per epoch.\
                        (default: %(default)s)")

    parser.add_argument('--epochs', '-e', type=int, default=128,
                        help="Number of epochs. (default: %(default)s)")

    parser.add_argument('--lr', type=float, default=0.001,
                        help="Learning rate. (default: %(default)s)")

    parser.add_argument('--config', '-c',
                        default='ddsp_piano/configs/default.gin',
                        help="A .gin configuration file.")

    parser.add_argument('--phase', '-p', type=int, default=1,
                        help="Training phase strategy to apply. \
                        Set to even for fine-tuning only the detuner and \
                        inharmonicity sub-modules.\
                        (default: %(default)s)")

    parser.add_argument('--restore', '-r', type=str, default=None,
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


def lock_gpu(soft=True, gpu_device_id=-1):
    import socket

    if "ircam.fr" in socket.gethostname():
        import manage_gpus as gpl
    else:
        return None

    try:
        id_locked = gpl.get_gpu_lock(gpu_device_id=gpu_device_id, soft=soft)
        print(f"Locked GPU {id_locked}")
    except gpl.NoGpuManager:
        id_locked = None
        print("No gpu manager available - will use all available GPUs")
    except gpl.NoGpuAvailable:
        # no GPU available for locking, continue with CPU
        id_locked = None
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    return id_locked


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
    _ = [lock_gpu() for _ in range(args.n_gpus)]

    # Format training phase strategy
    first_phase_strat = ((args.phase % 2) == 1)

    # Build/Load and put the model in the available strategy scope
    gin.parse_config_file(args.config)
    strategy = train_util.get_strategy()
    with strategy.scope():
        model = get_model()
        model.alternate_training(first_phase=first_phase_strat)

        trainer = trainers.Trainer(model=model,
                                   strategy=strategy,
                                   learning_rate=args.lr)
        # Model building
        _ = trainer.run(
            tf.function(trainer.model.__call__),
            get_dummy_data(batch_size=args.batch_size,
                           sample_rate=model.sample_rate),
            training=True,
            return_losses=True,
        )
        trainer.model.summary()
        # Restore model and optimizer states
        if args.restore is not None:
            trainer.restore(args.restore)
            print(f"Restored model from {args.restore} at step {trainer.step.numpy()}")

    # Dataset loading
    val_path = args.maestro_path if args.val_path is None else args.val_path

    training_dataset = get_training_dataset(args.maestro_path,
                                            batch_size=args.batch_size,
                                            max_polyphony=model.n_synths,
                                            sample_rate=model.sample_rate)
    val_dataset = get_validation_dataset(val_path,
                                         batch_size=args.batch_size,
                                         max_polyphony=model.n_synths,
                                         sample_rate=model.sample_rate)
    # Dataset distribution
    with strategy.scope():
        training_dataset = trainer.distribute_dataset(training_dataset)
        val_dataset = trainer.distribute_dataset(val_dataset)

        train_iterator = iter(training_dataset)

    # Inits before the training loop
    exp_dir = osjoin(args.exp_dir, f'phase_{args.phase}')

    os.makedirs(osjoin(exp_dir, "logs"), exist_ok=True)
    os.makedirs(osjoin(exp_dir, "last_iter"), exist_ok=True)
    os.makedirs(osjoin(exp_dir, "best_iter"), exist_ok=True)

    summary_writer = create_file_writer(osjoin(exp_dir, "logs"))

    # Training loop
    loss_keys = model._losses_dict.keys()
    lowest_val_loss = 9999999.
    try:
        with summary_writer.as_default():
            for epoch in range(args.epochs):
                print("Epoch:", epoch, "\nTraining...")
                step = trainer.step  # step != epoch if resuming training

                # Fit training data
                epoch_losses = {k: 0. for k in loss_keys}
                for _ in tqdm(range(args.steps_per_epoch), ncols=64):
                    # Train step
                    losses = trainer.train_step(train_iterator)
                    # Retrieve loss values
                    for k in loss_keys:
                        epoch_losses[k] += float(tf.debugging.check_numerics(
                            losses[k],
                            message=f"Nan loss at step {trainer.step.numpy()} with loss {k}"))

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
                trainer.save(osjoin(exp_dir, "last_iter"))
                print('Last iteration model saved at',
                      osjoin(exp_dir, "last_iter"))

                # Skip validation during early training
                if trainer.step < 3 * args.steps_per_epoch:
                    # Just add an audio summary without computing the val loss
                    val_batch = next(iter(val_dataset))
                    summaries.audio_summary(
                        model(val_batch, training=True)["audio_synth"],
                        step,
                        sample_rate=model.sample_rate,
                        name='synthesized_audio')
                    collect_garbage()
                    continue

                # Evaluate on validation data
                print("Validation...")
                val_outs_summary = None
                epoch_val_losses = {k: 0. for k in loss_keys}
                for val_step, val_batch in enumerate(tqdm(val_dataset,
                                                          ncols=64)):
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
                                        sample_rate=model.sample_rate,
                                        name='synthesized_audio')
                summaries.spectrogram_summary(val_outs_summary['audio'],
                                              val_outs_summary['audio_synth'],
                                              step=step)
                # Save if better epoch
                if epoch_val_losses['audio_stft_loss'] < lowest_val_loss:
                    lowest_val_loss = epoch_val_losses['audio_stft_loss']
                    trainer.save(osjoin(exp_dir, "best_iter"))

                # Collect garbage
                collect_garbage()

    except tf.errors.InvalidArgumentError as e:
        trainer.save(osjoin(exp_dir, "crashed_iter"))
        print(e)

    except KeyboardInterrupt:
        trainer.save(osjoin(exp_dir, "stopped_iter"))


if __name__ == '__main__':
    main(process_args())
