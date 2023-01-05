import argparse
import tensorflow as tf
import manage_gpus as gpl

from tqdm import tqdm
from ddsp.training import trainers, train_util
from ddsp_piano.data_pipeline import single_track_dataset
from ddsp_piano.default_model import build_model, get_model
# from ddsp_piano.jaes_relu import get_model


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

    parser.add_argument('midi_file', type=str, default=None,
    					help=".mid of the track to train on.")

    parser.add_argument('audio_file', type=str, default=None,
    					help=".wav file of the track to train on.")

    parser.add_argument('exp_dir', type=str,
                        help="Folder to store experiment results and logs.")

    return parser.parse_args()


def lock_gpu(soft=True, gpu_device_id=-1):
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
	# GPU slot
	_ = lock_gpu()

	# Get training data
	dataset = single_track_dataset(args.midi_file,
								   args.audio_file,
								   batch_size=args.batch_size)
	# Build and distribute model
	strategy = train_util.get_strategy()
	with strategy.scope():
		model = build_model(get_model(),
							batch_size=args.batch_size,
							first_phase=True)
		trainer = trainers.Trainer(model=model,
								   strategy=strategy,
								   learning_rate=1e-3)
		dataset = trainer.distribute_dataset(dataset)

	loss_keys = model._losses_dict.keys()  # Retrieve losses to extract

	# Training
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
