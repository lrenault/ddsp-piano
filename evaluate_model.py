import os
import gin
import argparse
import pandas as pd
import tensorflow as tf
from tqdm import tqdm
from soundfile import write
from ddsp.training import trainers, train_util
from ddsp.training.models import get_model
from ddsp_piano.data_pipeline import get_dummy_data, get_test_dataset

osjoin = os.path.join


def process_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, help="A .gin model config",
                        default='ddsp_piano/configs/maestro-v2.gin')
    parser.add_argument('--ckpt', type=str, help="Model checkpoint to load.",
                        default='ddsp_piano/model_weights/v2/')
    parser.add_argument('--warm_up', '-wu', type=float, default=0.5,
                        help="Warm-up duration (in s, default: %(default)s)")
    parser.add_argument('--get_wav', '-w', action='store_true',
                        help="Generate wav files.")
    parser.add_argument('maestro_dir', type=str,
                        help="Path to the MAESTRO dataset.")
    parser.add_argument('out_dir', type=str,
                        help="Folder containing the synthesized test wav files.")
    return parser.parse_args()


@tf.function
def validation_step(trainer, inputs):
    """Distributed training step."""
    # Wrap iterator in tf.function, slight speedup passing in iter vs batch.
    batch = next(inputs) if hasattr(inputs, '__next__') else inputs
    outputs, losses = trainer.run(
        tf.function(trainer.model.__call__),
        batch,
        training=True,
        return_losses=True
    )
    # Add up the scalar losses across replicas.
    n_replicas = trainer.strategy.num_replicas_in_sync
    return trainer.strategy.gather(outputs, axis=0), \
           {k: trainer.psum(v, axis=None) / n_replicas for k, v in losses.items()}


def main(args):
    # Parse and override gin-config
    gin.parse_config_file(args.config)
    gin.bind_parameter('%inference', True)
    gin.bind_parameter('%duration', 10.0)

    strategy = train_util.get_strategy()
    with strategy.scope():
        # Model contruction
        model = get_model()
        trainer = trainers.Trainer(model=model, strategy=strategy)
        trainer.build(get_dummy_data(batch_size=1,
                                     duration=10.0,
                                     sample_rate=model.sample_rate))

        # Load model checkpoint
        trainer.restore(args.ckpt)

    # Retrieve test dataset
    test_dataset = get_test_dataset(filename=args.maestro_dir,
                                    batch_size=1,
                                    duration=10.0,
                                    sample_rate=model.sample_rate)
    test_dataset = trainer.distribute_dataset(test_dataset)

    # Init
    os.makedirs(args.out_dir, exist_ok=True)
    evaluations = []

    # Save synthesized audio
    if args.get_wav:
        os.makedirs(osjoin(args.out_dir, 'wav'), exist_ok=True)

    for i, batch in tqdm(enumerate(test_dataset), ncols=64):
        pth=osjoin(args.out_dir, 'wav',
                   batch['filename'][0][0].numpy().decode('utf-8').split('/')[-1] + '.wav')
        outs, losses = validation_step(trainer, batch)
        loss_val = losses['audio_stft_loss']

        evaluations.append({
            'filename': batch['filename'].numpy()[0],
            'piano_model': batch['piano_model'].numpy()[0],
            'loss_val': loss_val.numpy()
        })
        if i % 100 == 0:
            df = pd.DataFrame(data=evaluations)
            df.to_csv(osjoin(args.out_dir, "spectral_losses.csv"),
                      index=False)

        if args.get_wav:
            write(pth,
                  outs['audio_synth'].numpy()[0],
                  model.sample_rate)
    df = pd.DataFrame(data=evaluations)
    df.to_csv(osjoin(args.out_dir, "spectral.csv"),
              index=False)


if __name__ == '__main__':
    main(process_args())