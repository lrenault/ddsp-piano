import tensorflow as tf
import ddsp_piano.utils.io_utils as io_utils

from os.path import join


def get_first_batch(*args, **kwargs):
    dataset = get_training_dataset(*args,
                                   shuffle=False,
                                   infinite_generator=False,
                                   num_parallel_calls=1,
                                   **kwargs)
    batch_value = next(iter(dataset))
    return batch_value


def get_dummy_data(batch_size=6,
                   duration=3,
                   sample_rate=16000,
                   frame_rate=250,
                   n_synths=16):
    """Create random input data. Same arguments as for get_dataset()."""
    # Shapes definition
    n_frames = int(duration * frame_rate)
    n_samples = int(duration * sample_rate)

    conditioning_shape = [batch_size, n_frames, n_synths, 2]
    pedal_shape = [batch_size, n_frames, 4]
    piano_model_shape = [batch_size, 1, ]
    audio_shape = [batch_size, n_samples, ]

    features = {}
    features['conditioning'] = tf.random.uniform(
        shape=tf.TensorShape(conditioning_shape),
        minval=0., maxval=1.,
        seed=0
    )
    features['pedal'] = tf.random.uniform(
        shape=tf.TensorShape(pedal_shape),
        minval=0., maxval=1.,
        seed=0
    )
    features['audio'] = tf.random.uniform(
        shape=tf.TensorShape(audio_shape),
        minval=0., maxval=1.,
        seed=0
    )
    features['piano_model'] = tf.random.uniform(
        shape=tf.TensorShape(piano_model_shape),
        minval=0, maxval=1, dtype=tf.int32,
        seed=0
    )

    return features


def get_training_dataset(*args, **kwargs):
    return get_dataset(*args, split='train', **kwargs)


def get_validation_dataset(*args, **kwargs):
    return get_dataset(*args,
                       split='validation',
                       infinite_generator=False,
                       shuffle=False,
                       **kwargs)


def get_test_dataset(*args, duration=30, **kwargs):
    return get_dataset(*args,
                       split='test',
                       duration=duration,
                       filter_over_polyphony=False,
                       infinite_generator=False,
                       shuffle=False,
                       num_parallel_calls=1,
                       **kwargs)


def get_dataset(dataset_dir,
                split='train',
                only_first_seg=False,
                duration=3.0,
                batch_size=6,
                shuffle=True,
                infinite_generator=True,
                sample_rate=16000,
                frame_rate=250,
                max_polyphony=16,
                filter_over_polyphony=True,
                num_parallel_calls=16):
    """Tensorflow dataset pipeline for feeding the training with conditioning
    MIDI inputs and audio target outputs. Automatically splits full tracks into
    segments.
    Args:
        - dataset_dir (path): path to the maestro-v3.0.0/ folder.
        - split (str): which dataset subset to use (among 'train', 'validation'
        and 'test').
        - only_first_seg (bool): retrieve only the first segment for each track
        - duration (float): duration of audio segments (in s).
        - batch_size (int): number of segments per batch.
        - shuffle (bool): apply shuffling between tracks.
        - infinite_generator (bool): provide data indefinitely.
        - sample_rate (int): number of audio samples per second.
        - frame_rate (int): number of conditioning input frames per second.
        - max_polyphony (int): filter out segments with more simultaneous notes
        than the model polyphonic capacity. Does not filter anything if set to
        `None`.
        - num_parallel_calls (int): number of threads.
    Returns:
        - dataset (tf.data.Dataset): segments dataset
    """
    # Init tf.dataset from .csv file
    dataset, n_examples, piano_models = io_utils.dataset_from_csv(
        join(dataset_dir, "maestro-v3.0.0.csv"),
        split=split
    )
    # Shapes definition
    n_frames = int(duration * frame_rate)
    n_samples = int(duration * sample_rate)

    conditioning_shape = [n_frames, max_polyphony, 2]
    pedal_shape = [n_frames, 4]
    piano_model_shape = [1, ]
    audio_shape = [n_samples, ]

    # Shuffle on tracks
    if shuffle:
        dataset = dataset.shuffle(buffer_size=n_examples,
                                  seed=0,
                                  reshuffle_each_iteration=True)
    # Encode piano model as one-hot
    dataset = dataset.map(
        lambda sample: dict(
            sample,
            piano_model=tf.where(tf.equal(piano_models,
                                          sample['year']))[0]),
        num_parallel_calls=num_parallel_calls
    )
    # Load segments of audio and MIDI data
    dataset = dataset.map(
        lambda sample: dict(
            sample,
            **io_utils.load_convert_split_data_tf(
                tf.strings.join([dataset_dir, sample['audio_filename']]),
                tf.strings.join([dataset_dir, sample['midi_filename']]),
                duration,
                max_polyphony)),
        num_parallel_calls=num_parallel_calls
    )
    # Split or replace track dataset into segment dataset
    if only_first_seg:
        # Only keep the first segment of each track
        dataset = dataset.map(
            lambda sample: dict(
                audio=sample["audio"][0],
                conditioning=sample["conditioning"][0],
                pedal=sample["pedal"][0],
                polyphony=tf.reduce_max(sample["polyphony"][0]),
                piano_model=sample["piano_model"]),
            num_parallel_calls=num_parallel_calls
        )
        # Filter out segments with polyphony exceeding supported polyphony
        if filter_over_polyphony:
            dataset = dataset.filter(
                lambda *sample: sample["polyphony"] <= max_polyphony
            )
        # Remove polyphony entry
        dataset = dataset.map(
            lambda *sample:
                {key: sample[key] for key in sample if key != "polyphony"},
            num_parallel_calls=num_parallel_calls
        )
    else:
        # Create a new dataset by making an entry for each segment
        dataset = dataset.flat_map(
            lambda sample: tf.data.Dataset.zip((
                tf.data.Dataset.from_tensor_slices(sample["audio"]),
                tf.data.Dataset.from_tensor_slices(sample["conditioning"]),
                tf.data.Dataset.from_tensor_slices(sample["pedal"]),
                tf.data.Dataset.from_tensor_slices(sample["polyphony"]),
                tf.data.Dataset.from_tensor_slices(
                    tf.repeat(sample["piano_model"],
                              repeats=sample["n_segments"]))
            ))
        )
        # Filter out segments with polyphony exceeding supported polyphony
        if filter_over_polyphony:
            dataset = dataset.filter(
                lambda *sample: tf.reduce_max(sample[3]) <= max_polyphony
            )
        # Rename keys
        dataset = dataset.map(
            lambda *sample: {
                "audio": sample[0],
                "conditioning": sample[1],
                "pedal": sample[2],
                "piano_model": sample[4][..., tf.newaxis]},
            num_parallel_calls=num_parallel_calls
        )
    # Infinite generator
    if infinite_generator:
        dataset = dataset.repeat(count=-1)

    # Make batch
    dataset = dataset.padded_batch(
        batch_size,
        padded_shapes={"conditioning": tf.TensorShape(conditioning_shape),
                       "pedal": tf.TensorShape(pedal_shape),
                       "audio": tf.TensorShape(audio_shape),
                       "piano_model": tf.TensorShape(piano_model_shape)}
    )
    # Drop batch if dataset exhausted and batch size is not met
    dataset = dataset.filter(
        lambda sample: tf.equal(batch_size, tf.shape(sample['audio'])[0])
    )
    # Prefetch next batches
    dataset = dataset.prefetch(4)

    # Sharding options
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
    dataset = dataset.with_options(options)

    return dataset
