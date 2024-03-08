import tensorflow as tf
import ddsp_piano.utils.io_utils as io_utils

from os.path import join


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

    features = {
        "conditioning": tf.random.uniform(
            shape=tf.TensorShape(conditioning_shape),
            minval=0., maxval=1., seed=0),

        "pedal": tf.random.uniform(
            shape=tf.TensorShape(pedal_shape),
            minval=0., maxval=1., seed=0),

        "audio": tf.random.uniform(
            shape=tf.TensorShape(audio_shape),
            minval=0., maxval=1., seed=0),

        "piano_model": tf.random.uniform(
            shape=tf.TensorShape(piano_model_shape),
            minval=0, maxval=10, seed=0, dtype=tf.int32)
    }

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


def get_preprocessed_dataset(dataset_dir,
                             split='train',
                             year=None,
                             sample_rate=16000,
                             frame_rate=250,
                             max_polyphony=16,
                             num_parallel_calls=8,
                             **kwargs):
    """Extract audio and midi data from the .csv metadata file.
    Args:
        - dataset_dir (path): folder location of maestro-v3.0.0/
        - split ('train', 'val', 'test'): which split to process.
        - year (int): specific piano model to keep.
        - sample_rate (int): number of audio samples per second.
        - frame_rate (int): number of conditioning frames per second.
        - max_polyphony (int): filter out segments with more simultaneous notes
        than the model polyphonic capacity. Does not filter anything if set to
        `None`.
        - num_parallel_calls (int): number of threads.
    """
    # Init tf.dataset from .csv file
    dataset, n_examples, piano_models = io_utils.dataset_from_csv(
        join(dataset_dir, "maestro-v3.0.0.csv"),
        # join(dataset_dir, "maps_MUS_dataset.csv"),
        # join(dataset_dir, "mel2mel_MUS_dataset.csv"),
        split=split,
        year=year,
        **kwargs
    )
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
            **io_utils.load_data_tf(
                tf.strings.join([dataset_dir, sample['audio_filename']]),
                tf.strings.join([dataset_dir, sample['midi_filename']]),
                max_polyphony,
                sample_rate,
                frame_rate)),
        num_parallel_calls=num_parallel_calls
    )
    return dataset


def get_dataset(filename,
                split='train',
                year=None,
                duration=3,
                batch_size=6,
                shuffle=True,
                infinite_generator=True,
                sample_rate=16000,
                frame_rate=250,
                max_polyphony=16,
                filter_over_polyphony=True,
                num_parallel_calls=8,
                **kwargs):
    """Tensorflow dataset pipeline for feeding the training with conditioning
    MIDI inputs and audio target outputs. Automatically splits full tracks into
    segments.
    Args:
        - filename (str): path to the maestro-v3.0.0/ folder OR a preprocessed
        .tfrecord file.
        - split (str): which dataset subset to use (among 'train', 'validation'
        and 'test').
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
    # Shapes definition
    n_frames  = int(duration * frame_rate)
    n_samples = int(duration * sample_rate)

    conditioning_shape = [n_frames, max_polyphony, 2]
    pedal_shape        = [n_frames, 4]
    piano_model_shape  = [1, ]
    audio_shape        = [n_samples, ]

    # Data loading
    if ".tfrecord" in filename:
        # Load preprocessed data from the .tfrecord
        dataset = tf.data.Dataset.load(filename)
    else:
        # Process data on the fly from the maestro-v3.0.0/ folder
        dataset = get_preprocessed_dataset(
            dataset_dir=filename,
            split=split,
            year=year,
            sample_rate=sample_rate,
            frame_rate=frame_rate,
            max_polyphony=max_polyphony,
            num_parallel_calls=num_parallel_calls,
            **kwargs
        )
    # Shuffle on tracks
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(dataset) // 2,
                                  seed=0,
                                  reshuffle_each_iteration=True)

    # Split tracks into segments
    dataset = dataset.map(
        lambda x: dict(
            x,
            audio=io_utils.split_sequence_tf(x['audio'], duration, sample_rate),
            conditioning=io_utils.split_sequence_tf(x['conditioning'], duration, frame_rate),
            pedal=io_utils.split_sequence_tf(x['pedal'], duration, frame_rate),
            polyphony=io_utils.split_sequence_tf(x['polyphony'], duration, frame_rate)),
        num_parallel_calls=num_parallel_calls)

    # Fix border issue
    dataset = dataset.map(
        lambda sample: dict(
            sample,
            n_segments=tf.reduce_min([len(sample["audio"]),
                                      len(sample["conditioning"])])),
        num_parallel_calls=num_parallel_calls)

    # Flatten the dataset with segments list into a dataset of segments
    dataset = dataset.flat_map(
        lambda sample: tf.data.Dataset.zip(dict(
            audio=tf.data.Dataset.from_tensor_slices(
                sample["audio"][:sample["n_segments"]]
            ),
            conditioning=tf.data.Dataset.from_tensor_slices(
                sample["conditioning"][:sample["n_segments"]]
            ),
            pedal=tf.data.Dataset.from_tensor_slices(
                sample["pedal"][:sample["n_segments"]]
            ),
            polyphony=tf.data.Dataset.from_tensor_slices(
                sample["polyphony"][:sample["n_segments"]]
            ),
            piano_model=tf.data.Dataset.from_tensor_slices(
                tf.repeat(sample["piano_model"],
                          repeats=sample["n_segments"])[..., tf.newaxis]
            )
        ))
    )
    # Filter out segments with polyphony exceeding supported polyphony
    if filter_over_polyphony:
        dataset = dataset.filter(
            lambda x: tf.reduce_max(x["polyphony"]) <= max_polyphony)

    # Keep relevant entries
    dataset = dataset.map(
        lambda x: {k: x[k] for k in ['audio',
                                     'conditioning',
                                     'pedal',
                                     'piano_model']},
        num_parallel_calls=num_parallel_calls)

    # Infinite generator
    if infinite_generator:
        dataset = dataset.repeat(count=-1)

    # Make batch
    dataset = dataset.padded_batch(
        batch_size,
        padded_shapes={"conditioning": tf.TensorShape(conditioning_shape),
                       "pedal": tf.TensorShape(pedal_shape),
                       "audio": tf.TensorShape(audio_shape),
                       "piano_model": tf.TensorShape(piano_model_shape)},
        drop_remainder=True
    )
    # Prefetch next batches
    dataset = dataset.prefetch(4)

    # Sharding options
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
    dataset = dataset.with_options(options)

    return dataset


def single_track_dataset(midi_filename,
                         audio_filename,
                         batch_size=1,
                         duration=3,
                         sample_rate=16000,
                         frame_rate=250,
                         num_parallel_calls=8):
    """Create a training dataset from a single pair of audio/midi track.
    Args:
        - midi_filename (str): path to the MIDI file.
        - audio_filename (str): path to the audio file.
        - batch_size (int): number of segments per batch.
        - duration (int): duration of segments (in s).
        - sample_rate (int): audio sample rate.
        - frame_rate (int): conditioning frame rate.
        - num_parallel_calls (int): number of tf.data.Dataset theads.
    """
    # Load audio and MIDI data
    audio, conditioning, pedal, polyphony = io_utils.load_data(
        audio_filename,
        midi_filename,
        max_polyphony=16,
        sample_rate=sample_rate,
        frame_rate=frame_rate,
    )
    # Split track into multiple segments
    if len(conditioning) / float(frame_rate) > duration:
        audio        = io_utils.split_sequence_tf(audio, duration, sample_rate)
        conditioning = io_utils.split_sequence_tf(conditioning, duration, frame_rate)
        pedal        = io_utils.split_sequence_tf(pedal, duration, frame_rate)
        polyphony    = io_utils.split_sequence_tf(polyphony, duration, frame_rate)

        n_segments = min(len(audio), len(conditioning))

        dataset = {"audio": audio,
                   "conditioning": conditioning,
                   "pedal": pedal,
                   "polyphony": polyphony}
        # Fix border issue
        for k in dataset.keys():
            dataset[k] = dataset[k][:n_segments]

    else:
        # Single segment available
        dataset = {"audio": [io_utils.ensure_sequence_length(audio, int(duration * sample_rate)), ],
                   "conditioning": [io_utils.ensure_sequence_length(conditioning, int(duration * frame_rate)), ],
                   "pedal": [io_utils.ensure_sequence_length(pedal, int(duration * frame_rate)), ],
                   "polyphony": [io_utils.ensure_sequence_length(polyphony, int(duration * frame_rate)), ]}

    # Convert to tf.data.Dataset
    dataset = tf.data.Dataset.from_tensor_slices(dataset)

    # Filter out segments with polyphony exceedings polyphonic capacity
    dataset = dataset.filter(
        lambda sample: tf.reduce_max(sample["polyphony"]) <= 16)
    # Add piano model
    dataset = dataset.map(
        lambda sample: dict(
            sample,
            piano_model=tf.zeros(1, dtype=tf.int32)))

    # Make batch
    dataset = dataset.padded_batch(
        batch_size,
        padded_shapes={
            "audio": tf.TensorShape([int(duration * sample_rate), ]),
            "conditioning": tf.TensorShape([int(duration * frame_rate), 16, 2]),
            "pedal": tf.TensorShape([int(duration * frame_rate), 4]),
            "polyphony": tf.TensorShape([int(duration * frame_rate), ]),
            "piano_model": tf.TensorShape([1, ])},
        drop_remainder=True
    )
    # Prefetch
    dataset = dataset.prefetch(4)

    # Sharding options
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
    dataset = dataset.with_options(options)

    return dataset


def preprocess_data_into_tfrecord(filename, **kwargs):
    """Parse through a maestro dataset and save preprocessing as a .tfrecord
    Typical usage would be:
    ```
    preprocess_data_into_tfrecord(
        </export/path>,
        dataset_dir=<path/to/base_dataset>,
        split='train',
        ...
    )
    ``` """
    dataset = get_preprocessed_dataset(**kwargs)
    dataset.save(filename)
