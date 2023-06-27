import gc
import pydub
import note_seq
import numpy as np
import tensorflow as tf

from pandas import read_csv
from ddsp.spectral_ops import pad_or_trim_to_expected_length
from ddsp_piano.utils.midi_encoders import MIDIRoll2Conditioning

seq_lib = note_seq.sequences_lib


def decode_tfstring(x):
    return x.numpy().decode('utf-8') if tf.is_tensor(x) else x


def tf_to_np(x):
    return x.numpy() if tf.is_tensor(x) else x


def dataset_from_csv(csv_path, split=None, year=None, **kwargs):
    """Load dataset from a csv file.
    Returns:
        - dataset (tf.data.Dataset): tensorflow dataset from .csv
        - n_samples (int): number of dataset entries.
        - piano_models (list): list of different piano models in the dataset.
    """
    # .csv reading in pandas dataframe
    df = read_csv(csv_path, **kwargs)

    # Filter by split
    if split:
        df = df[df.split == split]
    if year:
        df = df[df.year == year]

    # Convert dataframe to tf.data.Dataset
    dataset = (
        tf.data.Dataset.from_tensor_slices(
            {key: df[key].values for key in df})
    )
    # Get dataset length and piano models
    n_samples = len(df)
    piano_models = np.sort(df['year'].unique())

    return dataset, n_samples, piano_models


def load_audio_as_signal(audio_path, sample_rate=16000):
    """Load audio file at specified sample rate and return an array.
    In order to not use/install apache-beam, we've copied the function from
    ddsp.training.data_preparation.prepare_tfrecord_lib._load_audio_as_array
    Args:
        audio_path: path to audio file
        sample_rate: desired sample rate (can be different from original SR)
    Returns:
        audio: audio in np.float32
    """
    with tf.io.gfile.GFile(decode_tfstring(audio_path), 'rb') as f:
        # Load audio at original SR
        audio_segment = (pydub.AudioSegment.from_file(f).set_channels(1))
        # Compute expected length at given `sample_rate`
        expected_len = int(audio_segment.duration_seconds * sample_rate)
        # Resample to `sample_rate`
        audio_segment = audio_segment.set_frame_rate(sample_rate)
        sample_arr = audio_segment.get_array_of_samples()
        audio = np.array(sample_arr).astype(np.float32)
        # Zero pad missing samples, if any
        audio = pad_or_trim_to_expected_length(audio, expected_len)
    # Convert from int to float representation.
    audio /= np.iinfo(sample_arr.typecode).max
    return audio


def load_midi_as_note_sequence(mid_path):
    # Read MIDI file
    note_sequence = note_seq.midi_io.midi_file_to_note_sequence(mid_path)
    # Extend offset with sustain pedal
    note_sequence = note_seq.apply_sustain_control_changes(note_sequence)
    return note_sequence


def ensure_sequence_length(sequence, length):
    """Zero-pad or crop sequence to fit desired length."""
    original_length = sequence.shape[0]
    # Return as is
    if original_length == length:
        return sequence
    # Crop
    elif original_length > length:
        return sequence[:length]
    # Pad
    else:
        pad_width = [(0, int(length - original_length))]
        for _ in range(len(sequence.shape) - 1):
            pad_width += [(0, 0)]
        return np.pad(sequence, pad_width=pad_width)


def load_midi_as_conditioning(mid_path,
                              n_synths=16,
                              frame_rate=250,
                              duration=None):
    """Load MIDI file as conditioning and pedal inputs for inference.
    Args:
        - mid_path (path): path to .mid file.
        - n_synths (int): number of polyphonoic channels in the conditioning.
        - frame_rate (int): number of frames per second.
        - duration (float): crop file reading to this duration.
    Returns:
        - conditioning (1, n_frames, n_synths, 2): polyphonic note activity and
        onset inputs.
        - pedal (1, n_frames, 4): pedal information.
        - duration (float): length of the sequence (in s).
    """
    # File reading
    note_sequence = load_midi_as_note_sequence(mid_path)
    # Convert to pianoroll
    roll = seq_lib.sequence_to_pianoroll(note_sequence,
                                         frames_per_second=frame_rate,
                                         min_pitch=21,
                                         max_pitch=108)
    # Retrieve activity and onset velocities and pedals signals
    midi_roll = np.stack((roll.active, roll.onset_velocities), axis=-1)
    pedals = roll.control_changes[:, 64: 68] / 128.

    # Reduce pianoroll to conditioning while managing polyphonic information
    polyphony_manager = MIDIRoll2Conditioning(n_synths)
    conditioning, _ = polyphony_manager(midi_roll)

    # Set target length to an integer number of seconds
    if duration is None:
        target_n_frames = int(np.ceil(note_sequence.total_time) * frame_rate)
    else:
        target_n_frames = int(duration * frame_rate)

    # Crop/pad inputs
    conditioning = ensure_sequence_length(conditioning, target_n_frames)
    pedals = ensure_sequence_length(pedals, target_n_frames)

    # Return with a batch size of 1
    return {'conditioning': conditioning[np.newaxis, ...],
            'pedal': pedals[np.newaxis, ...],
            'duration': target_n_frames / frame_rate}


def load_data(audio_path,
              mid_path,
              max_polyphony=None,
              sample_rate=16000,
              frame_rate=250):
    """Load aligned audio and MIDI data (as conditioning sequence), then split
    into segments.
    Args:
        - audio_path (tf.path): absolute path to audio file.
        - mid_path (tf.path): absolute path to midi file.
        - max_polyphony (int): number of monophonic channels for the conditio-
        ning vector (return the piano rolls if None).
        - sample_rate (int): number of audio samples per second.
        - frame_rate (int): number of conditioning vectors per second.
    Returns:
        - segment_audio (list [n_samples,]): list of audio segments.
        - segment_rolls (list [n_frames, max_polyphony, 2]): list of segments
        conditioning vectors.
        - segment_pedals (list [n_frames, 4]): list of segments pedals condi-
        tioning.
        - polyphony (list [n_frames, 1]): list of polyphony information in the
        original piano roll.
    """
    # Read audio file
    audio = load_audio_as_signal(audio_path, sample_rate)

    # Read MIDI file
    note_sequence = load_midi_as_note_sequence(decode_tfstring(mid_path))

    # Convert to pianoroll
    roll = seq_lib.sequence_to_pianoroll(note_sequence,
                                         frames_per_second=frame_rate,
                                         min_pitch=21,
                                         max_pitch=108)
    # Retrieve activity and onset velocities
    midi_roll = np.stack((roll.active, roll.onset_velocities), axis=-1)

    # Pedals are CC64, 66 and 67
    pedals = roll.control_changes[:, 64: 68] / 128.0

    if max_polyphony is not None:
        polyphony_manager = MIDIRoll2Conditioning(max_polyphony)
        conditioning, polyphony = polyphony_manager(midi_roll)

        return audio, conditioning, pedals, polyphony

    else:
        return audio, midi_roll, pedals


@tf.function
def load_data_tf(audio_path, mid_path, max_polyphony):
    """tf.function wrapper for the load_and_split_data function."""
    audio, conditioning, pedal, polyphony = tf.py_function(
        load_data,
        [audio_path, mid_path, max_polyphony],
        Tout=(tf.float32, tf.float32, tf.float32, tf.int32)
    )
    return {"audio": audio,
            "conditioning": conditioning,
            "pedal": pedal,
            "polyphony": polyphony}


def split_sequence(x, segment_duration, rate, overlap=0.5):
    """Split sequence x into fixed size segments.
    Args:
        - x: a sequence to split.
        - segment_duration (float): duration (in s) of the segments.
        - rate (int): signal/frame rate of the sequence.
        - overlap (float): overlap ratio between two consecutive segments.
    Returns:
        - segments (list): list of split segments.
    """
    n_samples = int(segment_duration * rate)
    hop_size = int(n_samples * (1 - overlap))

    segments = []
    segments.append(ensure_sequence_length(x[:n_samples],
                                           n_samples))
    timestep = hop_size
    while timestep + n_samples < np.shape(x)[0]:
        segment = x[timestep: timestep + n_samples]
        segment = ensure_sequence_length(segment, n_samples)

        segments.append(segment)
        timestep += hop_size

    if len(segments) > 1:
        return np.stack(segments)
    else:
        return np.array(segments)


def collect_garbage():
    gc.collect()
    tf.keras.backend.clear_session()
    tf.compat.v1.reset_default_graph()
