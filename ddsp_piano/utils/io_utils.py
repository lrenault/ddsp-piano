import gc
import note_seq
import numpy as np
import tensorflow as tf

from pandas import read_csv
from ddsp.training.data_preparation import prepare_tfrecord_lib as ddsp_lib
from ddsp_piano.utils.midi_encoders import MIDIRoll2Conditioning

seq_lib = note_seq.sequences_lib


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
        target_n_frames = np.ceil(note_sequence.total_time) * frame_rate
    else:
        target_n_frames = int(duration * frame_rate)

    # Crop/pad inputs
    conditioning = ensure_sequence_length(conditioning, target_n_frames)
    pedals = ensure_sequence_length(pedals, target_n_frames)

    # Return with a batch size of 1
    return {'conditioning': conditioning[np.newaxis, ...],
            'pedal': pedals[np.newaxis, ...],
            'duration': target_n_frames / frame_rate}


def load_and_split_data(audio_path,
                        mid_path,
                        segment_duration=3.,
                        max_polyphony=None,
                        overlap=0.5,
                        sample_rate=16000,
                        frame_rate=250):
    """Load aligned audio and MIDI data (as conditioning sequence), then split
    into segments.
    Args:
        - audio_path (tf.path): absolute path to audio file.
        - mid_path (tf.path): absolute path to midi file.
        - segment_duration (float): length of segment chunks (in s).
        - max_polyphony (int): number of monophonic channels for the conditio-
        ning vector (return the piano rolls if None).
        - overlap (float): overlapping ratio between two consecutive segemnts.
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
    n_samples = int(segment_duration * sample_rate)
    n_frames = int(segment_duration * frame_rate)
    audio_hop_size = int(n_samples * (1 - overlap))
    midi_hop_size = int(n_frames * (1 - overlap))

    # Read audio file
    audio = ddsp_lib._load_audio_as_array(
        audio_path.numpy().decode("utf-8"),
        sample_rate
    )
    # Read MIDI file
    note_sequence = load_midi_as_note_sequence(
        mid_path.numpy().decode("utf-8")
    )
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
        midi_roll, polyphony = polyphony_manager(midi_roll)

    # Split into segments
    audio_t = 0
    midi_t = 0
    segment_audio = []
    segment_rolls = []
    segment_pedals = []
    segment_polyphony = []
    while midi_t + n_frames < np.shape(midi_roll)[0]:
        segment_audio.append(audio[audio_t: audio_t + n_samples])
        segment_rolls.append(midi_roll[midi_t: midi_t + n_frames])
        segment_pedals.append(pedals[midi_t: midi_t + n_frames])

        if max_polyphony:
            segment_polyphony.append(polyphony[midi_t: midi_t + n_frames])

        audio_t += audio_hop_size
        midi_t += midi_hop_size

    n_segments = len(segment_rolls)

    if max_polyphony is None:
        return np.array(segment_audio), np.array(segment_rolls), \
            np.array(segment_pedals), n_segments
    else:
        return np.array(segment_audio), np.array(segment_rolls), \
            np.array(segment_pedals), np.array(segment_polyphony), \
            n_segments


@tf.function
def load_convert_split_data_tf(audio_path,
                               mid_path,
                               segment_duration,
                               max_polyphony):
    """tf.function wrapper for the load_and_split_data function."""
    audio, conditioning, pedals, polyphony, n_segments = tf.py_function(
        load_and_split_data,
        [audio_path, mid_path, segment_duration, max_polyphony],
        Tout=(tf.float32, tf.float32, tf.float32, tf.int32, tf.int32)
    )
    return {"audio": audio,
            "conditioning": conditioning,
            "pedal": pedals,
            "polyphony": polyphony,
            "n_segments": n_segments}


def collect_garbage():
    gc.collect()
    tf.keras.backend.clear_session()
    tf.compat.v1.reset_default_graph()
