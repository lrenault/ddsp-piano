import numpy as np


class MIDIRoll2Conditioning(object):
    """Convert pianorolls into polyphonic conditioning vector in NumPy.
    Params:
        - n_synths (int): number of simultaneous notes supported
        - pitch_mul (88): multiplication matrix for converting active pianoroll
        - assigner (int): next available slot in the conditioning filler.
        - reorder (n_synths,): reordering for reduced pianoroll to conditioning
    """

    def __init__(self, n_synths=16):
        super(MIDIRoll2Conditioning, self).__init__()
        self.n_synths = n_synths
        self.pitch_mul = np.arange(21, 21 + 88)
        self.reorder = np.arange(self.n_synths)
        self.assigner = 0
        self.assigned_pitch = np.zeros(n_synths)

    def update_assigner(self):
        # Incremente assigner to next available channel
        self.assigner = (self.assigner + 1) % self.n_synths

        if 0 not in self.assigned_pitch:
            self.assigner = -1
        else:
            while self.assigned_pitch[self.assigner] != 0:
                self.assigner = (self.assigner + 1) % self.n_synths

    def __call__(self, roll):
        """ Convert active and onset veloctiy pianorolls into polyphonic
        conditioning vector.
        Args:
            - roll (n_frames, 88, 2): stacked active and onset velocity
            pianorolls.
        Returns:
            - conditioning (n_frames, n_synths, 2): stacked conditioning
            polyphonic vector.
        """
        note_activity = roll[..., 0]

        polyphony = np.sum(note_activity, axis=-1)

        note_activity *= self.pitch_mul

        # Get indices of n_synths highest notes at each frame
        idxs = np.argsort(note_activity)[:, -self.n_synths:]
        # Reduce rolls to top n_synths notes (n_frames, n_synths)
        note_activity = np.take_along_axis(note_activity, idxs, axis=-1)
        velocity = np.take_along_axis(roll[..., 1], idxs, axis=-1)

        # Rearange frame-wise
        t = 0
        for pitches in note_activity:
            # Apply same reordering if no change
            common_notes = np.intersect1d(pitches, self.assigned_pitch)
            if t > 0 \
                and len(common_notes) == len(np.unique(pitches)) \
                and len(common_notes) == len(np.unique(self.assigned_pitch)):
                note_activity[t] = np.take(pitches, self.reorder)
                velocity[t] = np.take(velocity[t], self.reorder)
                t += 1
                continue

            # Init reorder vector
            reorder = np.zeros(self.n_synths, dtype=int)
            # Free channels containing finished notes
            for c in range(self.n_synths):
                if self.assigned_pitch[c] not in pitches:
                    self.assigned_pitch[c] = 0

                    if self.assigner == -1:
                        self.update_assigner()

            # Keep sustained notes to assigned channels
            for c in range(self.n_synths):
                if pitches[c] in self.assigned_pitch \
                   and pitches[c] != 0:
                    reorder[np.where(self.assigned_pitch == pitches[c])[0][0]] = c

            # Assign new notes to unassigned channels
            for c in range(self.n_synths):
                if pitches[c] not in self.assigned_pitch:
                    reorder[self.assigner] = c
                    self.assigned_pitch[self.assigner] = pitches[c]
                    self.update_assigner()

            # Finish reordering with unassigned channels
            for c in range(self.n_synths):
                if pitches[c] == 0:
                    reorder[self.assigner] = c
                    self.update_assigner()

            note_activity[t] = np.take(pitches, reorder)
            velocity[t] = np.take(velocity[t], reorder)
            self.reorder = reorder
            t += 1

        return np.stack([note_activity, velocity], axis=-1), polyphony
