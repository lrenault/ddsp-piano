import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from ddsp.core import midi_to_hz


def inharm_summary(inharm_model, step):
    # Inharmonicity curve of the inharmonic model
    note_range = 21. + tf.range(88, dtype=tf.float32)
    inharm_coefs = []
    for note in note_range:
        inharm_coefs.append(inharm_model(note)['inharm_coef'])

    # Plot curve
    fig = plt.figure(figsize=(10,5))
    fig.add_subplot(111)
    fig.tight_layout(pad=0)
    plt.axis('off')
    plt.plot(note_range.numpy(), inharm_coefs)
    plt.yscale('log')
    plt.ylim(1e-4, 5e-2)

    # Draw canvas and get image data of the plot
    fig.canvas.draw()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    data = data.astype(np.float32) / 256.
    plt.close(fig)
    
    # Add to summary
    tf.summary.image("Inharmonicity curve", data[np.newaxis,:],
                     step=step, max_outputs=1)


def detune_summary(detuner, step):
    # Inharmonicity curve of the inharmonic model
    note_range = 21. + tf.range(88, dtype=tf.float32)
    detuning = []
    for note in note_range:
        f0_hz = detuner(note)['f0_hz']
        f0_ET = midi_to_hz(note)  # Equal temperament tuning
        detuning.append(1200. * tf.math.log(f0_hz / f0_ET))

    # Plot curve
    fig = plt.figure(figsize=(10,5))
    fig.add_subplot(111)
    fig.tight_layout(pad=0)
    plt.axis('off')
    plt.ylim(-30., 50.)
    plt.plot(note_range.numpy(), detuning)

    # Draw canvas and get image data of the plot
    fig.canvas.draw()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    data = data.astype(np.float32) / 256.
    plt.close(fig)
    
    # Add to summary
    tf.summary.image("Detuning curve", data[np.newaxis,:],
                     step=step, max_outputs=1)
