import gin
from ddsp_piano.modules.inharm_synth import MultiAdd


@gin.register
def polyphonic_dag(additive, noise, reverb=None,
                   additive_controls=['amps', 'harmonic_distribution', 'f0_hz'],
                   noise_controls=['noise_magnitudes'],
                   reverb_controls=[],
                   n_synths=16):
    """Return a polyphonic DAG for a Processor Group.
    Args:
        - additive (ddsp.processors.Processor): a Harmonic synthesizer
        - noise (ddsp.processors.Processor): filtered noise synthesizer.
        - additive_controls (list): list of associated monophonic controls.
        - noise_controls (list): list monophonic controls keys associated with the noise.
        - reverb (ddsp.processors.Processor): optional reverb model
        - reverb_controls (list): list of reverb control keys.
        - n_synths (int): polyphonic capacity.
    """
    # Init summing operator
    add = MultiAdd(name='add')

    dag = []
    # Initialization
    dag.append((additive, [c + '_0' for c in additive_controls]))
    dag.append((noise, [c + '_0' for c in noise_controls]))
    dag.append((add, [noise.name + '/signal',
                      additive.name + '/signal']))

    # Construct synth polyphony
    for i in range(1, n_synths):
        dag.append((additive, [c + f'_{i}' for c in additive_controls]))
        dag.append((noise, [c + f'_{i}' for c in noise_controls]))
        dag.append((add, ['add/signal',
                          noise.name + '/signal',
                          additive.name + '/signal']))
    # Apply reverb
    if reverb is not None:
        dag.append((reverb, ['add/signal'] + reverb_controls))

    return dag
