# DDSP-Piano: Differentiable Piano model for MIDI-to-Audio Performance Synthesis
DDSP-Piano is a piano sound synthesizer from MIDI based on [DDSP](https://github.com/magenta/ddsp).

### Links
- [Paper, published in DAFx20in22](https://dafx2020.mdw.ac.at/proceedings/papers/DAFx20in22_paper_48.pdf) 📄
- [Some audio examples](http://recherche.ircam.fr/anasyn/renault/DAFx22) 🔈

## Installation
This code relies on the official tensorflow implementation of [DDSP](https://github.com/magenta/ddsp) (tested on v3.2.0) without additional package required.
```
pip install --upgrade ddsp==3.2.0
```

## MIDI Synthesis
A piano MIDI file can be synthesized by the model with the command:
```
python synthesize_midi_file.py <input_midi_file.mid> <output_file.wav>
```
Additional arguments for the inference script include:
- `--duration`: the maximum duration of the synthesized file. If set to `None`, it will synthesize the whole file.
- `--piano_model`: the desired model among the 10 piano years learned from the MAESTRO dataset.
- `--ckpt`: a checkpoint folder with your own model weights.

It is recommended to limit the duration of the output file when synthesizing audio in batch.

## Model training
The paper model is trained with the [MAESTRO dataset](https://magenta.tensorflow.org/datasets/maestro#v300) (v3.0.0).

### Full training procedure

After following the instructions for downloading the dataset, a default piano model can be trained with the shell script:
```
sh train_ddsp_piano.sh <path-to-maestro-v3.0.0/> <experiment-directory>
```
This script reproduces the full training of the default model presented in the paper, by alternating between 2 training procedures (one for the layers involved in the partial frequencies computation and the other for the remaining layers).

The final model checkpoint should be located in `experiment-directory/phase_3/last_iter/`.


### Single phase training
According to our conducted listening test, decent synthesis quality can be achieved with only the first training phase.
Single phase training can be done with the python script:
```
python train_single_phase.py <path-to-maestro-v3.0.0/> <experiment-directory>
```
Additional arguments include:
- `--batch_size`, `--steps_per_epoch`, `--epochs`, `--lr`: your usual training hyper-parameters.
- `--phase`: the current training phase (which toggles the trainability of the corresponding layers).
- `--restore`: a checkpoint folder to restore weights from.

## Bibtex
If you use this code, please cite it as:
```
@inproceedings{
  renault2022diffpiano,
  title={Differentiable Piano Model for MIDI-to-Audio Performance Synthesis},
  author={Renault, Lenny and Mignot, Rémi and Roebel, Axel},
  booktitle={Proceedings of the 25th International Conference on Digital Audio Effects},
  year={2022}
}
```
