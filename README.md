# DDSP-Piano: Differentiable Piano model for MIDI-to-Audio Performance Synthesis
| [**Audio Samples 🔈**](http://recherche.ircam.fr/anasyn/renault/DAFx22)
| [**DAFx Conference Paper 📄**](https://dafx2020.mdw.ac.at/proceedings/papers/DAFx20in22_paper_48.pdf)
| [**JAES Article 📄**](https://doi.org/10.17743/jaes.2022.0102) 
|

DDSP-Piano is a piano sound synthesizer from MIDI based on [DDSP](https://github.com/magenta/ddsp).

![v2.0 Architecture](assets/4_ddsp-piano_v2.drawio.svg)

## Installation
This code relies on the official Tensorflow implementation of [DDSP](https://github.com/magenta/ddsp) (tested on v3.2.0 and v3.7.0) without additional package required.
```bash
pip install --upgrade ddsp==3.7.0
```

## Audio Synthesis from MIDI

### Single MIDI file
A piano MIDI file can be synthesized using the command:
```bash
python synthesize_midi_file.py <input_midi_file.mid> <output_file.wav>
```
Additional arguments for the inference script include:
- `-c`, `--config`: a `.gin` configuration file of a DDSP-Piano model architecture. By default, it is set to the `maestro-v2` architecture.
- `--ckpt`: a checkpoint folder with your own model weights.
- `--piano_type`: the desired model among the 10 piano years learned from the MAESTRO dataset (`0` to `9`).
- `-d`, `--duration`: the maximum duration of the synthesized file. It is set by default to `None`, which will synthesize the whole file.
- `-wu`, `--warm_up`: duration of recurrent layers warm-up (to avoid undesirable noise at the beginning of the synthesized audio).
- `-u`, `--unreverbed`: toggle it to also get the dry piano sound, without reverb applying.
- `-n`, `--normalize`: set the loudness of the output file to this amount of dBFS. Set by default to `None`, which does not apply any gain modification.

The default arguments will synthesize using the most recent version of DDSP-Piano.
If you want to use the default model presented in the published papers, the inference script should look like:
```bash
python synthesize_midi_file.py \
    --config ddsp_piano/configs/dafx22.gin \
    --ckpt ddsp_piano/model_weights/dafx22/ \
    <input_midi_file.mid> <output_file.wav>
```

### Synthesize multiple 

### Evaluation script

## Model training
The paper model is trained with the [MAESTRO dataset](https://magenta.tensorflow.org/datasets/maestro#v300) (v3.0.0).

### Full training procedure

After following the instructions for downloading the dataset, a default piano model can be trained with the shell script:
```bash
source train_ddsp_piano.sh <path-to-maestro-v3.0.0/> <experiment-directory>
```
This script reproduces the full training of the default model presented in the papers, by alternating between 2 training procedures (one for the layers involved in the partial frequencies computation and the other for the remaining layers).

The final model checkpoint should be located in `<experiment-directory>/phase_3/last_iter/`.


### Single phase training
According to our conducted listening test, decent synthesis quality can be achieved with only the first training phase.
Single phase training can be done with the python script:
```bash
python train_single_phase.py <path-to-maestro-v3.0.0/> <experiment-directory>
```
Additional arguments include:f
- `--batch_size`, `--steps_per_epoch`, `--epochs`, `--lr`: your usual training hyper-parameters.
- `--phase`: the current training phase (which toggles the trainability of the corresponding layers).
- `--restore`: a checkpoint folder to restore weights from.

## Bibtex
If you use this code for your research, please cite it as:
```latex
@article{renault2023ddsp_piano,
  title={DDSP-Piano: A Neural Sound Synthesizer Informed by Instrument Knowledge},
  author={Renault, Lenny and Mignot, Rémi and Roebel, Axel},
  journal={Journal of the Audio Engineering Society},
  volume={71},
  number={9},
  pages={552--565}
  year={2023},
  month={September}
}
```
or
```latex
@inproceedings{renault2022diffpiano,
  title={Differentiable Piano Model for MIDI-to-Audio Performance Synthesis},
  author={Renault, Lenny and Mignot, Rémi and Roebel, Axel},
  booktitle={Proceedings of the 25th International Conference on Digital Audio Effects},
  year={2022}
}
```
## Acknowledgments
This project is conducted at IRCAM and has been funded by the European Project [AI4Media](https://www.ai4media.eu/) (grant number 951911).

<p align="center">
  <a href="https://www.stms-lab.fr/"> <img src="assets/STMS-lab.png" width="9%"></a>
  &nbsp;
  <a href="https://www.ircam.fr/"> <img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTeJBTYBkufxwA0Us3JCcPAlGt0blX4LzRq1QxRb27xvQ&s" width="30%"></a>
  &nbsp;
  <img src="https://upload.wikimedia.org/wikipedia/fr/c/cd/Logo_Sorbonne_Universit%C3%A9.png" width="25%">
  &nbsp;
  <img src="https://upload.wikimedia.org/wikipedia/fr/thumb/7/72/Logo_Centre_national_de_la_recherche_scientifique_%282023-%29.svg/512px-Logo_Centre_national_de_la_recherche_scientifique_%282023-%29.svg.png" width="10%">
  &nbsp;
  <a href="http://www.idris.fr/jean-zay/"> <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/1/19/Logo-ministere-de-la-culture.png/788px-Logo-ministere-de-la-culture.png" width="12%"></a>
  &nbsp;
  <a href="https://ai4media.eu"> <img src="https://www.ai4europe.eu/sites/default/files/2021-06/Logo_AI4Media_0.jpg" width="14.5%"> </a>
</p>