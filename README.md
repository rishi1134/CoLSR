# [WACV 2026] CoLSR: Chain-of-Look Spatial Reasoning for Dense Surgical Instrument Counting
**Rishikesh Bhyri**, Brian R Quaranto, Junsong Yuan, Peter C W Kim*, Nan Xi*

_*Equal Advising_

Preprint Link: https://arxiv.org/abs/2602.11024 (Note: This version includes additional authors who contributed during the rebuttal phase)

# News
- 12/08/2025 - Released the SurgCount-HD dataset form
- 02/11/2026 - [Paper](https://arxiv.org/abs/2602.11024) preprint available on arXiv
- 02/15/2026 - Code released

## Contents
* [SurgCount-HD Dataset](#surgcount-hd-dataset)
* [Setup](#setup)
* [Inference](#inference)
* [Train](#train)
* [Acknowledgement](#acknowledgement)
* [License and Disclaimer](#license-and-disclaimer)

# SurgCount-HD Dataset
Please complete this [Data Agreement Form](https://forms.office.com/r/8vT8eBVCxZ) to receive access to the dataset download link.

## Setup

### Download

```
sudo apt update
sudo apt install build-essential
git clone https://github.com/rishi1134/CoLSR.git
```

### Install
We used [Anaconda version 2024.02-1](https://repo.anaconda.com/archive/Anaconda3-2024.02-1-Linux-x86_64.sh)
```
conda create -n surgcount python=3.9.19
conda activate surgcount
pip install -r requirements.txt
export CC=/usr/bin/gcc-11 # this ensures that gcc 11 is being used for compilation
cd models/GroundingDINO/ops
python setup.py build install
python test.py # should result in 6 lines of * True
```

### Download Pre-Trained Weights
```
mkdir checkpoints
python download_bert.py
wget -P checkpoints https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha2/groundingdino_swinb_cogcoor.pth
```

## Inference

The model weights used in the paper can be downloaded from [Google Drive link](https://drive.google.com/file/d/16IxliBhZufSf7Fizdr8OOptcDjBpOpyl/view?usp=sharing).
Place the downloaded checkpoint under the `results` folder.
```
./test.sh
```

## Train

- Download the SurgCount-HD dataset and place the annotation files under the appropriate `data\surgcount-hd` folder.
- Update the dataset paths in `config\datasets_surgcount.json`
- Update the training configurations in `config\cfg_surgcount_vit_b.py`, if necessary.
- Use the `train.sh` to start the training.

# Acknowledgement

A significant portion of the code in this repository is derived from the below projects. We gratefully acknowledge their efforts in making this work available as open source.

- [CountGD repository](https://github.com/niki-amini-naieni/CountGD)
- [GroundingDINO repository](https://github.com/IDEA-Research/GroundingDINO)

# License and Disclaimer

The License and Disclaimer statement can be found in the LICENSE file.
* MIT License - Code
* CC BY-NC 4.0 License - Dataset

# Contributors

We extend our sincere gratitude to the following contributors from the Jacobs School of Medicine and Biomed
ical Sciences for their invaluable support in the data collection and annotation process.
* Steven Schwaitzberg, MD
* Philip Seger, MD
* Katy Tung, MD
* Brendan Fox, B.Sc.
* Gene Yang, MD
