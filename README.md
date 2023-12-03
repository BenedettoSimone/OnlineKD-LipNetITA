# OnlineKD-LipNetITA
This repository is dedicated to my Master's thesis project, in which I will implement online Knowledge Distillation (KD) on the LipNet model with an Italian dataset.

## Project Overview

**Online KD Implementation**: the primary goal of this project is to apply an Online Knowledge Distillation technique described in the paper "_Effective Online Knowledge Distillation via Attention-Based Model Ensembling_" by Diana-Laura Borza et al. [[1](https://www.mdpi.com/2227-7390/10/22/4285)].

**LipNet Implementation**: this project is based on the Keras implementation of the method described in the paper "_LipNet: End-to-End Sentence-level Lipreading_" by Yannis M. Assael et al. [[2](https://arxiv.org/abs/1611.01599)], implemented by [[3](https://github.com/rizkiarm/LipNet)].

**Reference**: this project use a previous implementation of LipNet with an Italian dataset [[4](https://github.com/BenedettoSimone/Lipnet-ITA)]. Please follow the instructions in this repository to use the new updated version of LipNet.

## Prerequisites
- Python 3.10
- [ffmpeg](https://www.ffmpeg.org)

## 1. Getting started

First you need to clone the repository:
```
https://github.com/BenedettoSimone/OnlineKD-LipNetITA
```

Then you can install the package (if you are using an IDE, you can skip this step):
```
cd OnlineKD-LipNetITA/
pip install -e .
```

Next, install the requirements:

```
pip install -r requirements.txt
```
If an error occurs when installing the dlib package, install cmake.

## 2. Dataset ITA
This section will explain the process used to create the Italian dataset. _You can skip this section if you do not want to build a custom dataset._

### 2.1 Sentences
An Italian dataset containing the following sentences was used in this project.

|               Sentence                | ID  |                           Sentence                           | ID  |
|:-------------------------------------:|:---:|:--------------------------------------------------------:|:---:|
|  Salve quanto costa quell' articolo?  |  0  |                   Tutto bene, grazie.                    | 10  |
|     È in offerta, costa 10 euro.      |  1  |                Prendiamo un caffè al bar?                | 11  |
|    Perfetto, vorrei comprarne due.    |  2  |       Certo volentieri, io lo prenderò macchiato.        | 12  |
| Certo ecco a lei, vuole un sacchetto? |  3  |               A che ora arriva il pullman?               | 13  |
|       Sì, grazie e arrivederci.       |  4  |          Dovrebbe arrivare tra qualche minuto.           | 14  |
|     Le auguro una buona giornata.     |  5  |                Quanto costa il biglietto?                | 15  |
|      Buongiorno, io sono Mario.       |  6  | Purtroppo non lo so, però potresti chiedere all’autista. | 16  |
|       Buonasera, io sono Mario.       |  7  |                Va bene, grazie lo stesso.                | 17  |
|       Piacere Luigi, come stai?       |  8  |                          Prego.                          | 18  |
|            Tutto bene, tu?            |  9  |

<br>


### 2.2 Building
To build the dataset, a video recording tool was used (https://github.com/BenedettoSimone/Video-Recorder).
The videos have a size of ``360x288 x 4s``. Use the information provided in the repository and on the main page to replicate this work.
<br><br>After collecting the videos for each subject, the dataset should be organised with the following structure.
```
DATASET:
├───s1
│   ├─── 0-bs.mpg
│   ├─── 1-bs.mpg
│   └───...
├───s2
│   └───...
└───...
    └───...
```
Since ``25fps`` videos are needed, and since the experiment was conducted by recording with several devices, the videos have to be converted to 25fps. To do this, run the ``change_fps/change_fps.py`` script.
Then you have to replace the videos in the ``DATASET`` folder with the newly created videos.

### 2.3 Forced Alignment
For each video, audio and text synchronization (also known as forced alignment) was performed using [Aeneas](https://github.com/readbeyond/aeneas).

After installing Aeneas, create a copy of the dataset and organize the folder as shown below:

```
ForcedAlignment:
│   ├──DatasetCopy:
│       ├───s1
│       │   ├─── 0-bs.mpg
│       │   ├─── 1-bs.mpg
│       │   └───...
│       ├───s2
│       │   └───...
│       └───...
│           └───...
│        
```


Then, follow these steps in the `terminal`:
1. Use the script `alignment/create_fragments_txt.py` to create a `txt` file for each video, following the rules established by Aeneas.
2. Use the script `alignment/autorunAlign.py` to dynamically create the `config` file and generate the `align_json` folder in the `ForcedAlignment` directory.

After running the script, the `ForcedAlignment` folder will have the following structure:

```
ForcedAlignment:
│   ├──DatasetCopy:
│   │   ├───s1
│   │   │   ├─── 0-bs.mpg
│   │   │   ├─── 0-bs.txt
│   │   │   └───...
│   │   ├───s2
│   │       └───...
│   ├──align_json:
│       ├───s1
│       │   ├─── 0-bs.json
│       │   ├─── 1-bs.json
│       │   └───...
│       ├───s2
│       │   └───...   
```


3. Finally, use the script `alignment/alignment_converter.py` to transform each JSON file into an `.align` file with the following format:
```
0 46000 sil
46000 65000 Perfetto
65000 76000 vorrei
76000 88000 comprarne
88000 92000 due.
92000 99000 sil
```


The first number indicates the start of that word, and the second number indicates the stop. Each number represents the frame numbers multiplied by 1000 (e.g., frames 0-46 are silence, frames 46-65 are the word "Perfetto," etc).

Now, you will have the `align` folder in the `ForcedAlignment` directory.


### 2.4 Mouth extract
Before starting to extract frames and crop the mouth area insert the ``DATASET`` folder in the project folder and the ``align`` folder in ``Training/datasets/``.

After, execute the script ``MouthExtract/mouth_extract.py`` that return ``100 frames`` for each video in a new folder ``frames``. 

Finally, split this folder in ``Training/datasets/train`` and ``Training/datasets/val`` using 80% for training phase and 20% for validation phase.


## 3. Training
In the previous section, we examined how to build our Italian dataset. Now, we will focus on the training process of the LipNet model, dividing it into two main sections: the ``Training Vanilla`` and the ``Training with Knowledge Distillation``.

In the ``Training Vanilla`` section, we will explore how to train the LipNet model using the traditional approach without the use of Knowledge Distillation. This will provide us a basis for comparison to evaluate the performance of the model after the implementation of Knowledge Distillation.

Next, in the section ``Training with Knowledge Distillation`` we will explore how to train the LipNet model using the proposed KD framework.

If you are using a new dataset, please refer to the FAQ section to change the parameters.

The next table shows the details of the trainings carried out.

|        Training         |           Details            | Best model                |
|:-----------------------:|:----------------------------:|---------------------------|
|   2023_10_26_09_22_27   | LipNet vanilla 32 batch size | V32_weights567            |
|   2023_10_28_11_30_29   |    LipNet-128 vanilla 32b    | N/A                       |
|   2023_10_31_18_23_34   |        LipNet kd 16b         | KD16_weights590_peer_00, KD16_weights545_peer_01 |
|   2023_11_02_10_55_00   |    LipNet-256 vanilla 32b    | N/A                       |
|   2023_11_04_16_41_41   |      LipNet vanilla 16b      | V16_weights598            |
|   2023_11_06_17_35_37   |    LipNet-256 vanilla 16b    | N/A                       |
|   2023_11_06_17_36_26   |    LipNet-128 vanilla 16b    | N/A                       |



### 3.1 Training Vanilla
To train vanilla models use the ``Training/train.py`` script. You can find the different models in ``Lipnet/model.py``,``Lipnet/model2.py``,``Lipnet/model3.py``.

_N.B. before running vanilla training, disable the ``on_batch_end`` method in ``Lipnet/callbacks.py``._

### 3.1 Training with Knowledge Distillation
To train the models use the ``Training/train_KD.py`` script.

## FAQ
<details>
    <summary>What do the parameters "absolute_max_string_len" and "output_size" mean?</summary>

These parameters appear in the scripts ``model.py`` and in ``train.py``.

```python
# model.py script

class LipNet(object):
    def __init__(self, img_c=3, img_w=100, img_h=50, frames_n=100, absolute_max_string_len=54, output_size=28):
        pass
```

```python
# train.py script

def train(run_name, start_epoch, stop_epoch, img_c, img_w, img_h, frames_n, absolute_max_string_len, minibatch_size):
    pass
```
**absolute_max_string_len**: is the maximum length of the sentences, including spaces, in the dataset. In the dataset used:  len (purtroppo non lo so pero potresti chiedere all autista) = 54

**output_size**: is the number of letters in the alphabet (26) + blank + space = 28.


</details>