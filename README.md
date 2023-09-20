# OnlineKD-LipNetITA
This repository is dedicated to my Master's thesis project, in which I will implement online Knowledge Distillation (KD) on the LipNet model with an Italian dataset.

## Project Overview

**Online KD Implementation**: the primary goal of this project is to apply an Online Knowledge Distillation technique described in the paper "_Effective Online Knowledge Distillation via Attention-Based Model Ensembling_" by Diana-Laura Borza et al. [[1](https://www.mdpi.com/2227-7390/10/22/4285)].

**LipNet Implementation**: this project is based on the Keras implementation of the method described in the paper "_LipNet: End-to-End Sentence-level Lipreading_" by Yannis M. Assael et al. [[2](https://arxiv.org/abs/1611.01599)], implemented by [[3](https://github.com/rizkiarm/LipNet)].

**Reference**: this project use a previous implementation of LipNet with an Italian dataset [[4](https://github.com/BenedettoSimone/Lipnet-ITA)]. Please follow the instructions in this repository to use the new updated version of LipNet.

## Prerequisites
- Python 3.6.6

## 1. Getting started
In this section, we will first examine how to install and run the vanilla version of LipNet. Next, we will explore the version with Online KD.


### 1.1 Usage

To use the model, first you need to clone the repository:
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

<br>

If you have some issue, please follow the list of requirements showed in the following images:

<p align="center"><img src="./readme_images/req1.png"/></p>
<p align="center"><img src="./readme_images/req2.png"/></p>
<p align="center"><img src="./readme_images/req3.png"/></p>


## 2. Dataset ITA
This section will explain the process used to create the Italian dataset. If you want to do your own or use pre-trained weights to do lipreading go to section 3.

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

### 3.1 Training Vanilla

### 3.1 Training with Knowledge Distillation