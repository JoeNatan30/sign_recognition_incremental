# Sign Language Recognition - Incremental Methods

This repository have the code to train a Sign Language Recognition (SLR) model that use an incremental learning approach.

# Methods
We test three Methods:

- Fixed last layer 
- Expanded last layer
- Weighted last layer

(The frozen version of each method's feature extractor is also available)

# Spoter
The SLR model used in this repository is named 'Spoter' and was created by Matyáš Boháček. The original code is available in this [Repository](https://github.com/matyasbohacek/spoter).

The Spoter utilized in this context is a modified version derivate from the original Spoter version. 
In essence, the following modifications were implemented:
- A reduction in the number of model parameters
- Some modification to the data augmentation methods
- Adaptations to the model to align with our dataset format and keypoint configuration

# Dataset preparation
We utilized the [ConnectingPoints Repository](https://github.com/JoeNatan30/ConnectingPoints) to obtain the final version of AEC and PUCP-DGI305 dataset. For future dataset preparation, we will use the [Perusil framework Repository](https://github.com/gissemari/PeruvianSignLanguage).

# Instructions for Replicating Our Paper Results

We use `pip` to install the dependencies. To install these dependencies, use the following file `requirement.txt`.

The commands used for our experiment are available in the following shell script file (`SIMBig2023_Commands.sh`). Please note that the file contains a list of commands, but it should not be executed as a normal shell script, as it won't function properly in that context.

Additionally, we have included three CSV files containing the lists of word orders used to test the methods:
- incrementalList_V1.csv
- incrementalList_V2.csv
- incrementalList_V3.csv

# Citation

If you use this repository or part of it, You can cite our paper:

* (Citation currently unavailable, but will be provided soon)