# MHG-Predictor
MHG-Predictor: A Multi-Layer Heterogeneous Graph-based Predictor for Next Activity in Complex Business Processes
https://dl.acm.org/doi/10.1145/3701716.3715248

## Usage

### Preprocessing Event Logs

The code for preprocessing event logs is located within the Whole_Graph and Subgraph modules.
- **Whole_Graph**: This module is responsible for processing the entire raw event log data and constructing a multi-layer heterogeneous graph based on the complete event logs.
- **Subgraph**: This module is used to process the multi-layer heterogeneous graph based on the complete event logs, from which it extracts isomorphic and heterogeneous subgraphs.

### Training, Validating, and Testing the Model
To train, validate, and test the model, use the following command (example for event log BPI Challenge 2013 closed_problems):
#### Main Execution Script

All preprocessing of event logs and training of models are consolidated into a single script, `main.py`, for streamlined execution. This script allows you to choose the dataset you wish to test against.

#### Selecting a Dataset

You can specify the dataset within the `main.py` script. As an example, the `bpic2017_o` event log dataset can be selected for testing.

#### Running the Script

To run the preprocessing and model training on your chosen dataset, execute the following command in your terminal:

	python main.py

This command will trigger the processes as defined in the main.py script for `bpic2017_o` event log.

## Loading Saved Model and Testing on Test Set
To load a saved model and test it on the test set, run (example for event log `bpic2017_o`):

	python metrics.py 

## Tools
pytorch: Used for deep learning operations.
python: The programming language used for the project.
## Data
The event logs for predictive business process monitoring can be found at 4TU Research Data.


# - **Baselines**

This repository also includes implementations of various baseline methods for predictive business process monitoring, including MiDA, ProcessTransformer, MiTFM, PREMIERE, gcn-procesprediction, BIGDGCNN, and Multi-BIGDGCNN.

## Baseline Methods

### MiDA
- **Reference:** Vincenzo Pasquadibisceglie, Annalisa Appice, Giovanna Castellano, and Donato Malerba. A multi-view deep learning approach for predictive business process monitoring. IEEE Transactions on Services Computing, 15(4):2382–2395, 2022.

### ProcessTransformer
- **Reference:** Zaharah A Bukhsh, Aaqib Saeed, and Remco M Dijkman. Processtransformer: Predictive business process monitoring with transformer network. arXiv preprint arXiv:2104.00721, 2021.

### MiTFM
- **Reference:** Jiaxing Wang, Chengliang Lu, Bin Cao, and Jing Fan. MiTFM: A multi-view information fusion method based on transformer for next activity prediction of business processes. In Proceedings of the 14th Asia-Pacific Symposium on Internetware, pages 281–291, 2023.

### PREMIERE
- **Reference:** V. Pasquadibisceglie, A. Appice, G. Castellano, and D. Malerba. Predictive process mining meets computer vision. In Business Process Management Forum: BPM Forum 2020, Seville, Spain, September 13–18, 2020, Proceedings 18, pages 176–192, 2020.

### GCN-ProcessPrediction
- **Reference:** Ishwar Venugopal, Jessica Töllich, Michael Fairbank, and Ansgar Scherp. A comparison of deep-learning methods for analysing and predicting business processes. In 2021 International Joint Conference on Neural Networks (IJCNN), pages 1–8. IEEE, 2021.

### BIGDGCNN
- **Reference:** Andrea Chiorrini, Claudia Diamantini, Alex Mircoli, and Domenico Potena. Exploiting instance graphs and graph neural networks for next activity prediction. In International conference on process mining, pages 115–126, 2021.

### Multi-BIGDGCNN
- **Reference:** Andrea Chiorrini, Claudia Diamantini, Laura Genga, and Domenico Potena. Multi-perspective enriched instance graphs for next activity prediction through graph neural network. Journal of Intelligent Information Systems, 61(1):5–25, 2023.
