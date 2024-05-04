# MHG-Predictor

## Usage

### Preprocessing Event Logs

The code for preprocessing event logs is located within the DGLWhole_Graph and DGLSubgraph modules.
- **DGLWhole_Graph**: This module is responsible for processing the entire raw event log data and constructing a multi-layer heterogeneous graph based on the complete event logs.
- **DGLSubgraph**: This module is used to process the multi-layer heterogeneous graph based on the complete event logs, from which it extracts isomorphic and heterogeneous subgraphs.

### Training, Validating, and Testing the Model
To train, validate, and test the model, use the following command (example for event log BPI Challenge 2013 incidents):
#### Main Execution Script

All preprocessing of event logs and training of models are consolidated into a single script, `main_2.py`, for streamlined execution. This script allows you to choose the dataset you wish to test against.

#### Selecting a Dataset

You can specify the dataset within the `main_2.py` script. As an example, the `bpi13_closed_problems` event log dataset can be selected for testing.

#### Running the Script

To run the preprocessing and model training on your chosen dataset, execute the following command in your terminal:

	python main_2.py

This command will trigger the processes as defined in the main_2.py script for `bpi13_closed_problems` event log.

## Loading Saved Model and Testing on Test Set
To load a saved model and test it on the test set, run (example for event log `bpi13_closed_problems`):

python metrics.py 

## Tools
pytorch: Used for deep learning operations.
python: The programming language used for the project.
## Data
The event logs for predictive business process monitoring can be found at 4TU Research Data.
