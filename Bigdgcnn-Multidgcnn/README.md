This project was created in the context of the *Machine Learning Applications in Process Mining* Seminar. It contains an implementation of the BIG-DGCNN Algorithm by Chiorrini et al. [^big-dgcnn-algorithm].

The BIG-DGCNN Algorithm predicts the next activity of a process instance using a Deep Graph Convolutional Neural Network (DGCNN). As input to the DGCNN, the algorithm uses graphical representations of process instances, Instance Graphs, which are computed using the BIG Algorithm[^big-algorithm].

[^big-dgcnn-algorithm]: Chiorrini et al. Exploiting Instance Graphs and Graph Neural Networks for Next Activity Prediction. https://doi.org/10.1007/978-3-030-98581-3_9
[^big-algorithm]: Diamantini et al. Building instance graphs for highly variable processes. https://doi.org/10.1016/j.eswa.2016.04.021

# Project Structure
- The implementation of the algorithm is in the [bigdgcnn](./bigdgcnn/) directory
- Reproducibility scripts and requirements files are in the root directory
- Event Logs used in the evaluation are in the [Event Logs](./Event%20Logs/) directory
- The CPNTools model used to generate the synthetic event log is in the [CPNTools](./CPNTools/) directory

# Usage

```python
from bigdgcnn.ml.model import BIG_DGCNN
from pm4py import read_xes

log = read_xes("path/to/event/log.xes")
model = BIG_DGCNN(
  sort_pooling_k = 30,
  layer_sizes = [32]*4,
  batch_size = 32,
  learning_rate = 1e-3,
  dropout_rate = 0.1,
  dense_layer_sizes = [32],
  epochs = 100
)

model.train(log, logname="name-to-save-dataset-as")
# Torch will save the extracted dataset using the given name
# Subsequent training using this name will load the dataset from disk
model.plot_training_history()
```

# Running The Experiments
## Event Logs
- [Helpdesk Event Log](https://data.mendeley.com/datasets/39bp3vv62t/1)
  - Event data from the ticketing management process of the help desk of an Italian software company
- [BPI Challenge 2012](https://data.4tu.nl/articles/_/12689204/1)
  - Loan Application Process
  - Only Work-item related events &rarr; BPI12_W
- [eXdpn Event Log](./Event%20Logs/eXdpn/exdpn_with_customs.xes)
  - A synthetic event log generated by altering the CPNTools model of Park et al. [^exdpn]

[^exdpn]: Park et al. Explainable Predictive Decision Mining for Operational Support. https://doi.org/10.1007/978-3-031-26507-5_6

## Experiments
### Reproducing Experiments
The BIG-DGCNN Algorithm was run on each event log 10 times with the parameters reported in the original paper:

| Dataset  | Sort-Pooling K | Number of Graph Convolutions | Batch Size | Initial Learning Rate            | Dropout Rate |
|----------|----------------|------------------------------|------------|----------------------------------|--------------|
| Helpdesk | 30             | 5                            | 32         | 10<sup>-3</sup>                  | 0.1          |
| BPI12_W  | 5              | 5                            | 32         | 10<sup>-3</sup>                  | 0.2          |

- To run these experiments, run the corresponding script:
  - `python run_bpi12_mp.py`
  - `python run_helpdesk_mp.py`

#### Results
The results of these experiments can be read from the `summary.txt` files in the [Experiments](./Experiments/) directory

- [Helpdesk Summary](./Experiments/Helpdesk/summary.txt)
- [BPI12_W Summary](./Experiments/BPI12/LR_1e-3/summary.txt)

### Further Experiments
- For the BPI12 Event Log, our results deviated from the original paper, so we ran the experiment again with Learning Rate 10<sup>-4</sup>
    - `python run_bpi12_mp.py --1e-4`
    - [Results Summary](./Experiments/BPI12/LR_1e-4/summary.txt)
- The Helpdesk Event Log contains a number of traces with duplicate events. To get an idea of the noise robustness of the algorithm, we run it again on a repaired version of the event log, where cases containing duplicate events have been removed.
  - `python run_helpdesk_repaired_mp.py`
  - [Results Summary](./Experiments/Helpdesk%20Repaired/summary.txt)

### Synthetic Event Log

![](CPNTools/exdpn_with_customs.svg)

To analyse the prediction capability of the algorithm, as well as the potential of the algorithm when adding a data perspective to the model, we run it on a synthetic event log containing complex decision points based in the data perspective.

- *Request Manager Approval* if the total price of the purchase order is at least $800. *Request Standard Approval* otherwise.
- *Approve Purchase* if the total price is at most $1000 and the item’s category is not *“Fun”* and the supplier is not *“Scamming Corp.”*. *Reject Purchase* otherwise.
- *Hold at Customs* if the total price is at least $300 or the origin is Non-EU. *Pass Customs* otherwise.

#### Experiments
- We first run the BIG-DGCNN Algorithm 10 times on the synthetic event log as-is, not taking the data perspective into account
- Then, we extend the algorithm to include the data perspective, and run it another 10 times with different hyperparameters

| Dataset      | Sort-Pool K | Num. Graph Convs | Batch Size | Initial LR | 1D Conv. Outputs | Dropout Rate | Dense Layer Outputs |
|--------------|-------------|------------------|------------|------------|------------------|--------------|---------------------|
| Without Data | 30          | 5                | 32         | 1e-3       | 32               | 0.1          | \[32\]              |
| With Data    | 5           | 5                | 32         | 1e-3       | 32               | 0.2          | \[32, 64\]          |

- To run these experiments, run the corresponding script:
  - `python run_exdpn_mp.py`
  - `python run_exdpn_with_data_mp.py`	
- The results can be found in the corresponding `summary.txt` files in the [Experiments](./Experiments/) directory
  - [eXdpn Without Data](./Experiments/eXdpn/With%20Customs/summary.txt)
  - [eXdpn With Data](./Experiments/eXdpn/With%20Customs%20and%20Data/summary.txt)