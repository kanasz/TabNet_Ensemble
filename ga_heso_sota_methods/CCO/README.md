# CCO: A Cluster Core-based Oversampling Technique for Improved Class-Imbalanced Learning
Workflow Diagram

![Image](https://github.com/priyomondal/CCO/blob/main/data/diagram_updated_workflow.png)


## Building

Main Dependencies

- Pytorch
- Pandas
- Numpy
- Scikit Learn
- Imbalanced Learn

To install requirements, run:
```
pip install -r requirements.txt
```

## Usage

Run main .py file

```
python3 main.py --k [neighbours] --D [feature] --beta [density thresshold] --t [temperature] --batch_size [Batch Size] --num_workers [Number of Workers] --epochs [Epoch Number] --state [SEED] --split_no [SPLIT NUMBER]
```




