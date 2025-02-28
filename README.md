# RRT*former

This is the implementation of RRT*former

### RRT*former: Environment-Aware Sampling-Based Motion Planning using Transformer

## Setup
A suitable conda environment named RRTformer can be created and activated with:
```bash
conda env create -f environment.yaml
conda activate RRTformer
```
If you want to record the training process, `wandb` is also needed.

## Quick Start

### Related Data
Download [rrt_net_data.zip](https://drive.google.com/file/d/1-1GqCMNWjuwJkmsNNJ0gX-vPvRRLpuhh/view?usp=drive_link) and move the zip file into the root folder of this repo. Run
```bash
cd RRTformer/
unzip rrt_net_data.zip
```

### Demo
- For 2D, run
```bash
 cd 2d
 python demo_planning.py --env_file='../data/random_2d/test/envs_fsg.json'
```
`env_file` is the json file of your environment. Visualization will be shown in GUI.

- For 3D, run
```bash
 cd 3d
 python demo_planning_3d.py --env_file='../data/random_3d/test/envs_fsg.json'
```
## Data Generation
Instructions for generate your own data for train or test.

### Collect 2D random world data
```bash
cd 2d
python gene_data_json.py --yaml_path='../data/random_2d/random_2d.yml' --json_path='../data/random_2d/test/envs.json' --mode='test'
 python gene_data_astar.py --envs_file='../data/random_2d/train/envs.json' # For train
```

For training data, `mode`=train, for testing data, `mode`=test. `gene_data_json` is to generate the random environment. `gene_data_star` is to generate A* path in an environment for training.

### Collect 3D random world data

```bash
cd 3d
python gene_data_json_3d.py --yaml_path='../data/random_3d/random_3d.yml' --json_path='../data/random_3d/test/envs.json' --mode='test'
 python gene_data_astar_3d.py --envs_file='../data/random_3d/train/envs.json' # For train
```

## Model Training
Instructions for training your own models.

### 2D
Run
```bash
cd 2d
python train.py --dataset_path='../data/astar_2d.npy' --batch_size=256 --epochs=2000 --lr=0.0001 --checkpoint_dir='../checkpoint/2d'
```
For better results, the relevant parameters can be changed.

### 3D

Run

```bash
cd 3d
python train_3d.py --dataset_path='../data/astar_3d.npy' --batch_size=128 --epochs=2000 --lr=0.0001 --checkpoint_dir='../checkpoint/3d'
```

## Evaluation

### 2D
Run
```bash
cd 2d
python evaluate.py --env_file='../data/random_2d/test/envs_fsg.json'
```
### 3D
Run
```bash
cd 3d
python evaluate_3d.py --env_file='../data/random_3d/test/envs_fsg.json'
```

### Result Analysis
Put files in `evaluation` to `nirrt_star/results/evaluation + 2d or 3d` in [tedhuang96/nirrt_star](https://github.com/tedhuang96/nirrt_star) and follow `nirrt_star`'s instructions.


## References

[tedhuang96/nirrt_star](https://github.com/tedhuang96/nirrt_star)

