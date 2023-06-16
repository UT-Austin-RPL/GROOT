# **Learning Generalizable Manipulation Policies with Object-Centric 3D Representations** 

Our real robot experiments are based off of the codebase [Deoxys](https://github.com/UT-Austin-RPL/deoxys_control)

# Getting Started

## Prerequisite

1. XMem installation


1. DINOv2 installation


1. SAM installation


## Installation

```shell
    pip install -r requirements.txt
```



# Get the datasets for training policies

## **Combining process**
1. create original dataset


2. prompt the ui for interactive annotation

``` python
python scripts/interactive_demo.py --dataset DATASET_NAME
--num_objects (specify your number of objects)
```

By default, the real robot datasets will have a special attributes `real` specified in the hdf5 dataset file. This script will directly take this flag into account.


### Single instance case (we focus for now)

A single script to run:

```python
python release_scripts/process_demonstration_data.py dataset_name=DATASET_NAME
```

Note that in this combined script, we will automatically read from the dataset file to check if this is a real robot dataset or not. If you run the following scripts separately, you will need to manually specify them. Specifying the real or sim datasets are mainly for purpose of image file processing, which is different between the real camera images and robosuite images. More information at `vos_3d_algo/o3d_modules.py`.

which does the following:

1. generate masks using xmem and directly do the tracking 

    ```
    python data_preprocessing/preprocess_datasetpy --dataset DATASET_NAME --save-video [--real]
    ````

    and we get `mask.hdf5`

2. create point clouds

    ```
    python data_preprocessing/pcd_dataset_generation.py --dataset DATASET_NAME
    ```

    and we get `pcd.hdf5`

3. augment

    ``` python
    python data_preprocessing/aug_demo.py --dataset DATASET_NAME
    ```

    this we get `*_aug_demo.hdf5`

    ``` python
    python data_preprocessing/aug_demo_generation.py --dataset DATASET_NAME
    ```

    after this we get `aug_pcd.hdf5`

4. grouping

    ```
    python data_preprocessing/pcd_grouping_preprocess.py
    ```

    after this we get `grouped_pcd.hdf5`

5. merge the generated data into the demonstration data

    `mask.hdf5`, `pcd.hdf5` -> original dataset

    `aug_pcd.hdf5`, `grouped_pcd.hdf5` -> `*_aug_demo.hdf5`



# Simulation Rollout Evaluation



# Real Robot Evaluation

1. Reset joints:

``` shell
	python real_robot_scripts/deoxys_reset_joints.py
```

1. Collect demonstrations:

Now we usually do name aliasing for commands, making it easy to launc
```shell
alias roller_stamp="python deoxys_data_collection.py --dataset-name demonstration_data/roller_stamp"
```

1. Create datasets

``` shell
python create_dataset_example.py --folder demonstration_data/roller_stamp/
```

1. Run camera nodes:

``` shell
 python run_camera_rec_node.py --camera-type rs --visualization --camera-id 0 --depth-visualization --drawing-json drawing_configs/stamp_paper.json  --eval
```


1. Run checkpoints:

``` shell
python eval_scripts/real_robot_eval_checkpoint.py --checkpoint-dir experiments_real/VOS_3D_Real_Robot_Benchmark_pogba/stamp_paper_aug/VOS3DSingleTask/VOS3DRealRobotMaskedNoWristTransformerPolicy_seed10000/run_002 --checkpoint-idx 100 --experiment-config real_robot_scripts/eval_new_instances/eval_new_instance_3.yaml
```
Some common configs are pre-defined in a yaml file, and you only need
to specify the yaml file in the argument `--experiment-config`.

If it's for testing new instances, you need to make sure that you specify the same experiment config for a single object. Otherwise the annotation used for propagating the VOS model will not match.


### Details:

In the new instance case, the following script will be triggered: `real_robot_scripts/real_robot_sam_result.py` and subsequently `auto_annotation/sam_amg.py`.

And there will be multiple files stored for the case of new instances:
- The first frame captured by the camera: `evaluation_results/DATASET_NAME/first_frame`
- The outputs of SAM: `evaluation_reulsts/DATASET_NAME/masks`
- The output from cosine similarity: `annotations/DATASET_NAME/evaluation_{ID}`

And the evaluation script will retrieve the image and annoation from `annotations/DATASET_NAME/evaluation_{ID}`.


# Evaluations
 All the scripts for evaluation in simulation are located in `paper_eval_scripts`. 





### Outdated



1. Preprocess dataset with masks

``` python
python data_preprocessing/preprocess_datasetpy --dataset DATASET_NAME --save-video
```

`--save-video` for saving overlayed videos for debugging purpose.

If processing a batch of datasets, use `data_preprocessing/batch_preprocess_datasets.py`.


1. Preprocess dataset with point cloud

``` python
python data_preprocessing/pcd_dataset_generation.py --dataset DATASET_NAME
```


1. Augment dataset with multiview point clouds

    a. first, augment the dataset itself

    ``` python
    python data_preprocessing/aug_demo.py --dataset DATASET_NAME
    ```

    alternatively:
    Expand the demo (just copy data, do not augment):
    ```python
    python data_preprocessing/expand_demo_ablation_augmentation.py --dataset datasets/libero_datasets/KITCHEN_SCENE3_put_the_moka_pot_on_the_stove_demo.hdf5
    ```

    b. augment the point clouds

    ``` python
    python data_preprocessing/aug_demo_generation.py --dataset DATASET_NAME
    ```

1. Preprocess masking:

```python
python data_preprocessing/pcd_grouping_preprocess.py
```


# Some 

1. Merge your dataset
use kaede.edit_demo to merge two datasets.

1. Calculate the camera transformation:

```data_preprocessing/get_aug_view_cfg.py```

1. Correspondence

    a. ```first_frame_segmentation.py```

    b. ```first_frame_annotation.py```



1.2 Features -- DINO

`vos_3d/dino_features.py`.



1.3 test on real data

`scripts/test_real_mask_dino.py`


1.4 batch random seed experiments

`scripts/batch_run_random_seeds.py`


### Multiple instances case

Change the first step of the previous procedure into the following steps:


```auto_annotation/first_frame_segmentation.py```

which gives first-frame segmentations on all target dataset, stored in `auto_annotation/first_frame/DATASET_NAME`


```auto_annotation/first_frame_annotation.py```

A simple version of annotation script is located in `proof_of_concept_first_frame_annotation.py`. After this step, a first-frame image, a first-frame segmentation will be saved for each demonstration trajectory. They can be found in `annotations/DATASET_NAME/DEMO_KEY`, with the name `first_frame.jpg` and `first_frame_annotation.png`

Then run:
    ```
    python data_preprocessing/preprocess_datasetpy --dataset DATASET_NAME --save-video --multi-instance [--real]
    ````


And this should be equivalent to the first step in the previous procedure. 









### Visualization
visualize if point clouds make sense:

```python
python visualization_scripts/visualize_point_cloud_sequence.py --dataset annotations/KITCHEN_SCENE3_put_the_frying_pan_on_the_stove_demo/pcd.hdf5
```


### Evaluation

#### Simulation:

main evaluation script: `masked_eval_checkpoint.py`


#### real world:

`eval_scripts/eval_real_robot.py`
`eval_scripts/real_robot_eval_checkpoint.py`


### Results

`result_summary.yaml` and `result_annotation.py`


And do `finalize_result.py`



