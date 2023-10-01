

groot/
    scripts/
        interactive_demo_from_datasets.py   # Annotate one image in the collected datasets using S2M, the annotation will be propagated through datasets using XMem
        interactive_demo_from_video.py      # Annotate one image from a video using S2M. This is mainly for the demo purpose.
        process_demonstration_data.py # a summary script of all the processing needed.
        single_task_training.py       # training script for simulation experiments
        single_tas_real_robot_training.py  # training script for real robot experiments


    groot_algo/
        dino_features.py      # class DinoV2ImageProcessor that gives a feature volume
        xmem_tracker.py       # class XMemTracker that trackes given initial masks.

        misc_utils.py         # miscellaneous functions related to repeated operations in GROOT

        o3d_modules.py        # a wrapper class of open3d operations for processing RGB-D point clouds
        point_mae_modules.py  # PointMAE encoder

        # evaluation
        eval_utils.py         # process the evaluation code

        # simulation specific
        env_wrapper.py        # class ViewChangingOffScreenRenderEnv that renders different camera angles for LIBERO tasks.

    segmentation_correspondence_model/ # segmentation correspondence model

        annotation_utils.py
        dinov2_masks_correspondence.py
        sam_amg.py


    real_robot_scripts/
        deoxys_data_collection.py # collect data using deoxys
        deoxys_reset_joints.py    # reset joints using deoxys control


        backend_node.js           # node.js code for backend. This is important if you want to use the deoxys UI. 

    paper_eval_scripts/
        eval_groot.py                  # model to evaluate groot
        real_robot_eval_checkpoint.py  # evaluation script 


    notebooks/
        module_example.ipynb               # TODO: a summary of example usage of all the vision module involved
                                           # TODO: visualize XMem video, SAM, DINOv2, point clouds
        visualize_datasets.ipynb           # TODO: a notebook that verifies the generated demonstration data by verifying through visual
        real_robot_eval_checkpoint.ipynb   # TODO: some naive jupyter notebook that implements very basic functionality for real robot evaluation.

    # data
    reference_images/              # default folder to save images for references
    target_images                  # saved annotation images from segmentation correspondence model