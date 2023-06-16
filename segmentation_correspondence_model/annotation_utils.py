import os

def get_first_frame_folder(dataset_name):
    first_frame_folder = os.path.join("auto_annotation", "first_frames", dataset_name.split("/")[-1].replace("_demo.hdf5", ""))
    os.makedirs(first_frame_folder, exist_ok=True)

    return first_frame_folder

def get_first_frame_segmentation_folder(dataset_name):
    first_frame_folder = get_first_frame_folder(dataset_name)
    first_frame_mask_folder = os.path.join(first_frame_folder, "masks")
    os.makedirs(first_frame_mask_folder, exist_ok=True)
    return first_frame_mask_folder