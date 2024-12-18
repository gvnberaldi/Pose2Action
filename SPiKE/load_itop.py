from datasets.itop import ITOP
from visualization.plot_pc_joints import create_gif, clean_create_gif


AUGMENT_TEST  = [
    {
        "name": "CenterAug",
        "p_prob": 1.0,
        "p_axes": [True, True, True],
        "apply_on_gt": True
    },
    {
        "name": "TranslationAug",
        "p_prob": 0.0,
        "p_max_aabb_ratio": 2.0,
        "apply_on_gt": True
    }
]
AUGMENT_TRAIN  = [

    {
        "name": "CenterAug",
        "p_prob": 1.0,
        "p_axes": [True, True, True],
        "p_apply_extra_tensors": True
    },

    {
        "name": "RotationAug",
        "p_prob": 0.0,
        "p_axis": 1,
        "p_min_angle": 1.57,
        "p_max_angle": 1.57,
        "p_apply_extra_tensors": True
    },
    {
        "name": "NoiseAug",
        "p_prob": 0.0,
        "p_stddev": 0.01,
        "p_clip": 0.02,
        "p_apply_extra_tensors": False
    },
    {
        "name": "LinearAug",
        "p_prob": 0.0,
        "p_min_a": 0.9,
        "p_max_a": 1.1,
        "p_min_b": 0.0,
        "p_max_b": 0.0,
        "p_channel_independent": True,
        "p_apply_extra_tensors": True
    },
    {
        "name": "MirrorAug",
        "p_prob": 0.0,
        "p_axes": [True, False, False],
        "p_apply_extra_tensors": True
    },
    {
        "name": "MirrorAug",
        "p_prob": 1.0,
        "p_axes": [False, True, False],
        "apply_on_gt": True
    },

]


label_frame = 'middle'

# Replace the root path with the actual path to your ITOP dataset
dataset= ITOP(root='/data/iballester/datasets/ITOP-CLEAN-GT/SIDE', num_points=4096, frames_per_clip=5, train=False, use_valid_only=False, aug_list=AUGMENT_TEST, label_frame=label_frame)

# Now you can access the data in the dataset
clip, label, frame_idx = dataset[3001]

print(clip)
print(label)
print(frame_idx)
create_gif(clip, label, frame_idx, output_directory='visualization/gifs', plot_lines=True, label_frame=label_frame)

