"""
    Textual representations of joint indices
"""
joint_indices = {
    0: "Head",
    1: "Neck",
    2: "R Shoulder",
    3: "L Shoulder",
    4: "R Elbow",
    5: "L Elbow",
    6: "R Hand",
    7: "L Hand",
    8: "Torso",
    9: "R Hip",
    10: "L Hip",
    11: "R Knee",
    12: "L Knee",
    13: "R Foot",
    14: "L Foot"
}

"""
    Connections between joints to form a skeleton.
    (idx1, idx, limb colour)
"""
joint_connections = [
    (14, 12, "black"),  # L foot -- L knee
    (12, 10, "red"),  # L knee -- L Hip
    (13, 11, "green"),  # R foot -- R knee
    (11, 9, "yellow"),  # R knee -- R Hip
    (10, 8, "blue"),  # L hip -- Torso
    (9, 8, "magenta"),  # R hip -- Torso
    (8, 1, "cyan"),  # Torso -- Neck
    (1, 0, "lime"),  # Neck -- Head
    (7, 5, "orange"),  # L Hand -- L Elbow
    (5, 3, "gray"),  # L Elbow -- L Shoulder
    (3, 1, "slategray"),  # L Shoulder -- Neck
    (6, 4, "olive"),  # R Hand -- R Elbow
    (4, 2, "gold"),  # R Elbow -- R Shoulder
    (2, 1, "indigo"),  # R Shoulder -- Neck
]


"""
    Indices of pairs of joint to exchange with each other when flipping sides. Format: (left, right)
"""
side_idx = [
    (14, 13),  # foot
    (12, 11),  # knee
    (10, 9),  # hip
    (7, 6),  # hand
    (5, 4),  # elbow
    (3, 2),  # shoulder
]

"""
    Exchange positions of left-side joints for right-side joints and vice-versa
"""
def flip_joint_sides(joints):
    flipped_joints = joints.clone()
    for l_idx, r_idx in side_idx:
        flipped_joints[l_idx, :], flipped_joints[r_idx, :] = joints[r_idx, :].clone(), joints[l_idx, :].clone()
    return flipped_joints