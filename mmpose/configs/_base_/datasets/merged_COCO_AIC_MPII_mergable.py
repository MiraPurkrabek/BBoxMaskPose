dataset_info = dict(
    dataset_name='merged_COCO_AIC_MPII_mergable',
    paper_info=dict(
        author='Miroslav Purkrabek',
        title='Merged Pose Estimation Dataset',
        container='',
        year='2024',
        homepage='',
    ),
    # COCO - 17 keypoints:
    #   -> nose, left_eye, right_eye, left_ear, right_ear,
    #   -> left_shoulder, right_shoulder, left_elbow, right_elbow, left_wrist, right_wrist
    #   -> left_hip, right_hip, left_knee, right_knee, left_ankle, right_ankle
    # AIC - 14 keypoints:
    #   -> right_shoulder, right_elbow, right_wrist, left_shoulder, left_elbow, left_wrist
    #   -> right_hip, right_knee, right_ankle, left_hip, left_knee, left_ankle
    #   -> head_top, thorax
    # MPII - 16 keypoints:
    #   -> right_ankle, right_knee, right_hip, left_hip, left_knee, left_ankle
    #   -> pelvis, thorax, lower_neck, head_top
    #   -> right_wrist, right_elbow, right_shoulder, left_shoulder, left_elbow, left_wrist
    # --> Altogether (COCO: 12+5, AIC: 12+2, MPII: 12+4) = 12+5+2+4 = 23 keypoints

    # Merged - 21 keypoints:
    # ----
        # Merged - 22 keypoints:
    #  0. Nose (COCO 0, AIC -, MPII -)
    #  1. Left Eye (COCO 1, AIC -, MPII -)
    #  2. Right Eye (COCO 2, AIC -, MPII -)
    #  3. Left Ear (COCO 3, AIC -, MPII -)
    #  4. Right Ear (COCO 4, AIC -, MPII -)
    #  5. Left Shoulder (COCO 5, AIC 3, MPII 13)
    #  6. Right Shoulder (COCO 6, AIC 0, MPII 12)
    #  7. Left Elbow (COCO 7, AIC 4, MPII 14)
    #  8. Right Elbow (COCO 8, AIC 1, MPII 11)
    #  9. Left Wrist (COCO 9, AIC 5, MPII 15)
    # 10. Right Wrist (COCO 10, AIC 2, MPII 10)
    # 11. Left Hip (COCO 11, AIC 6, MPII 3)
    # 12. Right Hip (COCO 12, AIC 9, MPII 2)
    # 13. Left Knee (COCO 13, AIC 10, MPII 4)
    # 14. Right Knee (COCO 14, AIC 7, MPII 1)
    # 15. Left Ankle (COCO 15, AIC 11, MPII 5)
    # 16. Right Ankle (COCO 16, AIC 8, MPII 0)
    # 17. MPII Thorax (COCO -, AIC -, MPII 7)
    # 18. MPII Pelvis (COCO -, AIC -, MPII 6)
    # 19. MPII Neck (COCO -, AIC -, MPII 8)
    # 20. MPII Head Top (COCO -, AIC -, MPII 9)
    # 21. AIC Neck (COCO -, AIC 13, MPII -)
    # 22. AIC Head Top (COCO -, AIC 12, MPII -)

    keypoint_info={
        0:
        dict(name='nose', id=0, color=[51, 153, 255], type='upper', swap=''),
        1:
        dict(
            name='left_eye',
            id=1,
            color=[51, 153, 255],
            type='upper',
            swap='right_eye'),
        2:
        dict(
            name='right_eye',
            id=2,
            color=[51, 153, 255],
            type='upper',
            swap='left_eye'),
        3:
        dict(
            name='left_ear',
            id=3,
            color=[51, 153, 255],
            type='upper',
            swap='right_ear'),
        4:
        dict(
            name='right_ear',
            id=4,
            color=[51, 153, 255],
            type='upper',
            swap='left_ear'),
        5:
        dict(
            name='left_shoulder',
            id=5,
            color=[0, 255, 0],
            type='upper',
            swap='right_shoulder'),
        6:
        dict(
            name='right_shoulder',
            id=6,
            color=[255, 128, 0],
            type='upper',
            swap='left_shoulder'),
        7:
        dict(
            name='left_elbow',
            id=7,
            color=[0, 255, 0],
            type='upper',
            swap='right_elbow'),
        8:
        dict(
            name='right_elbow',
            id=8,
            color=[255, 128, 0],
            type='upper',
            swap='left_elbow'),
        9:
        dict(
            name='left_wrist',
            id=9,
            color=[0, 255, 0],
            type='upper',
            swap='right_wrist'),
        10:
        dict(
            name='right_wrist',
            id=10,
            color=[255, 128, 0],
            type='upper',
            swap='left_wrist'),
        11:
        dict(
            name='left_hip',
            id=11,
            color=[0, 255, 0],
            type='lower',
            swap='right_hip'),
        12:
        dict(
            name='right_hip',
            id=12,
            color=[255, 128, 0],
            type='lower',
            swap='left_hip'),
        13:
        dict(
            name='left_knee',
            id=13,
            color=[0, 255, 0],
            type='lower',
            swap='right_knee'),
        14:
        dict(
            name='right_knee',
            id=14,
            color=[255, 128, 0],
            type='lower',
            swap='left_knee'),
        15:
        dict(
            name='left_ankle',
            id=15,
            color=[0, 255, 0],
            type='lower',
            swap='right_ankle'),
        16:
        dict(
            name='right_ankle',
            id=16,
            color=[255, 128, 0],
            type='lower',
            swap='left_ankle'),
        17:
        dict(
            name='thorax_mpii',
            id=17,
            color=[51, 153, 255],
            type='upper',
            swap=''),
        18:
        dict(
            name='pelvis_mpii',
            id=18,
            color=[51, 153, 255],
            type='lower',
            swap=''),
        19:
        dict(
            name='neck_mpii',
            id=19,
            color=[51, 153, 255],
            type='upper',
            swap=''),
        20:
        dict(
            name='head_top_mpii',
            id=20,
            color=[51, 153, 255],
            type='upper',
            swap=''),
        21:
        dict(
            name='neck_aic',
            id=21,
            color=[51, 153, 255],
            type='upper',
            swap=''),
        22:
        dict(
            name='head_top_aic',
            id=22,
            color=[51, 153, 255],
            type='upper',
            swap=''),
    },
    skeleton_info={
        0:
        dict(link=('left_ankle', 'left_knee'), id=0, color=[0, 255, 0]),
        1:
        dict(link=('left_knee', 'left_hip'), id=1, color=[0, 255, 0]),
        2:
        dict(link=('right_ankle', 'right_knee'), id=2, color=[255, 128, 0]),
        3:
        dict(link=('right_knee', 'right_hip'), id=3, color=[255, 128, 0]),
        4:
        dict(link=('left_hip', 'right_hip'), id=4, color=[51, 153, 255]),
        5:
        dict(link=('left_shoulder', 'left_hip'), id=5, color=[51, 153, 255]),
        6:
        dict(link=('right_shoulder', 'right_hip'), id=6, color=[51, 153, 255]),
        7:
        dict(
            link=('left_shoulder', 'right_shoulder'),
            id=7,
            color=[51, 153, 255]),
        8:
        dict(link=('left_shoulder', 'left_elbow'), id=8, color=[0, 255, 0]),
        9:
        dict(
            link=('right_shoulder', 'right_elbow'), id=9, color=[255, 128, 0]),
        10:
        dict(link=('left_elbow', 'left_wrist'), id=10, color=[0, 255, 0]),
        11:
        dict(link=('right_elbow', 'right_wrist'), id=11, color=[255, 128, 0]),
        12:
        dict(link=('left_eye', 'right_eye'), id=12, color=[51, 153, 255]),
        13:
        dict(link=('nose', 'left_eye'), id=13, color=[51, 153, 255]),
        14:
        dict(link=('nose', 'right_eye'), id=14, color=[51, 153, 255]),
        15:
        dict(link=('left_eye', 'left_ear'), id=15, color=[51, 153, 255]),
        16:
        dict(link=('right_eye', 'right_ear'), id=16, color=[51, 153, 255]),
        17:
        dict(link=('left_ear', 'left_shoulder'), id=17, color=[51, 153, 255]),
        18:
        dict(
            link=('right_ear', 'right_shoulder'), id=18, color=[51, 153, 255])
    },
    joint_weights=[
        1., 1., 1., 1., 1., 1., 1., 1.2, 1.2, 1.5, 1.5, 1., 1., 1.2, 1.2, 1.5, 1.5,
        1., 1., 1., 1., 1., 1.,
    ],
    sigmas=[
        # Face - 0.025
        # Shoulders, thorax, neck, top of head - 0.079
        # Elbows - 0.072
        # Wrists - 0.062
        # Hips, pelvis - 0.107
        # Knees - 0.087
        # Ankles - 0.089

        # COCO
        0.026, 0.025, 0.025, 0.035, 0.035, 0.079, 0.079, 0.072, 0.072, 0.062,
        0.062, 0.107, 0.107, 0.087, 0.087, 0.089, 0.089, 
        
        # Additional
        0.079, 0.107, 0.079, 0.079, 0.079, 0.079,
    ])
