### Detection and Tracking Methods:
- Detection: We use a Faster R-CNN model for fish detection. This model is chosen for its accuracy and robustness in detecting objects in images.
- Tracking: We use the Deep SORT algorithm to track detected fish across video frames. Deep SORT is effective in maintaining object identities even after occlusions.

- ======================================================================
- Layer (type:depth-idx)                                       Output Shape              Param #
- ============================================================
- FasterRCNN                                                   [0, 4]                    --
- ├─FasterRCNN: 1-1                                            [0, 4]                    --
- │    └─GeneralizedRCNNTransform: 2-1                         [2, 3, 320, 576]          --
- │    └─BackboneWithFPN: 2-2                                  [2, 256, 5, 9]            --
- │    │    └─IntermediateLayerGetter: 3-1                     [2, 960, 10, 18]          2,947,552
- │    │    └─FeaturePyramidNetwork: 3-2                       [2, 256, 5, 9]            1,467,392
- │    └─RegionProposalNetwork: 2-3                            [0, 4]                    --
- │    │    └─RPNHead: 3-3                                     [2, 15, 10, 18]           609,355
- │    │    └─AnchorGenerator: 3-4                             [6075, 4]                 --
- │    └─RoIHeads: 2-4                                         [0, 4]                    --
- │    │    └─MultiScaleRoIAlign: 3-5                          [0, 256, 7, 7]            --
- │    │    └─TwoMLPHead: 3-6                                  [0, 1024]                 13,895,680
- │    │    └─FastRCNNPredictor: 3-7                           [0, 2]                    10,250
- ==========================================================
- Total params: 18,930,229
- Trainable params: 18,871,333
- Non-trainable params: 58,896
- Total mult-adds (G): 2.59
- =========================================================
- Input size (MB): 49.77
- Forward/backward pass size (MB): 263.78
- Params size (MB): 75.72
- Estimated Total Size (MB): 389.26
- ======================================================




### Optimizations:
- Detection: The Faster R-CNN model is fine-tuned on a relevant dataset to improve detection accuracy.
- Tracking: Deep SORT parameters are adjusted to handle the specific characteristics of fish movement.
- Image Processing: CLAHE is used to enhance image contrast


### training setups:
- epoches=10
- learning rate=1e-4
- batch size=1
