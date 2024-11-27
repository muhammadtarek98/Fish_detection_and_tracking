Detection and Tracking Methods
Detection: We use a Faster R-CNN model for fish detection. This model is chosen for its accuracy and robustness in detecting objects in images.
Tracking: We use the Deep SORT algorithm to track detected fish across video frames. Deep SORT is effective in maintaining object identities even after occlusions.

model details
Total params: 18,930,229
Trainable params: 18,871,333
Non-trainable params: 58,896
Total mult-adds (G): 2.59
Input size (MB): 49.77
Forward/backward pass size (MB): 263.78
Params size (MB): 75.72
Estimated Total Size (MB): 389.26




Optimizations
Detection: The Faster R-CNN model is fine-tuned on a relevant dataset to improve detection accuracy.
Tracking: Deep SORT parameters are adjusted to handle the specific characteristics of fish movement.
Image Processing: CLAHE is used to enhance image contrast


training setups:
epoches=10
learning rate=1e-4
batch size=1
