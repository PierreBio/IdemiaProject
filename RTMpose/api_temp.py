from mmpose.apis import MMPoseInferencer

# replace this with your own image path
img_path = 'data\images\train\764_1222608.jpg'

# instantiate the inferencer using the model alias
inferencer = MMPoseInferencer('human')

# The MMPoseInferencer API employs a lazy inference approach,
# creating a prediction generator when given input
result_generator = inferencer(img_path, show=True)
result = next(result_generator)
