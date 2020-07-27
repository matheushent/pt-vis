from vis.core.gradcam import GradCam
import torchvision.models as models
from vis.utils import image
from vis.utils import saver
import os

input_path = os.path.join('inputs', 'cat_dog.png')
class_index = 281 # cat. Check https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a for a complete class list

model = models.vgg16(pretrained=True)

input_tensor = image.preprocess_input(input_path)

explainer = GradCam()
cam = explainer.explain(input_tensor, model, class_index, layer=11)

saver.save_gradient(cam, os.path.join('logs', 'gradcam'), 'cat_id.jpg')