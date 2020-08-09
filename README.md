# pt-vis
Visualisation techniques performed on top of PyTorch

![Alt text](img/bear.jpg)

# ```Requisites```

## ```PyTorch```

Follow https://pytorch.org/get-started/locally/ to install rightly to your system ```pytorch==1.5.1``` and ```torchvision==0.6.1```.

Code teste on:

* Windows 10
* Python 3.6
* NVIDIA RTX 2060
* i7 9750-H

# ```Usage```

```
from vis.core.gradcam import GradCam
import torchvision.models as models
from vis.utils import image
from vis.utils import saver
import os

input_path = 'image path here'
class_index = <class index here. Int required>

model = models.vgg16(pretrained=True)

input_tensor = image.preprocess_input(input_path)

explainer = GradCam()
cam = explainer.explain(input_tensor, model, class_index, layer=11)

saver.save_gradient(cam, output_folder_path, output_filename)
```

Logs will be written in the path you set above.

Assuming you installed all packages rightly, run ```python example.py```.