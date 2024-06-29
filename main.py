import torchvision.models as models
import torchvision.transforms as transforms
import os
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import urllib.request as request
import random
from skimage import io
from os import listdir
import supervision as sv
from inference import get_model
from PIL import Image
from inference_sdk import InferenceHTTPClient
    
def cutImage(imageName, image, positionsToCut ):
    for index, position in enumerate(positionsToCut):
        cutImage = image.crop((position["x"] - position["width"]/2, position["y"] - position["height"]/2, position["x"] + position["width"] / 2, position["y"] + position["height"] / 2))
        cutImage.save("Images-cut/" + str(index + 1) + " - " + imageName)
      #  cutImage.show()

def detectPuzzle(image):
    result = CLIENT.infer(image, model_id="yolo-rompecabezas/2")
    predictions = result['predictions']
    if len(predictions) == 1 and (predictions[0]['width'] == image.width and predictions[0]["height"] == image.height):
        print("Puzzle not found")
        return None
    else:
        # load the results into the supervision Detections api
        detections = sv.Detections.from_inference(result)

        # create supervision annotators
        bounding_box_annotator = sv.BoundingBoxAnnotator()
        label_annotator = sv.LabelAnnotator()

        # annotate the image with our inference results
        annotated_image = bounding_box_annotator.annotate(
            scene=image, detections=detections)
        annotated_image = label_annotator.annotate(
            scene=annotated_image, detections=detections)
        
        # display the image
       # sv.plot_image(annotated_image)
        return predictions

if __name__ == "__main__":
    imagesList = listdir("Images/")

    CLIENT = InferenceHTTPClient(
        api_url="https://detect.roboflow.com",
        api_key="tJeGrQHcijZ9A4T7J91p"
    )

    for imageName in imagesList:
        image = Image.open("Images/" + imageName)
        result = detectPuzzle(image)
        if result is not None:
            cutImage(imageName, image, result)


# tensors = []
# preprocess = transforms.Compose([transforms.ToTensor()])

# imagesList = listdir("Images/")
# loadedImages = []
# for image in imagesList:
#     img = Image.open("Images/" + image)
#     tensors.append(preprocess(img))
# #    plt.imshow(img); plt.axis('off'); plt.show()
# # for example in examples:
# #   img = io.imread(os.path.join(source, example))
# #   tensors.append(preprocess(img))
# #   plt.imshow(img); plt.axis('off'); plt.show()


# model = models.detection.maskrcnn_resnet50_fpn(pretrained=True).eval()
# predictions = model(tensors)

# for prediction in predictions:
    
#     mask = prediction['masks'][0][0].detach().numpy()

#     plt.imshow(mask); plt.axis('off'); plt.show()



