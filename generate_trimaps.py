import torch
import numpy as np
from torchvision import transforms
import cv2

IMG_EXT = ('.png', '.jpg', '.jpeg', '.JPG', '.JPEG')

CLASS_MAP = {"background": 0, "aeroplane": 1, "bicycle": 2, "bird": 3, "boat": 4, "bottle": 5, "bus": 6, "car": 7,
             "cat": 8, "chair": 9, "cow": 10, "diningtable": 11, "dog": 12, "horse": 13, "motorbike": 14, "person": 15,
             "potted plant": 16, "sheep": 17, "sofa": 18, "train": 19, "tv/monitor": 20}


def trimap(probs, dilation_size, erosion_size, conf_threshold):
    """
    This function creates a trimap based on erosion and dilation operations
    Inputs [3]: an image with probabilities of each pixel being the foreground, size of dilation kernel, erosion kernel,
    foreground confidence threshold
    Output    : a trimap
    """
    mask = (probs > 0.5).astype(np.uint8) * 255

    erosion_pixels = 2 * erosion_size + 1
    erosion_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erosion_pixels, erosion_pixels))
    eroded = cv2.erode(mask, erosion_kernel, iterations=1)

    dilation_pixels = 2 * dilation_size + 1
    dilation_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation_pixels, dilation_pixels))
    dilation = cv2.dilate(eroded, dilation_kernel, iterations=1)

    remake = np.zeros_like(mask)
    remake[dilation == 255] = 127  # Set every pixel within dilated region as probably foreground.
    remake[probs > conf_threshold] = 255  # Set every pixel with large enough probability as definitely foreground.

    return remake


def get_trimap(image, label, dilation_kernel_size, erosion_kernel_size, conf_threshold):
    model = torch.hub.load('pytorch/vision:v0.6.0', 'deeplabv3_resnet101', pretrained=True)
    model.eval()

    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)  # create a mini-batch as expected by the model

#    if torch.cuda.is_available():
#        input_batch = input_batch.to('cuda')
#        model.to('cuda')

#    with torch.no_grad():
#        output = model(input_batch)['out'][0]
#        output = output.argmax(0)
#    out = output.cpu().numpy()
#    out = np.where(out == CLASS_MAP[label], 255, 0).astype(np.uint8)

    with torch.no_grad():
        output = model(input_batch)['out'][0]
        output = torch.softmax(output, 0)

    output_cat = output[CLASS_MAP[label], ...].numpy()
    trimap_image = trimap(output_cat, dilation_kernel_size, erosion_kernel_size, conf_threshold)

    return trimap_image
