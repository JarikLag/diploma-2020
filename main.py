from detecto import core, utils
import generate_trimaps
import matting
import numpy as np
import cv2
from os import listdir
from os.path import join, isfile, basename


def read_image(name):
    return (cv2.imread(name) / 255.0)[:, :, ::-1]


def read_trimap(name):
    trimap_im = cv2.imread(name, 0) / 255.0
    h, w = trimap_im.shape
    trimap = np.zeros((h, w, 2))
    trimap[trimap_im == 1, 1] = 1
    trimap[trimap_im == 0, 0] = 1
    return trimap


def get_object_box(image, object_name):
    model = core.Model()
    labels, boxes, _ = model.predict_top(image)
    if object_name in labels:
        idx = labels.index(object_name)
        return boxes[idx]
    else:
        return None


def get_cropped_box(image, label):
    box = get_object_box(image, label)
    if box is not None:
        box = list(map(lambda x: round(float(x)), box))
        x_min, y_min, x_max, y_max = box
        return image[y_min:y_max, x_min:x_max]
    else:
        return None


def swap_bg(image, alpha):
    green_bg = np.full_like(image, 255).astype(np.float32)

    alpha = alpha[:, :, np.newaxis]
    result = alpha * image.astype(np.float32) + (1 - alpha) * green_bg
    result = np.clip(result, 0, 255).astype(np.uint8)

    return result


def process_image(dirname, image_path, label):
    image_name = basename(image_path)
    image = utils.read_image(image_path)
    cropped = get_cropped_box(image, label)
    if cropped is None:
        return
    cv2.imwrite(
        join('.', '_cropped.png'),
        cv2.cvtColor(cropped, cv2.COLOR_RGB2BGR)
    )
    dil = 10
    ero = 5
    conf_thred = 0.9
    print(f'Processing image {image_path}')
    trimap = generate_trimaps.get_trimap(cropped, label, dil, ero, conf_thred)
    cv2.imwrite(
        join(dirname, f'{image_name}_trimap.png'),
        cv2.cvtColor(trimap, cv2.COLOR_RGB2BGR)
    )
    img = read_image('_cropped.png')
    trimap = read_trimap(join(dirname, f'{image_name}_trimap.png'))

    fg, bg, alpha = matting.perform_matting(img, trimap)
    cv2.imwrite(
        join(dirname, f'{image_name}_fg.png'),
        fg[:, :, ::-1] * 255,
    )
    cv2.imwrite(
        join(dirname, f'{image_name}_bg.png'),
        bg[:, :, ::-1] * 255,
    )
    cv2.imwrite(
        join(dirname, f'{image_name}_alpha.png'), alpha * 255,
    )
    example_swap_bg = swap_bg(fg[:, :, ::-1] * 255, alpha)
    cv2.imwrite(
        join(dirname, f'{image_name}_swapped_bg.png'), example_swap_bg,
    )


def main(label):
    dirname = join('test_images', f'{label}')
    save_dir = join('results', f'{label}')
    files = [join(dirname, f) for f in listdir(dirname) if isfile(join(dirname, f))]
    for file in files:
        process_image(save_dir, file, label)


if __name__ == '__main__':
    labels = ['sofa', 'cat', 'car']
    for lbl in labels:
        main(lbl)
