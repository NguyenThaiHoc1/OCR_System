import numpy as np


def to_rec(box, image_size):
    """Finds minimum rectangle around some points and scales it to desired
    image size.

    # Arguments
        box: Box or points [x1, y1, x2, y2, ...] with values between 0 and 1.
        image_size: Size of output image.
    # Return
        xy_rec: Corner coordinates of rectangle, array of shape (4, 2).
    """
    image_h, image_w = image_size
    xmin = np.min(box[0::2]) * image_w
    xmax = np.max(box[0::2]) * image_w
    ymin = np.min(box[1::2]) * image_h
    ymax = np.max(box[1::2]) * image_h
    xy_rec = np.array([[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]])
    return xy_rec
