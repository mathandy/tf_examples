from __future__ import absolute_import, division, print_function
import os
from PIL import Image, ImageDraw
from simshow import simshow
import numpy as np
import scipy
from time import time
import webbrowser
import sys
import cv2
from time import time as current_time
from sys import stdout


class Timer:
    """A simple tool for timing code while keeping it pretty."""

    def __init__(self, mes='', pretty_time=True):
        self.mes = mes  # append after `mes` + '...'
        self.pretty_time = pretty_time

    @staticmethod
    def format_time(et):
        if et < 60:
            return '{:.1f} sec'.format(et)
        elif et < 3600:
            return '{:.1f} min'.format(et / 60)
        else:
            return '{:.1f} hrs'.format(et / 3600)

    def __enter__(self):
        stdout.write(self.mes + '...')
        stdout.flush()
        self.t0 = current_time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.t1 = current_time()
        if self.pretty_time:
            print("done (in {})".format(self.format_time(self.t1 - self.t0)))
        else:
            print("done (in {} seconds).".format(self.t1 - self.t0))
        stdout.flush()


def find_nth(big_string, substring, n):
    """Find the nth occurrence of a substring in a string."""
    idx = big_string.find(substring)
    while idx >= 0 and n > 1:
        idx = big_string.find(substring, idx + len(substring))
        n -= 1
    return idx


def parentdir(path_, n=1):
    for i in range(n):
        path_ = os.path.dirname(path_)
    return path_


class WorkingDirectory:
    """ A tool to temporarily change the current working directory, then change 
    it back upon `__exit__` (i.e. for use with `with`).

    Usage:
        with WorkingDirectory(directory_to_work_in):
            # do something
    """

    def __init__(self, working_directory):
        self.old_wd = os.getcwd()
        self.wd = working_directory

    def __enter__(self):
        os.chdir(self.wd)

    def __exit__(self, exc_type, exc_val, exc_tb):
        os.chdir(self.old_wd)


# class BoxedImage:
#     def __init__(self, textboxes, image_filename, gt_filename):
#         self.textboxes = textboxes
#         self.image_filename = image_filename
#         self.gt_filename = gt_filename
#
#     def __repr__(self):
#         s = "from: " + self.gt_filename + '\n'
#         s += "image: " + self.image_filename
#         for tb in self.textboxes:
#             s += '\n' + str(tb)
#         return s


class TextBox:
    def __init__(self, coords, text, image_filename=None, gt_filename=None):
        self.coords = coords
        self.text = text
        self.image_filename = image_filename
        self.gt_filename = gt_filename

    def __repr__(self):
        s = "from: " + self.gt_filename
        s += "image: " + self.image_filename
        return str(self.coords) + ' : ' + self.text


def draw_boxes_on_image(image, textboxes, savename=None, show=True):
    """
    Draw boxes on an image.

    Args:

        images (string): The filename of an image. 
        textboxes (iterable): A list of TextBox objects.
        savename (object): If none, won't save.
        show (bool): Whether or not to display immediately.
    """
    with Image.open(image) as img:
        draw = ImageDraw.Draw(img)

        def z2xy(z):
            return z.real, z.imag

        for tb in textboxes:
            for i in range(4):
                x1, y1 = z2xy(tb.coords[i])
                x2, y2 = z2xy(tb.coords[(i + 1) % 4])
                draw.line((x1, y1, x2, y2), fill=128)
        del draw
        if savename is not None:
            img.create_checkpoint(savename)
        if show:
            simshow(img)


# def color_in_polygon(points, im, color=1):
#     from svgpathtools import Path, Line, path_encloses_pt
#     p = Path(*[Line(points[i], points[(i + 1) % len(points)])
#                for i in range(len(points))])
#
#     x0 = min(z.real for z in points)
#     x1 = max(z.real for z in points)
#     y0 = min(z.imag for z in points)
#     y1 = max(z.imag for z in points)
#
#     for y in range(y0, y1+1):
#         for x in range(x0, x1+1):
#             if path_encloses_pt(p, x+1j*y):
#                 im[y, x] = 1


def color_in_polygon(img, points, color=255):
    """Colors in polygon in image (in place).

    Args:
        points: a list of (x,y) points
        img: a numpy.array
        color: 3-tuple or integer 0-255

    Returns:
        None
    """
    pts = np.array(points, dtype=np.int32).reshape(-1, 1, 2)
    cv2.fillConvexPoly(img, pts, True, color)


def boxes2silhouette(boxes, size, dtype=np.float32):
    """Creates a binary image displaying the boxes.

    Args:
        boxes: A list of 4-tuples of complex numbers.
        size: a 2-tuple or 3-tuple, the output image's size.

    Returns:
        numpy.array: A numpy.array of specified `dtype` -- though all values 
        are either 0 or 1.
    """

    def z2xy(z):
        return [(z.real, z.imag) for z in z]

    sil = np.zeros(size, dtype=np.uint8)
    for box in boxes:
        color_in_polygon(sil, z2xy(box), color=1)

    return np.array(sil).astype(dtype=dtype)


def boxes2pixellabels(boxes, size, dtype=np.float32):
    """Creates a 3-tensor (numpy.array) of shape `[size[0], size[1], 2]`.

    Note:  This is meant for (the 1-hot analog) of binary pixel labels.

    Args:
        boxes: A list of 4-tuples of complex numbers.
        size: a 2-tuple or 3-tuple, the output image's size.

    Returns:
        numpy.array: A numpy.array of specified `dtype` and dimensions
        `[size[0], size[1], 2]` -- though all values are either 0 or 1.
    """

    sil = boxes2silhouette(boxes=boxes, size=size, dtype=dtype)
    return np.stack([sil, 1 - sil], axis=2)
