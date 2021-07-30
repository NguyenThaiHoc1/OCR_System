"""
Base reader which contain information to make all data for model
Author: NTH
Date: 29-07-2021
"""
import sys

import numpy as np
from matplotlib import pyplot as plt


class BaseReader(object):
    def __init__(self):
        self.gt_path = ''
        self.image_path = ''
        self.classes = []
        self.image_names = []
        self.data_annotation = []

        # parameter
        self.list_statlabels = None
        self.num_image_without_annotation = None
        self.num_objects = None

        self.class_lower = None
        self.colors = None
        self.num_classes = None
        self.num_samples = None
        self.num_images = None

    def statistics_data(self):
        """
        This
        :return:
        """
        # check annotations that match for each image
        self.num_classes = len(self.classes)
        list_statlabels = np.zeros(self.num_classes)
        num_image_without_annotation = 0
        for index in range(len(self.data_annotation)):
            if len(self.data_annotation[index]) == 0:
                num_image_without_annotation += 1
            else:
                unique, counts = np.unique(self.data_annotation[index][:, -1].astype(np.int16), return_counts=True)
                list_statlabels[unique] += counts

        self.list_statlabels = list_statlabels
        self.num_image_without_annotation = num_image_without_annotation
        self.num_objects = sum(self.list_statlabels)
        self.colors = plt.cm.hsv(np.linspace(0, 1, len(self.classes) + 1)).tolist()  # random color for each class
        self.num_samples = len(self.image_names)
        self.num_images = len(self.data_annotation)

        if self.num_samples == 0:
            print('WARNING: empty dataset', sys.warnoptions)

    def __str__(self):
        """
        That function show all information of dataset
        :return: None
        """
        if not hasattr(self, 'list_statlabels'):
            self.statistics_data()

        string_text = ''
        for index in range(self.num_classes):
            string_text += '%-16s %8i\n' % (self.classes[index], self.list_statlabels[index])
        string_text += '\n'
        string_text += '%-16s %8i\n' % ('images', self.num_images)
        string_text += '%-16s %8i\n' % ('objects', self.num_objects)
        string_text += '%-16s %8.2f\n' % ('per image', self.num_objects / self.num_images)
        string_text += '%-16s %8i\n' % ('no annotation', self.num_image_without_annotation)
        return string_text

    def split(self, rate_split=0.8):
        """
        This function used to split dataset that become the train-set and the val-set
        :param rate_split: The rate which decide to divide a dataset 0.8 - train - 0.2 - val
        :return:
        """
        reader_split_one = BaseReader()
        reader_split_one.gt_path = self.gt_path
        reader_split_one.image_path = self.image_path
        reader_split_one.classes = self.classes

        reader_split_two = BaseReader()
        reader_split_two.gt_path = self.gt_path
        reader_split_two.image_path = self.image_path
        reader_split_two.classes = self.classes

        number_sample = int(round(rate_split * len(self.image_names)))
        reader_split_one.image_names = self.image_names[:number_sample]
        reader_split_two.image_names = self.image_names[number_sample:]

        reader_split_one.data_annotation = self.data_annotation[:number_sample]
        reader_split_two.data_annotation = self.data_annotation[number_sample:]

        reader_split_one.statistics_data()
        reader_split_two.statistics_data()
        return reader_split_one, reader_split_two
