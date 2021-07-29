import os
import numpy as np
from xml.etree import ElementTree
from UtlisData.base_reader import BaseReader


class ReaderVOC(BaseReader):
    """
        Reader for PASCAL VOC (Visual Object Classes) dataset.
    """

    def __init__(self, data_path):
        """
        :param data_path: path where contain VOC dataset
        """
        super(ReaderVOC, self).__init__()
        self.data_path = data_path
        self.image_path = os.path.join(self.data_path, "JPEGImages")
        self.gt_path = os.path.join(self.data_path, 'Annotations')
        self.classes = [
            'Background',
            'Aeroplane', 'Bicycle', 'Bird', 'Boat', 'Bottle',
            'Bus', 'Car', 'Cat', 'Chair', 'Cow', 'Diningtable',
            'Dog', 'Horse', 'Motorbike', 'Person', 'Pottedplant',
            'Sheep', 'Sofa', 'Train', 'Tvmonitor'
        ]
        self.class_lower = [s.lower() for s in self.classes]

        for filename in os.listdir(self.gt_path):
            tree = ElementTree.parse(os.path.join(self.gt_path, filename))
            root = tree.getroot()
            boxes = []
            size_tree = root.find('size')
            img_width = float(size_tree.find('width').text)
            img_height = float(size_tree.find('height').text)
            image_name = root.find('filename').text
            for object_tree in root.findall('object'):
                class_name = object_tree.find('name').text
                class_idx = self.classes_lower.index(class_name)
                for bbox in object_tree.findall('bndbox'):
                    xmin = float(bbox.find('xmin').text) / img_width
                    ymin = float(bbox.find('ymin').text) / img_height
                    xmax = float(bbox.find('xmax').text) / img_width
                    ymax = float(bbox.find('ymax').text) / img_height
                    box = [xmin, ymin, xmax, ymax, class_idx]
                    boxes.append(box)
            boxes = np.asarray(boxes)
            self.image_names.append(image_name)
            self.data_annotation.append(boxes)

        self.statistics_data()


