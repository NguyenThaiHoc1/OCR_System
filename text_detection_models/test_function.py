import time

import tensorflow as tf
from tqdm import tqdm

from DataLoader.dataloader import DataGenerator
from UtlisData.voc_reader import ReaderVOC


@tf.function
def read_data(iteritor_train):
    data = next(iteritor_train)
    return data


if __name__ == '__main__':
    reader = ReaderVOC(data_path='./dataset_OD/VOC_Dataset/VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007/')
    reader_train, reader_validate = reader.split(rate_split=0.8)

    generator_train = DataGenerator(reader=reader_train, batch_size=32, image_size=(300, 300, 3))
    generator_validate = DataGenerator(reader=reader_validate, batch_size=32, image_size=(300, 300, 3))

    dataset_train = generator_train.get_tf_dataset(num_parallel_calls=1)
    dataset_validate = generator_validate.get_tf_dataset(num_parallel_calls=1)

    iteritor_train = iter(dataset_train)

    for epoch in range(4):
        count = 0
        for index in tqdm(range(generator_train.num_batches), f'Training Epoch {epoch}', leave=True):
            data = read_data(iteritor_train)
            count += 1
        time.sleep(1)
        print(f"Epoch {epoch} -- Count: {count}")
    print("Done !")
