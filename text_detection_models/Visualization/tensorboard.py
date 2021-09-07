import tensorflow as tf


class WritterTB(object):
    writter = None

    def __init__(self, path_writter):
        self.path_writter = path_writter

        self._create_writter()

    def _create_writter(self):
        self.writter = tf.summary.create_file_writer(self.path_writter)

    def get_writter(self):
        return self.writter

    def writing_values(self, name, epoch, result):
        try:
            with self.writter.as_default():
                tf.summary.scalar(str(name), result, step=epoch)
        except Exception as exc:
            print(f"This Error appear in WritterTB at func {'writing_values'}")
            print(f"Error: {exc}")
