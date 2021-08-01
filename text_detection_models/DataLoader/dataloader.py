class DataGenerator(object):
    def __init__(self):
        self.__dict__.update(locals())  # avoid to self.properties

        # parameter input
        self.reader = None
        self.batch_size = None
        self.image_size = None

        # parameter init
        self.num_batches = None
        self.num_samples = None

    def setup_number(self):
        self.num_batches = self.reader.num_samples // self.batch_size
        self.num_samples = self.num_batches * self.batch_size

    def __str__(self):
        """
        Show information of dataloader
        :return:
        """
        f = '%-20s %s\n'
        s = ''
        s += f % ('batch_size', self.batch_size)
        s += f % ('image_size', self.image_size)
        s += f % ('num_batches', self.num_batches)
        s += f % ('num_samples', self.reader.num_samples)
        return s

    def _get_process(self, idx, encode=True, debug=False):
        """
        That function used to process data. It is lambda function or function default
        :param idx: the number of sort image that it used to get image from index
        :param encode:
        :param debug: debug image parameter
        :return: None
        """
        raise NotImplementedError

    def get_datagenerator(self, num_parallel_calls=-1, seed=1337):
        """

        :param num_parallel_calls:
        :param seed:
        :return:
        """
        raise NotImplementedError
