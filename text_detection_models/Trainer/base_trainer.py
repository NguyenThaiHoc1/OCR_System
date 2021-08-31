class BaseTrainer(object):
    def __init__(self):
        # required parameter
        self.epoch = None
        self.train_reader = None
        self.validate_reader = None
        self.batch_size = None
        self.learning_rate = None
        self.model = None
        self.loss = None
        self.metrics = None

        # parameter
        self.train_generator = None
        self.validate_generator = None
        self.optimizer = None
        self.regularizer = None

    def __str__(self):
        if self.model is None:
            raise ValueError("Please update model architecture !")

        if self.train_reader is None or self.validate_reader is None:
            raise ValueError("Please update Reader !")

        f = '%-20s %s\n'
        s = ''
        s += f % ('Epoch', self.epoch)
        s += f % ('Batch Size', self.batch_size)
        s += f % ('Train Reader', self.train_reader)
        s += f % ('Validate Reader', self.validate_reader)
        return s

    def _setup_required_parameter_training(self):
        raise NotImplementedError

    def _setup_loss_function(self):
        raise NotImplementedError

    def _setup_optimizers_training(self):
        raise NotImplementedError

    def _setup_metrics(self):
        raise NotImplementedError

    def _reset_state(self):
        raise NotImplementedError

    def _training_step(self, iteritor_train, trainable):
        raise NotImplementedError

    def _validate_step(self, iteritor_validate, trainable):
        raise NotImplementedError

    def trainer(self):
        raise NotImplementedError
