import dynet as dy

class Model(object):
    def create_feed_dict(self, inputs_batch, labels_batch=None):
        raise NotImplementedError("Each Model must re-implement this method.")

    def init_parameters(self):
        raise NotImplementedError("Each Model must re-implement this method.")

    def init_trainer(self):
        raise NotImplementedError("Each Model must re-implement this method.")

    def train_on_batch(self, inputs_batch, labels_batch):
        self.create_feed_dict(inputs_batch, labels_batch=labels_batch)
        pred = self.prediction()
        loss = self.compute_loss(pred)
        return loss

    def predict_on_batch(self, inputs_batch):
        self.create_feed_dict(inputs_batch)
        pred = self.prediction()
        return pred

    def build(self):
        self.init_trainer()
        self.init_parameters()
