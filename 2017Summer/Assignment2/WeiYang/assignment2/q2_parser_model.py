import os
import time
import dynet as dy
import numpy as np
import cPickle

from model import Model
from q1_softmax import cross_entropy_loss
from q2_initialization import xavier_weight_init
from utils.general_utils import Progbar
from utils.parser_utils import minibatches, load_and_preprocess_data


class Config(object):
    n_features = 36
    n_classes = 3
    dropout = 0.5
    embed_size = 50
    hidden_size = 200
    batch_size = 2048
    n_epochs = 10
    lr = 0.001


class ParserModel(Model):
    def init_trainer(self):
        self.m = dy.ParameterCollection()
        self.trainer = dy.AdamTrainer(self.m)
        self.trainer.learning_rate = self.config.lr

    def init_parameters(self):
        zeroInit = dy.ConstInitializer(0.0)
        # xavier = xavier_weight_init()

        self._pW = self.m.add_parameters((self.config.n_features * self.config.embed_size, self.config.hidden_size))
        self._pB1 = self.m.add_parameters((1, self.config.hidden_size), init=zeroInit)
        self._pU = self.m.add_parameters((self.config.hidden_size, self.config.n_classes))
        self._pB2 = self.m.add_parameters((1, self.config.n_classes), init=zeroInit)

        self.word_dict = self.m.lookup_parameters_from_numpy(self.pretrained_embeddings)

    def create_feed_dict(self, inputs_batch, labels_batch=None, dropout=1):
        self.input = inputs_batch
        # 2048*36
        self.labels = labels_batch
        self.dropout = dropout

    def add_embedding(self):
        embeddings = dy.concatenate([self.word_dict.batch(x) for x in np.transpose(self.input)])
        embeddings = dy.transpose(embeddings)
        # ((1, 50*36), 2048)
        return embeddings

    def prediction(self, dropout=False):
        x = self.add_embedding()
        W = dy.parameter(self._pW)
        U = dy.parameter(self._pU)
        b1 = dy.parameter(self._pB1)
        b2 = dy.parameter(self._pB2)

        z1 = x * W + b1
        h = dy.rectify(z1)
        h_drop = dy.dropout(h, self.dropout) if dropout else h

        z2 = h_drop * U + b2
        # print "z2: ", z2.dim()

        pred = dy.softmax(dy.reshape(z2, (self.config.n_classes,)))
        return pred

    def compute_loss(self, pred):
        y = dy.inputTensor(np.transpose(self.labels), batched=True)
        losses = cross_entropy_loss(y, pred)
        loss = dy.sum_batches(losses) / self.config.batch_size
        return loss

    def train_on_batch(self, inputs_batch, labels_batch):
        self.create_feed_dict(inputs_batch, labels_batch=labels_batch,
                                     dropout=self.config.dropout)
        pred = self.prediction(dropout=True)
        loss = self.compute_loss(pred)
        return loss

    def predict_on_batch(self, inputs_batch):
        self.create_feed_dict(inputs_batch)
        pred_dy = self.prediction()
        pred = np.transpose(pred_dy.npvalue())
        return pred

    def run_epoch(self, parser, train_examples, dev_set):
        for i, (train_x, train_y) in enumerate(minibatches(train_examples, self.config.batch_size)):
            dy.renew_cg()
            loss = self.train_on_batch(train_x, train_y)
            loss.forward()
            loss.backward()
            self.trainer.update()
        print "Training Loss: ", loss.value()
        print "Evaluating on dev set",
        dev_UAS, _ = parser.parse(dev_set)
        print "- dev UAS: {:.2f}".format(dev_UAS * 100.0)
        return dev_UAS

    def fit(self, saver, parser, train_examples, dev_set):
        best_dev_UAS = 0
        for epoch in range(self.config.n_epochs):
            print "Epoch {:} out of {:}".format(epoch + 1, self.config.n_epochs)
            dev_UAS = self.run_epoch(parser, train_examples, dev_set)
            if dev_UAS > best_dev_UAS:
                best_dev_UAS = dev_UAS
                if saver:
                    print "New best dev UAS! Saving model in ./data/weights/parser.weights"
                    dy.save('./data/weights/parser.weights')
            print

    def __init__(self, config, pretrained_embeddings):
        self.pretrained_embeddings = pretrained_embeddings
        self.config = config
        self.build()


def main(debug=False):
    print 80 * "="
    print "INITIALIZING"
    print 80 * "="
    config = Config()
    parser, embeddings, train_examples, dev_set, test_set = load_and_preprocess_data(debug)
    if not os.path.exists('./data/weights/'):
        os.makedirs('./data/weights/')

    print "Building model...",
    start = time.time()
    model = ParserModel(config, embeddings)
    parser.model = model
    print "took {:.2f} seconds\n".format(time.time() - start)

    saver = None if debug else True

    print 80 * "="
    print "TRAINING"
    print 80 * "="
    model.fit(saver, parser, train_examples, dev_set)

    if not debug:
        print 80 * "="
        print "TESTING"
        print 80 * "="
        print "Restoring the best model weights found on the dev set"
        saver.restore('./data/weights/parser.weights')
        print "Final evaluation on test set",
        UAS, dependencies = parser.parse(test_set)
        print "- test UAS: {:.2f}".format(UAS * 100.0)
        print "Writing predictions"
        with open('q2_test.predicted.pkl', 'w') as f:
            cPickle.dump(dependencies, f, -1)
        print "Done!"

if __name__ == '__main__':
    main()


