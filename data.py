from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import dpkt
import numpy
from numpy import random

sample_length = 784

protocol_map = {
    "arp": 0,
    "llmnr": 1,
    "mdns": 2,
    "dns": 3,
	"udp": 4,
}


def protocol_num():
    return len(protocol_map)


def extract_data(train_percentage):
    train = []
    evaluated = []
    protocols = protocol_map.keys()
    num_classes = len(protocols)
    for protocol, value in protocol_map.items():
        f = open("%s.pcap" % protocol, )
        if f == file("/dev/null"):
            print("can not open data file: %s" % protocol)
            exit(-1)
        else:
            pcap_file = dpkt.pcap.Reader(f)
            for ts, buf in pcap_file:
                if random.uniform() > train_percentage:
                    evaluated.append(
                        (value,
                         [ord(ch) / 225. for ch in (buf[max((len(buf)-sample_length),0):len(buf)].rjust(sample_length, '0'))],))
                else:
                    train.append(
                        (value,
                         [ord(ch) / 225. for ch in (buf[max((len(buf)-sample_length),0):len(buf)].rjust(sample_length, '0'))],))
    return train, evaluated


def dense_to_one_hot(labels_dense, num_classes=10):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    index_offset = numpy.arange(num_labels) * num_classes
    labels_one_hot = numpy.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot


class DataSet(object):
    def __init__(self, data_set):
        self._epochs_completed = 0
        self._index_in_epoch = 0
        self._num_examples = len(data_set)
        self._payloads = []
        self._labels = []

        labels = []
        payloads = []
        for item in data_set:
            payloads.append(item[1])
            labels.append(item[0])
        self._payloads = numpy.array(payloads)
        self._labels = dense_to_one_hot(numpy.array(labels), protocol_num())
		# Shuffle the data
        perm = numpy.arange(self._num_examples)
        numpy.random.shuffle(perm)
        self._payloads = self._payloads[perm]
        self._labels = self._labels[perm]

    @property
    def payloads(self):
        return self._payloads

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size):
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            perm = numpy.arange(self._num_examples)
            numpy.random.shuffle(perm)
            self._payloads = self._payloads[perm]
            self._labels = self._labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        return self._payloads[start:end], self._labels[start:end]
        pass


def packets_data():
    data = extract_data(0.7)
	
    train_dataset = DataSet(data[0])
    eval_dataset = DataSet(data[1])
    return train_dataset, eval_dataset


if __name__ == '__main__':
    #for item in packets_data()[0].labels:
    #    print(item)
	train,evalue = extract_data(0.7)
	#for item in train[:100]:
	#	print(len(item[1]))
	train_set, eval_set = packets_data()
	print (len(train_set.payloads))
