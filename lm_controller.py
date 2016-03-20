from __future__ import print_function
import binascii
import io
import json
import os
import shutil
import sys
import logging
from multiprocessing import Process
reload(sys)
sys.setdefaultencoding("utf-8")
import time

import numpy
from mimir import ServerLogger
from platoon.channel import Controller

logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s:%(levelname)s:%(name)s:%(message)s')
LOGGER = logging.getLogger(__name__)

class LMController(Controller):
    """
    This multi-process controller implements patience-based early-stopping SGD
    """

    def __init__(self, experiment_id, config):
        """
        Initialize the NMTController
        Parameters
        ----------
        experiment_id : str
            A string that uniquely identifies this run.
        config : dict
            The deserialized JSON configuration file.
        num_workers : int
            The number of workers (GPUs), used to calculate the alpha
            parameter for EASGD.
        """
        self.config = config
        LOGGER.info('Setting up controller ({})'
                            .format(config['multi']['control_port']))
        super(LMController, self).__init__(config['multi']['control_port'])
        self.patience = config['training']['patience']
        self.max_mb = config['training']['finish_after']

        self.valid_freq = config['management']['validFreq']
        self.uidx = 0
        self.bad_counter = 0
        self.min_valid_cost = numpy.inf

        self.valid = False
        self._stop = False

        self.experiment_id = experiment_id
        ServerLogger(filename='logs/{}.log.jsonl.gz'.format(self.experiment_id),
                     threaded=True, port=config['multi']['log_port'])



    def start_batch_server(self):
        self.p = Process(target=self._send_mb,
                args=(self.config['multi']['batch_port'],))
        self.p.daemon = True
        self.p.start()

    def _send_mb(self, batch_port):
        LOGGER.info('Loading training data stream')
        '''
        _, train_stream, _ = load_data(**self.config['data'])
        LOGGER.info('Connecting to socket ({})'.format(batch_port))
        self.init_data(batch_port)
        while True:
            LOGGER.info('Start new epoch sending batches')
            for x, x_mask, y, y_mask in train_stream.get_epoch_iterator():
                LOGGER.debug('Sending batch')
                self.send_mb([x.T, x_mask.T, y.T, y_mask.T])
                LOGGER.debug('Sent batch')
        '''

    def handle_control(self, req, worker_id):
        """
        Handles a control_request received from a worker
        Parameters
        ----------
        req : str or dict
            Control request received from a worker.
            The control request can be one of the following
            1) "next" : request by a worker to be informed of its next action
               to perform. The answers from the server can be 'train' (the
               worker should keep training on its training data), 'valid' (the
               worker should perform monitoring on its validation set and test
               set) or 'stop' (the worker should stop training).
            2) dict of format {"done":N} : used by a worker to inform the
                server that is has performed N more training iterations and
                synced its parameters. The server will respond 'stop' if the
                maximum number of training minibatches has been reached.
            3) dict of format {"valid_err":x, "test_err":x2} : used by a worker
                to inform the server that it has performed a monitoring step
                and obtained the included errors on the monitoring datasets.
                The server will respond "best" if this is the best reported
                validation error so far, otherwise it will respond 'stop' if
                the patience has been exceeded.
        """
        control_response = ""

        if req == 'config':
            control_response = self.config
        #elif req == 'alpha':
        #    tau = self.config['multi']['train_len']
        #    control_response = self.beta / tau / self.num_workers
        elif req == 'experiment_id':
            control_response = self.experiment_id
        elif req == 'next':
            if self.valid:
                self.valid = False
                control_response = 'valid'
            else:
                control_response = 'train'
        elif 'done' in req:
            self.uidx += req['done']

            if numpy.mod(self.uidx, self.valid_freq) == 0:
                self.valid = True
        elif 'valid_err' in req:
            valid_err = req['valid_err']

            if valid_err <= self.min_valid_cost:
                self.bad_counter = 0
                self.min_valid_cost = valid_err
                control_response = 'best'
            else:
                self.bad_counter += 1

        if self.uidx > self.max_mb or self.bad_counter > self.patience:
            control_response = 'stop'
            self.worker_is_done(worker_id)

        return control_response


if __name__ == '__main__':
    # Load the configuration file
    with io.open(sys.argv[1]) as f:
        config = json.load(f)
    #num_workers = int(sys.argv[2])
    # Create unique experiment ID and backup config file
    experiment_id = str(int(time.time()))#binascii.hexlify(os.urandom(3)).decode()
    shutil.copyfile(sys.argv[1], 'logs/{}.config.json'.format(experiment_id))
    # Start controller
    l = LMController(experiment_id, config)
    l.start_batch_server()
    l.serve()
