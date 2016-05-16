import numpy as np
from chainer import Variable, Chain
from chainer import functions as F

class LSTMVRAE(Chain):
    """
    Class: LSTMVRAE
    ===============
    Implements Variational Recurrent Autoencoders, described here: http://arxiv.org/pdf/1412.6581.pdf
    This specific architecture uses a single-layer LSTM for both the encoder and the decoder.
    """

    def __init__(self, n_input, n_hidden, n_latent, loss_func):
        """
        :param n_input: number of input dimensions
        :param n_hidden: number of LSTM cells for both generator and decoder
        :param n_latent: number of dimensions for latent code (z)
        :param loss_func: loss function to compute reconstruction error (e.g. F.mean_squared_error)
        """
        self.__dict__.update(locals())
        super(LSTMVRAE, self).__init__(

            # Encoder (recognition):
            recog_x_h=F.Linear(n_input, n_hidden*4),
            recog_h_h=F.Linear(n_hidden, n_hidden*4),
            recog_mean=F.Linear(n_hidden, n_latent),
            recog_log_sigma=F.Linear(n_hidden, n_latent),

            # Decoder (generation)
            gen_z_h=F.Linear(n_latent, n_hidden*4),
            gen_x_h=F.Linear(n_input, n_hidden*4),
            gen_h_h=F.Linear(n_hidden, n_hidden*4),
            output=F.Linear(n_hidden, n_input)
        )

    def make_initial_state(self):
        """Returns an initial state of the RNN - all zeros"""
        return {
            'h_rec':Variable(np.zeros((1, self.n_hidden), dtype=np.float32)),
            'c_rec':Variable(np.zeros((1, self.n_hidden), dtype=np.float32)),
            'h_gen':Variable(np.zeros((1, self.n_hidden), dtype=np.float32)),
            'c_gen':Variable(np.zeros((1, self.n_hidden), dtype=np.float32))
        }

    def forward(self, x_data, state):
        """
        Does encode/decode on x_data.
        :param x_data: input data (a single timestep) as a numpy.ndarray
        :param state: previous state of RNN
        :param nonlinear_q: nonlinearity used in q(z|x) (encoder)
        :param nonlinear_p: nonlinearity used in p(x|z) (decoder)
        :param output_f: #TODO#
        :return: output, recognition loss, KL Divergence, state
        """
        #=====[ Step 1: Compute q(z|x) - encoding step, get z ]=====
        # Forward encoding
        for i in range(x_data.shape[0]):
            x = Variable(x_data[i].reshape((1, x_data.shape[1])))
            h_in = self.recog_x_h(x) + self.recog_h_h(state['h_rec'])
            c_t, h_t = F.lstm(state['c_rec'], h_in)
            state.update({'c_rec':c_t, 'h_rec':h_t})
        # Compute q_mean and q_log_sigma
        q_mean = self.recog_mean( state['h_rec'] )
        q_log_sigma = 0.5 * self.recog_log_sigma( state['h_rec'] )
        # Compute KL divergence based on q_mean and q_log_sigma
        KLD = -0.0005 * F.sum(1 + q_log_sigma - q_mean**2 - F.exp(q_log_sigma))
        # Compute as q_mean + noise*exp(q_log_sigma)
        eps = Variable(np.random.normal(0, 1, q_log_sigma.data.shape ).astype(np.float32))
        z   = q_mean + F.exp(q_log_sigma) * eps

        #=====[ Step 2: Compute p(x|z) - decoding step ]=====
        # Initial step
        output = []
        h_in = self.gen_z_h(z)
        c_t, h_t = F.lstm(state['c_gen'], h_in)
        state.update({'c_gen':c_t, 'h_gen':h_t})
        rec_loss = Variable(np.zeros((), dtype=np.float32))
        for i in range(x_data.shape[0]):
            # Get output and loss
            x_t = self.output(h_t)
            output.append(x_t.data)
            rec_loss += self.loss_func(x_t, Variable(x_data[i].reshape((1, x_data.shape[1]))))
            # Get next hidden state
            h_in = self.gen_x_h(x_t) + self.gen_h_h(state['h_gen'])
            c_t, h_t = F.lstm(state['c_gen'], h_in)
            state.update({'c_gen':c_t, 'h_gen':h_t})

        #=====[ Step 3: Compute KL-Divergence based on all terms ]=====
        return output, rec_loss, KLD, state
