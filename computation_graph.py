
import tensorflow as TF
import utiltf as util
import modeltf as model

class Graph(object):
    def __init__(self, args):
        super(Graph, self).__init__()

        self.buf_ph = TF.placeholder(TF.uint8, shape=(None, None, None, 3))
        self.buf_plot_op = TF.summary.image('waveform', self.buf_ph, max_outputs=args.batchsize)

    def _set_conditional(self, args, e_g, e_d, cseq, clen):
        if args.conditional:
            self.char_seq = TF.placeholder(TF.int32, shape=(None, None))
            self.char_seq_len = TF.placeholder(TF.int32, shape=(None,))
            self.char_seq_wrong = TF.placeholder(TF.int32, shape=(None, None))
            self.char_seq_wrong_len = TF.placeholder(TF.int32, shape=(None,))

            self.c_g = e_g.embed(self.char_seq, self.char_seq_len)
            self.c_d = e_d.embed(self.char_seq, self.char_seq_len)
            self.c_wrong_d = e_d.embed(self.char_seq_wrong, self.char_seq_wrong_len)

            self.c_g_fixed = e_g.embed(cseq, clen)

            self.cseq = cseq
            self.clen = clen
        else:
            self.c_g_fixed = self.c_g = self.c_d = self.c_wrong_d = self.cseq = self.clen = None

    def _create_summaries(self):
        self.d_summaries = [
                util.summarize_var(self.comp, 'comp', mean=True),
                util.summarize_var(self.d_real, 'd_real', mean=True),
                util.summarize_var(self.d_fake, 'd_fake', mean=True),
                util.summarize_var(self.pen, 'pen', mean=True, std=True),
                TF.summary.histogram('x_real', self.x_real),
                TF.summary.histogram('x_fake', self.x_fake),
                ]
        self.g_summaries = [
                util.summarize_var(self.d_fake, 'd_fake_g', mean=True),
                TF.summary.histogram('x_fake_g', self.x_fake),
                ]
        self.d_valid_summaries = self.d_summaries + [
                util.summarize_var(self.comp_verify, 'comp_verify', mean=True),
                util.summarize_var(self.d_verify_1, 'd_verify_1', mean=True),
                util.summarize_var(self.d_verify_2, 'd_verify_2', mean=True),
                util.summarize_var(self.d_verify_2 - self.d_verify_1, 'd_verify_diff', mean=True),
                util.summarize_var(self.pen_verify, 'pen_verify', mean=True, std=True),
                ]

    def _create_trainer(self, args, g, d, loss_g, loss_d):
        self.opt_g = TF.train.AdamOptimizer()
        self.opt_d = TF.train.AdamOptimizer()
        with TF.control_dependencies(TF.get_collection(TF.GraphKeys.UPDATE_OPS)):
            self.grad_g = self.opt_g.compute_gradients(loss_g, var_list=g.get_trainable_weights())
            self.grad_d = self.opt_d.compute_gradients(loss_d, var_list=d.get_trainable_weights())
        if args.ggradclip:
            self.pre_clipped_grad_g = [TF.norm(g_) for g_, v in self.grad_g if g_ is not None]
            self.grad_g = [(TF.clip_by_norm(_g, args.ggradclip), _v) for _g, _v in self.grad_g if _g is not None]
            self.g_summaries.append(
                util.summarize_var(self.pre_clipped_grad_g, 'pre_clipped_grad_g', min_=True, max_=True, mean=True, std=True)
                )
        if args.dgradclip:
            self.pre_clipped_grad_d = [TF.norm(g_) for g_, v in self.grad_d if g_ is not None]
            self.grad_d = [(TF.clip_by_norm(_g, args.dgradclip), _v) for _g, _v in self.grad_d if _g is not None]
            self.d_summaries.append(
                util.summarize_var(self.pre_clipped_grad_d, 'pre_clipped_grad_d', min_=True, max_=True, mean=True, std=True)
                )
        self.train_g = 0 if args.just_run == 'dis' else self.opt_g.apply_gradients(self.grad_g)
        self.train_d = 0 if args.just_run == 'gen' else self.opt_d.apply_gradients(self.grad_d)

    def _finalize(self, args, maxlen):
        self.d_summaries = TF.summary.merge(self.d_summaries)
        self.d_valid_summaries = TF.summary.merge(self.d_valid_summaries)
        self.g_summaries = TF.summary.merge(self.g_summaries)
        self.audio_gen = TF.summary.audio('sample', self.x, maxlen, max_outputs=args.batchsize)


class GAN(Graph):
    def __init__(self, args, maxlen, d, g, z, e_g=None, e_d=None, cseq=None, clen=None):
        Graph.__init__(self, args)

        self.x_real = TF.placeholder(TF.float32, shape=(None, maxlen))
        self.x_real2 = TF.placeholder(TF.float32, shape=(None, maxlen))
        self.lambda_ = TF.placeholder(TF.float32, shape=())

        self._set_conditional(args, e_g, e_d, cseq, clen)

        self.x_fake = g.generate(batch_size=args.batchsize, length=maxlen, c=self.c_g)
        self.comp, self.d_real, self.d_fake, self.pen, _, _ = d.compare(
                self.x_real, self.x_fake, c=self.c_d, c_wrong=self.c_wrong_d)
        self.comp_verify, self.d_verify_1, self.d_verify_2, self.pen_verify, _, _ = d.compare(
                self.x_real, self.x_real2, c=self.c_d, c_wrong=self.c_wrong_d)

        self.loss_d = self.comp + self.lambda_ * TF.reduce_mean(self.pen)
        metric_g = args.metric + '_g'
        if hasattr(util, metric_g):
            self.loss_g = TF.reduce_mean(getattr(util, metric_g)(self.d_fake))
        else:
            raise ValueError('not an eligible loss function')

        self.z = z
        self.x = g.generate(z=self.z, c=self.c_g_fixed)

        self._create_summaries()

        self._create_trainer(args, g, d, self.loss_g, self.loss_d)

        self._finalize(args, maxlen)


class DynamicGAN(GAN):
    def __init__(self, args, maxlen, d, g, z, e_g=None, e_d=None, cseq=None, clen=None):
        Graph.__init__(self, args)

        self.x_real = TF.placeholder(TF.float32, shape=(None, None))
        self.x_real2 = TF.placeholder(TF.float32, shape=(None, None))
        self.length_real = TF.placeholder(TF.int32, shape=(None,))
        self.length_real2 = TF.placeholder(TF.int32, shape=(None,))
        self.lambda_ = TF.placeholder(TF.float32, shape=())

        self._set_conditional(args, e_g, e_d, cseq, clen)

        self.x_fake, self.logprob_fake, self.action_fake, self.length_fake = g.generate(
                batch_size=args.batchsize,
                length=maxlen,
                c=self.c_g,
                )
        self.comp, self.d_real, self.d_fake, self.pen, _, _ = d.compare(
                self.x_real,
                self.x_fake,
                self.length_real,
                self.length_fake,
                c=self.c_d,
                c_wrong=self.c_wrong_d
                )
        self.comp_verify, self.d_verify_1, self.d_verify_2, self.pen_verify, _, _ = d.compare(
                self.x_real,
                self.x_real2,
                self.length_real,
                self.length_real2,
                c=self.c_d,
                c_wrong=self.c_wrong_d
                )

        self.loss_d = self.comp + self.lambda_ * TF.reduce_mean(self.pen)
        metric_g = args.metric + '_g'
        if hasattr(util, metric_g):
            self.loss_g = TF.reduce_mean(getattr(util, metric_g)(self.d_fake))
        else:
            raise ValueError('not an eligible loss function')
        self.ema = TF.train.ExponentialMovingAverage(decay=0.999)
        self.reward = -self.loss_g
        self.baseline_update_op = self.ema.apply([self.reward])
        self.baseline = self.ema.average(self.reward)
        self.alpha = TF.placeholder(TF.float32, shape=())
        self.cost_action = TF.losses.sparse_softmax_cross_entropy(
                self.action_fake,
                self.logprob_fake,
                )
        self._loss_g = self.loss_g + self.alpha * (self.reward - self.baseline) * self.cost_action
        TF.add_to_collection(TF.GraphKeys.UPDATE_OPS, self.baseline_update_op)

        self.z = z
        self.x, self.logprob, self.action, self.length = g.generate(z=self.z, c=self.c_g_fixed)

        self._create_summaries()

        self._create_trainer(args, g, d, self._loss_g, self.loss_d)

        self._finalize(args, maxlen)
