from modules import ph, mlp, batch_norm, linear, prelu
import tensorflow as tf
import numpy as np
import pdb
from matplotlib import pyplot as plt
rng = np.random
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)


D_DATA = 28**2
D_CODE = 100
# K_COMPONENTS = 1

train_flag = ph((None), dtype=tf.bool)
data = ph((None, D_DATA))

# encoder = mlp([D_DATA, 100, D_CODE*K_COMPONENTS*3], act_fn=prelu, bn=train_flag)
# w_code, mu_code, lv_code = tf.unstack(tf.reshape(code, [-1, 3, D_CODE, K_COMPONENTS]), axis=1)
encoder = mlp([D_DATA, 100, D_CODE*3], act_fn=prelu, bn=train_flag)
code = encoder(data)  # bs x dc*kc*3
w_code, mu_code, lv_code = tf.unstack(tf.reshape(code, [-1, 3, D_CODE]), axis=1)

# encoder = mlp([D_DATA, 50, 50, D_CODE], act_fn=tf.nn.relu, bn=None)
# mu_code = encoder(data)  # bs x dc*kc*3

# decoder = mlp([D_CODE*K_COMPONENTS*3, 100, D_DATA*2], act_fn=prelu, bn=train_flag)
# recon = decoder(mu_code)
# recon_logit_mu, recon_logit_lv = tf.unstack(tf.reshape(recon, [-1, 2, D_DATA]), axis=1)

decoder = mlp([D_CODE, 50, 50, D_DATA], act_fn=tf.nn.relu, bn=None)
recon_logit_mu = decoder(mu_code)

# mu_guide = tf.Variable(tf.random_normal([D_CODE, K_COMPONENTS]))
# lv_guide = tf.Variable(tf.random_normal([D_CODE, K_COMPONENTS]))

# from http://dx.doi.org.sci-hub.cc/10.1016/j.imavis.2010.12.002
def sym_kl(mu1, var1, mu2, var2):
    dist = 0.5 * ((var1 / var2) + (var2 / var1) +\
                  (tf.square(mu1 - mu2) * ((1./var1) + (1./var2))) - 2.)
    return dist


def gmm_emd(data_w, data_mu, data_var, guide_mu, guide_var):
    """
        batch of bs gaussian mixtures, each with d gms with k components
        data_w: bs x d x k
        data_mu: bs x d x k
        guide_mu: d x k
        guide_var: d x k
    """

    duv = sym_kl(data_mu, data_var, guide_mu, guide_var)  # bs x d x k
    fduv = data_w * duv
    fduv = fduv / tf.reduce_sum(data_w, [2], keep_dims=True)  # normalize
    gmm_emd = tf.reduce_sum(fduv, [1, 2])
    return gmm_emd


# kl bt our inferred mean and var and the unit gaussian
code_losses = 0.5 * tf.reduce_sum(tf.square(mu_code) +\
                    tf.exp(lv_code) -\
                    tf.log(tf.exp(lv_code)) - 1., [1])

# code_losses = log_gaussian_loss(tf.squeeze(mu_code, 2), tf.squeeze(lv_code, 2))
# code_losses = gmm_emd(tf.exp(w_code), mu_code, tf.exp(lv_code), mu_guide, tf.exp(lv_guide))

code_loss = tf.reduce_sum(code_losses)
recon_losses = tf.nn.sigmoid_cross_entropy_with_logits(logits=recon_logit_mu, labels=data)
recon_loss = tf.reduce_sum(recon_losses)

train_losses = code_loss + recon_loss
train_loss = tf.reduce_sum(train_losses)

trainer = tf.train.AdamOptimizer(1e-3).minimize(train_loss)

n_train = mnist.train.num_examples
BS = 100
N_UPDATE = int(1e5)
fig, ax = plt.subplots(1, 2)
cmap = plt.get_cmap('gray')
with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    for i_step in xrange(N_UPDATE):
        if i_step % 1000 == 0:
            i_start = 0
            rl0, cl0 = 0., 0.
            while i_start < n_train:
                i_end = np.min([i_start + 10000, n_train])
                x = mnist.train.images[i_start:i_end, :]
                fd = {data: x, train_flag: False}
                # rl00, cl00 = sess.run([recon_loss, code_loss], feed_dict=fd)
                rl00 = sess.run(recon_loss, feed_dict=fd)
                recon0 = tf.nn.sigmoid(recon_logit_mu).eval(feed_dict=fd)
                i_recon = rng.randint(recon0.shape[0])

                # PLOT
                plt.ion()
                ax[0].clear(); ax[1].clear()
                ax[1].matshow(recon0[i_recon].reshape(28, 28), vmin=0, vmax=1, aspect='auto', cmap=cmap)
                ax[0].matshow(x[i_recon].reshape(28, 28), vmin=0, vmax=1, aspect='auto', cmap=cmap)
                plt.draw()
                plt.show()
                plt.pause(0.01)
                plt.ioff()

                # if np.isnan(cl00):
                #     pdb.set_trace()
                rl0 += rl00
                # cl0 += cl00
                i_start = i_end

            print 'update %d recon loss: %03f' % (i_step, rl0 / n_train)
            # print 'update %d code loss: %03f' % (i_step, cl0 / n_train)

        x, _ = mnist.train.next_batch(BS)
        sess.run([trainer], feed_dict = {data: x, train_flag: True})
