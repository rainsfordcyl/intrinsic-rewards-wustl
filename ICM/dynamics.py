import numpy as np
import tensorflow as tf

from auxiliary_tasks import JustPixels
from utils import small_convnet, flatten_two_dims, unflatten_first_dim, getsess, unet


class Dynamics(object):
    def __init__(self, auxiliary_task, predict_from_pixels, feat_dim=None, scope='dynamics'):
        self.scope = scope
        self.auxiliary_task = auxiliary_task
        self.hidsize = self.auxiliary_task.hidsize
        self.feat_dim = feat_dim
        self.obs = self.auxiliary_task.obs
        self.last_ob = self.auxiliary_task.last_ob
        self.ac = self.auxiliary_task.ac
        self.ac_space = self.auxiliary_task.ac_space
        self.ob_mean = self.auxiliary_task.ob_mean
        self.ob_std = self.auxiliary_task.ob_std
        self.intrinsic_reward = 0

        # if predicting from pixels, extract features from the current observation
        # else, use the features from the auxiliary_task
        if predict_from_pixels:
            self.features = self.get_features(self.obs, reuse=False)
        else:
            self.features = tf.stop_gradient(self.auxiliary_task.features)

        self.out_features = self.auxiliary_task.next_features

        with tf.variable_scope(self.scope + "_loss"):
            self.loss = self.get_loss()

    def get_features(self, x, reuse):
        nl = tf.nn.leaky_relu
        x_has_timesteps = (x.get_shape().ndims == 5)
        if x_has_timesteps:
            sh = tf.shape(x)
            x = flatten_two_dims(x)
        with tf.variable_scope(self.scope + "_features", reuse=reuse):
            x = (tf.to_float(x) - self.ob_mean) / self.ob_std
            x = small_convnet(x, nl=nl, feat_dim=self.feat_dim, last_nl=nl, layernormalize=False)
        if x_has_timesteps:
            x = unflatten_first_dim(x, sh)
        return x

    # Loss function for the Forward Model
    # calculate L_F
    def get_loss(self):
        # Converts the actions into a one-hot encoded format
        ac = tf.one_hot(self.ac, self.ac_space.n, axis=2) # [batch_size, timesteps, action_dim]
        sh = tf.shape(ac)
        ac = flatten_two_dims(ac) # [batch_size * timesteps, action_dim]

        def add_ac(x):
            return tf.concat([x, ac], axis=-1)

        with tf.variable_scope(self.scope):
            x = flatten_two_dims(self.features) # [batch_size * timesteps, feature_dim]
            
            # Adding Actions to Features and Passing Through a Dense Layer
                # x: phi(s_t)
                # add_ac(x): concatenate the action to the features
                # purpose: to predict the next state features
            x = tf.layers.dense(add_ac(x), self.hidsize, activation=tf.nn.leaky_relu) # [batch_size * timesteps, self.hidsize]
            def residual(x):
                res = tf.layers.dense(add_ac(x), self.hidsize, activation=tf.nn.leaky_relu)
                res = tf.layers.dense(add_ac(res), self.hidsize, activation=None)
                return x + res

            # Applies the residual function 4 times, creating a deep residual network
            for _ in range(4):
                x = residual(x) # [batch_size * timesteps, self.hidsize]

            n_out_features = self.out_features.get_shape()[-1].value
            # final dense later to predict the next state features
            x = tf.layers.dense(add_ac(x), n_out_features, activation=None) # [batch_size * timesteps, n_out_features]
            x = unflatten_first_dim(x, sh) # [batch_size, timesteps, n_out_features]
        
        # calcualte the difference between predicted feature (x) and actual feature (out_features)
        self.intrinsic_reward = tf.reduce_mean((x - tf.stop_gradient(self.out_features)) ** 2, -1)
        return self.intrinsic_reward
    
    # computes the loss for the dynamics model in chunks to handle large batches
    def calculate_loss(self, ob, last_ob, acs):
        n_chunks = 8
        n = ob.shape[0]
        chunk_size = n // n_chunks
        assert n % n_chunks == 0
        sli = lambda i: slice(i * chunk_size, (i + 1) * chunk_size)
        return np.concatenate([getsess().run(self.loss,
                                             {self.obs: ob[sli(i)], self.last_ob: last_ob[sli(i)],
                                              self.ac: acs[sli(i)]}) for i in range(n_chunks)], 0)


# predict feauters that are normalized pixels
# for JustPixels in auxiliary_tasks.py
class UNet(Dynamics):
    def __init__(self, auxiliary_task, predict_from_pixels, feat_dim=None, scope='pixel_dynamics'):
        assert isinstance(auxiliary_task, JustPixels)
        assert not predict_from_pixels, "predict from pixels must be False, it's set up to predict from features that are normalized pixels."
        super(UNet, self).__init__(auxiliary_task=auxiliary_task,
                                   predict_from_pixels=predict_from_pixels,
                                   feat_dim=feat_dim,
                                   scope=scope)

    def get_features(self, x, reuse):
        raise NotImplementedError

    def get_loss(self):
        nl = tf.nn.leaky_relu
        ac = tf.one_hot(self.ac, self.ac_space.n, axis=2)
        sh = tf.shape(ac)
        ac = flatten_two_dims(ac)
        ac_four_dim = tf.expand_dims(tf.expand_dims(ac, 1), 1)

        def add_ac(x):
            if x.get_shape().ndims == 2:
                return tf.concat([x, ac], axis=-1)
            elif x.get_shape().ndims == 4:
                sh = tf.shape(x)
                return tf.concat(
                    [x, ac_four_dim + tf.zeros([sh[0], sh[1], sh[2], ac_four_dim.get_shape()[3].value], tf.float32)],
                    axis=-1)

        with tf.variable_scope(self.scope):
            x = flatten_two_dims(self.features)
            x = unet(x, nl=nl, feat_dim=self.feat_dim, cond=add_ac)
            x = unflatten_first_dim(x, sh)
        self.prediction_pixels = x * self.ob_std + self.ob_mean
        return tf.reduce_mean((x - tf.stop_gradient(self.out_features)) ** 2, [2, 3, 4])


    # def get_loss(self):
    #     nl = tf.nn.leaky_relu
    #     ac = tf.one_hot(self.ac, self.ac_space.n, axis=2)
    #     sh = tf.shape(ac)
    #     ac = flatten_two_dims(ac)
    #     ac_four_dim = tf.expand_dims(tf.expand_dims(ac, 1), 1)

    #     def add_ac(x):
    #         if x.get_shape().ndims == 2:
    #             return tf.concat([x, ac], axis=-1)
    #         elif x.get_shape().ndims == 4:
    #             sh = tf.shape(x)
    #             return tf.concat(
    #                 [x, ac_four_dim + tf.zeros([sh[0], sh[1], sh[2], ac_four_dim.get_shape()[3].value], tf.float32)],
    #                 axis=-1)

    #     with tf.variable_scope(self.scope):
    #         x = flatten_two_dims(self.features)
    #         x = unet(x, nl=nl, feat_dim=self.feat_dim, cond=add_ac)
    #         x = unflatten_first_dim(x, sh)
    #     self.prediction_pixels = x * self.ob_std + self.ob_mean
    #     return tf.reduce_mean((x - tf.stop_gradient(self.out_features)) ** 2, [2, 3, 4])

