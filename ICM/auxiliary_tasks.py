import tensorflow as tf

from utils import small_convnet, fc, activ, flatten_two_dims, unflatten_first_dim, small_deconvnet


class FeatureExtractor(object):
    def __init__(self, policy, features_shared_with_policy, feat_dim=None, layernormalize=None,
                 scope='feature_extractor'):
        self.scope = scope
        self.features_shared_with_policy = features_shared_with_policy
        self.feat_dim = feat_dim
        self.layernormalize = layernormalize
        self.policy = policy
        self.hidsize = policy.hidsize
        self.ob_space = policy.ob_space
        self.ac_space = policy.ac_space
        self.obs = self.policy.ph_ob # [o_{t-2}, o_{t-1}, o_t]
        self.ob_mean = self.policy.ob_mean
        self.ob_std = self.policy.ob_std
        # get observations and features
        with tf.variable_scope(scope):
            self.last_ob = tf.placeholder(dtype=tf.int32,
                                          shape=(None, 1) + self.ob_space.shape, name='last_ob') # [o_{t+1}]
            self.next_ob = tf.concat([self.obs[:, 1:], self.last_ob], 1) # [o_{t-1}, o_t, o_{t+1}]
            print('self.features_shared_with_policy 1: ', self.features_shared_with_policy)
            if features_shared_with_policy:
                self.features = self.policy.features
                self.last_features = self.policy.get_features(self.last_ob, reuse=True)
            else:
                self.features = self.get_features(self.obs, reuse=False)
                self.last_features = self.get_features(self.last_ob, reuse=True)
            self.next_features = tf.concat([self.features[:, 1:], self.last_features], 1)

            self.ac = self.policy.ph_ac
            self.scope = scope

            self.loss = self.get_loss()

    def get_features(self, x, reuse):
        nl = tf.nn.leaky_relu
        x_has_timesteps = (x.get_shape().ndims == 5) # [batch_size, timesteps, height, width, channels]
        if x_has_timesteps:
            sh = tf.shape(x)
            x = flatten_two_dims(x) # flattens (batch_size and timesteps) into one, treating each time step independently for feature extraction
        with tf.variable_scope(self.scope + "_features", reuse=reuse):
            x = (tf.to_float(x) - self.ob_mean) / self.ob_std
            x = small_convnet(x, nl=nl, feat_dim=self.feat_dim, last_nl=None, layernormalize=self.layernormalize)
        if x_has_timesteps:
            x = unflatten_first_dim(x, sh)
        return x

    def get_loss(self):
        return tf.zeros((), dtype=tf.float32)

class InverseDynamics(FeatureExtractor):
    def __init__(self, policy, features_shared_with_policy, feat_dim=None, layernormalize=None):
        super(InverseDynamics, self).__init__(scope="inverse_dynamics", policy=policy,
                                              features_shared_with_policy=features_shared_with_policy,
                                              feat_dim=feat_dim, layernormalize=layernormalize)
    # WHY: calculate the loss L1 generated from the inverse model
    def get_loss(self):

        with tf.variable_scope(self.scope):
            x = tf.concat([self.features, self.next_features], 2) # SHAPE: [batch_size, timesteps, 2 * feature_dim]
            sh = tf.shape(x)
            x = flatten_two_dims(x) # SHAPE: [batch_size * timesteps, 2 * feature_dim]

            # WHAT:
            # initialize weight W shape of [input_dim, self.policy.hidsize]
            #            bias b shape of [self.policy.hidsize]
            # computes x = x @ W + b
            # introduce non-linearity x = activ(x @ w + b)
            # 
            # WHY: S, S_{t+1} -> phi(S), phi(S_{t+1})

            x = fc(x, units=self.policy.hidsize, activation=activ) # SHAPE: [batch_size * timesteps, self.policy.hidsize]
            # WHY: phi(S), phi(S_{t+1}) -> a_hat_t
            x = fc(x, units=self.ac_space.n, activation=None) # SHAPE: [batch_size * timesteps, self.ac_space.n]
            
            param = unflatten_first_dim(x, sh) # SHAPE: [batch_size, timesteps, self.ac_space.n]
            
            # convert logits (from nn) to probability distribution over predicted action
            # DEF: logits are raw, unnormalized scores for each possible action
            idfpd = self.policy.ac_pdtype.pdfromflat(param)
            # returns the negative log probability of the actual actions under the predicted distribution
            return idfpd.neglogp(self.ac)


class VAE(FeatureExtractor):
    def __init__(self, policy, features_shared_with_policy, feat_dim=None, layernormalize=False, spherical_obs=False):
        assert not layernormalize, "VAE features should already have reasonable size, no need to layer normalize them"
        self.spherical_obs = spherical_obs
        super(VAE, self).__init__(scope="vae", policy=policy,
                                  features_shared_with_policy=features_shared_with_policy,
                                  feat_dim=feat_dim, layernormalize=False)
        self.features = tf.split(self.features, 2, -1)[0]  # use mean only for features exposed to the dynamics
        self.next_features = tf.split(self.next_features, 2, -1)[0]

    def get_features(self, x, reuse):
        nl = tf.nn.leaky_relu
        x_has_timesteps = (x.get_shape().ndims == 5)
        if x_has_timesteps:
            sh = tf.shape(x)
            x = flatten_two_dims(x)
        with tf.variable_scope(self.scope + "_features", reuse=reuse):
            x = (tf.to_float(x) - self.ob_mean) / self.ob_std
            x = small_convnet(x, nl=nl, feat_dim=2 * self.feat_dim, last_nl=None, layernormalize=False)
        if x_has_timesteps:
            x = unflatten_first_dim(x, sh)
        return x

    def get_loss(self):
        with tf.variable_scope(self.scope):
            posterior_mean, posterior_scale = tf.split(self.features, 2, -1)
            posterior_scale = tf.nn.softplus(posterior_scale)
            posterior_distribution = tf.distributions.Normal(loc=posterior_mean, scale=posterior_scale)

            sh = tf.shape(posterior_mean)
            prior = tf.distributions.Normal(loc=tf.zeros(sh), scale=tf.ones(sh))

            posterior_kl = tf.distributions.kl_divergence(posterior_distribution, prior)

            posterior_kl = tf.reduce_sum(posterior_kl, [-1])
            assert posterior_kl.get_shape().ndims == 2

            posterior_sample = posterior_distribution.sample()
            reconstruction_distribution = self.decoder(posterior_sample)
            norm_obs = self.add_noise_and_normalize(self.obs)
            reconstruction_likelihood = reconstruction_distribution.log_prob(norm_obs)
            assert reconstruction_likelihood.get_shape().as_list()[2:] == [84, 84, 4]
            reconstruction_likelihood = tf.reduce_sum(reconstruction_likelihood, [2, 3, 4])

            likelihood_lower_bound = reconstruction_likelihood - posterior_kl
            return - likelihood_lower_bound

    def add_noise_and_normalize(self, x):
        x = tf.to_float(x) + tf.random_uniform(shape=tf.shape(x), minval=0., maxval=1.)
        x = (x - self.ob_mean) / self.ob_std
        return x

    def decoder(self, z):
        nl = tf.nn.leaky_relu
        z_has_timesteps = (z.get_shape().ndims == 3)
        if z_has_timesteps:
            sh = tf.shape(z)
            z = flatten_two_dims(z)
        with tf.variable_scope(self.scope + "decoder"):
            z = small_deconvnet(z, nl=nl, ch=4 if self.spherical_obs else 8, positional_bias=True)
            if z_has_timesteps:
                z = unflatten_first_dim(z, sh)
            if self.spherical_obs:
                scale = tf.get_variable(name="scale", shape=(), dtype=tf.float32,
                                        initializer=tf.ones_initializer())
                scale = tf.maximum(scale, -4.)
                scale = tf.nn.softplus(scale)
                scale = scale * tf.ones_like(z)
            else:
                z, scale = tf.split(z, 2, -1)
                scale = tf.nn.softplus(scale)
            # scale = tf.Print(scale, [scale])
            return tf.distributions.Normal(loc=z, scale=scale)


class JustPixels(FeatureExtractor):
    def __init__(self, policy, features_shared_with_policy, feat_dim=None, layernormalize=None,
                 scope='just_pixels'):
        assert not layernormalize
        assert not features_shared_with_policy
        super(JustPixels, self).__init__(scope=scope, policy=policy,
                                         features_shared_with_policy=False,
                                         feat_dim=None, layernormalize=None)

    def get_features(self, x, reuse):
        with tf.variable_scope(self.scope + "_features", reuse=reuse):
            x = (tf.to_float(x) - self.ob_mean) / self.ob_std
        return x

    def get_loss(self):
        return tf.zeros((), dtype=tf.float32)
