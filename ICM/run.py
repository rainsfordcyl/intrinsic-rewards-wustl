#!/usr/bin/env python
import numpy as np

try:
    from OpenGL import GLU
except:
    print("no OpenGL.GLU")
import functools
import os
import os.path as osp
from functools import partial

import gym
import ale_py

import time
import random
from datetime import datetime

import tensorflow as tf
from baselines import logger
from baselines.bench import Monitor
from baselines.common.atari_wrappers import NoopResetEnv, FrameStack
from mpi4py import MPI

from auxiliary_tasks import FeatureExtractor, InverseDynamics, VAE, JustPixels
from cnn_policy import CnnPolicy
from cppo_agent import PpoOptimizer
from dynamics import Dynamics, UNet
from utils import random_agent_ob_mean_std
from wrappers import MontezumaInfoWrapper, make_mario_env, make_robo_pong, make_robo_hockey, \
    make_multi_pong, AddRandomStateToInfo, MaxAndSkipEnv, ProcessFrame84, ExtraTimeLimit

print('\n\ngym:', gym.__version__)
print('\n\nale_py:', ale_py.__version__)

def start_experiment(**args):
    make_env = partial(make_env_all_params, add_monitor=True, args=args)

    trainer = Trainer(make_env=make_env,
                      num_timesteps=args['num_timesteps'], hps=args,
                      envs_per_process=args['envs_per_process'])
    log, tf_sess = get_experiment_environment(**args)
    
    if train:
        with log, tf_sess:
            cwd = os.getcwd()
            now = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
            log_dir = os.path.join(cwd, 'logs/{}_{}_{}'.format(args['env'], args['feat_learning'], now))
            logger.configure(log_dir)
            logdir = logger.get_dir()
            # logdir = './logs'
            print("results will be saved to ", logdir)
            trainer.train()
    else:
        trainer.test()


class Trainer(object):
    def __init__(self, make_env, hps, num_timesteps, envs_per_process):
        self.make_env = make_env
        self.hps = hps
        self.envs_per_process = envs_per_process
        self.num_timesteps = num_timesteps
        self._set_env_vars()

        self.policy = CnnPolicy(
            scope='pol',
            ob_space=self.ob_space,
            ac_space=self.ac_space,
            hidsize=512,
            feat_dim=512,
            ob_mean=self.ob_mean,
            ob_std=self.ob_std,
            layernormalize=False,
            nl=tf.nn.leaky_relu)

        self.feature_extractor = {"none": FeatureExtractor,
                                  "idf": InverseDynamics, # use this one to regenerate the paper's result
                                  "vaesph": partial(VAE, spherical_obs=True),
                                  "vaenonsph": partial(VAE, spherical_obs=False),
                                  "pix2pix": JustPixels}[hps['feat_learning']]
        self.feature_extractor = self.feature_extractor(policy=self.policy,
                                                        features_shared_with_policy=False,
                                                        feat_dim=512,
                                                        layernormalize=hps['layernorm'])

        self.dynamics = Dynamics if hps['feat_learning'] != 'pix2pix' else UNet
        self.dynamics = self.dynamics(auxiliary_task=self.feature_extractor,
                                      predict_from_pixels=hps['dyn_from_pixels'],
                                      feat_dim=512)

        self.agent = PpoOptimizer(
            scope='ppo',
            ob_space=self.ob_space,
            ac_space=self.ac_space,
            stochpol=self.policy,
            use_news=hps['use_news'],
            gamma=hps['gamma'],
            lam=hps["lambda"],
            nepochs=hps['nepochs'],
            nminibatches=hps['nminibatches'],
            lr=hps['lr'],
            cliprange=0.1,
            nsteps_per_seg=hps['nsteps_per_seg'],
            nsegs_per_env=hps['nsegs_per_env'],
            ent_coef=hps['ent_coeff'],
            normrew=hps['norm_rew'],
            normadv=hps['norm_adv'],
            ext_coeff=hps['ext_coeff'],
            int_coeff=hps['int_coeff'],
            dynamics=self.dynamics
        )

        self.agent.to_report['aux'] = tf.reduce_mean(self.feature_extractor.loss)
        self.agent.total_loss += self.agent.to_report['aux']
        self.agent.to_report['dyn_loss'] = tf.reduce_mean(self.dynamics.loss)
        self.agent.total_loss += self.agent.to_report['dyn_loss']
        self.agent.to_report['feat_var'] = tf.reduce_mean(tf.nn.moments(self.feature_extractor.features, [0, 1])[1])

    def _set_env_vars(self):
        env = self.make_env(0, add_monitor=False)
        self.ob_space, self.ac_space = env.observation_space, env.action_space
        self.ob_mean, self.ob_std = random_agent_ob_mean_std(env)
        del env
        self.envs = [functools.partial(self.make_env, i) for i in range(self.envs_per_process)]

    # def train(self):
    #     self.agent.start_interaction(self.envs, nlump=self.hps['nlumps'], dynamics=self.dynamics)

    #     checkpoint_dir = "checkpoints"
    #     if not os.path.exists(checkpoint_dir):
    #         os.makedirs(checkpoint_dir)
    #     checkpoint_file = os.path.join(checkpoint_dir, "checkpoint.chk")
    #     saver = tf.train.Saver(name="saver")
    #     session = tf.get_default_session()
    #     saver.save(session, checkpoint_file)

    #     while True:
    #         print('step:', self.agent.rollout.stats['tcount'])
    #         info = self.agent.step()
    #         if info['update']:
    #             logger.logkvs(info['update'])
    #             logger.dumpkvs()
    #             session = tf.get_default_session()
    #             saver.save(session, checkpoint_file)
    #         if self.agent.rollout.stats['tcount'] > self.num_timesteps:
    #             break

    #     self.agent.stop_interaction()

    def train(self):
        self.agent.start_interaction(self.envs, nlump=self.hps['nlumps'], dynamics=self.dynamics)

        checkpoint_dir = "checkpoints"
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        
        checkpoint_file_name = "checkpoint.chk"  # Change this to the desired checkpoint file name
        checkpoint_file = os.path.join(checkpoint_dir, checkpoint_file_name)
        saver = tf.train.Saver(name="saver")
        session = tf.get_default_session()

        # Restore from the specified checkpoint file if it exists
        if osp.exists(checkpoint_file + ".index"):  # Check if the checkpoint file exists
            print("Loading model checkpoint from: ", checkpoint_file)
            saver.restore(session, checkpoint_file)
        else:
            # Restore from the latest checkpoint if the specified checkpoint file does not exist
            latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
            if latest_checkpoint:
                print("Loading model checkpoint from: ", latest_checkpoint)
                saver.restore(session, latest_checkpoint)
            else:
                print("No checkpoint found. Starting fresh training.")

        while True:
            print('step:', self.agent.rollout.stats['tcount'])
            info = self.agent.step()
            if info['update']:
                logger.logkvs(info['update'])
                logger.dumpkvs()
                logger.logkv('int_rew', np.mean(self.agent.rollout.int_rew))
                session = tf.get_default_session()
                saver.save(session, checkpoint_file)
            if self.agent.rollout.stats['tcount'] > self.num_timesteps:
                break

        self.agent.stop_interaction()



    def test(self):
        env = gym.make(self.hps['env'])
        print('actions: ', env.unwrapped.get_action_meanings())
        assert 'NoFrameskip' in env.spec.id
        env = NoopResetEnv(env, noop_max=self.hps['noop_max'])
        env = MaxAndSkipEnv(env, skip=4)
        env = ProcessFrame84(env, crop=False)
        env = FrameStack(env, 4)
        # print('ob shape: ', env.observation_space.shape)

        episodes = eval_episode
        for episode in range(0, episodes):
            state = env.reset()
            done = False
            score = 0

            while not done:
                env.render()
                # action = self.policy.get_action(tf.expand_dims(tf.convert_to_tensor(state), axis=0))
                action = self.policy.get_action(np.expand_dims(np.array(state), axis=0))
                state_, reward, done, info = env.step(action)
                score += reward
                state = state_
            print('Episode: {} Score: {}'.format(episode, score))

        env.close()


def make_env_all_params(rank, add_monitor, args):
    if args["env_kind"] == 'atari':
        env = gym.make(args['env'])
        assert 'NoFrameskip' in env.spec.id
        env = NoopResetEnv(env, noop_max=args['noop_max'])
        env = MaxAndSkipEnv(env, skip=4)
        env = ProcessFrame84(env, crop=False)
        env = FrameStack(env, 4)
        env = ExtraTimeLimit(env, args['max_episode_steps'])
        if 'Montezuma' in args['env']:
            env = MontezumaInfoWrapper(env)
        env = AddRandomStateToInfo(env)
    elif args["env_kind"] == 'mario':
        env = make_mario_env()
    elif args["env_kind"] == "retro_multi":
        env = make_multi_pong()
    elif args["env_kind"] == 'robopong':
        if args["env"] == "pong":
            env = make_robo_pong()
        elif args["env"] == "hockey":
            env = make_robo_hockey()

    if add_monitor:
        env = Monitor(env, osp.join(logger.get_dir(), '%.2i' % rank))

    return env


def get_experiment_environment(**args):
    from utils import setup_mpi_gpus, setup_tensorflow_session
    from baselines.common import set_global_seeds
    from gym.utils.seeding import hash_seed
    process_seed = args["seed"] + 1000 * MPI.COMM_WORLD.Get_rank()
    process_seed = hash_seed(process_seed, max_bytes=4)
    set_global_seeds(process_seed)
    setup_mpi_gpus()

    logger_context = logger.scoped_configure(dir=None,
                                             format_strs=['stdout', 'log',
                                                          'csv'] if MPI.COMM_WORLD.Get_rank() == 0 else ['log'])

    tf_context = setup_tensorflow_session()
    return logger_context, tf_context


def add_environments_params(parser, id):
    # parser.add_argument('--env', help='environment ID', default='BreakoutNoFrameskip-v4',
    #                     type=str)
    parser.add_argument('--env', help='environment ID', default=id,
                        type=str)
                        # default = 4500
    parser.add_argument('--max-episode-steps', help='maximum number of timesteps for episode', default=4500, type=int)
    parser.add_argument('--env_kind', type=str, default="atari")
    parser.add_argument('--noop_max', type=int, default=30)


# For actual training purposes
def add_optimization_params(parser):
    parser.add_argument('--lambda', type=float, default=0.95)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--nminibatches', type=int, default=8)
    parser.add_argument('--norm_adv', type=int, default=1)
    parser.add_argument('--norm_rew', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--ent_coeff', type=float, default=0.001)
    parser.add_argument('--nepochs', type=int, default=3)
    # 2.5e7: 100 frames in SpaceInvader
    #      200 frames in Breakout
    parser.add_argument('--num_timesteps', type=int, default=int(1e8))
    # parser.add_argument('--num_timesteps', type=int, default=int(1e3))


def add_rollout_params(parser):
    parser.add_argument('--nsteps_per_seg', type=int, default=128)
    parser.add_argument('--nsegs_per_env', type=int, default=1)
    parser.add_argument('--envs_per_process', type=int, default=128)
    parser.add_argument('--nlumps', type=int, default=1)

# For quick debugging and testing purposes:
# def add_optimization_params(parser):
#     parser.add_argument('--lambda', type=float, default=0.95)
#     parser.add_argument('--gamma', type=float, default=0.99)
#     parser.add_argument('--nminibatches', type=int, default=4)  # Adjusted for quicker computation
#     parser.add_argument('--norm_adv', type=int, default=1)
#     parser.add_argument('--norm_rew', type=int, default=1)
#     parser.add_argument('--lr', type=float, default=1e-4)
#     parser.add_argument('--ent_coeff', type=float, default=0.001)
#     parser.add_argument('--nepochs', type=int, default=1)  # Reduced for speed
#     parser.add_argument('--num_timesteps', type=int, default=int(1e4))  # Significantly reduced for quick test

# def add_rollout_params(parser):
#     parser.add_argument('--nsteps_per_seg', type=int, default=64)  # Reduced to collect data more frequently
#     parser.add_argument('--nsegs_per_env', type=int, default=1)
#     parser.add_argument('--envs_per_process', type=int, default=8)  # Reduced to lower the computational load
#     parser.add_argument('--nlumps', type=int, default=1)

if __name__ == '__main__':
    import argparse
    
    train = True
    eval_episode = 5
    
    # env_id = 'PongNoFrameskip-v4'
    # env_id = 'BreakoutNoFrameskip-v4'
    env_id = 'SpaceInvadersNoFrameskip-v4'
    # env_id = 'MontezumaRevengeNoFrameskip-v4'
    # env_id = 'Frogger-v4'
    os.environ["CUDA_VISIBLE_DEVICES"]= '0'
    feat = 'idf'
    
    if tf.test.gpu_device_name():
      print('\n\nDefault GPU Device:{} \n\n'.format(tf.test.gpu_device_name()))
    else:
      print("\n\nPlease install GPU version of TF\n\n")
    time.sleep(1)

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_environments_params(parser, env_id)
    add_optimization_params(parser)
    add_rollout_params(parser)

    parser.add_argument('--exp_name', type=str, default='')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--dyn_from_pixels', type=int, default=0)
    parser.add_argument('--use_news', type=int, default=0)
    parser.add_argument('--ext_coeff', type=float, default=0.)
    parser.add_argument('--int_coeff', type=float, default=1.)
    parser.add_argument('--layernorm', type=int, default=0)
    parser.add_argument('--feat_learning', type=str, default=feat,
                        choices=["none", "idf", "vaesph", "vaenonsph", "pix2pix", "vime"])

    args = parser.parse_args()

    print(args)

    start_experiment(**args.__dict__)
