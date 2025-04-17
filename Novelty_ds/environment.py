import gymnasium as gym
from minigrid.wrappers import ImgObsWrapper
from stable_baselines3.common.vec_env import VecMonitor, DummyVecEnv, VecTransposeImage

def make_env(wrapper_class=None, render_mode="rgb_array"):
    def _init():
        env = gym.make("MiniGrid-Empty-16x16-v0", render_mode=render_mode)
        env = ImgObsWrapper(env)
        if wrapper_class:
            env = wrapper_class(env)
        return env
    return _init

def create_vector_env(wrapper_class=None):
    env = DummyVecEnv([make_env(wrapper_class)])
    env = VecTransposeImage(env)
    # Add Monitor wrapper for proper episode tracking
    env = VecMonitor(env)
    return env