"""
Define functions to be called by dashboard
"""
import time
import random
import os 

import vizdoom as vzd
import gymnasium 
import streamlit as st
import stable_baselines3
import pandas as pd

from vizdoom import gymnasium_wrapper # Register doom environments
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor



def set_training_parameters(scenario):
    """
    Set training parameters (episode length, timeouts etc.)
    
    Arguments:
    -------------

    scenario : str - the chosen scenario to provide to the gym wrapper
    """

    env_kwargs = {
        "render_mode": "rgb_array", 
        "visible": False #Stop popout window appearing
    }

    #Define training wrapper
    match scenario:
        case "basic.cfg":
            env = gymnasium.make("VizdoomBasic-v1", **env_kwargs)
        
        case "deadly_corridor.cfg":
            env = gymnasium.make("VizdoomDeadlyCorridor-v1", **env_kwargs)
        
        case "defend_the_center.cfg":
            env = gymnasium.make("VizdoomDefendCenter-v1", **env_kwargs)
    
    return Monitor(env) 


#Define a class that will extract data from the ongoing instance
class StreamlitCallback(BaseCallback):
    """
    Callback to capture gym environment frames
    and stream them in real-time to a Streamlit placeholder.

    Callback function also updates the streamlit training data placeholder.
    And plots a live visualisation of training reward
    """
    def __init__(self, frame_placeholder, info_placeholder, graph_placeholder, verbose=0):
        super(StreamlitCallback, self).__init__(verbose)
        self.frame_placeholder = frame_placeholder
        self.info_placeholder = info_placeholder
        self.graph_placeholder = graph_placeholder 

        self.running_reward = 0.0
        self.last_episode_reward = 0.0
        self.episodes_completed = 0
        self.reward_history = []

    def _on_step(self) -> bool:
        # 1. Update visual frame feed
        try:
            frame = self.training_env.render()
            if frame is not None:
                self.frame_placeholder.image(frame, channels="RGB", use_container_width=True)
        except Exception as e:
            st.warning(f"Frame render warning: {e}")

        # 2. Update training metrics
        step_rewards = self.locals.get("rewards")
        infos = self.locals.get("infos", [])

        if step_rewards is not None:
            self.running_reward += float(step_rewards[0])

        for info in infos:
            if "episode" in info:
                self.last_episode_reward = info["episode"]["r"]
                self.episodes_completed += 1
                self.running_reward = 0.0  # Reset counter for next episode
                self.reward_history.append(self.last_episode_reward)

        # Display metrics to Streamlit UI
        self.info_placeholder.markdown(
            f"**Episodes Completed:** `{self.episodes_completed}`\n\n"
            f"**Current Episode Total Reward:** `{self.running_reward:.2f}`\n\n"
            f"**Last Episode Final Reward:** `{self.last_episode_reward:.2f}`"
        )

        self.graph_placeholder.line_chart(
            self.reward_history,
            x_label="Episode",
            y_label="Reward")
        
        return True

def train_agent(env, model_choice, training_steps, frame_placeholder, info_placeholder, graph_placeholder) -> None:
    """
    Train RL agent and stream real-time frames directly into Streamlit via callback hooks
    """
    # Hook our Streamlit streaming engine to SB3
    callback = StreamlitCallback(frame_placeholder, info_placeholder, graph_placeholder)
    
    match model_choice:
        case "A2C":
            model = stable_baselines3.A2C("MultiInputPolicy", env, verbose=1)
            model.learn(total_timesteps=training_steps, callback=callback, )
        
        case "PPO":
            model = stable_baselines3.PPO("MultiInputPolicy", env, verbose=1)
            model.learn(total_timesteps=training_steps, callback=callback)





