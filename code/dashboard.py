#Import packages
import streamlit as st
from training_funcs import *


#Create basic page layout
st.set_page_config(page_title="ViZDoom Machine Learning Dashboard", layout = 'wide')
st.title("ML Monitoring Dashboard")

#Create sidebar for params and config
st.sidebar.header("ML Agent Settings")

#Allow user to choose scenario, model to train, and number of training runs
scenario = st.sidebar.selectbox(
    "Choose a Scenario to Run:",
    ["basic.cfg", "deadly_corridor.cfg", "defend_the_center.cfg"]
)

model_choice = st.sidebar.selectbox(
        "Choose a Model to Train:",
        ["A2C", "PPO"]
    )

training_steps = st.sidebar.number_input(
    "Total Training Time (ticks)?",
    0,
    step=500
)

#Begin simulation training:
start_run = st.sidebar.button("Start Simulation Run")

#Create two columns for live data, and any additional information
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Live Game Stream")
    frame_placeholder = st.empty()

with col2:
    st.subheader("Training Information")
    info_placeholder = st.empty()
    graph_placeholder = st.empty()

if start_run:
    #Initiate game env
    environment = set_training_parameters(scenario)
    
    #Begin model training
    train_agent(
        environment,
        model_choice,
        training_steps,
        frame_placeholder=frame_placeholder,
        info_placeholder=info_placeholder,
        graph_placeholder=graph_placeholder
    )

    #Upon completing training - exit the simulation.
    environment.close()
    
