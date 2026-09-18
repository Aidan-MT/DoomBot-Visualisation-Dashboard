# DoomBot-Visualisation-Dashboard
A basic dashboard visualising the performance of a Reinforcement Learning algorithm, trained to play Doom. Uses the VizDoom library and Streamlit.

# Pre-Requisites
Requires Python 3.14.2 or newer. 

## Quickstart (Linux/MacOS)
Clone repository to any appropriate location
```bash
   git clone https://github.com/Aidan-MT/DoomBot-Visualisation-Dashboard.git
   cd DoomBot-Visualisation-Dashboard
```

Install dependences 
```bash
   pip install -r requirements.txt
```
_NOTE: if desired, a virtual python environment may be created to avoid package conflicts_

Run Streamlit dashboard in the terminal
```python
   streamlit run code/dashboard.py
```

## Dashboard Overview
The dashboard contains user-defined parameters. Upon selection, an instance of doom will be generated, and training information provided. 

NOTE: This tool is for demonstration purposes only, under normal circumstances model training takes many more iterations to gain competence! At the human-interpretable frame-rates used here this would take a huge length of time...

<img width="1469" height="874" alt="Example_Dashboard" src="https://github.com/user-attachments/assets/53d397db-fecc-4120-8a80-6db3a8dc68e4" />

   
