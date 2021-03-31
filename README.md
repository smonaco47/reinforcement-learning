# Lunar Lander Exploration

This repo was used to learn tensorforce in the context of the Lunar Lander openAI gym.

Files of note:
- train_tensorforce_model.py: Runs a single training instance for the agent
- grid_search.py: Explores the hyperparameter space specified in the randomize() function in Hyperparameter
- analyze_results.py: Makes a graph of the different hyperparameters vs. the highest bucket reward achieved
- consolidate_results.py: I ran multiple grid searches at once, this consolidates the output files for easy
    consumption 
    
I used Anaconda to manage packages, the environment I used for my testing is stored in tensorforce.yml.