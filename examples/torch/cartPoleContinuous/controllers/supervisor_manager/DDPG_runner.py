import numpy as np

from supervisor_controller import CartPoleSupervisor
from keyboard_controller_cartpole import KeyboardControllerCartPole
from agent.DDPG_agent import DDPGAgent
from utilities import plotData


def run():
    ########
    # Setup
    ########
    # Initialize supervisor object
    supervisorPre = CartPoleSupervisor()
    # Wrap the CartPole supervisor in the custom keyboard controller
    supervisorEnv = KeyboardControllerCartPole(supervisorPre)

    agent = DDPGAgent(supervisorPre.observationSpace, supervisorPre.actionSpace, lr_actor=0.000025, lr_critic=0.00025,
                      layer1_size=30, layer2_size=50, layer3_size=30, batch_size=64)

    solved = False  # Whether the solved requirement is met
    averageEpisodeActionProbs = []  # Save average episode taken actions probability to plot later
    episodeCount = 0
    episodeLimit = 10000

    ########
    # Train
    ########
    # Run outer loop until the episodes limit is reached or the task is solved
    while not solved and episodeCount < episodeLimit:
        state = supervisorPre.reset()  # Reset robot and get starting observation
        supervisorPre.episodeScore = 0
        actionProbs = []  # This list holds the probability of each chosen action

        # Inner loop is the episode loop
        for step in range(supervisorPre.stepsPerEpisode):
            # In training mode the agent returns the action plus OU noise for exploration
            action = agent.choose_action_train(state)
            # Step the supervisor to get the current action reward, the new state and whether we reached
            # the done condition
            newState, reward, done, info = supervisorEnv.step(action)

            # Save the current state transition in agent's memory
            agent.remember(state, action, reward, newState, int(done))
            # Perform a learning step

            supervisorPre.episodeScore += reward  # Accumulate episode reward
            agent.learn()
            if done or step == supervisorPre.stepsPerEpisode - 1:
                # Save the episode's score
                supervisorPre.episodeScoreList.append(supervisorPre.episodeScore)
                # agent.learn(batch_size=step + 1)
                solved = supervisorPre.solved()  # Check whether the task is solved
                break

            state = newState  # state for next step is current step's newState

        if supervisorPre.test:  # If test flag is externally set to True, agent is deployed
            break

        print("Episode #", episodeCount, "score:", supervisorPre.episodeScore)

        episodeCount += 1  # Increment episode counter

    ###############
    # Create plots
    ###############
    # np.convolve is used as a moving average, see https://stackoverflow.com/a/22621523
    # this is done to smooth out the plots
    movingAvgN = 10
    plotData(np.convolve(supervisorPre.episodeScoreList, np.ones((movingAvgN,)) / movingAvgN, mode='valid'),
             "episode", "episode score", "Episode scores over episodes")

    #############
    # Test agent
    #############
    if not solved and not supervisorPre.test:
        print("Reached episode limit and task was not solved.")
    else:
        if not solved:
            print("Task is not solved, deploying agent for testing...")
        elif solved:
            print("Task is solved, deploying agent for testing...")
    state = supervisorPre.reset()
    supervisorPre.test = True
    while True:
        action = agent.choose_action_test(state)
        state, _, _, _ = supervisorEnv.step(action)
