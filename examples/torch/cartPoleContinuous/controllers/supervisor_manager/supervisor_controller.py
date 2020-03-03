import numpy as np
from deepbots.supervisor.controllers.supervisor_emitter_receiver import SupervisorCSV
from utilities import normalizeToRange, plotData


class CartPoleSupervisor(SupervisorCSV):
    """
    CartPoleSupervisor acts as an environment having all the appropriate methods such as get_reward(), but also contains
    the agent. The agent used here is trained with the PPO algorithm (https://arxiv.org/abs/1707.06347).

    Taken from https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py and modified for Webots.
    Description:
        A pole is attached by an un-actuated joint to a cart, which moves forwards and backwards. The pendulum
        starts upright, and the goal is to prevent it from falling over by increasing and reducing the cart's
        velocity.
    Source:
        This environment corresponds to the version of the cart-pole problem described
        by Barto, Sutton, and Anderson
    Observation:
        Type: Box(4)
        Num	Observation                 Min         Max
        0	Cart Position z axis      -0.4            0.4
        1	Cart Velocity             -Inf            Inf
        2	Pole Angle                -1.3 rad        1.3 rad
        3	Pole Velocity At Tip      -Inf            Inf

    Actions:
        Type: Continuous(1)
        Num     Min     Max
        0       -inf    inf

        Note: The amount the velocity that is reduced or increased is not fixed; it depends on the angle the pole is
        pointing. This is because the center of gravity of the pole increases the amount of energy needed to move the
        cart underneath it
    Reward:
        Reward is 1 for every step taken, including the termination step
    Starting State:
        [0.0, 0.0, 0.0, 0.0]
    Episode Termination:
        Pole Angle is more than 0.261799388 rad (15 degrees)
        Cart Position is more than 0.39 on z axis (cart has reached arena edge)
        Episode length is greater than 200
        Solved Requirements (average episode score in last 100 episodes > 195.0)
    """

    def __init__(self, stepsPerEpisode=200):
        """
        In the constructor, the agent object is created, the robot is spawned in the world via respawnRobot().
        References to robot and the pole endpoint are initialized here, used for building the observation.
        When in test mode (self.test = True) the agent stops being trained and picks actions in a non stochastic way.

        :param stepsPerEpisode: int, how many steps to run each episode (changing this messes up the solved condition)
        """
        print("Robot is spawned in code, if you want to inspect it pause the simulation.")
        super().__init__()
        self.observationSpace = (4,)
        self.actionSpace = (1,)

        self.robot = None
        self.respawnRobot()

        self.poleEndpoint = self.supervisor.getFromDef("POLE_ENDPOINT")
        self.robot = self.supervisor.getFromDef("ROBOT")

        self.stepsPerEpisode = stepsPerEpisode
        self.episodeScore = 0  # score accumulated during an episode
        self.episodeScoreList = []  # a list to save all the episode scores, used to check if task is solved
        self.test = False  # whether the agent is in test mode

    def get_observations(self):
        """
        This get_observation implementation builds the required observation for the CartPole problem.
        All values apart from pole angle are gathered here from the robot and poleEndpoint objects.
        The pole angle value is taken from the message sent by the robot.
        All values are normalized appropriately to [-1, 1].

        :return: list, observation: [cartPosition, cartVelocity, poleAngle, poleTipVelocity]
        """
        # Position on z axis
        cartPosition = normalizeToRange(self.robot.getPosition()[2], -0.4, 0.4, -1.0, 1.0)
        # Linear velocity on z axis
        cartVelocity = normalizeToRange(self.robot.getVelocity()[2], -0.2, 0.2, -1.0, 1.0, clip=True)

        self.handle_receiver()  # update _last_message received from robot, which contains pole angle
        if self._last_message is not None:
            poleAngle = normalizeToRange(float(self._last_message[0]), -0.23, 0.23, -1.0, 1.0, clip=True)
        else:
            # method is called before _last_message is initialized
            poleAngle = 0.0

        # Angular velocity x of endpoint
        endpointVelocity = normalizeToRange(self.poleEndpoint.getVelocity()[3], -1.5, 1.5, -1.0, 1.0, clip=True)

        return [cartPosition, cartVelocity, poleAngle, endpointVelocity]

    def get_reward(self, action):
        """
        Reward is +1 for each step taken, apart for termination step, which is handled after step() is called.

        :param action: float
        :return: int, always 1
        """
        return 1

    def is_done(self):
        """
        An episode is done if the score is over 195.0, or if the pole is off balance, or the cart position is on the
        arena's edges.

        :return: bool, True if termination conditions are met, False otherwise
        """
        if self.episodeScore > 195.0:
            return True

        if self._last_message is not None:
            poleAngle = round(float(self._last_message[0]), 2)
        else:
            # method is called before _last_message is initialized
            poleAngle = 0.0
        if abs(poleAngle) > 0.261799388:  # 15 degrees off vertical
            return True

        cartPosition = round(self.robot.getPosition()[2], 2)  # Position on z axis
        if abs(cartPosition) > 0.39:
            return True

        return False

    def reset(self):
        """
        Reset calls respawnRobot() method and returns starting observation.
        :return: list, starting observation filled with zeros
        """
        self.respawnRobot()
        return [0.0 for _ in range(self.observationSpace[0])]

    def respawnRobot(self):
        """
        This method reloads the saved CartPole robot in its initial state from the disk.
        """
        if self.robot is not None:
            # Despawn existing robot
            self.robot.remove()

        # Respawn robot in starting position and state
        rootNode = self.supervisor.getRoot()  # This gets the root of the scene tree
        childrenField = rootNode.getField('children')  # This gets a list of all the children, ie. objects of the scene
        childrenField.importMFNode(-2, "CartPoleRobot.wbo")  # Load robot from file and add to second-to-last position

        # Get the new robot and pole endpoint references
        self.robot = self.supervisor.getFromDef("ROBOT")
        self.poleEndpoint = self.supervisor.getFromDef("POLE_ENDPOINT")
        # Reset the simulation physics to start over
        self.supervisor.simulationResetPhysics()

        self._last_message = None

    def get_info(self):
        return None

    def solved(self):
        """
        This method checks whether the CartPole task is solved, so training terminates.
        :return: bool, True if task is solved, False otherwise
        """
        if len(self.episodeScoreList) > 100:  # Over 100 trials thus far
            if np.mean(self.episodeScoreList[-100:]) > 195.0:  # Last 100 episodes' scores average value
                return True
        return False
