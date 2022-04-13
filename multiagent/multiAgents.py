# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).
import math

from util import manhattanDistance
from game import Directions
import random, util

from game import Agent


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        for food in newFood.asList():
            food_distance = util.manhattanDistance(food, newPos)

        for ghost in newGhostStates:
            ghost_distance = util.manhattanDistance(ghost, newPos)

        return successorGameState.getScore() + food_distance + ghost_distance


def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"

        def minimax(gameState, index, depth):  # maximizer helper function (pacman)
            if gameState.isWin() or gameState.isLose() or depth == self.depth:  # if terminal state,
                return [self.evaluationFunction(gameState)]  # return score

            if index == 0:
                maxval = -math.inf  # initialize max value
                for action in gameState.getLegalActions(index):  # for all actions pacman can take
                    succ = gameState.generateSuccessor(index, action)  # get all successors of node
                    val = minimax(succ, index + 1, depth)[0]  # call minimizer of successor
                    if val > maxval:  # if val is max
                        maxval = val  # set max value to value
                        a = action  # set action
                return [maxval, a]  # return max val and action

            else:
                minval = math.inf  # initialize min value
                for action in gameState.getLegalActions(index):  # for all ghosts actions
                    succ = gameState.generateSuccessor(index, action)  # get all successors of node
                    if index == gameState.getNumAgents() - 1:  # if pacman state next
                        val = minimax(succ, 0, depth + 1)[0]  # call maximizer
                    else:
                        val = minimax(succ, index + 1, depth)[0]  # call maximizer
                    if val < minval:  # if value is min
                        minval = val  # set min value to value
                        a = action  # set action
                return [minval, a]  # return min value and action

        return minimax(gameState, self.index, 0)[1]  # call minimax on pacman


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """

        def alpha_beta(gameState, index, depth, alpha, beta):  # maximizer helper function (pacman)
            if gameState.isWin() or gameState.isLose() or depth == self.depth:  # if terminal state,
                return [self.evaluationFunction(gameState)]  # return score

            if index == 0:
                maxval = -math.inf  # initialize max value
                for action in gameState.getLegalActions(index):  # for all actions pacman can take
                    succ = gameState.generateSuccessor(index, action)  # get all successors of node
                    val = alpha_beta(succ, index + 1, depth, alpha, beta)[0]  # call minimizer of successor
                    if val > maxval:  # if value is max
                        maxval = val  # set max value to val
                        a = action  # set action
                    if maxval > beta:  # check is max value is larger than beta
                        return [maxval]  # if so, return max value
                    alpha = max(alpha, maxval)  # else, reset alpha value
                return [maxval, a]  # return max value and best action

            else:
                minval = math.inf  # initialize min value
                for action in gameState.getLegalActions(index):  # for all ghosts actions
                    succ = gameState.generateSuccessor(index, action)  # get all successors of node
                    if index == gameState.getNumAgents() - 1:  # if pacman state next
                        val = alpha_beta(succ, 0, depth + 1, alpha, beta)[0]  # choose max value
                    else:
                        val = alpha_beta(succ, index + 1, depth, alpha, beta)[0]  # choose max value
                    if val < minval:  # if value is min
                        minval = val  # set min value to val
                        a = action  # set action
                    if minval < alpha:  # check if min value is less than alpha
                        return [minval]  # if so, return min value
                    beta = min(beta, minval)  # else, reset beta value
                return [minval, a]  # return min value and best action

        return alpha_beta(gameState, self.index, 0, -math.inf, math.inf)[1]  # get alpha-beta for pacman


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """

        def exp_max(gameState, index, depth):  # maximizer helper function (pacman)
            if gameState.isWin() or gameState.isLose() or depth == self.depth:  # if terminal state,
                return [self.evaluationFunction(gameState)]  # return score

            if index == 0:  # if pacman:
                maxval = -math.inf  # initialize max value
                for action in gameState.getLegalActions(index):  # for all actions pacman can take
                    succ = gameState.generateSuccessor(index, action)  # get all successors of node
                    val = exp_max(succ, index + 1, depth)[0]  # call expectimax on successor
                    if val > maxval:  # if value is max
                        maxval = val  # set max value to value
                        a = action  # set action of max value
                return [maxval, a]  # return max value and best action

            else:
                val = 0  # initialize value to 0
                p = 1 / len(gameState.getLegalActions(index))  # initialize probability
                for action in gameState.getLegalActions(index):  # for all ghosts actions
                    succ = gameState.generateSuccessor(index, action)  # get all successors of node
                    if index == gameState.getNumAgents() - 1:  # if pacman state next
                        val += p * exp_max(succ, 0, depth + 1)[0]  # increase val by chance prob
                    else:
                        val += p * exp_max(succ, index + 1, depth)[0]  # increase val by chance prob
                    a = action  # set action of min value
                return [val, a]  # return expectimax value and best action

        return exp_max(gameState, self.index, 0)[1]  # get pacman expectimax


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()


# Abbreviation
better = betterEvaluationFunction
