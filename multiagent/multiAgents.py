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


from util import manhattanDistance
from game import Directions
import random, util
from math import inf
from functools import partial, reduce

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
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

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
        # newPos = successorGameState.getPacmanPosition()
        # newFood = successorGameState.getFood()
        # newGhostStates = successorGameState.getGhostStates()
        # #newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newPower = successorGameState.getCapsules()
        newGhost = successorGameState.getGhostStates()
        newScaredTimes = [ghost.scaredTimer for ghost in newGhost]

        minScared = min(newScaredTimes)

        if len(newPower) != 0:
            pelletDistances = [manhattanDistance(newPos, pel) for pel in newPower]
            distPower = min(pelletDistances)
        else:
            distPower = 10000000000

        ghostScore = self.ghostScore(newGhost, newPos)
        foodScore = self.foodScore(newFood.asList(), newPos)

        return successorGameState.getScore() + 1.0*ghostScore + 1.0*foodScore + 3.0/distPower + minScared

    def ghostScore(self, ghostStates, position):
        distances = [manhattanDistance(position, ghost.getPosition()) for ghost in ghostStates]
        minDist = min(distances)

        if minDist <= 1:
            return -100
        else:
            return 1/minDist

    def foodScore(self, foodList, position):
        distances = [manhattanDistance(position, food) for food in foodList]
        if (len(distances) == 0):
            return 1/100000000
        minDist = min(distances)
        return 1/minDist

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

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
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
        def evalCheck(depth, state):
            return state.isWin() or state.isLose() or depth == self.depth

        def max_value(state, agIndex, curDepth):
            maxScores = []

            if evalCheck(curDepth, state):
                return self.evaluationFunction(state)

            for action in state.getLegalActions(agIndex):
                nextState = state.generateSuccessor(agIndex, action)
                maxScores.append(min_value(nextState, 1, curDepth))

            return max(maxScores)

        def min_value(state, agIndex, curDepth):
            minScores = []

            if evalCheck(curDepth, state):
                return self.evaluationFunction(state)

            for action in state.getLegalActions(agIndex):
                nextState = state.generateSuccessor(agIndex, action)

                if agIndex == state.getNumAgents() - 1:
                    minScores.append(max_value(nextState, 0, curDepth + 1))
                else:
                    minScores.append(min_value(nextState, agIndex + 1, curDepth))

            return min(minScores)

        dict = {}
        for action in gameState.getLegalActions(0):
            nextState = gameState.generateSuccessor(0, action)
            #start with depth 0 then go up to our depth
            score = min_value(nextState, 1, 0)
            dict[action] = score

        return max(dict, key = dict.get)

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        alpha = -inf
        beta = inf

        def evalCheck(depth, state):
            return state.isWin() or state.isLose() or depth == self.depth

        def max_value(state, agIndex, curDepth, alpha, beta):

            if evalCheck(curDepth, state):
                # print("max state result: ", self.evaluationFunction(state))

                return self.evaluationFunction(state)

            v = -inf
            for action in state.getLegalActions(agIndex):
                nextState = state.generateSuccessor(agIndex, action)
                nextScore = min_value(nextState, 1, curDepth,  alpha, beta)

                v = max(v, nextScore)

                if v> beta:
                    return v
                else:
                    alpha = max(alpha, v)

            return v

        def min_value(state, agIndex, curDepth, alpha, beta):

            if evalCheck(curDepth, state):
                # print("min state result: ", self.evaluationFunction(state))
                return self.evaluationFunction(state)

            v = inf

            # print("min state legal actions: ", state.getLegalActions(agIndex))
            for action in state.getLegalActions(agIndex):
                nextState = state.generateSuccessor(agIndex, action)

                if agIndex == state.getNumAgents() - 1:
                    nextScore = max_value(nextState, 0, curDepth + 1,  alpha, beta)
                else:
                    nextScore = (min_value(nextState, agIndex + 1, curDepth, alpha, beta))

                v = min(v, nextScore)
                if v < alpha:
                    return v
                else:
                    beta = min(beta, v)

            return v

        dict = {}
        bestScore= []
        for action in gameState.getLegalActions(0):
            nextState = gameState.generateSuccessor(0, action)
            score = min_value(nextState, 1, 0, alpha, beta)

            bestScore.append(score)
            dict[action] = score
            alpha = max(max(bestScore), alpha)

            if score > beta:
                return max(dict, key = dict.get)

        return list(dict.keys())[list(dict.values()).index(alpha)]


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

        #same as minimax
        def evalCheck(depth, state):
            return state.isWin() or state.isLose() or depth == self.depth

        #same as minimax
        def max_value(state, agIndex, curDepth):
            maxScores = []

            if evalCheck(curDepth, state):
                return self.evaluationFunction(state)

            for action in state.getLegalActions(agIndex):
                nextState = state.generateSuccessor(agIndex, action)
                maxScores.append(min_value(nextState, 1, curDepth))

            return max(maxScores)

        def min_value(state, agIndex, curDepth):
            minScores = []

            if evalCheck(curDepth, state):
                return self.evaluationFunction(state)

            for action in state.getLegalActions(agIndex):
                nextState = state.generateSuccessor(agIndex, action)

                if agIndex == state.getNumAgents() - 1:
                    minScores.append(max_value(nextState, 0, curDepth + 1))
                else:
                    minScores.append(min_value(nextState, agIndex + 1, curDepth))

            #only thing diff from minimax
            return sum(minScores)/float(len(minScores))

        actionDict = {}
        for action in gameState.getLegalActions(0):
            nextState = gameState.generateSuccessor(0, action)
            score = min_value(nextState, 1, 0)
            actionDict[action] = score

        return max(actionDict, key = actionDict.get)

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>

    Things I calculated:
    distFood = distance  to the nearest food
    distPower = distance to the nearest power pellet
    distGhost = distance to the nearest ghost
    minScared = the minimum time the ghosts willbe scared for

    My function was a linear combination of:
    - current game score
    - minScared
    - 1/distFood
    - 1/distGhost
    - 1/distPower

    """
    curPos = currentGameState.getPacmanPosition()
    curFood= currentGameState.getFood()
    curPower = currentGameState.getCapsules()
    curGhost = currentGameState.getGhostStates()
    curScaredTimes = [ghost.scaredTimer for ghost in curGhost]

    minScared = min(curScaredTimes)

    if len(curPower) != 0:
        pelletDistances = [manhattanDistance(curPos, pel) for pel in curPower]
        distPower = min(pelletDistances)
    else:
        distPower = 10000000000

    ghostDistances = [manhattanDistance(curPos, ghost.getPosition()) for ghost in curGhost]
    distGhost = min(ghostDistances)
    if distGhost<=1:
        ghostScore = -100
    else:
        ghostScore = 1/distGhost

    if len(curFood.asList()) != 0:
        foodDistances = [manhattanDistance(curPos, food) for food in curFood.asList()]
        distFood = min(foodDistances)
    else:
        distFood = 10000000000

    return currentGameState.getScore() + minScared + 1.0/distFood + 1.0*ghostScore + 3.0/distPower

# Abbreviation
better = betterEvaluationFunction
