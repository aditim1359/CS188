# valueIterationAgents.py
# -----------------------
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


# valueIterationAgents.py
# -----------------------
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


import mdp, util

from learningAgents import ValueEstimationAgent
import collections


class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """

    def __init__(self, mdp, discount=0.9, iterations=100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter()  # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        for i in range(self.iterations):  # for each iteration
            newValues = self.values.copy()
            for state in self.mdp.getStates():  # for each state
                if self.mdp.isTerminal(state):  # if terminal state, nee to stop
                    continue  # breaks this state for loop
                actions = self.mdp.getPossibleActions(state)
                maxValue = max([self.getQValue(state, action) for action in actions])
                newValues[state] = maxValue

            self.values = newValues

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]

    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        qVal = 0

        transition = self.mdp.getTransitionStatesAndProbs(state, action)
        for sPrime, tVal in transition:
            qVal += tVal*(self.mdp.getReward(state, action, sPrime) + self.discount*self.values[sPrime])
        return qVal

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """

        if self.mdp.isTerminal(state):
            return None

        # optimalAction = {a:[] for a in self.mdp.getPossibleActions(state)}
        optimalAction = util.Counter() # Dict with a default 0!!
        for action in self.mdp.getPossibleActions(state):
            optimalAction[action] = self.getQValue(state, action)
        return optimalAction.argMax()

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        """Returns the policy at the state (no exploration)."""
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)


class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """

    def __init__(self, mdp, discount=0.9, iterations=1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        states = self.mdp.getStates()

        for i in range(self.iterations):
            state = states[i%len(states)]
            if self.mdp.isTerminal(state):
                continue
            actions = self.mdp.getPossibleActions(state)
            maxValue = max([self.getQValue(state, action) for action in actions])
            self.values[state] = maxValue

class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """

    def __init__(self, mdp, discount=0.9, iterations=100, theta=1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"

        def updatePred(pred):
            if self.mdp.isTerminal(pred):
                return None
            predValue = self.values[pred]
            predActions = self.mdp.getPossibleActions(pred)
            PredmaxValue = max([self.getQValue(pred, action) for action in predActions])
            diff = abs(predValue - PredmaxValue)
            return diff

        # compute the predecessors of stte
        predecessors = {}

        states = self.mdp.getStates()
        for state in states:
            if self.mdp.isTerminal(state):
                continue
            for action in self.mdp.getPossibleActions(state):
                for nextState, trans in self.mdp.getTransitionStatesAndProbs(state, action):
                    if nextState in predecessors:
                        predecessors[nextState].add(state)
                    else:
                        predecessors[nextState] = {state}

        #create prioirity Queue
        pq = util.PriorityQueue()

        # absvaldiff between current state val and maxQval of actions: find states' priority
        for state in states:
            diff = updatePred(state)
            if (diff!=None):
                pq.push(state, -1*diff)

        # do iterations
        for iteration in range(self.iterations):
            if pq.isEmpty():
                break
            state = pq.pop()
            if self.mdp.isTerminal(state):
                continue

            #update values of state with bellman update
            actions = self.mdp.getPossibleActions(state)
            maxValue = max([self.getQValue(state, action) for action in actions])
            self.values[state] = maxValue

            #for each predecessor of s
            for pred in predecessors[state]:
                diff = updatePred(pred)
                if (diff!=None):
                    if (diff > self.theta):
                        pq.update(pred, -1*diff) # need to UPDATE not PUSH!



