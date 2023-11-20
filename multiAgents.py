import random
import statistics

import util
from game import Agent, Directions
from pacman import GameState


class ReflexAgent(Agent):

    def getAction(self, gameState: GameState):
        # Storing all legal movements that agent can have
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        "Add more of your code here if you want to"
        # print(legalMoves[chosenIndex])

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"

        return successorGameState.getScore()


def scoreEvaluationFunction(currentGameState: GameState):
    return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):
    def __init__(self, evalFn='betterEvaluationFunction', depth='2'):
        self.index = 0
        self.evaluationFunction = better
        self.depth = int(depth)
        self.BEST_ACTION = None


class MinimaxAgent(MultiAgentSearchAgent):
    def getAction(self, state):
        def minimax(state, depth, agent):
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state)
            if agent == 0:
                max_value = float("-inf")
                actions = state.getLegalActions(agent)
                for action in actions:
                    successor = state.generateSuccessor(agent, action)
                    eval = minimax(successor, depth, agent + 1)
                    max_value = max(max_value, eval)
                return max_value
            else:
                next_agent = agent + 1
                if next_agent == state.getNumAgents():
                    next_agent = 0
                    depth += 1
                min_value = float("inf")
                actions = state.getLegalActions(agent)
                for action in actions:
                    successor = state.generateSuccessor(agent, action)
                    eval = minimax(successor, depth, next_agent)
                    min_value = min(min_value, eval)
                return min_value
        actions = state.getLegalActions(0)
        move = Directions.STOP
        value2 = float("-inf")
        for action in actions:
            compare = minimax(state.generateSuccessor(0, action), 0, 1)
            if compare > value2:
                value2 = compare
                move = action
        return move

class ExpectimaxAgent(MultiAgentSearchAgent):
    def expectimax(self, gameState, depth, agent):
        if gameState.isWin() or gameState.isLose() or depth == 0:
            return self.evaluationFunction(gameState)
        # agent = player % gameState.getNumAgents()
        if agent == 0:
            value = float("-inf")
            actions = gameState.getLegalActions(agent)
            for action in actions:
                successor = gameState.generateSuccessor(agent, action)
                eval = self.expectimax(successor, depth - 1, agent + 1)
                if depth == self.depth and eval > value:
                    self.BEST_ACTION = action
                value = max(value, eval)
            return value
        else:
            value = float("inf")
            scores = []
            n_agent = (agent + 1) % gameState.getNumAgents()
            if agent + 1 == gameState.getNumAgents():
                depth -= 1
            for action in gameState.getLegalActions(agent):
                successor = gameState.generateSuccessor(agent, action)
                list.append(scores, self.expectimax(successor, depth, n_agent))
                value = statistics.mean(scores)
            return value

    def getAction(self, gameState: GameState):
        self.expectimax(gameState, self.depth, 0)
        return self.BEST_ACTION


def nearest_food_distance(state):
    state.getFood()
    walls = state.getWalls()
    row, col = 0, 0
    for i in walls:
        for j in walls[0]:
            col += 1
        row += 1

    def is_in_bounds(i, j):
        if 0 < i < row and 0 < j < col:
            return True
        else:
            return False

    pac_position = state.getPacmanPosition()
    visited = set()
    queue = util.Queue()
    queue.push([pac_position, 0])
    while not queue.isEmpty():
        temp_position = queue.pop()
        x, y = temp_position[0]

        if state.hasFood(x, y):
            return temp_position[1]

        if temp_position[0] in visited:
            continue

        visited.add(temp_position[0])

        x, y = temp_position[0]
        if not walls[x - 1][y] and is_in_bounds(x - 1, y):
            queue.push([(x - 1, y), temp_position[1] + 1])
        if not walls[x + 1][y] and is_in_bounds(x + 1, y):
            queue.push([(x + 1, y), temp_position[1] + 1])
        if not walls[x][y - 1] and is_in_bounds(x, y - 1):
            queue.push([(x, y - 1), temp_position[1] + 1])
        if not walls[x][y + 1] and is_in_bounds(x, y + 1):
            queue.push([(x, y + 1), temp_position[1] + 1])

    return float('inf')



def nearest_scared_ghost(ghoststate, state):
    state.getNumAgents()
    walls = state.getWalls()
    row, col = 0, 0
    for i in walls:
        for j in walls[0]:
            col += 1
        row += 1
    def is_in_bounds(i, j):
        if 0 < i < row and 0 < j < col:
            return True
        else:
            return False

    pac_position = state.getPacmanPosition()
    visited = set()
    queue = util.Queue()
    queue.push([pac_position, 0])
    while not queue.isEmpty():
        temp_position = queue.pop()
        x, y = temp_position[0]

    if ghoststate == (x, y):
        return temp_position[1]









def better_evaluation_function(currentGameState: GameState, ghostState=None):
    score = 0
    pac_pos = currentGameState.getPacmanPosition()
    food_remain = currentGameState.getNumFood()
    ghost_states = currentGameState.getGhostStates()
    ghost_distance = 0


    if currentGameState.isWin():
        return currentGameState.getScore() + 10000
    if currentGameState.isLose():
        return -10000

    score += currentGameState.getScore() / 2

    score -= 100 * food_remain

    score += 10 / nearest_food_distance(currentGameState)

    for ghost in ghost_states:
        d = util.manhattanDistance(ghost.getPosition(), pac_pos)
        ghost_distance += d
        if d < 3:
            score -= d * 10


    return score



# Abbreviation
better = better_evaluation_function
