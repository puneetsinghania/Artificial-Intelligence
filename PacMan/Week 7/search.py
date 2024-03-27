# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """
    "*** YOUR CODE HERE ***"

    # Import the Stack
    from util import Stack

    # Create a stack for nodes to explore and create a list for explored nodes
    to_explore = Stack()
    explored = []

    # Initial check to get the start state
    initial_state = problem.getStartState()

    # Push the initial state onto the stack if it's not a goal state. Otherwise return an empty list
    to_explore.push((initial_state, [])) if not problem.isGoalState(initial_state) else []

    # Continue searching while there are nodes to explore
    while not to_explore.isEmpty():
        current, path = to_explore.pop()

        # If the current state is a goal state, return the path to it
        if problem.isGoalState(current):
            return path

        # Update the current state as explored
        explored.append(current)

        # Generate successors of the current state and their corresponding paths
        successors = [(state, path + [direction]) for state, direction, _ in problem.getSuccessors(current) if state not in explored]

        # Push the successors onto the stack for further exploration
        for state, updated_path in successors:
            to_explore.push((state, updated_path))

    # If goal state not found, then return an empty list
    return []



    #util.raiseNotDefined()

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"


    #Import the Queue
    from util import Queue

    # Create a queue for exploration and create a list for tracking of visited nodes
    visited = []
    explore_queue = Queue()
    
    # Get the initial state of the problem
    initial = problem.getStartState()

    # Check if the initial state is the goal and return if it is
    if problem.isGoalState(initial):
        return []

    # Start exploration from the initial state
    explore_queue.push((initial, []))

    # Continue exploration while the queue is not empty
    while not explore_queue.isEmpty():
        current, path_taken = explore_queue.pop()

        # If we get the current state as a goal state, return the path taken to reach it
        if problem.isGoalState(current):
            return path_taken

        # Update the current state as visited
        visited.append(current)

        # Get successors and update the queue with unvisited successors
        successors = [
            (next_state, path_taken + [direction])
            for next_state, direction, _ in problem.getSuccessors(current)
            if next_state not in visited and not any(item[0] == next_state for item in explore_queue.list)
        ]

        # Perform pushing of the unvisited successors onto the queue
        for state, path_new in successors:
            explore_queue.push((state, path_new))

    # If goal state is not found, then return an empty list
    return []

    #util.raiseNotDefined()

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"

    # Import the PriorityQueue
    from util import PriorityQueue

    # Create a priority queue for exploration based on cost
    search_queue = PriorityQueue()

    # Use a set to track the explored states
    explored = set()

    # Get the initial state of the problem
    initial = problem.getStartState()

    # If the initial state is the goal, then return an empty list
    if problem.isGoalState(initial):
        return []

    # Starting the exploration from the initial state with a cost of 0
    search_queue.push((initial, []), 0)

    # Continue exploration while the queue is not empty
    while not search_queue.isEmpty():
        current, path = search_queue.pop()

        # Skip nodes already explored
        if current in explored:
            continue

        # If the current state is a goal state, then return the path to it
        if problem.isGoalState(current):
            return path

        # Update the current state as explored
        explored.add(current)

        # Handle successors and their costs
        for next_node, action, _ in problem.getSuccessors(current):
            path_new = path + [action]
            new_cost = problem.getCostOfActions(path_new)

            # Check if the successor node is in the queue with a greater cost and update
            if next_node not in explored:
                search_queue.update((next_node, path_new), new_cost)
            else:
                # Push the successor onto the queue alongwith its cost
                search_queue.push((next_node, path_new), new_cost)

    # If goal state is not found, then return an empty list
    return []


    #util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

# Import the PriorityQueue
from util import PriorityQueue

class OptimizedPriorityQueue(PriorityQueue):
    """
    New class OptimizedPriorityQueue inherits from PriorityQueue
    """
    def __init__(self, task, computePriority):
        self.computePriority = computePriority
        self.task = task
        PriorityQueue.__init__(self)

    def enqueue(self, item, heuristicFn):
        
        #Queue an item using the priority determined by the computePriority function
        
        # Calculate the priority for the item utilizing the computePriority function
        priority = self.computePriority(self.task, item, heuristicFn)
        PriorityQueue.push(self, item, priority)

# Calculate the cost of actions for the given state using the task's getCostOfActions method
def priorityFunction(task, state, heuristicFn):
    
    cost = task.getCostOfActions(state[1])
    # Calculate the heuristic value for the current state utilizing the given heuristic function
    heuristic = heuristicFn(state[0], task)
    return cost + heuristic



def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"

    # Create an instance of the OptimizedPriorityQueue class
    queue = OptimizedPriorityQueue(problem, priorityFunction)
    
    # Initialize a set for tracking the explored states
    explored = set()
    
    # Get the initial state of the problem
    initial = problem.getStartState()
    
    # Push the initial state onto the priority queue with its heuristic value if it's not a goal state, otherwise return an empty list
    queue.enqueue((initial, []), heuristic) if not problem.isGoalState(initial) else []

    # Continue the search while the priority queue is not empty
    while not queue.isEmpty():
        current, path = queue.pop()
        
        # Skip nodes that have already been explored
        if current in explored:
            continue
        
        # If the current state is a goal state, then return the path to it
        if problem.isGoalState(current):
            return path
        
        # Update the current state as explored
        explored.add(current)
        
        # Generate successors and their corresponding paths
        successors = [(s, path + [a]) for s, a, _ in problem.getSuccessors(current) if s not in explored]
        
        # Push the unexplored successors onto the priority queue with their heuristic values
        for item in successors:
            queue.enqueue(item, heuristic)

    # If goal state is not found, then return an empty list
    return []


    #util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
