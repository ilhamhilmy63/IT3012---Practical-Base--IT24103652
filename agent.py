# agent.py
# IT3012 - Intelligent Agents
# Practical 1: Simple Reflex Agent
# Practical 2: Model-Based (Reflex) Agent
# Practical 3: Problem-Solving / Search Agent (BFS, DFS, UCS)

import random
from collections import deque
import heapq
import itertools

# All four movement actions, listed in clockwise order so that "turning"
# means simply advancing to the next entry in the list.
DIRECTIONS = ['Up', 'Right', 'Down', 'Left']

# How each action changes an (x, y) coordinate.
# y increases upwards, x increases to the right.
MOVES = {
    'Up': (0, 1),
    'Down': (0, -1),
    'Left': (-1, 0),
    'Right': (1, 0),
}

# The SimpleReflexAgent is not allowed to store state, so "forward" and
# "turn left" must be fixed constants rather than instance attributes.
FORWARD = 'Right'
TURN_LEFT = 'Up'      # turning left while facing Right = facing Up


# ---------------------------------------------------------------------------
# Practical 1 - Simple Reflex Agent
# ---------------------------------------------------------------------------
class SimpleReflexAgent:
    """
    Practical 1 - a purely reactive agent.

    NOTE: this class deliberately has NO __init__ method. The practical sheet
    forbids storing any history, and that restriction is the entire point -
    the agent's action is a pure function of the current percept.

    Because it is stateless it cannot tell one corner from another, so an
    identical percept always produces an identical action. In a corner it
    repeats the same move forever.
    """

    def sense_and_act(self, percept: dict) -> str:
        wall_ahead = percept.get('wall_ahead', False)
        food_here = percept.get('food_here', False)

        # ---- CONDITION-ACTION RULES (the whole agent program) -------------
        if food_here:                 # RULE 1
            return FORWARD            # pellet is taken on entry, so carry on
        if wall_ahead:                # RULE 2
            return TURN_LEFT
        return FORWARD                # RULE 3 (default)


# ---------------------------------------------------------------------------
# Practical 2 - Model-Based Reflex Agent
# ---------------------------------------------------------------------------
class ModelBasedAgent:
    """
    Practical 2 - a reflex agent WITH internal state.

    The percept is only {'wall_ahead': bool, 'food_here': bool}, so the agent
    is blind to its own coordinates. To build `visited_cells` at all it must
    estimate its position by DEAD RECKONING:

        start at (0, 0), remember which way you are facing, and update the
        estimate after every action according to whether that action could
        actually have succeeded.

    TRANSITION MODEL - how the world evolves (`_update_state`)
    SENSOR MODEL     - how percepts map to the world (`wall_ahead` marks the
                       faced neighbour as a known wall)
    """

    def __init__(self):
        # ---- INTERNAL STATE ----------------------------------------------
        self.visited_cells = set()      # every cell the agent believes it entered
        self.position = (0, 0)          # dead-reckoned position estimate
        self.facing = FORWARD           # direction the agent believes it faces
        self.known_walls = set()        # cells the sensor has reported as walls
        self.last_action = None         # what it did on the previous turn
        self.last_percept = None        # what it saw on the previous turn
        self.repeat_count = 0           # identical percepts seen in a row
        self.uncertain_steps = 0        # moves made without sensor evidence
        self.steps = 0

    # -- TRANSITION MODEL ---------------------------------------------------
    def _update_state(self, percept: dict) -> None:
        """Advance the internal world model using the last action + percept.

        The conditional is the important part: a move only succeeds if the
        target cell was not blocked. The agent knows a cell is blocked only
        when it was FACING that cell and the sensor said wall_ahead.
        """
        if self.last_action is not None:
            dx, dy = MOVES[self.last_action]
            target = (self.position[0] + dx, self.position[1] + dy)

            if self.last_action == self.facing:
                # We were ALREADY facing this way, so last turn's wall_ahead
                # genuinely described this move. The percept is informative.
                if self.last_percept.get('wall_ahead', False):
                    self.known_walls.add(target)      # sensor model -> map
                else:
                    self.position = target            # move must have worked
            else:
                # We TURNED. Last turn's wall_ahead described the old facing
                # direction, so it says nothing about this move. Resolve it
                # with the CURRENT percept instead, which now describes the
                # new direction:
                #   wall_ahead True  -> we are still here, blocked
                #   wall_ahead False -> the cell ahead is clear, so we moved
                if percept.get('wall_ahead', False):
                    self.known_walls.add(target)
                else:
                    self.position = target
                self.uncertain_steps += 1

            self.facing = self.last_action

        # Record where we now believe we are.
        self.visited_cells.add(self.position)

        # If a wall is ahead right now, mark that neighbour on the map too.
        if percept.get('wall_ahead', False):
            dx, dy = MOVES[self.facing]
            self.known_walls.add((self.position[0] + dx, self.position[1] + dy))

    # -- helpers ------------------------------------------------------------
    def _target_of(self, action: str):
        dx, dy = MOVES[action]
        return (self.position[0] + dx, self.position[1] + dy)

    # -- main decision procedure -------------------------------------------
    def sense_and_act(self, percept: dict) -> str:
        self.steps += 1
        self._update_state(percept)

        wall_ahead = percept.get('wall_ahead', False)
        food_here = percept.get('food_here', False)

        # LOOP DETECTION: an unchanging percept means the world is not
        # responding to us. This is the safety net that guarantees escape even
        # when the dead-reckoned map has drifted out of sync with reality.
        if self.last_percept is not None and percept == self.last_percept:
            self.repeat_count += 1
        else:
            self.repeat_count = 0

        if self.repeat_count >= 2:
            # Memory says we are cycling. Discard the stale map and rotate.
            self.known_walls.clear()
            self.visited_cells = {self.position}
            self.repeat_count = 0
            idx = DIRECTIONS.index(self.facing)
            action = DIRECTIONS[(idx + 1) % len(DIRECTIONS)]
            self.last_action = action
            self.last_percept = dict(percept)
            return action

        # ---- MEMORY-AWARE CONDITION-ACTION RULES -------------------------
        # RULE 1: on food with a clear path -> keep going
        if food_here and not wall_ahead:
            action = self.facing

        else:
            # Rank the four directions using the internal map.
            candidates = []
            for direction in DIRECTIONS:
                target = self._target_of(direction)

                if target in self.known_walls:
                    continue                                  # known dead end
                if wall_ahead and direction == self.facing:
                    continue                                  # sensor says no

                # RULE 2: strongly prefer somewhere we have never been.
                priority = 0 if target not in self.visited_cells else 1
                candidates.append((priority, direction))

            if candidates:
                candidates.sort()
                action = candidates[0][1]
            else:
                # Fully boxed in by memory - forget the map and sweep again.
                self.known_walls.clear()
                self.visited_cells = {self.position}
                idx = DIRECTIONS.index(self.facing)
                action = DIRECTIONS[(idx + 1) % len(DIRECTIONS)]

        self.last_action = action
        self.last_percept = dict(percept)
        return action


# ---------------------------------------------------------------------------
# Practical 3 - Problem-Solving (Search) Agent
# ---------------------------------------------------------------------------
class SearchAgent:
    """
    A goal-based, problem-solving agent.

    Unlike the reflex agents above, this one does OFFLINE planning: given a
    model of the environment (Step 1.1's 'grid_size' / 'walls' / 'all_food')
    it searches for a COMPLETE sequence of actions to the nearest food BEFORE
    it moves at all, then simply executes that plan one step at a time.

    Step 1.2 - three uninformed strategies share one node-expansion skeleton;
    only the Frontier's data structure (and therefore the expansion order)
    differs:
        BFS - FIFO queue   (deque.popleft) -> shallowest node first
        DFS - LIFO stack   (list.pop)      -> deepest node first
        UCS - priority queue (heapq)       -> cheapest g(n) first

    All three keep a `reached` set (dict for UCS, so it can also remember the
    best cost seen) of every state already generated. That is what turns a
    Tree Search into a Graph Search and stops the agent from looping forever
    on a grid full of cycles.
    """

    def __init__(self):
        self.plan = []
        self.active_algo = 'BFS'      # Step 1.3: 'BFS' | 'DFS' | 'UCS'
        self.position = (0, 0)        # where the agent believes it is (dead reckoning)
        self.nodes_expanded = 0

    # -- Step 1.2a: Breadth-First Search ------------------------------------
    def bfs_search(self, start_pos, goal_pos, walls, grid_size):
        """
        Breadth-First Search over a rectangular grid.

        Args:
            start_pos: (x, y) tuple - where the agent begins.
            goal_pos:  (x, y) tuple - the target cell.
            walls:     iterable of (x, y) tuples that cannot be entered.
            grid_size: (width, height) of the grid.

        Returns:
            A list of action strings, e.g. ['Up', 'Right', 'Right'].
            [] if the agent is already standing on the goal.
            None if the goal is unreachable.
        """
        width, height = grid_size
        start = tuple(start_pos)
        goal = tuple(goal_pos)
        wall_set = {tuple(w) for w in walls}

        self.nodes_expanded = 0

        # --- Guard clauses -------------------------------------------------
        if start == goal:
            return []
        if start in wall_set or goal in wall_set:
            return None

        # --- BFS -----------------------------------------------------------
        # Frontier is a FIFO queue: this is what makes the search breadth-first
        # and therefore optimal on a uniform-cost grid.
        frontier = deque([(start, [])])
        reached = {start}

        while frontier:
            current, path = frontier.popleft()
            self.nodes_expanded += 1

            for action in DIRECTIONS:
                dx, dy = MOVES[action]
                nx, ny = current[0] + dx, current[1] + dy
                neighbour = (nx, ny)

                # Skip anything off the grid or inside a wall.
                if not (0 <= nx < width and 0 <= ny < height):
                    continue
                if neighbour in wall_set:
                    continue
                if neighbour in reached:
                    continue

                new_path = path + [action]

                # Goal test on generation - the first time BFS reaches the
                # goal it has done so by a shortest path.
                if neighbour == goal:
                    return new_path

                reached.add(neighbour)
                frontier.append((neighbour, new_path))

        # Frontier exhausted without reaching the goal => unreachable.
        return None

    # -- Step 1.2b: Depth-First Search ---------------------------------------
    def dfs_search(self, start_pos, goal_pos, walls, grid_size):
        """
        Depth-First Search over a rectangular grid.

        Same node-expansion skeleton as BFS, but the Frontier is a LIFO
        stack (list.pop() removes the LAST item added), so the search plunges
        down one branch before backtracking. It is complete (thanks to the
        `reached` set below) but NOT optimal - the first path it stumbles
        into is whatever depth-first order happened to produce, which is why
        it tends to wind, overshoot and backtrack instead of taking the
        shortest route.

        Returns: same contract as bfs_search.
        """
        width, height = grid_size
        start = tuple(start_pos)
        goal = tuple(goal_pos)
        wall_set = {tuple(w) for w in walls}

        self.nodes_expanded = 0

        if start == goal:
            return []
        if start in wall_set or goal in wall_set:
            return None

        # Frontier is a LIFO stack: list.pop() (no index) pops the last item.
        frontier = [(start, [])]
        reached = {start}

        while frontier:
            current, path = frontier.pop()
            self.nodes_expanded += 1

            # DFS goal-tests on EXPANSION (not generation), unlike BFS above -
            # that is exactly what lets it wander past shorter routes before
            # finally reaching the goal.
            if current == goal:
                return path

            for action in DIRECTIONS:
                dx, dy = MOVES[action]
                nx, ny = current[0] + dx, current[1] + dy
                neighbour = (nx, ny)

                if not (0 <= nx < width and 0 <= ny < height):
                    continue
                if neighbour in wall_set:
                    continue
                if neighbour in reached:
                    continue

                # Mark reached the moment it's generated so the same cell can
                # never be pushed onto the stack twice (Graph Search, not
                # Tree Search - without this a DFS on a grid full of cycles
                # would never terminate).
                reached.add(neighbour)
                frontier.append((neighbour, path + [action]))

        return None

    # -- Step 1.2c: Uniform-Cost Search --------------------------------------
    def ucs_search(self, start_pos, goal_pos, walls, grid_size):
        """
        Uniform-Cost Search over a rectangular grid.

        Frontier is a min-heap (heapq) ordered by the total path cost g(n).
        Every move on this grid costs 1, so UCS degenerates to BFS's shortest
        path here - but the mechanics (cost-ordered frontier, reached mapped
        to its best-known cost, re-expanding only on improvement) are exactly
        what you would need if some cells later cost more to enter than
        others.

        Returns: same contract as bfs_search.
        """
        width, height = grid_size
        start = tuple(start_pos)
        goal = tuple(goal_pos)
        wall_set = {tuple(w) for w in walls}

        self.nodes_expanded = 0

        if start == goal:
            return []
        if start in wall_set or goal in wall_set:
            return None

        # A monotonically increasing tie-breaker stops heapq from ever trying
        # to compare two (cost, node, path) tuples' node/path elements.
        tie_break = itertools.count()

        # Frontier entries: (g_cost, tie_break, node, path).
        frontier = [(0, next(tie_break), start, [])]
        reached = {start: 0}          # best g(n) found so far per state

        while frontier:
            cost, _, current, path = heapq.heappop(frontier)
            self.nodes_expanded += 1

            # A cheaper route to `current` may have been pushed after this
            # (now stale) entry - skip it rather than re-expand.
            if cost > reached.get(current, float('inf')):
                continue

            if current == goal:
                return path

            for action in DIRECTIONS:
                dx, dy = MOVES[action]
                nx, ny = current[0] + dx, current[1] + dy
                neighbour = (nx, ny)

                if not (0 <= nx < width and 0 <= ny < height):
                    continue
                if neighbour in wall_set:
                    continue

                new_cost = cost + 1     # uniform step cost
                if new_cost < reached.get(neighbour, float('inf')):
                    reached[neighbour] = new_cost
                    heapq.heappush(frontier, (new_cost, next(tie_break), neighbour, path + [action]))

        return None

    # -- helpers for driving the live environment ---------------------------
    def _closest_food(self, start_pos, food_positions):
        """Rank candidate goals by Manhattan distance so we search towards a
        sensible pellet rather than an arbitrary one."""
        if not food_positions:
            return None
        sx, sy = start_pos
        return min(
            (tuple(f) for f in food_positions),
            key=lambda f: abs(f[0] - sx) + abs(f[1] - sy),
        )

    def _search(self, start_pos, goal_pos, walls, grid_size):
        """Dispatch to whichever algorithm self.active_algo names."""
        strategies = {
            'BFS': self.bfs_search,
            'DFS': self.dfs_search,
            'UCS': self.ucs_search,
        }
        strategy = strategies.get(self.active_algo, self.bfs_search)
        return strategy(start_pos, goal_pos, walls, grid_size)

    # -- Step 1.3: form a plan, then execute it ------------------------------
    def sense_and_act(self, percept: dict) -> str:
        """
        If there is no queued plan, find the nearest food pellet from
        percept['all_food'] and run self.active_algo (BFS/DFS/UCS) to build
        a brand-new plan. Either way, pop and return the plan's next action.
        """
        if not self.plan:
            grid_size = percept.get('grid_size')
            walls = percept.get('walls', [])
            all_food = percept.get('all_food', [])

            goal = self._closest_food(self.position, all_food)

            if grid_size is None or goal is None:
                return random.choice(DIRECTIONS)   # no world model / no food yet

            new_plan = self._search(self.position, goal, walls, grid_size)
            self.plan = new_plan if new_plan else []

        if not self.plan:
            return random.choice(DIRECTIONS)       # goal unreachable - don't freeze

        action = self.plan.pop(0)

        # The plan was built from a full, accurate world model, so every
        # planned move is guaranteed to succeed - dead reckoning here is
        # exact, not an estimate as it was for ModelBasedAgent.
        dx, dy = MOVES[action]
        self.position = (self.position[0] + dx, self.position[1] + dy)
        return action


# ---------------------------------------------------------------------------
# Original base-code agent, kept for comparison (missing import now fixed)
# ---------------------------------------------------------------------------
class GreedyGridAgent:
    """The random-sweep agent supplied in the practical base code."""

    def __init__(self):
        self.actions_pool = ['Up', 'Down', 'Left', 'Right']

    def sense_and_act(self, percept: dict) -> str:
        return random.choice(self.actions_pool)