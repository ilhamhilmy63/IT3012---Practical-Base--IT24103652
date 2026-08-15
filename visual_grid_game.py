import random

# tkinter is only needed for the GUI. Importing it lazily lets the environment
# and the agents be tested headlessly (e.g. in Google Colab or a CI runner).
try:
    import tkinter as tk
    TK_AVAILABLE = True
except ImportError:
    tk = None
    TK_AVAILABLE = False

# How each action changes an (x, y) coordinate. y increases upwards.
MOVES = {
    'Up': (0, 1),
    'Down': (0, -1),
    'Left': (-1, 0),
    'Right': (1, 0),
}


class VisualGridHuntGame:
    """A flexible Pacman-style grid environment with support for configurable opponents and larger scales."""

    def __init__(self, width=10, height=10, num_food=10, num_opponents=2, custom_walls=None):
        self.width = width
        self.height = height
        self.agent_pos = [0, 0]  # Starting position (x, y)

        if custom_walls is not None:
            self.walls = set(custom_walls)
        else:
            # Generate some default scattered walls for a larger grid
            self.walls = {(2, 2), (2, 3), (5, 5), (6, 5), (3, 7)}

        # Dynamically generate random food positions avoiding walls and agent start
        self.food_positions = set()
        while len(self.food_positions) < num_food:
            fx = random.randint(0, self.width - 1)
            fy = random.randint(0, self.height - 1)
            pos_tuple = (fx, fy)
            if pos_tuple != (0, 0) and pos_tuple not in self.walls:
                self.food_positions.add(pos_tuple)

        # Generate adversarial opponents
        self.opponents = []
        while len(self.opponents) < num_opponents:
            ox = random.randint(0, self.width - 1)
            oy = random.randint(0, self.height - 1)
            op_pos = [ox, oy]
            if tuple(op_pos) != (0, 0) and tuple(op_pos) not in self.walls and tuple(op_pos) not in self.food_positions:
                self.opponents.append(op_pos)

        self.score = 0
        self.steps = 0
        self.collision = False

        # Step 1.1: the environment now owns the agent's facing direction.
        # The agent is blind to coordinates, so it cannot track this itself.
        self.facing = 'Right'

    def _wall_ahead(self) -> bool:
        """Is the cell directly in front of the agent a wall or the grid edge?"""
        dx, dy = MOVES.get(self.facing, (0, 0))
        nx, ny = self.agent_pos[0] + dx, self.agent_pos[1] + dy

        if not (0 <= nx < self.width and 0 <= ny < self.height):
            return True
        return (nx, ny) in self.walls

    def get_percept(self) -> dict:
        """Percept for the reflex agents (Practicals 1 & 2) PLUS the world
        model needed by the goal-based SearchAgent (Practical 3, Step 1.1).

        The two local booleans ('wall_ahead', 'food_here') keep the reflex
        agents partially observable exactly as before - they cannot tell one
        corner of the grid from another. The three new keys below instead
        expose the environment's abstract state space so a planning agent can
        SIMULATE future states (BFS/DFS/UCS) before ever moving:

            'grid_size': (self.width, self.height)
            'walls':     list(self.walls)
            'all_food':  list(self.food_positions)
        """
        return {
             'wall_ahead': self._wall_ahead(),
             'food_here': tuple(self.agent_pos) in self.food_positions,

             # Practical 3 - Step 1.1
             'grid_size': (self.width, self.height),
             'walls': list(self.walls),
             'all_food': list(self.food_positions),
        }

    def execute_action(self, action: str):
        self.steps += 1
        if action in MOVES:
            self.facing = action
        new_pos = list(self.agent_pos)

        if action == 'Up':
            new_pos[1] = min(self.height - 1, new_pos[1] + 1)
        elif action == 'Down':
            new_pos[1] = max(0, new_pos[1] - 1)
        elif action == 'Left':
            new_pos[0] = max(0, new_pos[0] - 1)
        elif action == 'Right':
            new_pos[0] = min(self.width - 1, new_pos[0] + 1)

        if tuple(new_pos) in self.walls:
            self.score -= 5
        else:
            self.agent_pos = new_pos

        tuple_pos = tuple(self.agent_pos)
        if tuple_pos in self.food_positions:
            self.food_positions.remove(tuple_pos)
            self.score += 20

        for op in self.opponents:
            move = random.choice(['Up', 'Down', 'Left', 'Right', 'Stay'])
            if move == 'Up' and op[1] < self.height - 1:
                op[1] += 1
            elif move == 'Down' and op[1] > 0:
                op[1] -= 1
            elif move == 'Left' and op[0] > 0:
                op[0] -= 1
            elif move == 'Right' and op[0] < self.width - 1:
                op[0] += 1

            if op == self.agent_pos:
                self.score -= 50
                self.collision = True

    def is_done(self) -> bool:
        return len(self.food_positions) == 0 or self.steps >= 60 or self.collision


class GridGameGUI:
    """Tkinter wrapper that dynamically scales cell sizes to keep larger grids on screen."""

    def __init__(self, root, width=10, height=10, num_food=12, num_opponents=2, walls=None, agent=None):
        self.root = root
        self.agent = agent
        self.root.title("IT3012 - Scalable Multi-Agent Grid Hunt")

        self.env = VisualGridHuntGame(width=width, height=height, num_food=num_food, num_opponents=num_opponents,
                                      custom_walls=walls)

        # Dynamically calculate cell size so the total canvas fits nicely within a 600x600 window ceiling
        max_canvas_dim = 600
        self.cell_size = max(20, min(max_canvas_dim // self.env.width, max_canvas_dim // self.env.height))

        canvas_w = self.env.width * self.cell_size
        canvas_h = self.env.height * self.cell_size

        self.canvas = tk.Canvas(root, width=canvas_w, height=canvas_h, bg="white")
        self.canvas.pack()

        self.label = tk.Label(root, text="Score: 0 | Steps: 0", font=("Arial", 14))
        self.label.pack(pady=10)

        self.btn = tk.Button(root, text="Start Simulation", command=self.run_loop, font=("Arial", 12), bg="#000066",
                             fg="white")
        self.btn.pack(pady=5)

        self.draw_grid()

    def draw_grid(self):
        self.canvas.delete("all")

        for x in range(self.env.width):
            for y in range(self.env.height):
                x1 = x * self.cell_size
                y1 = (self.env.height - 1 - y) * self.cell_size
                x2 = x1 + self.cell_size
                y2 = y1 + self.cell_size

                color = "#f1f5f9" if (x, y) not in self.env.walls else "#64748b"
                self.canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline="#cbd5e1")

                # Only draw text if cell is large enough
                if self.cell_size >= 40 and (x, y) in self.env.walls:
                    self.canvas.create_text(x1 + self.cell_size / 2, y1 + self.cell_size / 2, text="W", fill="white",
                                            font=("Arial", 8, "bold"))

        for fx, fy in self.env.food_positions:
            offset = self.cell_size * 0.25
            x1 = fx * self.cell_size + offset
            y1 = (self.env.height - 1 - fy) * self.cell_size + offset
            self.canvas.create_oval(x1, y1, x1 + self.cell_size * 0.5, y1 + self.cell_size * 0.5, fill="#f59e0b",
                                    outline="#d97706")

        for ox, oy in self.env.opponents:
            offset = self.cell_size * 0.2
            x1 = ox * self.cell_size + offset
            y1 = (self.env.height - 1 - oy) * self.cell_size + offset
            self.canvas.create_rectangle(x1, y1, x1 + self.cell_size * 0.6, y1 + self.cell_size * 0.6, fill="#990000",
                                         outline="#7a0000")

        ax, ay = self.env.agent_pos
        offset = self.cell_size * 0.15
        x1 = ax * self.cell_size + offset
        y1 = (self.env.height - 1 - ay) * self.cell_size + offset
        self.canvas.create_oval(x1, y1, x1 + self.cell_size * 0.7, y1 + self.cell_size * 0.7, fill="#000066",
                                outline="#1e3a8a")

    def run_loop(self):
        self.btn.config(state="disabled")

        def step():
            if not self.env.is_done():
                if self.agent is not None:
                    percept = self.env.get_percept()
                    action = self.agent.sense_and_act(percept)
                else:
                    action = random.choice(['Up', 'Down', 'Left', 'Right'])
                self.env.execute_action(action)

                self.draw_grid()
                self.label.config(text=f"Score: {self.env.score} | Steps: {self.env.steps} | Action: {action}")
                self.root.after(250, step)
            else:
                end_text = f"Collision! Game Over! Final Score: {self.env.score}" if self.env.collision else f"Finished! Final Score: {self.env.score}"
                self.label.config(text=end_text)
                self.btn.config(state="normal")

        step()

    def get_neighbors(self, position, grid_size, walls):

     x, y = position
     width, height = grid_size

     moves = [
        ("UP", (x, y - 1)),
        ("DOWN", (x, y + 1)),
        ("LEFT", (x - 1, y)),
        ("RIGHT", (x + 1, y))
     ]

     neighbors = []

     for action, new_position in moves:

         nx, ny = new_position

         # Check map boundary
         if nx < 0 or nx >= width:
             continue

         if ny < 0 or ny >= height:
            continue

        # Check wall
         if new_position in walls:
             continue

         neighbors.append((action, new_position))

     return neighbors


if __name__ == "__main__":
    from agent import SearchAgent

    if not TK_AVAILABLE:
        raise SystemExit("tkinter is not installed - run practical2_demo.py instead.")

    root = tk.Tk()
    # Practical 3 - Step 1.3 Observation Task: swap SearchAgent().active_algo
    # between 'BFS', 'DFS' and 'UCS' (or the class default in agent.py) and
    # re-run to compare the paths. Swap in ModelBasedAgent()/SimpleReflexAgent()
    # to see the earlier reflex agents, or pass agent=None for the random walker.
    search_agent = SearchAgent()
    search_agent.active_algo = 'BFS'   # try 'DFS' / 'UCS' too
    app = GridGameGUI(root, width=12, height=12, num_food=15, num_opponents=0,
                      agent=search_agent)
    root.mainloop()