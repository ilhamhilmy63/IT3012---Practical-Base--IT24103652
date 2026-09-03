# grid_game.py

import random


class GridHuntGame:
    """A small Pacman-style grid environment (4x4) where an agent collects food."""

    def __init__(self, width=4, height=4):
        self.width = width
        self.height = height

        # Starting position of the agent (x, y)
        self.agent_pos = [0, 0]

        # Place food pellets
        # Use tuples because they can be stored inside a set
        self.food_positions = {
            (1, 2),
            (2, 3),
            (3, 0),
            (2, 1)
        }

        # Place walls
        self.walls = {
            (1, 1),
            (2, 2)
        }

        # Game state
        self.score = 0
        self.steps = 0

    def get_percept(self, agent=None) -> dict:
        """
        Return the information perceived by the agent.
        """

        return {
            'agent_pos': list(self.agent_pos),

            # True if food is at the current position
            'smells_food': tuple(self.agent_pos) in self.food_positions,

            # True if the current position is a wall
            'hit_wall': tuple(self.agent_pos) in self.walls,

            # Current score
            'score': self.score,

            # Number of food pellets remaining
            'remaining_food': len(self.food_positions)
        }

    def execute_action(self, agent=None, action: str = 'Stay'):
        """
        Execute an action selected by the agent.

        Possible actions:
        Up, Down, Left, Right
        """

        self.steps += 1

        # Start with the current position
        new_pos = list(self.agent_pos)

        # Move Up
        if action == 'Up':
            new_pos[1] = min(
                self.height - 1,
                new_pos[1] + 1
            )

        # Move Down
        elif action == 'Down':
            new_pos[1] = max(
                0,
                new_pos[1] - 1
            )

        # Move Left
        elif action == 'Left':
            new_pos[0] = max(
                0,
                new_pos[0] - 1
            )

        # Move Right
        elif action == 'Right':
            new_pos[0] = min(
                self.width - 1,
                new_pos[0] + 1
            )

        # Stay in the same position
        elif action == 'Stay':
            pass

        # Invalid action
        else:
            print(f"Invalid action: {action}")

        # Check collision with a wall
        if tuple(new_pos) in self.walls:

            # Penalty for hitting a wall
            self.score -= 5

            print("Hit a wall!")

        else:
            # Move the agent
            self.agent_pos = new_pos

        # Check whether the agent landed on food
        current_position = tuple(self.agent_pos)

        if current_position in self.food_positions:

            # Remove the food
            self.food_positions.remove(current_position)

            # Give reward
            self.score += 20

            print("Food collected! +20 points")

    def is_done(self) -> bool:
        """
        Check whether the game is finished.

        Game ends when:
        1. All food is collected, OR
        2. The agent reaches 20 steps.
        """

        return (
            len(self.food_positions) == 0
            or self.steps >= 20
        )


# ---------------------------------------------------------
# Simple test
# ---------------------------------------------------------

if __name__ == "__main__":

    game = GridHuntGame()

    print("=== Grid Hunt Game ===")
    print()

    print("Starting position:", game.agent_pos)
    print("Food positions:", game.food_positions)
    print("Walls:", game.walls)
    print()

    # Example actions
    actions = [
        'Right',
        'Up',
        'Up',
        'Right',
        'Down',
        'Right'
    ]

    for action in actions:

        print(f"Action: {action}")

        game.execute_action(action=action)

        percept = game.get_percept()

        print("Percept:", percept)
        print()

        if game.is_done():
            print("Game Over!")
            break

    print("=== Final Result ===")
    print("Position:", game.agent_pos)
    print("Score:", game.score)
    print("Steps:", game.steps)
    print("Remaining food:", len(game.food_positions))