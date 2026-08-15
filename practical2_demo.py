# practical2_demo.py
# IT3012 - Practical 02: Agent Architectures
# Headless proof for Step 1.2 / Step 1.3.
# Runs both agents on the SAME partially observable environment and shows
# that the reflex agent cycles while the model-based agent escapes.

from visual_grid_game import VisualGridHuntGame
from agent import SimpleReflexAgent, ModelBasedAgent


def run(agent, name, steps=25, seed_walls=None):
    env = VisualGridHuntGame(width=6, height=6, num_food=6,
                             num_opponents=0, custom_walls=seed_walls)
    env.agent_pos = [0, 0]

    print(f"\n=== {name} ===")
    print(" step | percept (wall_ahead, food_here) | action | true pos")
    print(" " + "-" * 58)

    history = []
    seen_cells = {tuple(env.agent_pos)}
    for _ in range(steps):
        if env.is_done():
            break
        percept = env.get_percept()
        action = agent.sense_and_act(percept)
        env.execute_action(action)

        history.append((percept['wall_ahead'], percept['food_here'], action))
        seen_cells.add(tuple(env.agent_pos))
        print(f" {env.steps:>4} | {str(percept['wall_ahead']):<5} "
              f"{str(percept['food_here']):<21} | {action:<6} | "
              f"{tuple(env.agent_pos)}")

    # Detect a repeating cycle in the last portion of the run.
    tail = history[-8:]
    looping = len(tail) == 8 and len(set(tail)) <= 2

    print(f" -> Score {env.score} | distinct cells actually visited: {len(seen_cells)}")
    print(f" -> Stuck in a repeating cycle? {'YES' if looping else 'NO'}")

    if isinstance(agent, ModelBasedAgent):
        print(f" -> Internal memory: {len(agent.visited_cells)} cells in "
              f"visited_cells, {len(agent.known_walls)} known walls")
        print(f" -> Dead-reckoned position: {agent.position}")

    return looping


def main():
    # A U-shaped trap in the bottom-left corner, plus scattered walls.
    walls = [(2, 0), (2, 2), (0, 3), (1, 3), (4, 4)]   # gap at (2,1) = the only exit

    print("=== IT3012 Practical 02: Partial Observability ===")
    print("Percept is ONLY {'wall_ahead': bool, 'food_here': bool}")
    print("Grid 6x6 | walls form a pocket the agent must reason its way out of.")

    run(SimpleReflexAgent(), "Step 1.2: SimpleReflexAgent (stateless)",
        seed_walls=walls)
    run(ModelBasedAgent(), "Step 1.3: ModelBasedAgent (internal state)",
        seed_walls=walls)


if __name__ == "__main__":
    main()
