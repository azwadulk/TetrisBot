# TetrisBot
Reinforcement learning bot adjusting weights attached to heuristics to find optimal policy 

# Evolutionary Heuristic-Driven Tetris Agent

This project implements an autonomous Tetris agent designed as a **Markov Decision Process (MDP)** solver that optimizes a **multi-objective linear value function**. By leveraging a **stochastic evolutionary strategy** and **multi-level beam search**, the agent navigates a high-dimensional state space to achieve long-horizon stability. The architecture is specifically designed to handle **non-stationary environments** via a "Rising Floor" mechanism, testing the agent’s robustness and adaptive control capabilities under dynamic constraints.

## Core Architecture

### 1. Heuristic Evaluation (Linear Value Function)
The agent evaluates the "goodness" of a resulting board state $V(s)$ using a weighted combination of five state features:

$$V(s) = w_0 \cdot \text{Lines} + w_1 \cdot \text{Holes} + w_2 \cdot \text{Bumpiness} + w_3 \cdot \text{Height} + w_4 \cdot \text{Tetris}$$

* **Lines Cleared ($w_0$)**: Directly rewards objective completion and state-space reduction.
* **Holes ($w_1$)**: Penalizes inaccessible grid cells (empty spaces with at least one block above them) that increase board entropy.
* **Bumpiness ($w_2$)**: Measures surface irregularity (absolute height difference between adjacent columns) to maintain a manageable horizon.
* **Aggregate Height ($w_3$)**: Penalizes the total height of all columns to avoid proximity to the terminal state.
* **Tetris ($w_4$)**: A specific weight to encourage high-yield, 4-line clears for maximum score efficiency.

### 2. Search Strategy: Beam Search
To mitigate the "greedy" nature of single-step optimization, the bot employs a **Beam Search** with a lookahead depth of 2:
* **Beam Width (8)**: The bot evaluates all legal permutations (rotations and positions) for the current piece and retains the top 8 candidates based on heuristic value.
* **Stochastic Lookahead**: For each candidate, it simulates the optimal placement of the next piece in the queue, accounting for future state transitions.
* **Optimization**: The agent executes the move that maximizes the expected total value (current + future) across the search horizon.

## Technical Nuances

### Well-Aware Bumpiness (Heuristic Refinement)
To promote **Tetris-ready topography**, the bumpiness algorithm includes a custom modification for "well" maintenance. It identifies the column with the minimum height and ignores height differentials associated with it. By applying a threshold (treating differences $\leq 4$ as 0 and reducing larger differences by 4), the agent is incentivized to build vertical stacks while maintaining an open well for an I-piece.



### Evolutionary Lifecycle and Robustness
The agent undergoes a three-phase lifecycle to ensure the selection of statistically significant policies:
* **Primordial**: Initial discovery phase using random weight generation to identify genomes reaching the `EVOLUTION_START_SCORE` (100,000).
* **The Gauntlet (Validation)**: To filter out "lucky" outcomes, candidates must pass a consistency test over 3 independent validation runs.
* **Civilization**: The "King" (optimal policy) undergoes continuous refinement through aggressive mutation (applying `BASE_NOISE`) and localized hill-climbing.

### Stagnation Kick (Escape from Local Optima)
To prevent the agent from converging on a sub-optimal "survival-only" policy, a stagnation detection system monitors performance plateaus. If the policy fails to improve after 10 episodes, the mutation noise scale is increased from 1.5 to 20.0, forcing a global exploration of the weight space to break out of local optima.

### Rising Floor (Dynamic Constraint Handling)
The environment simulates resource scarcity and increasing complexity via a "Rising Floor" mechanism. At set intervals (decreasing as the level increases), indestructible "Bedrock" rows are injected from the bottom, reducing the effective state space and requiring the agent to maintain high clearing efficiency to prevent terminal collision.

## Execution and Performance
The bot consistently hits scores of **250,000+** and has peaked above **300,000**.

### Controls:
* **TAB**: Toggle rendering (disable for high-speed training).
* **R**: Hard reset of the agent and weights.
* **Mouse Click**: Toggle between **Turbo** (999 FPS) and **Normal** (3 FPS) speed.
