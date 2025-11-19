import networkx as nx
import cvxpy as cp
import numpy as np


def envy_free_room_allocation(valuations: list[list[float]], rent: float, mode: int = 0, zero: float = 1e-4):
    """
    Finds a room allocation maximizing total value and calculates envy-free prices.

    Args:
        valuations: Matrix where valuations[i][j] is Player i's value for Room j.
        rent: Total rent to be divided among the rooms.
        mode: 0 = Allow negative prices. Maximize Minimum Utility.
              1 = Prices must be >= 0. Maximize Minimum Utility.
              2 = Prices must be > 0. Sequential Optimization:
                  (1) Maximize 'z' (Positivity Margin).
                  (2) Maximize Minimum Utility (subject to optimal z).
        zero: custom "zero" value, to account for floating-point errors.

    Returns:
        None (prints output).

    Examples:
        >>> # Mode 0: allow negative prices.
        >>> envy_free_room_allocation([[150, 0], [140, 10]], rent=100.0, mode=0, zero=0)
        Player 0 gets Room 0 (Value: 150.00), pays 120.00
        Player 1 gets Room 1 (Value: 10.00), pays -20.00

        >>> # Mode 1: non-negative prices (>= 0).
        >>> # Same input as above. Since P1 needs to be negative in any case, this is now infeasible.
        >>> envy_free_room_allocation([[150, 0], [140, 10]], rent=100, mode=1)
        Solution is infeasible with the current constraints.

        >>> # balanced case works in Mode 1.
        >>> envy_free_room_allocation([[100, 100], [100, 100]], rent=100, mode=1)
        Player 0 gets Room 1 (Value: 100.00), pays 50.00
        Player 1 gets Room 0 (Value: 100.00), pays 50.00

        >>> # Mode 2: strictly positive prices (> 0).
        >>> envy_free_room_allocation([[60, 30], [50, 40]], rent=100, mode=2)
        Player 0 gets Room 0 (Value: 60.00), pays 55.00
        Player 1 gets Room 1 (Value: 40.00), pays 45.00

        >>> # case where strictly positive is impossible.
        >>> # Room 1 has 0 value to everyone; it must be free (price 0) or negative.
        >>> envy_free_room_allocation([[10, 0], [10, 0]], rent=10, mode=2)
        Solution is infeasible (strictly positive prices not possible).

        >>> envy_free_room_allocation([[100, 50], [100, 50]], rent=100.0, mode=0, zero=0)
        Player 0 gets Room 1 (Value: 50.00), pays 25.00
        Player 1 gets Room 0 (Value: 100.00), pays 75.00

        >>> envy_free_room_allocation([[150, 0], [140, 10]], rent=100, mode=1)
        Solution is infeasible with the current constraints.

        >>> vals = [[2400, 2100, 2000, 1900], [2300, 2100, 1900, 2000],
        ... [2100, 2200, 2000, 2300], [2200, 1800, 2000, 1700]]
        >>> envy_free_room_allocation(vals, rent=8000, mode=2)
        Player 0 gets Room 0 (Value: 2400.00), pays 2150.00
        Player 1 gets Room 1 (Value: 2100.00), pays 1950.00
        Player 2 gets Room 3 (Value: 2300.00), pays 1950.00
        Player 3 gets Room 2 (Value: 2000.00), pays 1950.00
    """
    val_matrix = np.array(valuations)
    num_players, num_rooms = val_matrix.shape

    if num_players != num_rooms:
        raise ValueError("Number of players must equal number of rooms for one-to-one assignment.")

    # Find room-player allocation using bipartite graph
    B = nx.Graph()
    B.add_nodes_from(range(num_players), bipartite=0)  # Add players as nodes. 0 is the name of the player nodes set.
    B.add_nodes_from(range(num_players, num_players + num_rooms), bipartite=1)  # Add rooms as nodes.
    #  Note: 0 and 1 are just standard names. Could just the same be 'A' and 'B'.

    # Add edges: weight[i][j] is valuation of room j to player i.
    for p in range(num_players):
        for r_j in range(num_rooms):
            B.add_edge(p, num_players + r_j, weight=val_matrix[p][r_j])

    # Maximize sum of values
    matching = nx.max_weight_matching(B, maxcardinality=True)

    # Parse matching to map player -> room.
    # In each pair of the matching, the node that is < n is the player. The other (>= n) is the room.
    allocation = {}
    for u, v in matching:
        if u < num_players:
            allocation[u] = v - num_players
        else:
            allocation[v] = u - num_players

    # Verify one-to-one allocation
    if len(allocation) != num_players:
        print("Not all players assigned.")
        return

    # Find pricing with Linear Programming
    prices = cp.Variable(num_rooms)
    # Init constraint list; start with sum(prices) == rent
    constraints = [cp.sum(prices) == rent]
    min_utility = cp.Variable()

    # Constraints for Envy-Free:
    # For all i,j: (V = value, p = price, r = room)
    # V_i (r(i)) - p_r(i) >= V_i (r(j)) - p_r(j)
    for i in range(num_players):
        r_i = allocation[i]
        v_i = val_matrix[i][r_i]
        constraints.append((v_i - prices[r_i]) >= min_utility)
        for r_j in range(num_rooms):
            if r_j == r_i:  # no need to compare i to i
                continue
            v_j = val_matrix[i][r_j]
            constraints.append((v_i - prices[r_i]) >= (v_j - prices[r_j]))

    # Set objective based on mode
    if mode == 2:
        # First, check if strictly positive prices solution is feasible
        z = cp.Variable()
        constraints_p1 = constraints + [prices >= z]
        prob1 = cp.Problem(cp.Maximize(z), constraints_p1)
        prob1.solve()
        if prob1.status in ["infeasible", "unbounded"] or z.value is None:
            print("Solution is infeasible.")
            return
        z_opt = z.value
        if z_opt <= zero:  # to account for floating-point errors. can be set to 0 if we want.
            print("Solution is infeasible (strictly positive prices not possible).")
            return

        # next, maximize fairness with constraint prices >= z_opt
        constraints_p2 = constraints + [prices >= z_opt]
        prob2 = cp.Problem(cp.Maximize(min_utility), constraints_p2)
        prob2.solve()
        prob = prob2

    else:
        # Mode 0 or 1: Maximize Minimum Utility
        objective = cp.Maximize(min_utility)
        if mode == 1:
            constraints.append(prices >= zero)

        prob = cp.Problem(objective, constraints)
        prob.solve()

    if prob.status in ["infeasible", "unbounded"]:
        print("Solution is infeasible with the current constraints.")
        return

    final_prices = prices.value
    if final_prices is None:
        print("Error solving optimization.")
        return

    # Sort by player ID for clean output
    for i in range(num_players):
        room_idx = allocation[i]
        val = val_matrix[i][room_idx]
        price = final_prices[room_idx]
        print(f"Player {i} gets Room {room_idx} (Value: {val:.2f}), pays {price:.2f}")


if __name__ == "__main__":
    import doctest
    doctest.testmod()
