"""
Yedidya Even-chen
Used chatGPT-5 for debugging and cvxpy tutorial
"""
import cvxpy as cp
import numpy as np
import itertools
import random


def egalitarian_allocation(values: list[list[float]], precision: int = 2) -> list[list[float]]:
    """finds an egalitarian allocation for the given resources and agents

    Args:
        values: an m by n matrix of agent valuations. values[i][j] is the value that agent i assigns to resource j.
        precision: how many decimal places to display (default is 2).

    Returns:
        Allocation matrix (n by m) where each column sums to 1.

    Raises:
        ValueError: if matrix is empty or bad (negatives, etc.).

    Examples:
        >>> egalitarian_allocation([[81,19,1],[70,1,29]], 2)
        [[0.53, 1.0, 0.0], [0.47, 0.0, 1.0]]
    """
    # input validation
    if not values or not all(len(row) == len(values[0]) for row in values):
        raise ValueError("Input must be a non-empty rectangular matrix.")

    # convert to numpy array for convenience
    V = np.array(values, dtype=float)
    n, m = V.shape

    # create X, the solution (allocations) matrix
    X = cp.Variable((n, m), nonneg=True)
    z = cp.Variable()

    # vector of total utility (allocation * value) for each agent
    utilities = cp.sum(cp.multiply(V, X), axis=1)
    # constraints: sum of each column must be 1, each utility at least z
    constraints = [cp.sum(X, axis=0) == 1, utilities >= z]

    objective = cp.Maximize(z)
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.ECOS)

    if prob.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"Solver failed with status {prob.status}")

    if np.any(X.value < -1e-6) or np.any(X.value > 1 + 1e-6):
        raise RuntimeError("Invalid solution: variable outside [0,1].")
    X_opt = np.clip(X.value, 0, 1)

    if not np.allclose(X_opt.sum(axis=0), 1, atol=1e-7):
        raise RuntimeError("Invalid allocation: columns do not sum to 1.")

    return np.round(X_opt, decimals=precision).tolist()


def leximin_egalitarian_allocation(values: list[list[float]], precision: int = 2) -> list[list[float]]:
    """Compute a leximin (lexicographically max-min) egalitarian allocation

    Args:
        values: an m by n matrix of agent valuations. values[i][j] is the value that agent i assigns to resource j.
        precision: how many decimal places to display (default is 2).

    Returns:
        Allocation matrix (n by m) where each column sums to 1.

    Raises:
        ValueError: if matrix is empty or bad (negatives, etc.).

    Examples:
        >>> weights = [[81, 19, 1], [70, 1, 29]]
        >>> alloc = leximin_egalitarian_allocation(weights, 2)
        >>> for line in format_allocation(alloc):
        ...     print(line)
        Agent #1 gets 0.53 of resource #1, 1.0 of resource #2, and 0.0 of resource #3.
        Agent #2 gets 0.47 of resource #1, 0.0 of resource #2, and 1.0 of resource #3.
    """
    if not values or not all(len(row) == len(values[0]) for row in values):
        raise ValueError("Input must be a non-empty rectangular matrix.")

    V = np.array(values, dtype=float)
    n, m = V.shape

    X = cp.Variable((n, m), nonneg=True)
    utilities = cp.sum(cp.multiply(V, X), axis=1)
    base_constraints = [cp.sum(X, axis=0) == 1]
    fixed_z = np.zeros(n)
    fixed_z_constraints = []

    # Enumerate every subset of size i for 1 to n
    for i in range(n):
        z = cp.Variable()
        constraints = (base_constraints + fixed_z_constraints.copy() +
                       [cp.sum(utilities[list(S)]) >= z for S in itertools.combinations(range(n), i + 1)])

        prob = cp.Problem(cp.Maximize(z), constraints)
        prob.solve(solver=cp.ECOS)

        if prob.status not in ("optimal", "optimal_inaccurate"):
            raise RuntimeError(f"Tier {i} solve failed: {prob.status}")

        fixed_z[i] = float(z.value)
        for S in itertools.combinations(range(n), i + 1):
            fixed_z_constraints.append(cp.sum(utilities[list(S)]) >= fixed_z[i])

    X_raw = X.value
    if X_raw is None:
        raise RuntimeError("Solver returned no solution for X.")
    if np.any(X_raw < -1e-6) or np.any(X_raw > 1 + 1e-6):
        raise RuntimeError("Invalid solution: entries outside [0,1] by >1e-6.")

    X_opt = np.clip(X_raw, 0, 1)
    if not np.allclose(X_opt.sum(axis=0), 1, atol=1e-7):
        raise RuntimeError("Invalid allocation: columns do not sum to 1.")

    return np.round(X_opt, decimals=precision).tolist()


def format_allocation(allocation: list[list[float]]) -> list[str]:
    """Convert an allocation matrix to formatted agent sentences.

    >>> x = [[0.47, 1.0, 0.0],
    ...      [0.53, 0.0, 1.0]]
    >>> format_allocation(x)[0]
    'Agent #1 gets 0.47 of resource #1, 1.0 of resource #2, and 0.0 of resource #3.'
    """
    lines = []
    for i, row in enumerate(allocation, start=1):
        parts = [f"{str(v)} of resource #{j}" for j, v in enumerate(row, start=1)]
        if len(parts) > 1:
            parts[-1] = "and " + parts[-1]
        line = f"Agent #{i} gets " + ", ".join(parts) + "."
        lines.append(line)
    return lines


def get_different_solutions(n=2, m=3, lower=None, upper=None, diff=0.0, max_iter=100, seed=42):
    """
    generates an example in which the egalitarian solution is different from the leximin-egalitarian solution

    Args:
        n: number of agents
        m: number of resources
        lower: vector (length n) of lower bounds for random values
        upper: vector (length n) of upper bounds for random values
        diff: required diff (default is 0)
        max_iter: maximum number of iterations
        seed: random seed (optional)

    Returns:
        matrix (n by m) of values for an example run of the above algorithms
    """
    if lower is None:
        lower = [i for i in range(n)]
    if upper is None:
        lower = [(i + 1) * 2 for i in range(n)]

    sol1, sol2, vals, similar, count = [], [], [], True, 0
    random.seed(seed)
    while similar and count < max_iter:
        count += 1
        vals = [[random.randint(lower[i], upper[i]) for _ in range(m)] for i in range(n)]
        sol1 = egalitarian_allocation(vals)
        sol2 = leximin_egalitarian_allocation(vals)
        for i in range(n):
            for j in range(m):
                if abs(sol1[i][j] - sol2[i][j]) > diff:
                    similar = True
                    break
    return vals if count < max_iter else None


if __name__ == '__main__':
    example = [[0, 3, 0, 5, 3], [11, 10, 13, 11, 7], [14, 15, 16, 14, 19]]
    for line in format_allocation(egalitarian_allocation(example)):
        print(line)
    print()
    for line in format_allocation(leximin_egalitarian_allocation(example)):
        print(line)
