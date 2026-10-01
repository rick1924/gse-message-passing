# This file provides all of the methods (exact + primal SDP relaxations) and helper functions necessary to compute the ground state energy in a chain of particles.
# For the moment, we assume particles are qubits.

import numpy as np
import cvxpy as cp
import itertools
from typing import Optional, Union

# ===========================================================================================================================
# --------------------------------------------------- Helper functions ------------------------------------------------------
# ===========================================================================================================================


def embed_op_in_chain(h: np.ndarray, pos: int, N: int) -> np.ndarray:
    """
    Embed a k-local operator into a chain of N sites, where each site is assumed to be a qubit, i.e. creates I ⊗ ... ⊗ I ⊗ h ⊗ I ⊗ ... ⊗ I.
    The operator h acts on sites starting at position `pos`.

    Args
    -------
    h: numpy.ndarray
        The local operator to embed.
    pos: int
        The starting position in the chain.
    N: int
        Total number of sites in the chain.

    Returns
    -------
    np.ndarray
        The embedded operator of size (local_dim^N, local_dim^N).
    """

    assert N < 12, f"A chain with size N={N} is too large. Choose N < 12."
    assert h.shape[0] in [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]

    k = int(np.log(h.shape[0]) / np.log(2))  # number of sites h acts on

    if pos == 0:
        res = np.kron(h, np.eye(2 ** (N - k)))
    else:
        res = np.eye(2**pos)
        res = np.kron(res, h)
        res = np.kron(res, np.eye(2 ** (N - pos - k)))
    return res


def chain_edges(N: int) -> list[tuple]:
    """Return a list of all the edges in a chain of length `N`."""
    return [(i, i + 1) for i in range(N - 1)]


def consecutive_chain_triples(N: int) -> list[tuple]:
    """Return a list of all triples of consecutive nodes in a chain of length `N`."""
    return [(i, i + 1, i + 2) for i in range(N - 2)]


def all_chain_triples(N: int) -> list[tuple]:
    """Return ALL triples of nodes in a chain of length `N`, i.e. all NC3 groups."""
    return list(itertools.combinations([i for i in range(N)], 3))


def partial_trace_B(
    X: np.ndarray | cp.Variable | cp.Expression, dimA: int, dimB: int
) -> Union[np.ndarray, cp.Variable, cp.Expression]:
    """
    Compute Tr_B(X) where X acts on A ⊗ B. It is Cvxpy friendly.

    Args
    ----
    X: numpy.array | cvxpy.Variable | cvxpy.Expression
        A cvxpy object of size (dimA*dimB, dimA*dimB)
    dimA: int
        Dimension of the A subsystem.
    dimB: int
        Dimension of the B subsystem.

    Returns
    -------
    The matrix of size (dimA, dimA) on subsystem A. Same data type as X.
    """

    dA, dB = dimA, dimB
    result = 0
    for kb in range(dB):
        # block indices for this kb
        rows = slice(kb, dA * dB, dB)  # i_A dB + kb
        cols = slice(kb, dA * dB, dB)  # j_A dB + kb
        result += X[rows, cols]
    return result


def partial_trace_keep(
    rho: np.ndarray, keep: list[int], dims: list[int]
) -> np.ndarray:
    """
    Partial trace of an operator keeping only the subsystems listed in 'keep'.

    Args
    -----
    rho: numpy.array
        (D, D) density matrix, D = prod(dims).
    keep: list
        Sorted list of subsystem indices to keep.
    dims:
        List of subsystem dimensions.
    """

    dims = list(dims)
    n = len(dims)
    keep = sorted(keep)
    trace_out = [i for i in range(n) if i not in keep]

    d_keep = int(np.prod([dims[i] for i in keep])) if keep else 1
    d_trace = int(np.prod([dims[i] for i in trace_out])) if trace_out else 1

    new_order = keep + trace_out
    rho_t = rho.reshape(dims + dims)
    perm = new_order + [i + n for i in new_order]
    rho_t = rho_t.transpose(perm).reshape(d_keep, d_trace, d_keep, d_trace)
    return np.einsum("ikjk->ij", rho_t)


def partial_trace_A(
    X: np.ndarray | cp.Variable | cp.Expression, dimA: int, dimB: int
) -> Union[np.ndarray, cp.Variable, cp.Expression]:
    """
    Compute Tr_A(X) where X acts on A ⊗ B. It is Cvxpy friendly.

    Args
    ----
    X: numpy.array | cvxpy.Variable | cvxpy.Expression
        A matrix of size (dimA*dimB, dimA*dimB)
    dimA: int
        The dimension of the A subsystem.
    dimB: int
        The dimension of the B subsystem.

    Returns
    -------
    The matrix of size (dimB, dimB) on subsystem B. It is the same data type as X.
    """

    dA, dB = dimA, dimB
    result = 0
    for ka in range(dA):
        rows = slice(ka * dB, (ka + 1) * dB)
        cols = slice(ka * dB, (ka + 1) * dB)
        result += X[rows, cols]
    return result


def partial_trace_middle(
    X: np.ndarray | cp.Variable | cp.Expression, d1: int, d2: int, d3: int
):
    """
    Trace out the middle subsystem for a system X of the form A ⊗ B ⊗ C.

    Args
    ----
    X: numpy.array | cvxpy.Variable | cvxpy.Expression
        A matrix of size (d1*d2*d3, d1*d2*d3).
    d1: int
        Dimension of the first subsystem.
    d2: int
        Dimension of the second subsystem.
    d3: int
        Dimension of the third subsystem.

    Returns
    -------
    The matrix of size (d1*d3, d1*d3) representing A ⊗ C.
    """

    result = 0
    for i in range(d1):
        for j in range(d1):
            for k in range(d3):
                for l in range(d3):
                    block_sum = 0
                    for m in range(d2):
                        # Index in flattened basis: i*(d2*d3) + m*d3 + k
                        row_idx = i * (d2 * d3) + m * d3 + k
                        col_idx = j * (d2 * d3) + m * d3 + l
                        block_sum += X[row_idx, col_idx]
                    # This gives element (i,k), (j,l) in A⊗C basis
                    basis = np.zeros((d1 * d3, d1 * d3))
                    basis[i * d3 + k, j * d3 + l] = 1
                    result = result + block_sum * basis
    return result


def max_edge_diff(
    mu_edges_1: list[cp.Variable], mu_edges_2: list[cp.Variable]
) -> tuple[complex, list[complex]]:
    """
    Compare two lists of density matrices using the Frobenius norm. We compute ||sum_i ρ_i - ρ'_i || where the norm is the Frobenius norm and ρ_i and ρ'_i are 2-site RDMs.

    Args
    -----
    mu_edge_1: list[cp.Variable]
        Variables corresponding to the 2-site marginals.
    mu_edge_2: list[cp.Variable]
        Variables corresponding to the 2-site marginals.
    """
    assert len(mu_edges_1) == len(mu_edges_2)

    diffs = []
    for i in range(len(mu_edges_1)):
        A = mu_edges_1[i].value
        B = mu_edges_2[i].value
        diff = np.linalg.norm(A - B, ord="fro")
        diffs.append(diff)

    return max(diffs), diffs


# ==================================================================================================================================
# --------------------------------------------- Main functions ---------------------------------------------------------------------
# ==================================================================================================================================


def exact_diagonalization(
    edges: list[tuple[int, int]],
    edge_potentials: dict[tuple[int, int], np.ndarray],
    node_potentials: Optional[dict[int, np.ndarray]] = None,
):
    """
    Exactly diagonalize the Hamiltonian H = sum_{(i,j)} h_{i,j} + sum_i h_i of a system on an arbitrary graph on qubits.

    This is the general implementation. There is also `exact_diagonalization_chain` which can be used for convenience when the graph is a chain.

    TODO: implement for qudits.

    Args
    ----------
    edges: list[tuple[int, int]]
        List of edges in the graph. Each edge is a tuple (i, j).
    edge_potentials: dict[tuple[int, int], numpy.ndarray]
        Dictionary mapping edges to their 4x4 Hamiltonian terms.
    node_potentials: Optional[dict[int, numpy.ndarray]], default=None
        Dictionary mapping node indices to their 2x2 Hamiltonian terms.
        If None, assumes zero potential on all nodes.

    Returns
    --------
    E: array[np.ndarray]
        The list of eigenvalues.
    V: array[np.ndarray]
        The list of eigenvectors.
    """

    nodes = sorted(set(n for e in edges for n in e))
    N = len(nodes)
    node_to_idx = {n: i for i, n in enumerate(nodes)}

    hamiltonian = np.zeros((2**N, 2**N), dtype=complex)

    # Add edge potentials
    for (i, j), h in edge_potentials.items():
        idx_i, idx_j = node_to_idx[i], node_to_idx[j]
        # Build the full operator: need to place h on qubits idx_i and idx_j
        # This is more complex for non-adjacent qubits
        full_op = np.zeros((2**N, 2**N), dtype=complex)
        for a in range(2):
            for b in range(2):
                for c in range(2):
                    for d in range(2):
                        # h[a*2+b, c*2+d] is the matrix element <ab|h|cd>
                        val = h[a * 2 + b, c * 2 + d]
                        if val == 0:
                            continue
                        # Build basis state indices
                        for other_bits in range(2 ** (N - 2)):
                            # Insert bits a,c at position idx_i and b,d at position idx_j
                            row_idx = 0
                            col_idx = 0
                            other_pos = 0
                            for k in range(N):
                                if k == idx_i:
                                    row_idx += a * (2 ** (N - 1 - k))
                                    col_idx += c * (2 ** (N - 1 - k))
                                elif k == idx_j:
                                    row_idx += b * (2 ** (N - 1 - k))
                                    col_idx += d * (2 ** (N - 1 - k))
                                else:
                                    bit = (other_bits >> other_pos) & 1
                                    row_idx += bit * (2 ** (N - 1 - k))
                                    col_idx += bit * (2 ** (N - 1 - k))
                                    other_pos += 1
                            full_op[row_idx, col_idx] += val
        hamiltonian += full_op

    # Add node potentials
    if node_potentials is not None:
        for n, h in node_potentials.items():
            idx = node_to_idx[n]
            # I ⊗ ... ⊗ h ⊗ ... ⊗ I
            op = np.eye(1)
            for k in range(N):
                if k == idx:
                    op = np.kron(op, h)
                else:
                    op = np.kron(op, np.eye(2, dtype=complex))
            hamiltonian += op

    E, V = np.linalg.eigh(hamiltonian)
    return E, V


def exact_diagonalization_chain(
    N: int,
    edge_potentials: np.ndarray | list[np.ndarray],
    node_potentials: Optional[np.ndarray | list[np.ndarray]] = None,
):
    """
    Exactly diagonalize the chain Hamiltonian H = sum_i h_{i,i+1}, where each h_{i,i+1} is an operator acting on the edges of a chain.

    This is a convenience wrapper around `exact_diagonalization` for the common case of a chain
    topology: it builds the edges (i, i+1) and the corresponding edge/node potential dicts, then
    delegates to the general graph implementation.

    TODO: implement for qudits.

    Args
    ----------
    N: int
        The length of the chain.
    edge_potentials: numpy.ndarray | list[numpy.ndarray]
        The list of the h_{i,i+1} terms. If only one is provided, it is taken to be the same for all edges of the chain.
    node_potentials: Optional[numpy.ndarray | list[numpy.ndarray]], default=None
        A list containing the operators h_i acting on single sites of the system. If only one is provided, it is taken to be the same for all nodes of the chain.

    Returns
    --------
    E: array[np.ndarray]
        The list of eigenvalues.
    V: array[np.ndarray]
        The list of eigenvectors.
    """

    # If only one edge potential is given, broadcast to the rest
    if isinstance(edge_potentials, np.ndarray):
        edge_potentials = [edge_potentials.copy() for _ in range(N - 1)]

    # If only one node potential is given, broadcast to the rest
    if isinstance(node_potentials, np.ndarray):
        node_potentials = [node_potentials.copy() for _ in range(N)]

    edges = [(i, i + 1) for i in range(N - 1)]
    edge_potentials_dict = dict(zip(edges, edge_potentials))
    node_potentials_dict = (
        {i: h for i, h in enumerate(node_potentials)}
        if node_potentials is not None
        else None
    )

    return exact_diagonalization(
        edges, edge_potentials_dict, node_potentials_dict
    )


def sdp_2_site_rdms(
    edges: list[tuple[int, int]],
    edge_potentials: dict[tuple[int, int], np.ndarray],
    node_potentials: Optional[dict[int, np.ndarray]] = None,
    minmax: str = "min",
    verbose: bool = False,
    solver: str = "SCS",
    solver_kwargs: Optional[dict] = None,
):
    """
    Use an SDP relaxation (primal) to compute the ground state energy of a system on an arbitrary graph. It optimizes over 2-qubit RDMs for each edge and 1-qubit RDMs for each node.

    This is the general implementation. There is also `sdp_2_site_rdms_chain` which can be used for convenience when the graph is a chain.

    Args
    ----
    edges: list[tuple[int, int]]
        List of edges in the graph. Each edge is a tuple (i, j) where i < j.
    edge_potentials: dict[tuple[int, int], numpy.ndarray]
        Dictionary mapping edges to their 4x4 Hamiltonian terms.
        Keys should be tuples (i, j) with i < j.
    node_potentials: Optional[dict[int, numpy.ndarray]], default=None
        Dictionary mapping node indices to their 2x2 Hamiltonian terms.
        If None, assumes zero potential on all nodes.
    minmax: str, default="min"
        Choose whether to maximize or minimize the energy.
    verbose: bool, default=False
        Provide information while computing.
    solver: str, default="SCS"
        The cvxpy solver to use.
    solver_kwargs: dict, optional
        Extra keyword arguments forwarded to prob.solve() (e.g. eps_abs, eps_rel for SCS).

    Returns
    -------
    energy: float
        The optimal energy value.
    mu_edges: dict[tuple[int, int], cvxpy.Variable]
        Dictionary mapping edges to their 2-site RDM variables.
    mu_nodes: dict[int, cvxpy.Variable]
        Dictionary mapping nodes to their 1-site RDM variables.

    Example
    -------
    # Triangle graph with Heisenberg interactions
    edges = [(0, 1), (1, 2), (0, 2)]
    J = 1.0
    heisenberg = J * (np.kron(X, X) + np.kron(Y, Y) + np.kron(Z, Z))
    edge_potentials = {e: heisenberg for e in edges}
    node_potentials = {i: -0.5 * Z for i in range(3)}  # magnetic field
    energy, mu_edges, mu_nodes = sdp_2_site_rdms(edges, edge_potentials, node_potentials)
    """

    edges = [tuple(sorted(e)) for e in edges]  # Normalize edges to have i < j
    edges = list(
        dict.fromkeys(edges)
    )  # Remove duplicates while preserving order

    nodes = sorted(
        set(n for e in edges for n in e)
    )  # Extract all nodes from edges

    if node_potentials is None:
        node_potentials = {}

    # Normalize edge potential keys
    edge_potentials = {tuple(sorted(k)): v for k, v in edge_potentials.items()}

    # Create cvxpy variables
    mu_edges = {e: cp.Variable((4, 4), hermitian=True) for e in edges}
    mu_nodes = {n: cp.Variable((2, 2), hermitian=True) for n in nodes}

    # Constraints
    constraints = []

    # Positivity and normalization for edge RDMs
    for e in edges:
        constraints += [mu_edges[e] >> 0]
        constraints += [cp.trace(mu_edges[e]) == 1]

    # Positivity and normalization for node RDMs
    for n in nodes:
        constraints += [mu_nodes[n] >> 0]
        constraints += [cp.trace(mu_nodes[n]) == 1]

    # Compatibility: edge RDMs must marginalize to node RDMs
    for i, j in edges:
        # Tr_j(ρ_{ij}) = ρ_i (trace out second subsystem)
        constraints += [partial_trace_B(mu_edges[(i, j)], 2, 2) == mu_nodes[i]]
        # Tr_i(ρ_{ij}) = ρ_j (trace out first subsystem)
        constraints += [partial_trace_A(mu_edges[(i, j)], 2, 2) == mu_nodes[j]]

    # Objective: sum of edge and node energies
    obj_terms = []
    for e in edges:
        if e in edge_potentials:
            obj_terms.append(cp.trace(edge_potentials[e] @ mu_edges[e]))
    for n in nodes:
        if n in node_potentials:
            obj_terms.append(cp.trace(node_potentials[n] @ mu_nodes[n]))

    if obj_terms:
        obj = cp.real(sum(obj_terms))
    else:
        obj = 0  # No potentials specified

    if minmax == "min":
        prob = cp.Problem(cp.Minimize(obj), constraints)
    elif minmax == "max":
        prob = cp.Problem(cp.Maximize(obj), constraints)
    else:
        raise ValueError(f"minmax must be 'min' or 'max', got {minmax}")

    prob.solve(solver=solver, verbose=verbose, **(solver_kwargs or {}))

    return prob.value, mu_edges, mu_nodes


def sdp_2_site_rdms_chain(
    N: int,
    edge_potentials: np.ndarray | list[np.ndarray],
    node_potentials: Optional[np.ndarray | list[np.ndarray]] = None,
    minmax: str = "min",
    verbose: bool = False,
    solver: str = "SCS",
    solver_kwargs: Optional[dict] = None,
):
    """
    Use an SDP relaxation (primal) to compute the ground state energy of a system on a chain. It may either include edge and node potentials or edge potentials only.

    This is a convenience wrapper around `sdp_2_site_rdms` for the common case of a chain topology.

    Args
    ----
    N: int
        The length of the chain.
    edge_potentials: numpy.ndarray | list[numpy.ndarray]
        A list containing the operators h_{i,i+1}.  If only one is provided, it is taken to be the same for all edges of the chain.
    node_potentials: Optional[numpy.ndarray | list[numpy.ndarray]]
        A list containing the operators h_i acting on single sites of the system.  If only one is provided, it is taken to be the same for all nodes of the chain.
    minmax: str, default="min"
        Choose whether we want to maximize or minimize the energy.
    verbose: bool, default=False
        Provide information while computing.
    solver: str, default="SCS"
        The cvxpy solver to use.
    solver_kwargs: dict, optional
        Extra keyword arguments forwarded to prob.solve() (e.g. eps_abs, eps_rel for SCS).

    Returns
    --------
    min_energy: float
        Its best estimate for the ground state energy.
    rdms: list[cvxpy.Variable] | tuple[list[cvxpy.Variable], list[cvxpy.Variable]]
        The 2-site edge RDMs if node_potentials is None, otherwise a tuple of
        (edge RDMs, node RDMs), each in chain order.
    """

    # If only one edge potential is given, broadcast to the rest
    if isinstance(edge_potentials, np.ndarray):
        edge_potentials = [edge_potentials.copy() for _ in range(N - 1)]

    # If only one node potential is given, broadcast to the rest
    if isinstance(node_potentials, np.ndarray):
        node_potentials = [node_potentials.copy() for _ in range(N)]

    edges = [(i, i + 1) for i in range(N - 1)]
    edge_potentials_dict = dict(zip(edges, edge_potentials))
    node_potentials_dict = (
        {i: h for i, h in enumerate(node_potentials)}
        if node_potentials is not None
        else None
    )

    min_energy, mu_edges, mu_nodes = sdp_2_site_rdms(
        edges,
        edge_potentials_dict,
        node_potentials_dict,
        minmax=minmax,
        verbose=verbose,
        solver=solver,
        solver_kwargs=solver_kwargs,
    )

    optimal_edge_rdms = [mu_edges[e] for e in edges]

    if node_potentials is None:
        return min_energy, optimal_edge_rdms
    else:
        optimal_node_rdms = [mu_nodes[i] for i in range(N)]
        return min_energy, (optimal_edge_rdms, optimal_node_rdms)


def sdp_3_site_rdms(
    N: int,
    edge_potentials: np.ndarray | list,
    minmax: str = "min",
    verbose: bool = False,
    eps_abs: float = 1e-7,
    eps_rel: float = 1e-7,
):
    """
    Use an SDP relaxation (primal) to compute the ground state energy of a system on a chain. It optimizes over 3-site RDMs,taken to be consecutive regions of the chain,
    e.g. [(1,2,3),(2,3,4), ...,(N-2, N-1, N)]. We impose the following constraints:
     - On a shared site, the 2-site marginals must be the same, i.e. Tr_i(ρ_{i,i+1}) = Tr_{i+2}(ρ_{i+1,i+2}) for all i.
     - The 2-site marginals are consistent with a larger 3-site state. In other words,

    Assumes that the particles are of dim=2.

    Parameters
    ----------
    N: int
        The length of the chain.
    h_edge: list
        A list containing the operators h_{i,i+1}.
    minmax: str
        Either minimize or maximize the expression: sum_{i,i+1} Tr(h_{i,j} μ_{i,j}).
    verbose: bool, default=False
        Provide information while computing.
    solver: str, default="SCS"
        The cvxpy solver to use.
    solver_kwargs: dict, optional
        Extra keyword arguments forwarded to prob.solve() (e.g. eps_abs, eps_rel for SCS).

    Returns
    -------
    prob.value: float
        The ground state energy.
    mu_pairs: list[numpy.ndarray]
        The 2-site RDMs of the ground state.
    mu_triplets: list[numpy.ndarray]
        The 3-site RDMs of the ground state.
    """

    # If given a single hamiltonian term, broadcast it to all other edges.
    if isinstance(edge_potentials, np.ndarray):
        p = [edge_potentials for _ in range(N - 1)]
        edge_potentials = p.copy()

    # Define the 2-site and 3-site RDMs
    mu_edges = [cp.Variable((4, 4), hermitian=True) for _ in range(N - 1)]
    mu_triplets = [cp.Variable((8, 8), hermitian=True) for _ in range(N - 2)]

    # Constraints
    # Positivity and normalization for all RDMs
    constraints = []
    for x in mu_edges + mu_triplets:
        constraints += [x >> 0, cp.trace(x) == 1]

    # Compatibility: on a shared qubit, the 2-site marginals must reduce to the same state
    for i in range(N - 2):
        constraints += [
            partial_trace_A(mu_edges[i], 2, 2)
            == partial_trace_B(mu_edges[i + 1], 2, 2)
        ]

    # Compatibility: 3-site RDMs must marginalize to 2-site RDMs
    for i in range(N - 2):
        # Trace out last site: (i, i+1, i+2) -> (i, i+1)
        constraints += [mu_edges[i] == partial_trace_B(mu_triplets[i], 4, 2)]

        # Trace out first site: (i, i+1, i+2) -> (i+1, i+2)
        constraints += [
            mu_edges[i + 1] == partial_trace_A(mu_triplets[i], 2, 4)
        ]

    print(len(constraints))

    obj = cp.real(
        sum(cp.trace(edge_potentials[i] @ mu_edges[i]) for i in range(N - 1))
    )
    if minmax == "min":
        prob = cp.Problem(cp.Minimize(obj), constraints)
    if minmax == "max":
        prob = cp.Problem(cp.Maximize(obj), constraints)
    prob.solve(verbose=verbose, eps_abs=eps_abs, eps_rel=eps_rel)

    return prob.value, mu_edges, mu_triplets


def sdp_3_site_rdms_v2(
    N: int,
    edge_potentials: np.ndarray | list,
    minmax: str = "min",
    verbose=False,
    eps_abs=1e-7,
    eps_rel=1e-7,
):
    """
    Use an SDP relaxation to compute the ground state energy of the Heisenberg Hamiltonian. Uses 2-site and 3-site RDMs. We assume that the particles are qubits.

    This implementation now actually constrains variables like rho_{0,2} and rho_{1,3}.

    Parameters
    ----------
    N: int
        The length of the chain.
    edge_potentials: numpy.ndarray | list[numpy.ndarray]
        A list containing the of the operators h_{i,i+1}.
    minmax: str
        Either minimize or maximize the expression: sum_{i,i+1} Tr(h_{i,j} μ_{i,j}).
    verbose: bool, default=False
        Provide information while computing.
    solver: str, default="SCS"
        The cvxpy solver to use.
    solver_kwargs: dict, optional
        Extra keyword arguments forwarded to prob.solve() (e.g. eps_abs, eps_rel for SCS).


    Returns
    -------
    prob.value: float
        The approximation to the ground state energy.
    mu_pairs: list[numpy.ndarray]
        The 2-site RDMs of the ground state.
    mu_edges: list[numpy.ndarray]
        The 2-site RDMs of the ground state that also correspond to edges in the graph.
    mu_triplets: list[numpy.ndarray]
        The 3-site RDMs of the ground state.
    """

    # If given a single hamiltonian term, broadcast it to all other edges.
    if isinstance(edge_potentials, np.ndarray):
        p = [edge_potentials for _ in range(N - 1)]
        edge_potentials = p.copy()

    edges = chain_edges(N)
    triples = consecutive_chain_triples(N)

    # Create a set of all pairs of variables coming from all triples. In other words, for a triple (i,j,k), we create all pairs from this triple: (i,j), (j,k), (i,k).
    all_pairs = set()
    for i, j, k in triples:
        all_pairs.update(
            [
                tuple(sorted((i, j))),
                tuple(sorted((i, k))),
                tuple(sorted((j, k))),
            ]  # create (i,j), (j,k), (i,k)
        )
    all_pairs = sorted(all_pairs)
    pair_index = {
        p: idx for idx, p in enumerate(all_pairs)
    }  # Hash the set of pairs
    triples_index = {
        p: idx for idx, p in enumerate(triples)
    }  # Hash the set of triples

    # Define 2-site and 3-site RDMs.
    mu_pairs = [
        cp.Variable((4, 4), hermitian=True) for _ in range(len(all_pairs))
    ]  # List of variables corresponding to all pairs in a triple; for all triples.
    mu_edges = [
        mu_pairs[pair_index[t]] for t in edges
    ]  # Select those corresponding to actual edges in the graph
    mu_triplets = [
        cp.Variable((8, 8), hermitian=True) for _ in range(len(triples))
    ]

    # Constraints
    # Positivity and normalization for all RDMs.
    constraints = []
    for x in mu_pairs + mu_triplets:
        constraints += [x >> 0, cp.trace(x) == 1]

    # Compatibility: the 2-site RDMs that lie on a chain must agree on the shared qubit.
    for i in range(N - 2):
        constraints += [
            partial_trace_A(mu_edges[i], 2, 2)
            == partial_trace_B(mu_edges[i + 1], 2, 2)
        ]

    # Compatibility: 3-site RDMs must marginalize to 2-site RDMs.
    for i, j, k in triples:

        # Trace out last site: (i, j, k) -> (i, j)
        constraints += [
            mu_pairs[pair_index[(i, j)]]
            == partial_trace_B(mu_triplets[triples_index[(i, j, k)]], 4, 2)
        ]

        # Trace out first site: (i, j, k) -> (j, k)
        constraints += [
            mu_pairs[pair_index[(j, k)]]
            == partial_trace_A(mu_triplets[triples_index[(i, j, k)]], 2, 4)
        ]

        # Trace out middle site: (i, j, k) -> (i, k)
        constraints += [
            mu_pairs[pair_index[(i, k)]]
            == partial_trace_middle(
                mu_triplets[triples_index[(i, j, k)]], 2, 2, 2
            )
        ]

    print(len(constraints))

    # Define the objective and solve.
    obj = cp.real(
        sum(
            cp.trace(edge_potentials[i] @ mu_pairs[pair_index[edges[i]]])
            for i in range(N - 1)
        )
    )
    if minmax == "min":
        prob = cp.Problem(cp.Minimize(obj), constraints)
    if minmax == "max":
        prob = cp.Problem(cp.Maximize(obj), constraints)
    prob.solve(verbose=verbose, eps_abs=eps_abs, eps_rel=eps_rel)

    mu_edges = [
        mu_pairs[pair_index[i]] for i in edges
    ]  # pick out the 2-site variables that correspond to actual edges in the chain.

    return prob.value, mu_pairs, mu_edges, mu_triplets


def sdp_3_site_rdms_v3(
    N: int,
    edge_potentials: np.ndarray | list,
    minmax: str = "min",
    verbose: bool = False,
    eps_abs: float = 1e-7,
    eps_rel: float = 1e-7,
):
    """
    Use an SDP relaxation to compute the ground state energy of a system on a chain. Uses 1-site, 2-site, and 3-site RDMs. We assume that the particles are qubits.

    This implementation now considers all possible triples in the chain, which allows the existence of 2-site marginals like ρ_{0,7}.

    Parameters
    ----------
    N: int
        The length of the chain.
    edge_potentials: numpy.ndarray | list[numpy.ndarray]
        A list containing the of the operators h_{i,i+1}.
    minmax: str
        Either minimize or maximize the expression: sum_{i,i+1} Tr(h_{i,j} μ_{i,j}).
    verbose: bool, default=False
        Provide information while computing.
    solver: str, default="SCS"
        The cvxpy solver to use.
    solver_kwargs: dict, optional
        Extra keyword arguments forwarded to prob.solve() (e.g. eps_abs, eps_rel for SCS).


    Returns
    -------
    prob.value: float
        The approximation to the ground state energy.
    mu_vertices: list[numpy.ndarray]
        The 1-site RDMs of the ground state.
    mu_pairs: list[numpy.ndarray]
        The 2-site RDMs of the ground state.
    mu_edges: list[numpy.ndarray]
        The 2-site RDMs of the ground state that also correspond to edges in the graph.
    mu_triplets: list[numpy.ndarray]
        The 3-site RDMs of the ground state.
    """

    # If given a single hamiltonian term, broadcast it to all other edges.
    if isinstance(edge_potentials, np.ndarray):
        p = [edge_potentials for _ in range(N - 1)]
        edge_potentials = p.copy()

    edges = chain_edges(N)
    triples = all_chain_triples(N)  # obtain all NC3 possible triples

    # Create a set of all pairs in the triples.
    pair_set = set()
    for i, j, k in triples:
        pair_set.update(
            [
                tuple(sorted((i, j))),
                tuple(sorted((i, k))),
                tuple(sorted((j, k))),
            ]
        )
    pairs = sorted(pair_set)
    pair_index = {p: idx for idx, p in enumerate(pairs)}
    triples_index = {p: idx for idx, p in enumerate(triples)}

    # Define 1-site, 2-site, and 3-site RDMs.
    mu_vertices = [cp.Variable((2, 2), hermitian=True) for _ in range(N)]
    mu_pairs = [cp.Variable((4, 4), hermitian=True) for _ in range(len(pairs))]
    mu_triplets = [
        cp.Variable((8, 8), hermitian=True) for _ in range(len(triples))
    ]

    # Constraints
    # Positivity and normalization for all RDMs
    constraints = []
    for x in mu_vertices + mu_pairs + mu_triplets:
        constraints += [x >> 0, cp.trace(x) == 1]

    # Compatibility: 2-site RDMs must marginalize to 1-site RDMs
    for i, j in pairs:
        constraints += [
            mu_vertices[i]
            == partial_trace_B(mu_pairs[pair_index[(i, j)]], 2, 2),
            mu_vertices[j]
            == partial_trace_A(mu_pairs[pair_index[(i, j)]], 2, 2),
        ]

    # Compatibility: 3-site RDMs must marginalize to 2-site RDMs
    for i, j, k in triples:
        # Trace out last site: (i, j, k) -> (i, j)

        constraints += [
            mu_pairs[pair_index[(i, j)]]
            == partial_trace_B(mu_triplets[triples_index[(i, j, k)]], 4, 2)
        ]

        # Trace out first site: (i, j, k) -> (j, k)
        constraints += [
            mu_pairs[pair_index[(j, k)]]
            == partial_trace_A(mu_triplets[triples_index[(i, j, k)]], 2, 4)
        ]

        # Trace out middle site: (i, j, k) -> (i, k)
        constraints += [
            mu_pairs[pair_index[(i, k)]]
            == partial_trace_middle(
                mu_triplets[triples_index[(i, j, k)]], 2, 2, 2
            )
        ]

    obj = cp.real(
        sum(
            cp.trace(edge_potentials[i] @ mu_pairs[pair_index[edges[i]]])
            for i in range(N - 1)
        )
    )
    if minmax == "min":
        prob = cp.Problem(cp.Minimize(obj), constraints)
    if minmax == "max":
        prob = cp.Problem(cp.Maximize(obj), constraints)
    prob.solve(verbose=verbose, eps_abs=eps_abs, eps_rel=eps_rel)

    mu_edges = [
        mu_pairs[pair_index[i]] for i in edges
    ]  # pick out the 2-site variables that correspond to actual edges in the chain.

    return prob.value, mu_vertices, mu_pairs, mu_edges, mu_triplets
