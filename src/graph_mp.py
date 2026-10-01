# This file contains functions to solve the first level of the geometrically local hierarchy applied to 2-local Hamiltonians (the marginals are supported on the edges of the Hamiltonian) using
# either one of our message passing algorithms: subgradient or Nesterov-type accelerated gradient descent with smoothed objective.
# For the subgradient algorithm we allow several variants: (1) gradient update with constant step size, (2) gradient update with constant step length, (3) gradient update modulated by the
# iteration number, and (4) accelerated gradient update based on the heavy ball method. For more details see "Subgradient Methods" by Boyd.

import numpy as np
import copy
from collections import deque
from typing import Optional


def grad1(nu: np.ndarray, step_size: float, grad: np.ndarray, iter_num: int):
    """
    Gradient update with constant step size.

    Args
    ----
    nu: numpy.array
        A Lagrange multiplier.
    step_size: float
        Step size taken in the direction of `grad`.
    grad: numpy.array
        Gradient.
    iter_num: int
        Iteration number. Not used here, it is included for convenience.
    """

    return nu + step_size * grad


def grad2(nu: np.ndarray, step_size: float, grad: np.ndarray, iter_num: int):
    """
    Gradient update with constant step length.

    Args
    ----
    nu: numpy.array
        A Lagrange multiplier.
    step_size: float
        Step size taken in the direction of `grad`.
    grad: numpy.array
        Gradient.
    iter_num: int
        Iteration number. Not used here, it is included for convenience.
    """

    grad_norm = np.linalg.norm(grad)
    if grad_norm > 1e-10:
        return nu + (step_size / grad_norm) * grad
    else:
        return nu


def grad3(nu: np.ndarray, step_size: float, grad: np.ndarray, iter_num: int):
    """
    Gradient update modulated by the iteration number.

    Args
    ----
    nu: numpy.array
        A Lagrange multiplier.
    step_size: float
        Step size taken in the direction of `grad`.
    grad: numpy.array
        Gradient.
    iter_num: int
        Current iteration number.
    """

    return nu + (step_size / iter_num) * grad


def grad_momentum(
    nu: np.ndarray,
    step_size: float,
    grad: np.ndarray,
    iter_num: int,
    velocity: np.ndarray,
    momentum_coeff: float = 0.9,
    momentum_exp: float = 0.5,
):
    """
    Gradient update with momentum (heavy ball method).

    v_t = momentum * v_{t-1} + grad
    nu_t = nu_{t-1} + step_size * v_t

    Args
    ----
    nu: np.ndarray
        Current Lagrange multiplier.
    step_size: float
        Learning rate.
    grad: np.ndarray
        Current gradient.
    iter_num: int
        Current iteration number (unused, kept for API consistency).
    velocity: np.ndarray
        Previous velocity (momentum term).
    momentum_coeff: float, default = 0.9
        Momentum coefficient (beta).
    momentum_exp: float, default = 0.5
        The exponent on the iteration modulation

    Returns
    -------
    new_nu: np.ndarray
        Updated Lagrange multiplier.
    new_velocity: np.ndarray
        Updated velocity for next iteration.
    """
    # new_velocity = momentum * velocity + grad
    # new_nu = nu + (step_size / iter_num) * new_velocity

    new_nu = (
        nu
        + (step_size / iter_num**momentum_exp) * grad
        + momentum_coeff * velocity
    )
    new_velocity = new_nu - nu

    return new_nu, new_velocity


def sample_min_eigenvector(A: np.ndarray):
    """
    Given a Hermitian matrix, diagonalize it and compute the reduced density matrix |ψ><ψ| where ψ is an eigenvector with the smallest eigenvalue.

    Args
    ----
    A: numpy.ndarray
        A single Hermitian matrix.

    Returns
    -------
    rdm: np.ndarray
        The density matrix |ψ><ψ|.
    min_eigval: float
        The smallest eigenvalue.
    """
    eigvals, eigvecs = np.linalg.eigh(A)  # Returns sorted eigenvalues
    psi = eigvecs[:, 0].reshape(
        -1, 1
    )  # Smallest eigenvalue's eigenvector as column vector
    rdm = psi @ psi.conj().T

    return rdm, eigvals[0]


def sample_min_eigenvector_batched(A_batch: np.ndarray):
    """
    Vectorized version of `sample_min_eigenvector`. Given a batch of matrices, diagonalize each and compute the RDMs of the ground state.

    Args
    -----
    A_batch: numpy.ndarray
        Array of shape (batch_size, N, N) where N is the number of nodes in the graph.

    Returns
    -------
    rdms: numpy.ndarray
        Array of shape (batch_size, N, N), containing all of the reduced density matrices formed by |ψ><ψ| where ψ is the eigenvector with smallest eigenvalue.
    min_eigvals: numpy.ndarray
        Array of shape (batch_size,) containing the smaller eigenvalues.
    """
    eigvals, eigvecs = np.linalg.eigh(
        A_batch
    )  # Shape: (batch, n), (batch, n, n)
    psi = eigvecs[
        :, :, 0
    ]  # Shape: (batch, n) - smallest eigenvalue eigenvectors (there might be many, we always get the first one)
    psi = psi[:, :, np.newaxis]  # Shape: (batch, n, 1)
    rdms = psi @ psi.conj().transpose(0, 2, 1)  # Shape: (batch, n, n)

    return rdms, eigvals[:, 0]


def compute_gibbs(A: np.ndarray, beta: float):
    """
    Compute the Gibbs state of a matrix A at inverse temperature β. Used in the smoothed algorithm.

    Assuming A is Hermitian, uses an eigendecomposition and shifts by the minimum eigenvalue before exponentiating so that
    exp(-β(λ - λ_min)) <= 1 for every eigenvalue λ regardless of how large β is. This avoids the overflow that naive exp(-β A) suffers from once β*||A|| gets large (e.g. for small `eps` in the smooth algorithm, where β = 2S/eps).

    Args
    ----
    A: numpy.ndarray
        An effective Hamiltonian matrix.
    beta: float
        The inverse temperature.

    Returns
    -------
    rdm: numpy.ndarray
        The Gibbs state exp(-β A) / Tr[exp(-β A)].
    min_eigval_smoothed: float
        The smoothed minimum eigenvalue estimate -1/β * log(Tr[exp(-β A)]).
    """
    eigvals, eigvecs = np.linalg.eigh(A)
    min_eigval = eigvals[0]

    # exp(-β(λ - λ_min)) is in (0, 1] for every eigenvalue, so this never overflows.
    shifted_weights = np.exp(-beta * (eigvals - min_eigval))
    normalization = np.sum(shifted_weights)

    rdm = (eigvecs * (shifted_weights / normalization)) @ eigvecs.conj().T

    # -1/β * log(Tr[exp(-β A)]) = λ_min - 1/β * log(Σ exp(-β(λ_i - λ_min)))  (log-sum-exp trick)
    min_eigval_smoothed = min_eigval - np.log(normalization) / beta

    return rdm, min_eigval_smoothed


def compute_gibbs_batched(A_batch: np.ndarray, beta: float):
    """
    Vectorized version of `compute_gibbs`. Given a batch of effective Hamiltonians, compute their Gibbs states at a fixed temperature. Used in the smoothed algorithm.

    Uses the same shifted log-sum-exp approach as `compute_gibbs` to avoid overflow.

    Args
    ----
    A_batch: numpy.ndarray
        Array of shape (batch_size, N, N) containing the effective Hamiltonians (each Hermitian).
    beta: float
        The inverse temperature.

    Returns
    -------
    rdms: numpy.ndarray
        Array of shape (batch_size, N, N) containing all of the Gibbs states.
    min_eigvals_smoothed: numpy.ndarray
        Array of shape (batch_size,) of smoothed minimum eigenvalue estimates.
    """
    eigvals, eigvecs = np.linalg.eigh(
        A_batch
    )  # eigvals: (batch, N), eigvecs: (batch, N, N)
    min_eigvals = eigvals[:, 0]

    shifted_weights = np.exp(
        -beta * (eigvals - min_eigvals[:, None])
    )  # (batch, N)
    normalizations = np.sum(shifted_weights, axis=-1)  # (batch,)

    rdms = np.einsum(
        "bik,bk,bjk->bij",
        eigvecs,
        shifted_weights / normalizations[:, None],
        eigvecs.conj(),
    )

    min_eigvals_smoothed = min_eigvals - np.log(normalizations) / beta

    return rdms, min_eigvals_smoothed


def random_hermitian_sparse(
    n: int, rng=None, density: float = 0.3, norm: Optional[float] = None
):
    """
    Generate a random sparse Hermitian matrix that matches Qutip's 'rand_herm()' behavior.

    Uses the uniform distribution [-1, 1] + i*[-1, 1] for each element and it then applies sparsification based on density parameter.

    Args
    ----
    n: int
        Matrix dimension
    rng: numpy.random.Generator
        Random number generator.
    density: float, default: 0.3
        Fraction of non-zero elements. Defaults to 0.3 to match Qutip
    norm: Optional[float], default = None
        Target Frobenius norm. If None, the matrix is returned unscaled.
    """
    if rng is None:
        rng = np.random.default_rng()

    # Generate random complex matrix with uniform distribution in [-1,1]
    M = (2 * rng.random((n, n)) - 1) + 1j * (2 * rng.random((n, n)) - 1)

    M = 0.5 * (M + M.conj().T)  # Make it hermitian

    # Apply sparsification to match Qutip's density behavior. Qutip removes some elements to reach target density
    target = 1 - density**0.5
    num_remove = n * (n - 0.660) * target + 0.660 * n * (1 - density)
    num_remove = max(int(num_remove), 1)

    for index in rng.choice(n * n, num_remove, replace=False):
        row, col = divmod(index, n)
        M[col, row] = 0
        M[row, col] = 0

    if norm is not None:
        M *= norm / np.linalg.norm(M, "fro")

    return M


def random_hermitian_dense(n: int, rng=None, norm: Optional[float] = None):
    """
    Generate a random dense Hermitian matrix using the standard normal distribution.

    Args
    -----
    n: int
        Matrix dimension.
    rng: numpy.random.Generator
        Random number generator.
    norm: Optional[float], default = None
        Target Frobenius norm. If None, the matrix is returned unscaled.
    """
    if rng is None:
        rng = np.random.default_rng()

    # Generate random complex matrix with standard normal distribution
    real_part = rng.standard_normal((n, n))
    imag_part = rng.standard_normal((n, n))
    A = real_part + 1j * imag_part

    M = (A + A.conj().T) / 2

    if norm is not None:
        M *= norm / np.linalg.norm(M, "fro")

    return M


def zero_multipliers(n: int, eps: float = 0.0):
    return np.zeros((n, n), dtype=complex) + np.array([[0, 0], [0, eps + 0j]])


class Node:

    def __init__(self, id: int, node_potential: np.ndarray):
        """
        A qubit.

        Args
        ----
        id: int
          The position of the node in the graph.
        node_potential: np.ndarray
             The potential function acting on this qubit (2x2 Hermitian matrix).
        """
        self.id = id
        self.edges = []  # List of Edge objects where this node appears.
        self.potential = node_potential
        self.rdm = None

    def __repr__(self):
        return f"Node({self.id})"

    def __eq__(self, other):
        if self.id == other.id:
            return True
        else:
            return False


class Edge:

    def __init__(
        self,
        node_1: Node,
        node_2: Node,
        edge_potential: np.ndarray,
    ):
        """
        An edge connecting two nodes.

        Args
        ----
        node_1, node_2: Node
            The two nodes connected by this edge.
        edge_potential: numpy.ndarray
            A 4x4 Hermitian matrix, which represents h_ij.
        """
        self.i, self.j = node_1, node_2
        self.potential = edge_potential
        self.rdm = None

        self.multipliers = {
            node_1.id: np.zeros((2, 2), dtype=complex),
            node_2.id: np.zeros((2, 2), dtype=complex),
        }

        # Variable for the momentum algorithm. These are fixed.
        self.velocity = {
            node_1.id: np.zeros((2, 2), dtype=complex),
            node_2.id: np.zeros((2, 2), dtype=complex),
        }

        # Variables for the smooth algorithm. These are fixed.
        self.multipliers_zeta = {
            node_1.id: np.zeros((2, 2), dtype=complex),
            node_2.id: np.zeros((2, 2), dtype=complex),
        }

        self.multipliers_theta = {
            node_1.id: 1 + 0j,
            node_2.id: 1 + 0j,
        }

    def __repr__(self):
        return f"Edge({self.i},{self.j})"

    def __str__(self):
        return f"({self.i.id}, {self.j.id})"

    def __eq__(self, other):
        if (self.i, self.j) == (other.i, other.j) or (self.j, self.i) == (
            other.i,
            other.j,
        ):
            return True
        else:
            return False

    def get_node_from_id(self, node_id):
        if self.i.id == node_id:
            return self.i
        else:
            return self.j

    def get_other_node_from_id(self, node_id):
        if self.i.id == node_id:
            return self.j
        elif self.j.id == node_id:
            return self.i
        else:
            raise ValueError(f"Node {node_id} is not part of this edge")

    def init_multipliers(
        self,
        multiplier_init_type: str = "zeros",
        multiplier_norm: Optional[float] = None,
        multiplier_perturbation: float = 0.0,
        multiplier_rng: Optional[np.random.Generator] = None,
    ):
        """
        Initialize the ν Lagrange multipliers, with particular different options for initialization.

        Args
        -----
        multiplier_init_type: str, default = zeros

        """

        # Different solvers may want to use different starting Lagrange multipliers.
        if multiplier_init_type == "sparse":
            initial_multipliers = random_hermitian_sparse(
                2, multiplier_rng, norm=multiplier_norm
            )
        elif multiplier_init_type == "dense":
            initial_multipliers = random_hermitian_dense(
                2, multiplier_rng, norm=multiplier_norm
            )
        elif multiplier_init_type == "zeros":
            initial_multipliers = zero_multipliers(2, multiplier_perturbation)
        else:
            raise ValueError(
                f"Unknown multiplier_init_type: {multiplier_init_type}. Use either 'sparse', 'dense', or 'zeros'."
            )

        # Initialize the Lagrange multipliers as desired
        self.multipliers[self.i.id] = initial_multipliers
        self.multipliers[self.j.id] = initial_multipliers


class InteractionGraph:
    """Main structure for the vectorized implementation of our distributed algorithm. Accepts arbitrary graphs."""

    def __init__(
        self,
        edge_list: list[tuple],
        edge_potentials: np.ndarray | list[np.ndarray],
        node_potentials: np.ndarray | list[np.ndarray],
    ):
        """
        Initialize the graph with edges and nodes.

        Args
        -----
        edge_list: list[tuple]
            List of tuples (i, j) defining edges.
        edge_potential: numpy.ndarray | list[numpy.ndarray]
            A 4x4 Hermitian matrix which represents h_ij
        node_potential: numpy.ndarray | list[numpy.ndarray]
            A 2x2 Hermitian matrix which represents h_i
        """
        self.edges = []
        self.nodes = {}

        # The distinct nodes actually named by the edges. Taken from the edge list rather
        # than assumed to be len(edge_list) + 1, which only holds for a tree.
        node_ids = sorted({n for e in edge_list for n in e})

        # If a single edge/node potential is given, broadcast it to all other edges/nodes.
        if isinstance(edge_potentials, np.ndarray):
            edge_potentials = [
                edge_potentials.copy() for _ in range(len(edge_list))
            ]
        if isinstance(node_potentials, np.ndarray):
            node_potentials = {n: node_potentials.copy() for n in node_ids}
        else:
            # Node potentials are looked up by node id, which need not be a contiguous
            # 0..N-1 range on a general graph, so key them by id rather than by position.
            node_potentials = dict(zip(node_ids, node_potentials))

        for edge_idx, (node_id_i, node_id_j) in enumerate(edge_list):

            # Return the nodes if they exist or otherwise create them.
            node_i = self.nodes.setdefault(
                node_id_i, Node(node_id_i, node_potentials[node_id_i])
            )
            node_j = self.nodes.setdefault(
                node_id_j, Node(node_id_j, node_potentials[node_id_j])
            )

            # Create and connect edge. Index the potentials by the edge's position in
            # `edge_list`, matching the edge order used by the vectorized routines below.
            edge_obj = Edge(node_i, node_j, edge_potentials[edge_idx])
            self.edges.append(edge_obj)
            node_i.edges.append(edge_obj)
            node_j.edges.append(edge_obj)

        self.num_edges = len(self.edges)
        self.num_nodes = len(self.nodes)

        # Variables for smooth algorithm.
        # L, δ, and β are initialized at a later time since they depend on ϵ.
        self.S = self.num_edges * np.log(4) + self.num_nodes * np.log(
            2
        )  # NOTE: assumes all nodes are qubits
        self.delta = 0
        self.L = 0.0
        self.beta = 0.0

        # Pre-compute identity matrices
        self.I2 = np.eye(2, dtype=complex)
        self.I4 = np.eye(4, dtype=complex)

        # Cache numpy versions of potentials
        self.edge_potentials = edge_potentials
        self.node_potentials = node_potentials

    def __repr__(self):
        cls = self.__class__.__name__
        return f"{cls}({[edge.__str__() for edge in self.edges]})"

    def _prepare_multipliers(self):
        """Collect all multipliers for vectorized edge updates."""

        multipliers_at_i = np.array(
            [e.multipliers[e.i.id] for e in self.edges]
        )  # This can be improved by already tensoring with the identity as it will later be subtracted
        multipliers_at_j = np.array([e.multipliers[e.j.id] for e in self.edges])

        return multipliers_at_i, multipliers_at_j

    def _prepare_multipliers_smooth(self):
        """Collect all 'multipliers' for vectorized edge updates. For the smooth algorithm, there are 3 such parameters of interest."""

        multipliers_at_i = [
            (
                e.multipliers[e.i.id],
                e.multipliers_zeta[e.i.id],
                e.multipliers_theta[e.i.id],
            )
            for e in self.edges
        ]  # TODO: This can be improved by already tensoring with the identity as it will later be subtracted

        multipliers_at_j = [
            (
                e.multipliers[e.j.id],
                e.multipliers_zeta[e.j.id],
                e.multipliers_theta[e.j.id],
            )
            for e in self.edges
        ]

        return multipliers_at_i, multipliers_at_j

    def update_multipliers(self, step_size: float, grad_func, iter_num: int):
        """
        Update all edge multipliers simultaneously. This processes all edges in batch operations for better performance.

        Returns
        -------
        dual_value: float
            The dual function value at this current state.
        """

        nu_is, nu_js = self._prepare_multipliers()
        num_edges = self.num_edges

        # Arrays to store new multipliers
        new_nu_is = np.zeros_like(nu_is)  # Same shape as 'mult_i'
        new_nu_js = np.zeros_like(nu_js)

        # We compute the value of the dual function at this stage.
        dual_value = 0
        node_eigvals = (
            {}
        )  # Track min eigenvalue per node to avoid double-counting
        node_rdms = {}  # Store node RDMs before multiplier update

        # ============== Update multipliers for node i on each edge using vectorization ===============

        # ----- Construct h_ij - nu_i ⊗ I - I ⊗ nu_j for each edge ----------

        # copy h_ij (a 4x4 matrix) num_edges-many times
        mats_ij_for_i = np.broadcast_to(
            self.edge_potentials, (num_edges, 4, 4)
        ).copy()

        # Subtract nu_i ⊗ I and I ⊗ nu_j for each edge. This could be simplified and vectorized.
        for e_idx in range(num_edges):
            mats_ij_for_i[e_idx] -= np.kron(nu_is[e_idx], self.I2)
            mats_ij_for_i[e_idx] -= np.kron(self.I2, nu_js[e_idx])

        mus_ij_for_i, min_eigvals_ij = sample_min_eigenvector_batched(
            mats_ij_for_i
        )  # The batched 2-qubit RDMs

        # Calculate the value of the dual function. This is the contribution from the edges.
        dual_value += np.sum(min_eigvals_ij)

        # -------- Construct h_i + \sum_{i \in N(i)} nu_i ----------------

        # Harder to vectorize because each node has different degree
        for e_idx, edge in enumerate(self.edges):
            current_node = edge.i

            # Compute mat_i
            mat_i = self.node_potentials[
                current_node.id
            ].copy()  # Here, mat_i = h_i
            for incident_edge in current_node.edges:
                mat_i += incident_edge.multipliers[
                    current_node.id
                ]  # Add the multipliers incident on that node

            mu_i, min_eigval_i = sample_min_eigenvector(mat_i)

            # Compute gradient: μ_i - Tr_j(μ_ij)
            # ν_i is in the first position, so trace out second qubit to get reduced state on first qubit
            mu_ij = mus_ij_for_i[e_idx]
            traced_mu_ij = np.trace(
                mu_ij.reshape(2, 2, 2, 2), axis1=1, axis2=3
            )  # Trace over second qubit
            grad = mu_i - traced_mu_ij

            # Apply gradient update
            nu_i = nu_is[e_idx]
            new_nu_is[e_idx] = grad_func(nu_i, step_size, grad, iter_num)

            # Store node eigenvalue and RDM (will sum once per node at the end)
            node_eigvals[current_node.id] = min_eigval_i
            node_rdms[current_node.id] = mu_i

        # ============== Update multipliers for node j on each edge using vectorization ===============

        # -------- Construct h_j + \sum_{j \in N(j)} ν_j ----------------
        # Note: We reuse mus_ij_for_i since h_ij - ν_i ⊗ I - I ⊗ ν_j is the same regardless of which node's gradient we're computing.
        # The tensor structure of h_ij is fixed (i in position 1, j in position 2).

        for e_idx, edge in enumerate(self.edges):
            current_node = edge.j

            # Compute mat_j
            mat_j = self.node_potentials[current_node.id].copy()
            for incident_edge in current_node.edges:
                mat_j += incident_edge.multipliers[current_node.id]

            mu_j, min_eigval_j = sample_min_eigenvector(mat_j)

            # Compute gradient: μ_j - Tr_i(μ_ij)
            # j is in the second tensor position, so trace out first qubit to get reduced state on j
            mu_ij = mus_ij_for_i[e_idx]
            traced_mu_ij = np.trace(
                mu_ij.reshape(2, 2, 2, 2), axis1=0, axis2=2
            )  # Trace over first qubit (node i)
            grad = mu_j - traced_mu_ij

            # Apply gradient update
            nu_j = nu_js[e_idx]
            new_nu_js[e_idx] = grad_func(nu_j, step_size, grad, iter_num)

            # Store node eigenvalue and RDM (will sum once per node at the end)
            node_eigvals[current_node.id] = min_eigval_j
            node_rdms[current_node.id] = mu_j

        # ============== Store RDMs before updating multipliers ===============
        # This ensures the RDMs correspond to the dual value computed above

        for e_idx, edge in enumerate(self.edges):
            edge.rdm = mus_ij_for_i[e_idx]

        for node_id, rdm in node_rdms.items():
            self.nodes[node_id].rdm = rdm

        # ============== Replace old multipliers with new ones ===============

        for e_idx, edge in enumerate(self.edges):
            edge.multipliers[edge.i.id] = new_nu_is[e_idx]
            edge.multipliers[edge.j.id] = new_nu_js[e_idx]

        # Add node contributions (each node counted exactly once)
        dual_value += sum(node_eigvals.values())

        return dual_value

    def update_multipliers_momentum(
        self,
        step_size: float,
        momentum_coeff: float,
        momentum_exp: float,
        iter_num: int,
    ):
        """
        Update all edge multipliers using momentum gradient descent (heavy ball method).

        Args
        ----
        step_size: float
            Learning rate for the gradient update.
        momentum_coeff: float
            Momentum coefficient (typically 0.9).
        iter_num: int
            Current iteration number.

        Returns
        -------
        dual_value: float
            The dual function value at this current state.
        """
        nu_is, nu_js = self._prepare_multipliers()
        num_edges = self.num_edges

        # Arrays to store new multipliers and velocities
        new_nu_is = np.zeros_like(nu_is)
        new_nu_js = np.zeros_like(nu_js)
        new_vel_is = np.zeros_like(nu_is)
        new_vel_js = np.zeros_like(nu_js)

        # Collect current velocities
        vel_is = np.array([e.velocity[e.i.id] for e in self.edges])
        vel_js = np.array([e.velocity[e.j.id] for e in self.edges])

        dual_value = 0
        node_eigvals = {}
        node_rdms = {}

        # Construct h_ij - ν_i ⊗ I - I ⊗ ν_j for each edge
        mats_ij_for_i = np.broadcast_to(
            self.edge_potentials, (num_edges, 4, 4)
        ).copy()
        for e_idx in range(num_edges):
            mats_ij_for_i[e_idx] -= np.kron(nu_is[e_idx], self.I2)
            mats_ij_for_i[e_idx] -= np.kron(self.I2, nu_js[e_idx])

        mus_ij_for_i, min_eigvals_ij = sample_min_eigenvector_batched(
            mats_ij_for_i
        )
        dual_value += np.sum(min_eigvals_ij)

        # Update multipliers for node i on each edge
        for e_idx, edge in enumerate(self.edges):
            current_node = edge.i

            mat_i = self.node_potentials[current_node.id].copy()
            for incident_edge in current_node.edges:
                mat_i += incident_edge.multipliers[current_node.id]

            mu_i, min_eigval_i = sample_min_eigenvector(mat_i)

            mu_ij = mus_ij_for_i[e_idx]
            traced_mu_ij = np.trace(mu_ij.reshape(2, 2, 2, 2), axis1=1, axis2=3)
            grad = mu_i - traced_mu_ij

            # Apply momentum gradient update
            new_nu_is[e_idx], new_vel_is[e_idx] = grad_momentum(
                nu_is[e_idx],
                step_size,
                grad,
                iter_num,
                vel_is[e_idx],
                momentum_coeff,
                momentum_exp,
            )

            node_eigvals[current_node.id] = min_eigval_i
            node_rdms[current_node.id] = mu_i

        # Update multipliers for node j on each edge
        for e_idx, edge in enumerate(self.edges):
            current_node = edge.j

            mat_j = self.node_potentials[current_node.id].copy()
            for incident_edge in current_node.edges:
                mat_j += incident_edge.multipliers[current_node.id]

            mu_j, min_eigval_j = sample_min_eigenvector(mat_j)

            mu_ij = mus_ij_for_i[e_idx]
            traced_mu_ij = np.trace(mu_ij.reshape(2, 2, 2, 2), axis1=0, axis2=2)
            grad = mu_j - traced_mu_ij

            # Apply momentum gradient update
            new_nu_js[e_idx], new_vel_js[e_idx] = grad_momentum(
                nu_js[e_idx],
                step_size,
                grad,
                iter_num,
                vel_js[e_idx],
                momentum_coeff,
                momentum_exp,
            )

            node_eigvals[current_node.id] = min_eigval_j
            node_rdms[current_node.id] = mu_j

        # Store RDMs
        for e_idx, edge in enumerate(self.edges):
            edge.rdm = mus_ij_for_i[e_idx]

        for node_id, rdm in node_rdms.items():
            self.nodes[node_id].rdm = rdm

        # Replace old multipliers and velocities with new ones
        for e_idx, edge in enumerate(self.edges):
            edge.multipliers[edge.i.id] = new_nu_is[e_idx]
            edge.multipliers[edge.j.id] = new_nu_js[e_idx]
            edge.velocity[edge.i.id] = new_vel_is[e_idx]
            edge.velocity[edge.j.id] = new_vel_js[e_idx]

        dual_value += sum(node_eigvals.values())

        return dual_value

    def update_multipliers_smooth(self, beta: float):
        """
        Update the Lagrange multipliers as requires in the smooth algorithm.

        NOTE: At the moment it is not storing the Gibbs state into the edges of the graph.
        """

        num_edges = self.num_edges
        dual_value = 0.0
        nu_is, nu_js = (
            self._prepare_multipliers_smooth()
        )  # This function retrieves the 4 types of 'multipliers'
        multipliers_nu_i, multipliers_zeta_i, multipliers_theta_i = (
            np.array(x) for x in zip(*nu_is)
        )
        multipliers_nu_j, multipliers_zeta_j, multipliers_theta_j = (
            np.array(x) for x in zip(*nu_js)
        )

        # Thetas are one scalar per edge so we broadcast to size (num_edges, 1, 1) for algebra below with (num_edges, 2, 2) multiplier matrices,
        # but we also use the flat (num_edges,) above for scalar algebra.
        theta_i_mat = multipliers_theta_i.reshape(-1, 1, 1)
        theta_j_mat = multipliers_theta_j.reshape(-1, 1, 1)

        # Compute the η's. Step (a) of the algorithm.
        etas_i = (
            1 - theta_i_mat
        ) * multipliers_nu_i + theta_i_mat * multipliers_zeta_i
        etas_j = (
            1 - theta_j_mat
        ) * multipliers_nu_j + theta_j_mat * multipliers_zeta_j

        # η is derived fresh each call (not persisted on Edge like ν/ζ/θ), so build a
        # local edge -> eta lookup here rather than storing it as edge state.
        # Edge defines __eq__ (by node pair) without __hash__, so it's unhashable;
        # key by id(edge) instead, which is safe since Edge objects are unique and
        # shared by identity between self.edges and each Node.edges list.
        eta_i_of_edge = {
            id(edge): etas_i[e_idx] for e_idx, edge in enumerate(self.edges)
        }
        eta_j_of_edge = {
            id(edge): etas_j[e_idx] for e_idx, edge in enumerate(self.edges)
        }

        # Step (b) of the algorithm for nodes at sites i
        # Construct h_ij - η_i ⊗ I - I ⊗ η_j for each edge
        mats_ij_for_i = np.broadcast_to(
            self.edge_potentials, (num_edges, 4, 4)
        ).copy()
        for e_idx in range(num_edges):
            mats_ij_for_i[e_idx] -= np.kron(etas_i[e_idx], self.I2)
            mats_ij_for_i[e_idx] -= np.kron(self.I2, etas_j[e_idx])

        rhos_ij_for_i, min_eigvals_ij = compute_gibbs_batched(
            mats_ij_for_i, beta
        )
        dual_value += np.sum(min_eigvals_ij)

        # Used to track the minimum eigenvalue per node (keyed by node id) to avoid double-counting nodes with degree > 1, which appear once per incident edge below.
        node_eigvals = {}

        # Update multipliers for node i on each edge. Since each node has a different amount of neighbors, they are not computed all at once.
        grads_for_i = []
        for e_idx, edge in enumerate(self.edges):
            current_node = edge.i

            mat_i = self.node_potentials[current_node.id].copy()
            for incident_edge in current_node.edges:
                mat_i += (
                    eta_i_of_edge[id(incident_edge)]
                    if incident_edge.i is current_node
                    else eta_j_of_edge[id(incident_edge)]
                )

            rho_i, min_eigval_i = compute_gibbs(mat_i, beta)

            # Step (c) of the algorithm
            rho_ij = rhos_ij_for_i[e_idx]
            traced_mu_ij = np.trace(
                rho_ij.reshape(2, 2, 2, 2), axis1=1, axis2=3
            )
            grad = rho_i - traced_mu_ij

            grads_for_i.append(grad)

            node_eigvals[current_node.id] = min_eigval_i

        # Update multipliers for node j on each edge. Since each node has a different amount of neighbors, they are not computed all at once.
        grads_for_j = []
        for e_idx, edge in enumerate(self.edges):
            current_node = edge.j

            mat_j = self.node_potentials[current_node.id].copy()
            for incident_edge in current_node.edges:
                mat_j += (
                    eta_i_of_edge[id(incident_edge)]
                    if incident_edge.i is current_node
                    else eta_j_of_edge[id(incident_edge)]
                )

            rho_j, min_eigval_j = compute_gibbs(mat_j, beta)

            # Step (c) of the algorithm
            # NOTE: j sits in the second tensor slot, so trace out the first qubit (node i).
            rho_ij = rhos_ij_for_i[e_idx]
            traced_mu_ij = np.trace(
                rho_ij.reshape(2, 2, 2, 2), axis1=0, axis2=2
            )
            grad = rho_j - traced_mu_ij

            grads_for_j.append(grad)

            node_eigvals[current_node.id] = min_eigval_j

        dual_value += sum(node_eigvals.values())

        # Step (d) of the algorithm
        multipliers_zeta_i_new = (
            multipliers_zeta_i + 1 / (theta_i_mat * self.L) * grads_for_i
        )
        multipliers_zeta_j_new = (
            multipliers_zeta_j + 1 / (theta_j_mat * self.L) * grads_for_j
        )

        # Step (e) of the algorithm
        multipliers_nu_i_new = (
            1 - theta_i_mat
        ) * multipliers_nu_i + theta_i_mat * multipliers_zeta_i_new
        multipliers_nu_j_new = (
            1 - theta_j_mat
        ) * multipliers_nu_j + theta_j_mat * multipliers_zeta_j_new

        # Step (f) of the algorithm
        multipliers_theta_i_new = (
            np.sqrt(multipliers_theta_i**4 + 4 * multipliers_theta_i**2)
            - multipliers_theta_i**2
        ) / 2
        multipliers_theta_j_new = (
            np.sqrt(multipliers_theta_j**4 + 4 * multipliers_theta_j**2)
            - multipliers_theta_j**2
        ) / 2

        # Redistribute the new multipliers back to the corresponding edges
        for e_idx, edge in enumerate(self.edges):
            # zetas
            edge.multipliers_zeta[edge.i.id] = multipliers_zeta_i_new[e_idx]
            edge.multipliers_zeta[edge.j.id] = multipliers_zeta_j_new[e_idx]

            # nus
            edge.multipliers[edge.i.id] = multipliers_nu_i_new[e_idx]
            edge.multipliers[edge.j.id] = multipliers_nu_j_new[e_idx]

            # thetas
            edge.multipliers_theta[edge.i.id] = multipliers_theta_i_new[e_idx]
            edge.multipliers_theta[edge.j.id] = multipliers_theta_j_new[e_idx]

        return dual_value

    def update_multipliers_choose_grad(
        self, step_size, solver, iter_num, momentum_coeff=0.9, momentum_exp=0.5
    ):
        """Update all the multipliers sequentially. This particular function is useful when studying the convergence of algorithms, e.g. number of iterations needed."""

        # Choose the appropriate grad function based on the 'solver'
        if solver == "1":
            grad_func = grad1
        elif solver == "2":
            grad_func = grad2
        elif solver == "3":
            grad_func = grad3
        elif solver == "momentum":
            return self.update_multipliers_momentum(
                step_size, momentum_coeff, momentum_exp, iter_num
            )
        else:
            raise Exception("Not implemented yet.")

        return self.update_multipliers(step_size, grad_func, iter_num)

    def _compute_dual_value(self):
        """Compute the dual function value without updating multipliers."""

        nu_is, nu_js = self._prepare_multipliers()
        num_edges = self.num_edges

        # Compute edge contributions: min eigenvalue of h_ij - ν_i ⊗ I - I ⊗ ν_j
        mats_ij = np.broadcast_to(
            self.edge_potentials, (num_edges, 4, 4)
        ).copy()
        for e_idx in range(num_edges):
            mats_ij[e_idx] -= np.kron(nu_is[e_idx], self.I2)
            mats_ij[e_idx] -= np.kron(self.I2, nu_js[e_idx])

        _, min_eigvals_ij = sample_min_eigenvector_batched(mats_ij)
        dual_value = np.sum(min_eigvals_ij)

        # Compute node contributions: min eigenvalue of h_i + sum of incident multipliers
        node_eigvals = {}
        for node in self.nodes.values():
            mat_i = self.node_potentials[node.id].copy()
            for edge in node.edges:
                mat_i += edge.multipliers[node.id]
            _, min_eigval_i = sample_min_eigenvector(mat_i)
            node_eigvals[node.id] = min_eigval_i

        dual_value += sum(node_eigvals.values())
        return dual_value

    def tot_energy(self):
        """Compute total (primal) energy of the system. We count the contributions on the edges and nodes."""

        energy = 0

        # Edge contributions
        for edge in self.edges:
            energy += np.trace(edge.rdm @ edge.potential).real  # Tr(μ_ij h_ij)

        # Node contributions
        for node in self.nodes.values():
            energy += np.trace(node.rdm @ node.potential).real  # Tr(μ_i h_i)

        return energy

    def initialize_multipliers(
        self,
        multiplier_init_type,
        multiplier_norm=None,
        multiplier_perturbation=0.0,
        multiplier_rng=None,
    ):
        """Initialize the Lagrange multipliers of all edges."""

        for edge in self.edges:
            edge.init_multipliers(
                multiplier_init_type,
                multiplier_norm=multiplier_norm,
                multiplier_perturbation=multiplier_perturbation,
                multiplier_rng=multiplier_rng,
            )

    def solve(
        self,
        num_iters: int,
        solver: str,
        step_size: float = 0.2,
        momentum_coeff: float = 0.9,
        momentum_exp: float = 0.5,
        smooth_eps: float = 1e-2,
        stop_tol: Optional[float] = None,
        stop_patience: int = 100,
        multiplier_init_type: Optional[str] = None,
        multiplier_norm: Optional[float] = None,
        multiplier_perturbation: float = 0.0,
        multiplier_rng: Optional[np.random.Generator] = None,
        verbose: int = 0,
    ):
        """
        Main routine. Used to compute the minimum energy of the system.

        Args
        ----
        num_iters: int
            Number of iterations to run the message passing algorithm for.
        step_size: float
            Step size for gradient updates.
        solver: str
            * "1" for constant step size
            * "2" for constant step length
            * "3" for iteration modulated
            * "momentum" for momentum/heavy ball gradient descent
            * "smooth" for Nesterov's accelerated method with smoothed function.
        momentum_coeff: float, default = 0.9
            Momentum coefficient. Only used when solver="momentum".
        momentum_exp: float, default = 0.5
            The exponent on the iteration modulation. Only used when solver="momentum".
        smooth_eps: float, default = 1e-2
            Precision parameter for the smooth solver. Roughly, it represents how far the smoothed optimum sits below the true one. If max_iters is chosen accordingly, the solution is guaranteed to fall within `smooth_eps` of the actual solution. Only used when solver="smooth".
        stop_tol: Optional[float], default = None
            Early-stopping tolerance. If given, the iteration of all algorithms stops once the dual value has not changed by more than `stop_tol` over the last `stop_patience` iterations. Useful for stopping early once we think the algorithm has converged. None disables early stopping, in which case `dual_vals` always has `num_iters` entries.
        stop_patience: int, default = 100
            Number of iterations over which the range is measured for `stop_tol`.
        multiplier_init_type: Optional[str], default = None
            Type of random initialization for multipliers:
                * "sparse": Sparse uniform in [-1,1] (matches Qutip, seems to have faster convergence)
                * "dense": Dense standard normal (slower convergence)
                * "zeros": The multipliers are 0.
                * None: choose "zeros" for solver="smooth", "sparse" otherwise.
        multiplier_norm: Optional[float], default = None
            The target Frobenius norm for the initial set of Lagrange multipliers.
        multiplier_perturbation: float
            If the multipliers are initialized to zero, this adds a small perturbation. Sometimes it happened that zero multipliers caused numerical problems.
        multiplier_rng: Optional[numpy.random.Generator], default = None
            Random number generator used for "sparse"/"dense" multiplier initialization.
        verbose: int, default = 0
            * 1 - Prints the most basic information.
            * 2 - Also prints information about the current iteration.

        Returns
        -------
        best_dual_val: float
            The best energy achieved during the calculation.
        best_iter: int
            The iteration at which the best dual value was found.
        optimal_graph: InteractionGraph
            The graph object which attained the best dual value.
        dual_vals: list[float]
            A list of the dual value achieved per iteration. Has `num_iters` entries unless there was an early stop caused by `stop_tol` and `stop_patience`.
        """
        use_momentum = solver == "momentum"
        use_smooth = solver == "smooth"

        # Initialize the multipliers depending on the input. Default multiplier init: "zeros" for the smooth algorithm, "sparse" otherwise.
        if multiplier_init_type is None:
            multiplier_init_type = "zeros" if solver == "smooth" else "sparse"

        self.initialize_multipliers(
            multiplier_init_type,
            multiplier_norm=multiplier_norm,
            multiplier_perturbation=multiplier_perturbation,
            multiplier_rng=multiplier_rng,
        )

        if use_smooth:
            # Smoothing parameter and Lipschitz constant of the smooth alg.
            self.beta = 2 * self.S / smooth_eps
            self.delta = max(
                (len(node.edges) for node in self.nodes.values()), default=0
            )  # Maximum vertex degree
            self.L = self.beta * (self.delta + 2)  # assumes a qubit

        dual_vals = []
        best_iter = 0
        optimal_graph = None

        # Compute initial dual value (sum of min eigenvalues) to be consistent with iterations
        initial_dual_val = self._compute_dual_value()

        dual_vals.append(initial_dual_val)
        best_dual_val = initial_dual_val

        # Sliding-window max/min over the last `stop_patience` iterates, used by the early-stopping mechanism below. `deques` allow for a faster implementation. Each deque holds indices into `dual_vals` whose values are monotone, so the window's extremes are always at their fronts. This keeps the test O(1) amortized instead of O(stop_patience) per iteration.
        window_max_idx: deque[int] = deque([0])
        window_min_idx: deque[int] = deque([0])

        for i in range(1, num_iters):  # i starts at 1

            if use_momentum:
                dual_val_at_iter = self.update_multipliers_momentum(
                    step_size, momentum_coeff, momentum_exp, i
                )
            elif use_smooth:
                dual_val_at_iter = self.update_multipliers_smooth(self.beta)
            else:
                dual_val_at_iter = self.update_multipliers_choose_grad(
                    step_size,
                    solver,
                    i,
                    momentum_coeff=momentum_coeff,
                    momentum_exp=momentum_exp,
                )

            if dual_val_at_iter > best_dual_val:
                best_dual_val = dual_val_at_iter
                best_iter = i
                optimal_graph = copy.deepcopy(self)

            # energies.append(energy_at_iter)
            dual_vals.append(dual_val_at_iter)

            if verbose == 2:
                print(
                    f"Iteration: {i}  {'current dual value':>10}: {dual_val_at_iter:.6f}   {'best dual value':>10}: {best_dual_val:.6f}"
                )

            # Early stopping mechanism
            # Stops when the dual value stays within `stop_tol` over the whole trailing window, i.e. max - min < stop_tol.
            if stop_tol is not None:
                while (
                    window_max_idx
                    and dual_vals[window_max_idx[-1]] <= dual_val_at_iter
                ):
                    window_max_idx.pop()
                window_max_idx.append(i)
                while (
                    window_min_idx
                    and dual_vals[window_min_idx[-1]] >= dual_val_at_iter
                ):
                    window_min_idx.pop()
                window_min_idx.append(i)

                # Drop indices that have fallen out of the trailing window.
                oldest = i - stop_patience
                if window_max_idx[0] < oldest:
                    window_max_idx.popleft()
                if window_min_idx[0] < oldest:
                    window_min_idx.popleft()

                window_range = (
                    dual_vals[window_max_idx[0]] - dual_vals[window_min_idx[0]]
                )
                if i >= stop_patience and window_range < stop_tol:
                    if verbose != 0:
                        print(
                            f"Stopping early at iteration {i}: the dual value stayed within "
                            f"{stop_tol} over the last {stop_patience} iterations."
                        )
                    break

        return best_dual_val, best_iter, optimal_graph, dual_vals
