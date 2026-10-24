import numpy as np
import functools


def forward_pass_log(log_pi, log_A, log_B, observations):
    """Computes the forward probabilities (alpha) in log space.
    This is the first part of the equation below:
      p(z_t = j|x_{1:T}) = p(z_t = j|x_{1:t})p(x_{t+1:T}|z_t = j)
    """
    T = len(observations)
    # The number of hidden states
    N = log_A.shape[0]
    # Initialize with negative infinity (log(0))
    log_alpha = np.full((T, N), -np.inf)

    # Initialization (t=0)
    # Begin with the probabilities of seeing the first observation for the emissions of each hidden state
    # The observations are Bernoulli, so we can treat them as indicators
    # That means they select the correct elements of log_B
    log_alpha[0, :] = log_pi + log_B@observations[0]

    # Iterate over the history (t=1 to T-1)
    # This fills in the forward probabilities at each value
    for t in range(1, T):
        # log_B_t is the log emission probability for the observation at time t
        # That's p(z_t = j | x_t)
        log_B_t = log_B@observations[t]

        # Loop over the hidden states
        # See 17.54 through 17.58 of Murphy's book
        for j in range(N):
            # Remember, non-log space these are multiplications
            # We are computing p(z_t = j|x_{1:t}), but with a history of 1
            # log_alpha[t-1, i] + log_A[i, j] + log_B_t[j]
            # Sum log probabilities (using np.logaddexp.reduce for numerical stability)
            log_sum = np.logaddexp.reduce(log_alpha[t-1, :] + log_A[:, j])
            log_alpha[t, j] = log_sum + log_B_t[j]

    # Termination: Log-likelihood of the observation sequence
    log_likelihood = np.logaddexp.reduce(log_alpha[T-1, :])

    return log_alpha, log_likelihood

def backward_pass_log(log_A, log_B, observations):
    """Computes the backward probabilities (beta) in log space.
    This is the second part of the equation below:
      p(z_t = j|x_{1:T}) = p(z_t = j|x_{1:t})p(x_{t+1:T}|z_t = j)
    """
    T = len(observations)
    # The number of hidden states
    N = log_A.shape[0]
    # Initialize with negative infinity (log(0))
    log_beta = np.full((T, N), -np.inf)

    # Initialization (t=T-1): P(nothing | S_T-1) = 1, so log(1) = 0
    log_beta[T-1, :] = 0.0

    # Iterate from (t=T-2 down to 0)
    # This fills in the backward probabilities at each value
    for t in range(T - 2, -1, -1):
        # log_B_t_plus_1 is the log emission probability for the observation at time t+1
        # The observations are Bernoulli, so we can treat them as indicators
        # That means they select the correct elements of log_B
        log_B_t_plus_1 = log_B@observations[t+1]

        for i in range(N):
            # Remember, non-log space these are multiplications
            # We are computing p(x_{t+1:T}|z_t = j), but only looking ahead 1
            # log_A[i, j] + log_B_t_plus_1[j] + log_beta[t+1, j]
            log_sum = log_A[i, :] + log_B_t_plus_1 + log_beta[t+1, :]
            log_beta[t, i] = np.logaddexp.reduce(log_sum)

    return log_beta

def compute_expectations_log(log_alpha, log_beta, log_A, log_B, observations, log_likelihood):
    """
    Computes gamma (single-state expectation) and xi (two-state transition expectation)
    in log space.
    These are from Murphy 17.101 and 17.102
      gamma is p(z_t = j|x_{i,1:T_i},\Theta)
      xi is p(z_{t-1} = j,z_{t}=k|x_{i,1:T_i},\Theta)
    Keep in mind that T is actually just as long as our history, although we'll
    loop over all values in this function.
    """
    # T is the number of observations, N is the number of hidden states
    T, N = log_alpha.shape

    # Log P(O) is the log_likelihood from the forward pass
    log_PO = log_likelihood

    # log_gamma[t, i] = log P(St=i | O)
    # Gamma is analogous to the soft clustering via assignment from mixture models
    # alpha*beta has the forward-backward probabilities, normalized by the likelihood of the observations
    log_gamma = log_alpha + log_beta - log_PO

    # log_xi[t, i, j] = log P(St=i, St+1=j | O)
    log_xi = np.full((T - 1, N, N), -np.inf)

    for t in range(T - 1):
        # Sum of logs for the numerator (P(St=i, St+1=j, O))
        # log(alpha_t(i)) + log(A_ij) + log(B_j(O_t+1)) + log(beta_t+1(j))
        log_B_t_plus_1 = log_B@observations[t+1]

        # For all N hidden states, this is the probability from the forward pass * transition probability * emission probability * backward probabilities
        # This is 17.66 from Murphy's book
        # These are the transition probabilities over every possible transition
        # Also called the (smoothed) two-slice marginal
        log_terms = (
            log_alpha[t, :, np.newaxis] +   # log alpha_t(n),    shape (N, 1)
            log_A +                         # log A_ij,          shape (N, N)
            log_B_t_plus_1[np.newaxis, :] + # log B_j(x_{t+1}),  shape (1, N)
            log_beta[t+1, np.newaxis, :]    # log beta_{t+1}(j), shape (1, N)
        )

        # Divide (or subtract in log space) by the probability of this observation, normalizing this observation
        log_xi[t, :, :] = log_terms - log_PO

    return log_gamma, log_xi

def maximization_step(log_gamma, log_xi, observations, vocabulary):
    """
    Compute the new HMM parameters (pi_new, A_new, B_new) from the expectations.
    """
    T = len(observations)
    # Shape is T, N, N
    N = log_xi.shape[1]

    # 1. Update Initial Probabilities (pi)
    # pi'_i = gamma_0(i)
    # From 17.103 in Murphy
    log_pi = log_gamma[0, :]

    # 2. Update Transition Matrix (A)
    # Numerator: \sum_{t=0}^{T-2}(xi_t(i, j))
    # Denominator: \sum_{t=0}^{T-2}(gamma_t(i))
    # From 17.103 in Murphy
    log_xi_sum = np.logaddexp.reduce(log_xi)
    log_gamma_sum = np.logaddexp.reduce(log_gamma[:-1, :])

    # For A, we take the number of expected transitions from j to k,
    # and divide by the expected transitions from j to any other state.
    # This guarantees that each row sums to 1.

    # Use broadcasting for A_new[i, j] = xi_sum[i, j] / gamma_sum[i]
    # np.newaxis ensures the denominator broadcasts correctly across columns j
    # From 17.103 in Murphy
    log_A = log_xi_sum - log_gamma_sum[:, np.newaxis]

    # 3. Update Emission Matrix (B)
    # Numerator: sum(gamma_t(i) * I(O_t = k))
    # Denominator: sum(gamma_t(i)) from t=0 to T-1 (total expected time in state i)
    # Numerator from 17.104 in Murphy
    # It is the expectation of the multinomial, E[M_{jl}]
    # This is conceptually simple. It's the expected number of times we observe symbol l while in state j
    # We normalize by dividing by the expected number of times we are in state j
    log_gamma_sum_all_t = np.logaddexp.reduce(log_gamma)

    # If there was only one state, this would just be the expected number of
    # times we see symbol l divided by the total number of observations.
    # That would give us the emission probability into l for markov model
    # without a hidden state.

    # Populate the emission matrix. Initialize with 0 probabilities.
    log_B = np.zeros((N, len(vocabulary))) - np.inf

    for k in vocabulary:
        # Find indices where observation O_t equals k
        one_hot = np.array([1 if i == k else 0 for i in range(len(vocabulary))])
        indices_k = np.where((observations == one_hot).all(axis=1))[0]

        if len(indices_k) > 0:
            # Sum log_gamma over time t where O_t=k
            log_gamma_sum_k = np.logaddexp.reduce(log_gamma[indices_k, :])
            # log(B'_jk) = log_gamma_sum_k[j] - log_gamma_sum_all_t[j]
            log_B[:, k] = log_gamma_sum_k - log_gamma_sum_all_t
        else:
            # If an observation never appears, its log-probability will be -inf (or very small)
            #log_B[:, k] = -np.inf
            # We already initialized these to 0
            pass

    return log_pi, log_A, log_B


def train_hmm(log_A, log_B, log_pi, observations, vocabulary, n_iterations):
    """
    Train a hiddem Markov model with Baum-Welch
    """

    # Keep track of the likelihoods during training
    past_log_likelihoods = []

    while (len(past_log_likelihoods) < n_iterations):
        ##################
        # Expectation step

        # Use the forward-backward algorithm to solve for
        #    p(z_t = j|x_{1:T}) = p(z_t = j|x_{1:t})p(x_{t+1:T}|z_t = j)
        # using our current parameters.
        # This takes O(K^2T), which isn't too bad if our number of hidden states is small.
        # We also need O(KT) space, which can be worse.
        log_alpha, log_likelihood = forward_pass_log(log_pi, log_A, log_B, observations)
        log_beta = backward_pass_log(log_A, log_B, observations)
        past_log_likelihoods.append(log_likelihood)

        # See 17.5.2.1 from Murphy
        log_gamma, log_xi = compute_expectations_log(log_alpha, log_beta, log_A, log_B, observations, log_likelihood)

        ##################
        # Maximization step
        # Maximize the probability of the observations by adjusting our parameters
        log_pi, log_A, log_B = maximization_step(log_gamma, log_xi, observations, vocabulary)

        # See if we should stop training
        if len(past_log_likelihoods) > 2:
            if np.abs(past_log_likelihoods[-1] - past_log_likelihoods[-2]) < 10e-5:
                print(f"Converged after {len(past_log_likelihoods)} steps")
                break

    return log_pi, log_A, log_B, log_gamma, log_xi, log_alpha, log_beta, past_log_likelihoods


def initialize_hmm(k, vocabulary):
    """
    Arguments:
        k (int): The number of hidden states
        vocabulary (list): The vocabulary
    returns:
        A, B, pi: The transition matrix, emission matrix, and initial state estimation
    """
    # There are k hidden states. Initialize the transition matrix with evenly
    # distributed transitions
    transition_matrix = np.full((k,k), 1/k)

    # Now the emission matrix, B
    # Use the dirichlet (string cutting rv) to initialize
    # Every row will sum to 1
    emission_matrix = np.random.dirichlet(np.ones(len(vocabulary)),size=k)

    # We'll assume that all hidden states are equally likely to begin
    pi = np.full((k), 1/k)

    return transition_matrix, emission_matrix, pi
