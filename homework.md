---
title: 💀 Домашка
order: 3
toc: true
---

### Linear algebra basics


1. [5 points] **Sensitivity Analysis in Linear Systems** Consider a nonsingular matrix $A \in \mathbb{R}^{n \times n}$ and a vector $b \in \mathbb{R}^n$. Suppose that due to measurement or computational errors, the vector $b$ is perturbed to $\tilde{b} = b + \delta b$.  
    1. Derive an upper bound for the relative error in the solution $x$ of the system $Ax = b$ in terms of the condition number $\kappa(A)$ and the relative error in $b$.  
    1. Provide a concrete example using a $2 \times 2$ matrix where $\kappa(A)$ is large (say, $\geq 100500$).

1. [5 points] **Effect of Diagonal Scaling on Rank** Let $A \in \mathbb{R}^{n \times n}$ be a matrix with rank $r$. Suppose $D \in \mathbb{R}^{n \times n}$ is a diagonal matrix. Determine the rank of the product $DA$. Explain your reasoning.

1. [8 points] **Unexpected SVD** Compute the Singular Value Decomposition (SVD) of the following matrices:
    * $A_1 = \begin{bmatrix} 2 \\ 2 \\ 8 \end{bmatrix}$
    * $A_2 = \begin{bmatrix} 0 & x \\ x & 0 \\ 0 & 0 \end{bmatrix}$, where $x$ is the sum of your birthdate numbers (day + month).

1. [10 points] **Effect of normalization on rank** Assume we have a set of data points $x^{(i)}\in\mathbb{R}^{n},\,i=1,\dots,m$, and decide to represent this data as a matrix
    $$
    X =
    \begin{pmatrix}
     | & & | \\
     x^{(1)} & \dots & x^{(m)} \\
     | & & | \\
    \end{pmatrix} \in \mathbb{R}^{n \times m}.
    $$

    We suppose that $\text{rank}\,X = r$.

    In the problem below, we ask you to find the rank of some matrix $M$ related to $X$.
    In particular, you need to find relation between $\text{rank}\,X = r$ and $\text{rank}\,M$, e.g., that the rank of $M$ is always larger/smaller than the rank of $X$ or that $\text{rank}\,M = \text{rank}\,X \big / 35$.
    Please support your answer with legitimate arguments and make the answer as accurate as possible.

    Note that border cases are possible depending on the structure of the matrix $X$. Make sure to cover them in your answer correctly.

    In applied statistics and machine learning, data is often normalized.
    One particularly popular strategy is to subtract the estimated mean $\mu$ and divide by the square root of the estimated variance $\sigma^2$. i.e.
    $$
    x \rightarrow (x - \mu) \big / \sigma.
    $$
    After the normalization, we get a new matrix
    $$
    \begin{split}
    Y &:=
    \begin{pmatrix}
     | & & | \\
     y^{(1)} & \dots & y^{(m)} \\
     | & & | \\
    \end{pmatrix},\\
    y^{(i)} &:= \frac{x^{(i)} - \frac{1}{m}\sum_{j=1}^{m} x^{(j)}}{\sigma}.
    \end{split}
    $$
    What is the rank of $Y$ if $\text{rank} \; X = r$? Here $\sigma$ is a vector and the division is element-wise. The reason for this is that different features might have different scales. Specifically:
    $$
    \sigma_i = \sqrt{\frac{1}{m}\sum_{j=1}^{m} \left(x_i^{(j)}\right)^2 - \left(\frac{1}{m}\sum_{j=1}^{m} x_i^{(j)}\right)^2}.
    $$

1. [20 points] **Image Compression with Truncated SVD** Explore image compression using Truncated Singular Value Decomposition (SVD). Understand how varying the number of singular values affects the quality of the compressed image.
    Implement a Python script to compress a grayscale image using Truncated SVD and visualize the compression quality.
    
    * **Truncated SVD**: Decomposes an image $A$ into $U, S,$ and $V$ matrices. The compressed image is reconstructed using a subset of singular values.
    * **Mathematical Representation**: 
        $$
        A \approx U_k \Sigma_k V_k^T
        $$
        * $U_k$ and $V_k$ are the first $k$ columns of $U$ and $V$, respectively.
        * $\Sigma_k$ is a diagonal matrix with the top $k$ singular values.
        * **Relative Error**: Measures the fidelity of the compressed image compared to the original. 
            $$
            \text{Relative Error} = \frac{\| A - A_k \|}{\| A \|}
            $$

    ```python
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    import numpy as np
    from skimage import io, color
    import requests
    from io import BytesIO

    def download_image(url):
        response = requests.get(url)
        img = io.imread(BytesIO(response.content))
        return color.rgb2gray(img)  # Convert to grayscale

    def update_plot(i, img_plot, error_plot, U, S, V, original_img, errors, ranks, ax1, ax2):
        # Adjust rank based on the frame index
        if i < 70:
            rank = i + 1
        else:
            rank = 70 + (i - 69) * 10

        reconstructed_img = ... # YOUR CODE HERE 

        # Calculate relative error
        relative_error = ... # YOUR CODE HERE
        errors.append(relative_error)
        ranks.append(rank)

        # Update the image plot and title
        img_plot.set_data(reconstructed_img)
        ax1.set_title(f"Image compression with SVD\n Rank {rank}; Relative error {relative_error:.2f}")

        # Remove axis ticks and labels from the first subplot (ax1)
        ax1.set_xticks([])
        ax1.set_yticks([])

        # Update the error plot
        error_plot.set_data(ranks, errors)
        ax2.set_xlim(1, len(S))
        ax2.grid(linestyle=":")
        ax2.set_ylim(1e-4, 0.5)
        ax2.set_ylabel('Relative Error')
        ax2.set_xlabel('Rank')
        ax2.set_title('Relative Error over Rank')
        ax2.semilogy()

        # Set xticks to show rank numbers
        ax2.set_xticks(range(1, len(S)+1, max(len(S)//10, 1)))  # Adjust the step size as needed
        plt.tight_layout()

        return img_plot, error_plot


    def create_animation(image, filename='svd_animation.mp4'):
        U, S, V = np.linalg.svd(image, full_matrices=False)
        errors = []
        ranks = []

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5, 8))
        img_plot = ax1.imshow(image, cmap='gray', animated=True)
        error_plot, = ax2.plot([], [], 'r-', animated=True)  # Initial empty plot for errors

        # Add watermark
        ax1.text(1, 1.02, '@fminxyz', transform=ax1.transAxes, color='gray', va='bottom', ha='right', fontsize=9)

        # Determine frames for the animation
        initial_frames = list(range(70))  # First 70 ranks
        subsequent_frames = list(range(70, len(S), 10))  # Every 10th rank after 70
        frames = initial_frames + subsequent_frames

        ani = animation.FuncAnimation(fig, update_plot, frames=len(frames), fargs=(img_plot, error_plot, U, S, V, image, errors, ranks, ax1, ax2), interval=50, blit=True)
        ani.save(filename, writer='ffmpeg', fps=8, dpi=300)

        # URL of the image
        url = ""

        # Download the image and create the animation
        image = download_image(url)
        create_animation(image)
    ```


### Convergence rates

1. [6 points] Determine (it means to prove the character of convergence if it is convergent) the convergence or divergence of a given sequences
    * $r_{k} = \frac{1}{\sqrt{k+5}}$.
    * $r_{k} = 0.101^k$.
    * $r_{k} = 0.101^{2^k}$.

1. [8 points] Let the sequence $\{r_k\}$ be defined by
    $$
    r_{k+1} = 
    \begin{cases}
    \frac{1}{2}\,r_k, & \text{if } k \text{ is even}, \\
    r_k^2, & \text{if } k \text{ is odd},
    \end{cases}
    $$
    with initial value $0 < r_0 < 1$. Prove that $\{r_k\}$ converges to 0 and analyze its convergence rate. In your answer, determine whether the overall convergence is linear, superlinear, or quadratic.

1. [6 points] Determine the following sequence $\{r_k\}$ by convergence rate (linear, sublinear, superlinear). In the case of superlinear convergence, determine whether there is quadratic convergence.
    $$
    r_k = \dfrac{1}{k!}
    $$

1. [8 points] Consider the recursive sequence defined by
    $$
    r_{k+1} = \lambda\,r_k + (1-\lambda)\,r_k^p,\quad k\ge0,
    $$
    where $\lambda\in [0,1)$ and $p>1$. Which additional conditions on $r_0$ should be satisfied for the sequence to converge? Show that when $\lambda>0$ the sequence converges to 0 with a linear rate (with asymptotic constant $\lambda$), and when $\lambda=0$ determine the convergence rate in terms of $p$. In particular, for $p=2$ decide whether the convergence is quadratic.

### Line search

1. [10 points] Consider a strongly convex quadratic function $f: \mathbb{R}^n \rightarrow \mathbb{R}$, and let us start from a point $x_k \in \mathbb{R}^n$ moving in the direction of the antigradient $-\nabla f(x_k)$, note that $\nabla f(x_k)\neq 0$. Show that the minimum of $f$ along this direction as a function of the step size $\alpha$, for a decreasing function at $x_k$, satisfies Armijo's condition for any $c_1$ in the range $0 \leq c_1 \leq \frac{1}{2}$. Specifically, demonstrate that the following inequality holds at the optimal $\alpha^*$:
    $$
    \varphi(\alpha) = f(x_{k+1}) = f(x_k - \alpha \nabla f(x_k)) \leq f(x_k) - c_1 \alpha \|\nabla f(x_k)\|_2^2
    $$

1. **Implementing and Testing Line Search Conditions in Gradient Descent** [36 points] 
    $$
    x_{k+1} = x_k - \alpha \nabla f(x_k)
    $$
    In this assignment, you will modify an existing Python code for gradient descent to include various line search conditions. You will test these modifications on two functions: a quadratic function and the Rosenbrock function. The main objectives are to understand how different line search strategies influence the convergence of the gradient descent algorithm and to compare their efficiencies based on the number of function evaluations.

    ```python
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.optimize import minimize_scalar
    np.random.seed(214)

    # Define the quadratic function and its gradient
    def quadratic_function(x, A, b):
        return 0.5 * np.dot(x.T, np.dot(A, x)) - np.dot(b.T, x)

    def grad_quadratic(x, A, b):
        return np.dot(A, x) - b

    # Generate a 2D quadratic problem with a specified condition number
    def generate_quadratic_problem(cond_number):
        # Random symmetric matrix
        M = np.random.randn(2, 2)
        M = np.dot(M, M.T)

        # Ensure the matrix has the desired condition number
        U, s, V = np.linalg.svd(M)
        s = np.linspace(cond_number, 1, len(s))  # Spread the singular values
        A = np.dot(U, np.dot(np.diag(s), V))

        # Random b
        b = np.random.randn(2)

        return A, b

    # Gradient descent function
    def gradient_descent(start_point, A, b, stepsize_func, max_iter=100):
        x = start_point.copy()
        trajectory = [x.copy()]

        for i in range(max_iter):
            grad = grad_quadratic(x, A, b)
            step_size = stepsize_func(x, grad)
            x -= step_size * grad
            trajectory.append(x.copy())

        return np.array(trajectory)

    # Backtracking line search strategy using scipy
    def backtracking_line_search(x, grad, A, b, alpha=0.3, beta=0.8):
        def objective(t):
            return quadratic_function(x - t * grad, A, b)
        res = minimize_scalar(objective, method='golden')
        return res.x

    # Generate ill-posed problem
    cond_number = 30
    A, b = generate_quadratic_problem(cond_number)

    # Starting point
    start_point = np.array([1.0, 1.8])

    # Perform gradient descent with both strategies
    trajectory_fixed = gradient_descent(start_point, A, b, lambda x, g: 5e-2)
    trajectory_backtracking = gradient_descent(start_point, A, b, lambda x, g: backtracking_line_search(x, g, A, b))

    # Plot the trajectories on a contour plot
    x1, x2 = np.meshgrid(np.linspace(-2, 2, 400), np.linspace(-2, 2, 400))
    Z = np.array([quadratic_function(np.array([x, y]), A, b) for x, y in zip(x1.flatten(), x2.flatten())]).reshape(x1.shape)

    plt.figure(figsize=(10, 8))
    plt.contour(x1, x2, Z, levels=50, cmap='viridis')
    plt.plot(trajectory_fixed[:, 0], trajectory_fixed[:, 1], 'o-', label='Fixed Step Size')
    plt.plot(trajectory_backtracking[:, 0], trajectory_backtracking[:, 1], 'o-', label='Backtracking Line Search')

    # Add markers for start and optimal points
    plt.plot(start_point[0], start_point[1], 'ro', label='Start Point')
    optimal_point = np.linalg.solve(A, b)
    plt.plot(optimal_point[0], optimal_point[1], 'y*', markersize=15, label='Optimal Point')

    plt.legend()
    plt.title('Gradient Descent Trajectories on Quadratic Function')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.savefig("linesearch.svg")
    plt.show()
    ```

    ![The code above plots this](linesearch.svg)

    Start by reviewing the provided Python code. This code implements gradient descent with a fixed step size and a backtracking line search on a quadratic function. Familiarize yourself with how the gradient descent function and the step size strategies are implemented.

    1. [10/36 points] Modify the gradient descent function to include the following line search conditions:

        1. Dichotomy 
        1. Sufficient Decrease Condition
        1. Wolfe Condition
        1. Polyak step size
            $$
            \alpha_k = \frac{f(x_k) - f^*}{\|\nabla f(x_k)\|_2^2},
            $$
            where $f^*$ is the optimal value of the function. It seems strange to use the optimal value of the function in the step size, but there are options to estimate it even without knowing the optimal value.       
        1. Sign Gradient Method:
            $$
            \alpha_k = \frac{1}{\|\nabla f(x_k)\|_2},
            $$
        Test your modified gradient descent algorithm with the implemented step size search conditions on the provided quadratic function. Plot the trajectories over iterations for each condition. Choose and specify hyperparameters for inexact line search conditions. Choose and specify the **termination criterion**. Start from the point $x_0 = (-1, 2)^T$.

    1. [8/36 points] Compare these 7 methods from the budget perspective. Plot the graph of function value from the number of function evaluations for each method on the same graph.
    1. [10/36 points] Plot trajectory for another function with the same set of methods
        $$
        f(x_1, x_2) =  10(x_2 − x_1^2)^2 + (x_1 − 1)^2
        $$
        with $x_0 = (-1, 2)^T$. You might need to adjust hyperparameters.

    1. [8/36 points] Plot the same function value from the number of function calls for this experiment.

### Matrix calculus

1. [6 points] Find the gradient $\nabla f(x)$ and hessian $f^{\prime\prime}(x)$, if $f(x) = \frac{1}{2}\Vert A - xx^T\Vert ^2_F, A \in \mathbb{S}^n$

1. [6 points] Find the gradient $\nabla f(x)$ and hessian $f''(x)$, if $f(x) = \dfrac{1}{2} \Vert Ax - b\Vert^2_2$.

1. [8 points] Find the gradient $\nabla f(x)$ and hessian $f''(x)$, if 
    $$
    f(x) = \frac1m \sum\limits_{i=1}^m \log \left( 1 + \exp(a_i^{T}x) \right) + \frac{\mu}{2}\Vert x\Vert _2^2, \; a_i, x \in \mathbb R^n, \; \mu>0
    $$

1. [8 points] Compute the gradient $\nabla_A f(A)$ of the trace of the matrix exponential function $f(A) = \text{tr}(e^A)$ with respect to $A$. Hint: Use the definition of the matrix exponential. Use the definition of the differential $df = f(A + dA) - f(A) + o(\Vert dA \Vert)$ with the limit $\Vert dA \Vert \to 0$.

1. [20 points] **Principal Component Analysis through gradient calculation.** Let there be a dataset $\{x_i\}_{i=1}^N, x_i \in \mathbb{R}^D$, which we want to transform into a dataset of reduced dimensionality $d$ using projection onto a linear subspace defined by the matrix $P \in \mathbb{R}^{D \times d}$. The orthogonal projection of a vector $x$ onto this subspace can be computed as $P(P^TP)^{-1}P^Tx$. To find the optimal matrix $P$, consider the following optimization problem:
    $$
    F(P) = \sum_{i=1}^N \|x_i - P(P^TP)^{-1}P^Tx_i\|^2 = N \cdot \text{tr}\left((I - P(P^TP)^{-1}P^T)^2 S\right) \to \min_{P \in \mathbb{R}^{D \times d}},
    $$
    where $S = \frac{1}{N} \sum_{i=1}^N x_i x_i^T$ is the sample covariance matrix for the normalized dataset.

    1. Find the gradient $\nabla_P F(P)$, calculated for an arbitrary matrix $P$ with orthogonal columns, i.e., $P : P^T P = I$. 
    
        *Hint: When calculating the differential $dF(P)$, first treat $P$ as an arbitrary matrix, and then use the orthogonality property of the columns of $P$ in the resulting expression.*
    1. Consider the eigendecomposition of the matrix $S$: 
        $$
        S = Q \Lambda Q^T,
        $$
        where $\Lambda$ is a diagonal matrix with eigenvalues on the diagonal, and $Q = [q_1 | q_2 | \ldots | q_D] \in \mathbb{R}^{D \times D}$ is an orthogonal matrix consisting of eigenvectors $q_i$ as columns. Prove the following:

        1. The gradient $\nabla_P F(P)$ equals zero for any matrix $P$ composed of $d$ distinct eigenvectors $q_i$ as its columns.
        2. The minimum value of $F(P)$ is achieved for the matrix $P$ composed of eigenvectors $q_i$ corresponding to the largest eigenvalues of $S$.


### Automatic differentiation and jax

1. **Benchmarking Hessian-Vector Product (HVP) Computation in a Neural Network via JAX** [22 points]

    You are given a simple neural network model (an MLP with several hidden layers using a nonlinearity such as GELU). The model's parameters are defined by the weights of its layers. Your task is to compare different approaches for computing the Hessian-vector product (HVP) with respect to the model's loss and to study how the computation time scales as the model grows in size.

    **Model and Loss Definition:** [2/22 points] Here is the code for the model and loss definition. Write a method `get_params()` that returns the flattened vector of all model weights.

    ```python
    import jax
    import jax.numpy as jnp
    import time
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from tqdm.auto import tqdm
    from jax.nn import gelu

    # Определение MLP модели
    class MLP:
        def __init__(self, key, layer_sizes):
            self.layer_sizes = layer_sizes
            keys = jax.random.split(key, len(layer_sizes) - 1)
            self.weights = [
                jax.random.normal(k, (layer_sizes[i], layer_sizes[i + 1]))
                for i, k in enumerate(keys)
                ]
        
        def forward(self, x):
            for w in self.weights[:-1]:
                x = gelu(jnp.dot(x, w))
            return jnp.dot(x, self.weights[-1])
        
        def get_params(self):
            ### YOUR CODE HERE ###
            return None
    ```

    **Hessian and HVP Implementations:** [2/22 points] Write a function 
    
    ```python
    # Функция для вычисления Гессиана
    def calculate_hessian(model, params):
        def loss_fn(p):
            x = jnp.ones((1, model.layer_sizes[0]))  # Заглушка входа
            return jnp.sum(model.forward(x))
        
        ### YOUR CODE HERE ###
        #hessian_fn =           
        return hessian_fn(params)
    ```
    that computes the full Hessian $H$ of the loss function with respect to the model parameters using JAX's automatic differentiation.
    
    **Naive HVP via Full Hessian:** [2/22 points] Write a function ```naive_hvp(hessian, vector)``` that, given a precomputed Hessian $H$ and a vector $v$ (of the same shape as the parameters), computes the Hessian-vector product using a straightforward matrix-vector multiplication.

    **Efficient HVP Using Autograd:** [4/22 points] Write a function 
    ```python
     def hvp(f, x, v):
         return jax.grad(lambda x: jnp.vdot(jax.grad(f)(x), v))(x)
    ```
    that directly computes the HVP without explicitly forming the full Hessian. This leverages the reverse-mode differentiation capabilities of JAX.
     
    **Timing Experiment:** Consider a family of models with an increasing number of hidden layers. 

    ```python
    ns = np.linspace(50, 1000, 15, dtype=int)  # The number of hidden layers
    num_runs = 10  # The number of runs for averaging
    ```

    For each model configuration:
    * Generate the model and extract its parameter vector.
    * Generate a random vector $v$ of the same dimension as the parameters.
    * Measure (do not forget to use ```.block_until_ready()``` to ensure accurate timing and proper synchronization) the following:
       1. **Combined Time (Full Hessian + Naive HVP):** The total time required to compute the full Hessian and then perform the matrix-vector multiplication.
       2. **Naive HVP Time (Excluding Hessian Computation):** The time required to perform the matrix-vector multiplication given a precomputed Hessian.
       3. **Efficient HVP Time:** The time required to compute the HVP using the autograd-based function.
    * Repeat each timing measurement for a fixed number of runs (e.g., 10 runs) and record both the mean and standard deviation of the computation times. 
    
    **Visualization and Analysis:** [12/22 points]
    * Plot the timing results for the three methods on the same graph. For each method, display error bars corresponding to the standard deviation over the runs.
    * Label the axes clearly (e.g., "Number of Layers" vs. "Computation Time (seconds)") and include a legend indicating which curve corresponds to which method.
    * Analyze the scaling behavior. Try analytically derive the scaling of the methods and compare it with the experimental results.

1. [15 points] We can use automatic differentiation not only to calculate necessary gradients but also for tuning hyperparameters of the algorithm like learning rate in gradient descent (with gradient descent 🤯). Suppose, we have the following function $f(x) = \frac{1}{2}\Vert x\Vert^2$, select a random point $x_0 \in \mathbb{B}^{1000} = \{0 \leq x_i \leq 1 \mid \forall i\}$. Consider $10$ steps of the gradient descent starting from the point $x_0$:
    $$
    x_{k+1} = x_k - \alpha_k \nabla f(x_k)
    $$
    Your goal in this problem is to write the function, that takes $10$ scalar values $\alpha_i$ and return the result of the gradient descent on function $L = f(x_{10})$. And optimize this function using gradient descent on $\alpha \in \mathbb{R}^{10}$. Suppose that each of $10$ components of $\alpha$ is uniformly distributed on $[0; 0.1]$.
    $$
    \alpha_{k+1} = \alpha_k - \beta \frac{\partial L}{\partial \alpha}
    $$
    Choose any constant $\beta$ and the number of steps you need. Describe the obtained results. How would you understand, that the obtained schedule ($\alpha \in \mathbb{R}^{10}$) becomes better than it was at the start? How do you check numerically local optimality in this problem? 

### Convexity

1. [10 points] Show that this function is convex.:
    $$
    f(x, y, z) = z \log \left(e^{\frac{x}{z}} + e^{\frac{y}{z}}\right) + (z - 2)^2 + e^{\frac{1}{x + y}}
    $$
    where the function $f : \mathbb{R}^3 \to \mathbb{R}$ has its domain defined as:
    $$
    \text{dom } f = \{ (x, y, z) \in \mathbb{R}^3 : x + y > 0, \, z > 0 \}.
    $$

1. [5 points] The center of mass of a body is an important concept in physics (mechanics). For a system of material points with masses $m_i$ and coordinates $x_i$, the center of mass is given by:
    $$
    x_c = \frac{\sum_{i=1}^k m_i x_i}{\sum_{i=1}^k m_i}
    $$
    The center of mass of a body does not always lie inside the body. For example, the center of mass of a doughnut is located in its hole. Prove that the center of mass of a system of material points lies in the convex hull of the set of these points.
1. [8 points] Show, that $\mathbf{conv}\{xx^\top: x \in \mathbb{R}^n, \Vert x\Vert  = 1\} = \{A \in \mathbb{S}^n_+: \text{tr}(A) = 1\}$.
1. [5 points] Prove that the set of $\{x \in \mathbb{R}^2 \mid e^{x_1}\le x_2\}$ is convex.
1. [8 points] Consider the function $f(x) = x^d$, where $x \in \mathbb{R}_{+}$. Fill the following table with ✅ or ❎. Explain your answers (with proofs).

    | $d$ | Convex | Concave | Strictly Convex | $\mu$-strongly convex |
    |:-:|:-:|:-:|:-:|:-:|
    | $-2, x \in \mathbb{R}_{++}$| | | | |
    | $-1, x \in \mathbb{R}_{++}$| | | | |
    | $0$| | | | |
    | $0.5$ | | | | |
    |$1$ | | | | |
    | $\in (1; 2)$ | | | | |
    | $2$| | | | |
    | $> 2$| | | | 
    
    : {.responsive}

1. [6 points] Prove that the entropy function, defined as
    $$
    f(x) = -\sum_{i=1}^n x_i \log(x_i),
    $$
    with $\text{dom}(f) = \{x \in \R^n_{++} : \sum_{i=1}^n x_i = 1\}$, is strictly concave.  

1. [8 points] Show that the maximum of a convex function $f$ over the polyhedron $P = \text{conv}\{v_1, \ldots, v_k\}$ is achieved at one of its vertices, i.e.,
    $$
    \sup_{x \in P} f(x) = \max_{i=1, \ldots, k} f(v_i).
    $$

    A stronger statement is: the maximum of a convex function over a closed bounded convex set is achieved at an extreme point, i.e., a point in the set that is not a convex combination of any other points in the set. (you do not have to prove it). *Hint:* Assume the statement is false, and use Jensen's inequality.

1. [6 points] Show, that the two definitions of $\mu$-strongly convex functions are equivalent:
    1. $f(x)$ is $\mu$-strongly convex $\iff$ for any $x_1, x_2 \in S$ and $0 \le \lambda \le 1$ for some $\mu > 0$:
        $$
        f(\lambda x_1 + (1 - \lambda)x_2) \le \lambda f(x_1) + (1 - \lambda)f(x_2) - \frac{\mu}{2} \lambda (1 - \lambda)\|x_1 - x_2\|^2
        $$

    1. $f(x)$ is $\mu$-strongly convex $\iff$ if there exists $\mu>0$ such that the function $f(x) - \dfrac{\mu}{2}\Vert x\Vert^2$ is convex.

### Optimality conditions. KKT. Duality

In this section, you can consider either the arbitrary norm or the Euclidian norm if nothing else is specified.

1. **Toy example** [10 points] 
    $$
    \begin{split}
    & x^2 + 1 \to \min\limits_{x \in \mathbb{R} }\\
    \text{s.t. } & (x-2)(x-4) \leq 0
    \end{split}
    $$

    1. Give the feasible set, the optimal value, and the optimal solution.
    1.  Plot the objective $x^2 +1$ versus $x$. On the same plot, show the feasible set, optimal point, and value, and plot the Lagrangian $L(x,\mu)$ versus $x$ for a few positive values of $\mu$. Verify the lower bound property ($p^* \geq \inf_x L(x, \mu)$for $\mu \geq 0$). Derive and sketch the Lagrange dual function $g$.
    1. State the dual problem, and verify that it is a concave maximization problem. Find the dual optimal value and dual optimal solution $\mu^*$. Does strong duality hold?
    1.  Let $p^*(u)$ denote the optimal value of the problem

    $$
    \begin{split}
    & x^2 + 1 \to \min\limits_{x \in \mathbb{R} }\\
    \text{s.t. } & (x-2)(x-4) \leq u
    \end{split}
    $$

    as a function of the parameter $u$. Plot $p^*(u)$. Verify that $\dfrac{dp^*(0)}{du} = -\mu^*$ 

1. Consider a smooth convex function $f(x)$ at some point $x_k$. One can define the first-order Taylor expansion of the function as:
    $$
    f^I_{x_k}(x) = f(x_k) + \nabla f(x_k)^\top (x - x_k),
    $$
    where we can define $\delta x = x - x_k$ and $g = \nabla f(x_k)$. Thus, the expansion can be rewritten as:
    $$
    f^I_{x_k}(\delta x) = f(x_k) + g^\top \delta x.
    $$
    Suppose, we would like to design the family of optimization methods that will be defined as:
    $$
    x_{k+1} = \text{arg}\min_{x} \left\{f^I_{x_k}(\delta x) + \frac{\lambda}{2} \|\delta x\|^2\right\},
    $$
    where $\lambda > 0$ is a parameter.

    1. [5 points] Show, that this method is equivalent to the gradient descent method with the choice of Euclidean norm of the vector $\|\delta x\| = \|\delta x\|_2$. Find the corresponding learning rate.
    1. [5 points] Prove, that the following holds:
        $$
        \text{arg}\min_{\delta x \in \mathbb{R}^n} \left\{ g^T\delta x + \frac{\lambda}{2} \|\delta x\|^2\right\} = - \frac{\|g\|_*}{\lambda} \text{arg}\max_{\|t\|=1} \left\{ t^T g \right\},
        $$
        where $\|g\|_*$ is the [dual norm](https://fmin.xyz/docs/theory/Dual%20norm.html) of $g$.
    1. [3 points] Consider another vector norm $\|\delta x\| = \|\delta x\|_\infty$. Write down explicit expression for the corresponding method.
    1. [2 points] Consider induced operator matrix norm for any matrix $W \in \mathbb{R}^{d_{out} \times d_{in}}$
        $$
        \|W\|_{\alpha \to \beta} = \max_{x \in \mathbb{R}^{d_{in}}} \frac{\|Wx\|_{\beta}}{\|x\|_{\alpha}}.
        $$
        Typically, when we solve optimization problems in deep learning, we stack the weight matrices for all layers $l = [1, L]$ into a single vector.
        $$
        w = \text{vec}(W_1, W_2, \ldots, W_L) \in \mathbb{R}^{n},
        $$
        Can you write down the explicit expression, that relates
        $$
        \|w\|_\infty \qquad \text{ and } \qquad \|W_l\|_{\alpha \to \beta}, \; l = [1, L]?
        $$

1. [10 points] Derive the dual problem for the Ridge regression problem with $A \in \mathbb{R}^{m \times n}, b \in \mathbb{R}^m, \lambda > 0$:

    $$
    \begin{split}
    \dfrac{1}{2}\|y-b\|^2 + \dfrac{\lambda}{2}\|x\|^2 &\to \min\limits_{x \in \mathbb{R}^n, y \in \mathbb{R}^m }\\
    \text{s.t. } & y = Ax
    \end{split}
    $$

1. [20 points] Derive the dual problem for the support vector machine problem with $A \in \mathbb{R}^{m \times n}, \mathbf{1} \in \mathbb{R}^m \in \mathbb{R}^m, \lambda > 0$:

    $$
    \begin{split}
    \langle \mathbf{1}, t\rangle + \dfrac{\lambda}{2}\|x\|^2 &\to \min\limits_{x \in \mathbb{R}^n, t \in \mathbb{R}^m }\\
    \text{s.t. } & Ax \succeq \mathbf{1} - t \\
    & t \succeq 0
    \end{split}
    $$

1. [10 points] Give an explicit solution to the following LP.
    
    $$
    \begin{split}
    & c^\top x \to \min\limits_{x \in \mathbb{R}^n }\\
    \text{s.t. } & 1^\top x = 1, \\
    & x \succeq 0 
    \end{split}
    $$

    This problem can be considered the simplest portfolio optimization problem.

1. [20 points] Show, that the following problem has a unique solution and find it:

    $$
    \begin{split}
    & \langle C^{-1}, X\rangle - \log \det X \to \min\limits_{x \in \mathbb{R}^{n \times n} }\\
    \text{s.t. } & \langle Xa, a\rangle \leq 1,
    \end{split}
    $$

    where $C \in \mathbb{S}^n_{++}, a \in \mathbb{R}^n \neq 0$. The answer should not involve inversion of the matrix $C$.

1. [20 points] Give an explicit solution to the following QP.
    
    $$
    \begin{split}
    & c^\top x \to \min\limits_{x \in \mathbb{R}^n }\\
    \text{s.t. } & (x - x_c)^\top A (x - x_c) \leq 1,
    \end{split}
    $$

    where $A \in \mathbb{S}^n_{++}, c \neq 0, x_c \in \mathbb{R}^n$.

1. [10 points] Consider the equality-constrained least-squares problem
    
    $$
    \begin{split}
    & \|Ax - b\|_2^2 \to \min\limits_{x \in \mathbb{R}^n }\\
    \text{s.t. } & Cx = d,
    \end{split}
    $$

    where $A \in \mathbb{R}^{m \times n}$ with $\mathbf{rank }A = n$, and $C \in \mathbb{R}^{k \times n}$ with $\mathbf{rank }C = k$. Give the KKT conditions, and derive expressions for the primal solution $x^*$ and the dual solution $\lambda^*$.



1. **Supporting hyperplane interpretation of KKT conditions**. [10 points]  Consider a **convex** problem with no equality constraints
    
    $$
    \begin{split}
    & f_0(x) \to \min\limits_{x \in \mathbb{R}^n }\\
    \text{s.t. } & f_i(x) \leq 0, \quad i = [1,m]
    \end{split}
    $$

    Assume, that $\exists x^* \in \mathbb{R}^n, \mu^* \in \mathbb{R}^m$ satisfy the KKT conditions
    
    $$
    \begin{split}
    & \nabla_x L (x^*, \mu^*) = \nabla f_0(x^*) + \sum\limits_{i=1}^m\mu_i^*\nabla f_i(x^*) = 0 \\
    & \mu^*_i \geq 0, \quad i = [1,m] \\
    & \mu^*_i f_i(x^*) = 0, \quad i = [1,m]\\
    & f_i(x^*) \leq 0, \quad i = [1,m]
    \end{split}
    $$

    Show that

    $$
    \nabla f_0(x^*)^\top (x - x^*) \geq 0
    $$

    for all feasible $x$. In other words, the KKT conditions imply the simple optimality criterion or $\nabla f_0(x^*)$ defines a supporting hyperplane to the feasible set at $x^*$.
    
1. **A penalty method for equality constraints.** [10 points] We consider the problem of minimization

    $$
    \begin{split}
    & f_0(x) \to \min\limits_{x \in \mathbb{R}^{n} }\\
    \text{s.t. } & Ax = b,
    \end{split}
    $$
    
    where $f_0(x): \mathbb{R}^n \to\mathbb{R} $ is convex and differentiable, and $A \in \mathbb{R}^{m \times n}$ with $\mathbf{rank }A = m$. In a quadratic penalty method, we form an auxiliary function

    $$
    \phi(x) = f_0(x) + \alpha \|Ax - b\|_2^2,
    $$
    
    where $\alpha > 0$ is a parameter. This auxiliary function consists of the objective plus the penalty term $\alpha \Vert Ax - b\Vert_2^2$. The idea is that a minimizer of the auxiliary function, $\tilde{x}$, should be an approximate solution to the original problem. Intuition suggests that the larger the penalty weight $\alpha$, the better the approximation $\tilde{x}$ to a solution of the original problem. Suppose $\tilde{x}$ is a minimizer of $\phi(x)$. Show how to find, from $\tilde{x}$, a dual feasible point for the original problem. Find the corresponding lower bound on the optimal value of the original problem.

### Linear programming

1. **📱🎧💻 Covers manufacturing.** [20 points] Lyzard Corp is producing covers for the following products: 
    * 📱 phones
    * 🎧 headphones
    * 💻 laptops

    The company's production facilities are such that if we devote the entire production to headphone covers, we can produce 5000 of them in one day. If we devote the entire production to phone covers or laptop covers, we can produce 4000 or 2000 of them in one day. 

    The production schedule is one week (6 working days), and the week's production must be stored before distribution. Storing 1000 headphone covers (packaging included) takes up 30 cubic feet of space. Storing 1000 phone covers (packaging included) takes up 50 cubic feet of space, and storing 1000 laptop covers (packaging included) takes up 200 cubic feet of space. The total storage space available is 1500 cubic feet. 
    
    Due to commercial agreements with Lyzard Corp has to deliver at least 6000 headphone covers and 4000 laptop covers per week to strengthen the product's diffusion. 

    The marketing department estimates that the weekly demand for headphones covers, phone, and laptop covers does not exceed 15000, 12000 and 8000 units, therefore the company does not want to produce more than these amounts for headphones, phone, and laptop covers. 

    Finally, the net profit per headphone cover, phone cover, and laptop cover are \$5, \$7, and \$12, respectively.

    The aim is to determine a weekly production schedule that maximizes the total net profit.

    1. Write a Linear Programming formulation for the problem.  Use the following variables:

        * $y_1$ = number of headphones covers produced over the week,  
        * $y_2$ = number of phone covers produced over the week,  
        * $y_3$ = number of laptop covers produced over the week. 

    1. Find the solution to the problem using [PyOMO](http://www.pyomo.org)
    
        ```python
        !pip install pyomo
        ! sudo apt-get install glpk-utils --quiet  # GLPK
        ! sudo apt-get install coinor-cbc --quiet  # CoinOR
        ```

    1. Perform the sensitivity analysis. Which constraint could be relaxed to increase the profit the most? Prove it numerically.

1. Prove the optimality of the solution [10 points] 
    
    $$
    x = \left(\frac{7}{3} , 0, \frac{1}{3}\right)^T
    $$
    
    to the following linear programming problem:
    
    $$
    \begin{split}
    & 9x_1 + 3x_2 + 7x_3 \to \max\limits_{x \in \mathbb{R}^3 }\\
    \text{s.t. } & 2x_1 + x_2 + 3x_3 \leq 6 \\
    & 5x_1 + 4x_2 + x_3 \leq 12 \\
    & 3x_3 \leq 1,\\
    & x_1, x_2, x_3 \geq 0
    \end{split}
    $$

    but you cannot use any numerical algorithm here.

1. [10 points] Economic interpretation of the dual problem: Suppose a small shop makes wooden toys, where each toy train requires one piece of wood and $2$ tins of paint, while each toy boat requires one piece of wood and $1$ tin of paint. The profit on each toy train is $\$30$, and the profit on each toy boat is $\$20$. Given an inventory of $80$ pieces of wood and $100$ tins of paint, how many of each toy
should be made to maximize the profit?
    1. Write out the optimization problem in standard form, writing all constraints as inequalities.
    1. Sketch the feasible set and determine $p^*$ and $x^*$
    1. Find the dual problem, then determine $d^*$ and $\lambda^*$. Note that we can interpret the Lagrange multipliers $\lambda_k$ associated with the constraints on wood and paint as the prices for each piece of wood and tin of paint, so that $−d^*$ is how much money would be obtained from selling the inventory for those prices. Strong duality says a buyer should not pay more for the inventory than what the toy store would make by producing and selling toys from it, and that the toy store should not sell the inventory for less than that.
    1. The other interpretation of the Lagrange multipliers is as sensitivities to changes in the constraints. Suppose the toymaker found some more pieces of wood; the $\lambda_k$ associated with the wood constraint will equal the partial derivative of $−p^*$ with respect to how much more wood became available. Suppose the inventory increases by one piece of wood. Use $\lambda^*$ to estimate how much the profit would increase, without solving the updated optimization problem. How is this consistent with the price interpretation given above for the Lagrange multipliers? [source](https://tleise.people.amherst.edu/Math294Spring2017/TeXfiles/LagrangeDualityHW.pdf) 

### Gradient Descent

1. **Convergence of Gradient Descent in non-convex smooth case** [10 points]

    We will assume nothing about the convexity of $f$.  We will show that gradient descent reaches an $\varepsilon$-substationary point $x$, such that $\|\nabla f(x)\|_2 \leq \varepsilon$, in $O(1/\varepsilon^2)$ iterations. Important note: you may use here Lipschitz parabolic upper bound: 
    
    $$
    f(y) \leq f(x) + \nabla f(x)^T (y-x) + \frac{L}{2} \|y-x\|_2^2, \;\;\;
    \text{for all $x,y$}.  
    $$ {#eq-quad_ub}

    * Plug in $y = x^{k+1} = x^{k} - \alpha \nabla f(x^k), x = x^k$ to (@eq-quad_ub) to show that 

        $$
        f(x^{k+1}) \leq f(x^k) - \Big (1-\frac{L\alpha}{2} \Big) \alpha \|\nabla f(x^k)\|_2^2.
        $$

    * Use $\alpha \leq 1/L$, and rearrange the previous result, to get 

        $$
        \|\nabla f(x^k)\|_2^2 \leq \frac{2}{\alpha} \left( f(x^k) - f(x^{k+1}) \right).
        $$

    * Sum the previous result over all iterations from $1,\ldots,k+1$ to establish
    
        $$
        \sum_{i=0}^k \|\nabla f(x^{i})\|_2^2 \leq 
        \frac{2}{\alpha} ( f(x^{0}) - f^*).
        $$

    * Lower bound the sum in the previous result to get 

        $$
        \min_{i=0,\ldots,k} \|\nabla f(x^{i}) \|_2 
        \leq \sqrt{\frac{2}{\alpha(k+1)} (f(x^{0}) - f^*)}, 
        $$
        which establishes the desired $O(1/\varepsilon^2)$ rate for achieving $\varepsilon$-substationarity.  

1. **How gradient descent convergence depends on the condition number and dimensionality.** [20 points] Investigate how the number of iterations required for gradient descent to converge depends on the following two parameters: the condition number $\kappa \geq 1$ of the function being optimized, and the dimensionality $n$ of the space of variables being optimized.
    
    To do this, for given parameters $n$ and $\kappa$, randomly generate a quadratic problem of size $n$ with condition number $\kappa$ and run gradient descent on it with some fixed required precision. Measure the number of iterations $T(n, \kappa)$ that the method required for convergence (successful termination based on the stopping criterion).

    Recommendation: The simplest way to generate a random quadratic problem of size $n$ with a given condition number $\kappa$ is as follows. It is convenient to take a diagonal matrix $A \in S_{n}^{++}$ as simply the diagonal matrix $A = \text{Diag}(a)$, whose diagonal elements are randomly generated within $[1, \kappa]$, and where $\min(a) = 1$, $\max(a) = \kappa$. As the vector $b \in \mathbb{R}^n$, you can take a vector with random elements. Diagonal matrices are convenient to consider since they can be efficiently processed with even for large values of $n$.

    Fix a certain value of the dimensionality $n$. Iterate over different condition numbers $\kappa$ on a grid and plot the dependence of $T(n,\kappa)$ against $\kappa$. Since the quadratic problem is generated randomly each time, repeat this experiment several times. As a result, for a fixed value of $n$, you should obtain a whole family of curves showing the dependence of $T(n, \kappa)$ on $\kappa$. Draw all these curves in the same color for clarity (for example, red).

    Now increase the value of $n$ and repeat the experiment. You should obtain a new family of curves $T(n',\kappa)$ against $\kappa$. Draw all these curves in the same color but different from the previous one (for example, blue).

    Repeat this procedure several times for other values of $n$. Eventually, you should have several different families of curves - some red (corresponding to one value of $n$), some blue (corresponding to another value of $n$), some green, etc.

    Note that it makes sense to iterate over the values of the dimensionality $n$ on a logarithmic grid (for example, $n = 10, n = 100, n = 1000$, etc.). Use the following stopping criterion: $\|\nabla f(x_k)\|_2^2 \leq \varepsilon \|\nabla f(x_0)\|_2^2$ with $\varepsilon = 10^{-5}$. Select the starting point $x_0 = (1, \ldots, 1)^T$

    What conclusions can be drawn from the resulting picture?


### Accelerated methods

1. **Local Convergence of Heavy Ball Method.** [20 points] We will work with the heavy ball method in this problem

    $$
    \tag{HB}
    x_{k+1} = x_k - \alpha \nabla f(x_k) + \beta (x_k - x_{k-1})
    $$

    It is known, that for the quadratics the best choice of hyperparameters is $\alpha^* = \dfrac{4}{(\sqrt{L} + \sqrt{\mu})^2}, \beta^* = \dfrac{(\sqrt{L} - \sqrt{\mu})^2}{(\sqrt{L} + \sqrt{\mu})^2}$, which ensures accelerated linear convergence for a strongly convex quadratic function.

    Consider the following continuously differentiable, strongly convex with parameter $\mu$, and smooth function with parameter $L$:

    $$
    f(x) = 
    \begin{cases} 
    \frac{25}{2}x^2, & \text{if } x < 1 \\
    \frac12x^2 + 24x - 12, & \text{if } 1 \leq x < 2 \\
    \frac{25}{2}x^2 - 24x + 36, & \text{if } x \geq 2
    \end{cases}
    \quad
    \nabla f(x) = 
    \begin{cases} 
    25x, & \text{if } x < 1 \\
    x + 24, & \text{if } 1 \leq x < 2 \\
    25x - 24, & \text{if } x \geq 2
    \end{cases}
    $$

    1. How to prove, that the given function is convex? Strongly convex? Smooth?
    1. Find the constants $\mu$ and $L$ for a given function.
    1. Plot the function value for $x \in [-4, 4]$. 
    1. Run the Heavy Ball method for the function with optimal hyperparameters $\alpha^* = \dfrac{4}{(\sqrt{L} + \sqrt{\mu})^2}, \beta^* = \dfrac{(\sqrt{L} - \sqrt{\mu})^2}{(\sqrt{L} + \sqrt{\mu})^2}$ for quadratic function, starting from $x_0 = 3.5$. If you have done everything above correctly, you should receive something like 
    
        {{< video heavy_ball_conv.mp4 >}}

        You can use the following code for plotting:

        ```python
        import numpy as np
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation
        from IPython.display import HTML

        # Gradient of the function
        def grad_f(x):
            ...

        # Heavy Ball method implementation
        def heavy_ball_method(alpha, beta, x0, num_iterations):
            x = np.zeros(num_iterations + 1)
            x_prev = x0
            x_curr = x0  # Initialize x[1] same as x[0] to start the algorithm
            for i in range(num_iterations):
                x[i] = x_curr
                x_new = x_curr - alpha * grad_f(x_curr) + beta * (x_curr - x_prev)
                x_prev = x_curr
                x_curr = x_new
            x[num_iterations] = x_curr
            return x

        # Parameters
        L = ...
        mu = ...
        alpha_star = ...
        beta_star = ...
        x0 = ...
        num_iterations = 30

        # Generate the trajectory of the method
        trajectory = heavy_ball_method(alpha_star, beta_star, x0, num_iterations)

        # Setup the figure and axes for the animation
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3.5))
        fig.suptitle("Heavy ball method with optimal hyperparameters α* β*")

        # Function for updating the animation
        def update(i):
            ax1.clear()
            ax2.clear()

            # Plot f(x) and trajectory
            x_vals = np.linspace(-4, 4, 100)
            f_vals = np.piecewise(x_vals, [x_vals < 1, (x_vals >= 1) & (x_vals < 2), x_vals >= 2],
                                [lambda x: 12.5 * x**2, lambda x: .5 * x**2 + 24 * x - 12, lambda x: 12.5 * x**2 - 24 * x + 36])
            ax1.plot(x_vals, f_vals, 'b-')
            ax1.plot(trajectory[:i], [12.5 * x**2 if x < 1 else .5 * x**2 + 24 * x - 12 if x < 2 else 12.5 * x**2 - 24 * x + 36 for x in trajectory[:i]], 'ro-')
            # Add vertical dashed lines at x=1 and x=2 on the left subplot
            ax1.axvline(x=1, color='navy', linestyle='--')
            ax1.axvline(x=2, color='navy', linestyle='--')

            # Plot function value from iteration
            f_trajectory = [None for x in trajectory]
            f_trajectory[:i] = [12.5 * x**2 if x < 1 else .5 * x**2 + 24 * x - 12 if x < 2 else 12.5 * x**2 - 24 * x + 36 for x in trajectory[:i]]
            ax2.plot(range(len(trajectory)), f_trajectory, 'ro-')
            ax2.set_xlim(0, len(trajectory))
            ax2.set_ylim(min(f_vals), max(f_vals))
            # Add horizontal dashed lines at f(1) and f(2) on the right subplot
            f_1 = 12.5 * 1.0**2
            f_2 = .5 * 2.**2 + 24 * 2. - 12
            ax2.axhline(y=f_1, color='navy', linestyle='--')
            ax2.axhline(y=f_2, color='navy', linestyle='--')

            # ax1.set_title("Function f(x) and Trajectory")
            ax1.set_xlabel("x")
            ax1.set_ylabel("f(x)")
            ax1.grid(linestyle=":")

            # ax2.set_title("Function Value from Iteration")
            ax2.set_xlabel("Iteration")
            ax2.set_ylabel("f(x)")
            ax2.grid(linestyle=":")

            plt.tight_layout()

        # Create the animation
        ani = animation.FuncAnimation(fig, update, frames=num_iterations, repeat=False, interval=100)
        HTML(ani.to_jshtml())
        ```
    
    1. Change the starting point to $x_0 = 3.4$. What do you see? How could you name such a behavior of the method? 
    1. Change the hyperparameter $\alpha^{\text{Global}} = \frac2L, \beta^{\text{Global}} = \frac{\mu}{L}$ and run the method again from $x_0 = 3.4$. Check whether you have accelerated convergence here.

    Context: this counterexample was provided in the [paper](https://arxiv.org/pdf/1408.3595.pdf), while the global convergence of the heavy ball method for general smooth strongly convex function was introduced in another [paper](https://arxiv.org/pdf/1412.7457.pdf). Recently, it was [suggested](https://arxiv.org/pdf/2307.11291.pdf), that the heavy-ball (HB) method provably does not reach an accelerated convergence rate on smooth strongly convex problems. 

1. [40 points] In this problem we will work with accelerated methods applied to the logistic regression problem. A good visual introduction to the topic is available [here](https://mlu-explain.github.io/logistic-regression/). 
    
    Logistic regression is a standard model in classification tasks. For simplicity, consider only the case of binary classification. Informally, the problem is formulated as follows: There is a training sample $\{(a_i, b_i)\}_{i=1}^m$, consisting of $m$ vectors $a_i \in \mathbb{R}^n$ (referred to as features) and corresponding numbers $b_i \in \{-1, 1\}$ (referred to as classes or labels). The goal is to construct an algorithm $b(\cdot)$, which for any new feature vector $a$ automatically determines its class $b(a) \in \{-1, 1\}$. 

    In the logistic regression model, the class determination is performed based on the sign of the linear combination of the components of the vector $a$ with some fixed coefficients $x \in \mathbb{R}^n$:
    
    $$
    b(a) := \text{sign}(\langle a, x \rangle).
    $$

    The coefficients $x$ are the parameters of the model and are adjusted by solving the following optimization problem:
    
    $$
    \tag{LogReg}
    \min_{x \in \mathbb{R}^n} \left( \frac{1}{m} \sum_{i=1}^m \ln(1 + \exp(-b_i \langle a_i, x \rangle)) + \frac{\lambda}{2} \|x\|^2 \right),
    $$

    where $\lambda \geq 0$ is the regularization coefficient (a model parameter). 

    1. Will the LogReg problem be convex for $\lambda = 0$? What is the gradient of the objective function? Will it be strongly convex? What if you will add regularization with $\lambda > 0$?
    1. We will work with the real-world data for $A$ and $b$: take the mushroom dataset. Be careful, you will need to predict if the mushroom is poisonous or edible. A poor model can cause death in this exercise. 

        ```python
        import requests
        from sklearn.datasets import load_svmlight_file

        # URL of the file to download
        url = 'https://hse24.fmin.xyz/files/mushrooms.txt'

        # Download the file and save it locally
        response = requests.get(url)
        dataset = 'mushrooms.txt'

        # Ensure the request was successful
        if response.status_code == 200:
            with open(dataset, 'wb') as f:
                f.write(response.content)

            # Load the dataset from the downloaded file
            data = load_svmlight_file(dataset)
            A, b = data[0].toarray(), data[1]
            n, d = A.shape

            print("Data loaded successfully.")
            print(f"Number of samples: {n}, Number of features: {d}")
        else:
            print(f"Failed to download the file. Status code: {response.status_code}")

        ```
    1. Divide the data into two parts: training and test. We will train the model on the $A_{train}$, $b_{train}$ and measure the accuracy of the model on the $A_{test}$, $b_{test}$.

        ```python
        from sklearn.model_selection import train_test_split
        # Split the data into training and test sets
        A_train, A_test, b_train, b_test = train_test_split(A, b, test_size=0.2, random_state=214)
        ```
    1. For the training part $A_{train}$, $b_{train}$, estimate the constants $\mu, L$ of the training/optimization problem. Use the same small value $\lambda$ for all experiments
    1. Using gradient descent with the step $\frac{1}{L}$, train a model. Plot: accuracy versus iteration number. 

        $$
        \tag{HB}
        x_{k+1} = x_k - \alpha \nabla f(x_k) + \beta (x_k - x_{k-1})
        $$

        Fix a step $\alpha = \frac{1}{L}$ and search for different values of the momentum $\beta$ from $-1$ to $1$. Choose your own convergence criterion and plot convergence for several values of momentum on the same graph. Is the convergence always monotonic?
    
    1. For the best value of momentum $\beta$, plot the dependence of the model accuracy on the test sample on the running time of the method. Add to the same graph the convergence of gradient descent with step $\frac{1}{L}$. Draw a conclusion. Ensure, that you use the same value of $\lambda$ for both methods.
    1. Solve the logistic regression problem using the Nesterov method. 

        $$
        \tag{NAG}
        x_{k+1} = x_k - \alpha \nabla f(x_k + \beta (x_k - x_{k-1})) + \beta (x_k - x_{k-1})  
        $$

        Fix a step $\frac{1}{L}$ and search for different values of momentum $\beta$ from $-1$ to $1$. Check also the momentum values equal to $\frac{k}{k+3}$, $\frac{k}{k+2}$, $\frac{k}{k+1}$ ($k$ is the number of iterations), and if you are solving a strongly convex problem, also $\frac{\sqrt{L} - \sqrt{\mu}}{\sqrt{L} + \sqrt{\mu}}$. Plot the convergence of the method as a function of the number of iterations (choose the convergence criterion yourself) for different values of the momentum. Is the convergence always monotonic?
    1. For the best value of momentum $\beta$, plot the dependence of the model accuracy on the test sample on the running time of the method. Add this graph to the graphs for the heavy ball and gradient descent from the previous steps. Make a conclusion.
    1. Now we drop the estimated value of $L$ and will try to do it adaptively. Let us make the selection of the constant $L$ adaptive. 
    
        $$
        f(y) \leq f(x^k) + \langle \nabla f(x^k), y - x^k \rangle + \frac{L}{2}\|x^k - y\|_2^2
        $$
        
        In particular, the procedure might work:

        ```python
        def backtracking_L(f, grad, x, h, L0, rho, maxiter=100):
            L = L0
            fx = f(x)
            gradx = grad(x)
            iter = 0
            while iter < maxiter :
                y = x - 1 / L * h
                if f(y) <= fx - 1 / L gradx.dot(h) + 1 / (2 * L) h.dot(h):
                    break
                else:
                    L = L * rho
                
                iter += 1
            return L
        ```

        What should $h$ be taken as? Should $\rho$ be greater or less than $1$? Should $L_0$ be taken as large or small? Draw a similar figure as it was in the previous step for L computed adaptively (6 lines - GD, HB, NAG, GD adaptive L, HB adaptive L, NAG adaptive L)

### Conjugate gradients

1. **[Randomized Preconditioners for Conjugate Gradient Methods.](https://web.stanford.edu/class/ee364b/364b_exercises.pdf)**  (20 points)

    *Linear least squares*

    In this task, we explore the use of some randomization methods for solving overdetermined least-squares problems, focusing on conjugate gradient methods. Let $\hat{A} \in \mathbb{R}^{m \times n}$ be a matrix (we assume that $m \gg n$) and $\hat{b} \in \mathbb{R}^m$, we aim to minimize

    $$
    f(x) = \frac{1}{2} \|\hat{A}x - \hat{b}\|^2_2 = \frac{1}{2} \sum_{i=1}^m (\hat{a}_i^T x - \hat{b}_i)^2,
    $$

    where the $\hat{a}_i \in \mathbb{R}^n$ denote the rows of $\hat{A}$.

    *Preconditioners*

    We know, that the convergence bound of the CG applied for the problem depends on the condition number of the matrix. Note, that for the problem above we have the matrix $\hat{A}^T \hat{A}$ and the condition number is squared after this operation ($\kappa (X^T X) =  \kappa^2 \left(X \right)$). That is the reason, why we typically need to use *preconditioners* ([read 12. for more details](https://www.cs.cmu.edu/~quake-papers/painless-conjugate-gradient.pdf)) with CG.

    The general idea of using preconditioners implies switchwing from solving $Ax = b$ to $MAx = Mb$ with hope, that $\kappa \left( MA\right) \ll \kappa \left( A\right)$ or eigenvalues of $MA$ are better clustered than those of $A$ (note, that matrix $A$ here is for the general case, here we have $\hat{A}^T\hat{A}$ instead). 

    This idea can also be viewed as coordinate change $x = T \hat{x}, \; \hat{x} = T^{-1}x$, which leads to the problem $T^T A T \hat{x} = T^Tb$. Note, that the spectrum of $T^TAT$ is the same as the spectrum of $MA$. 
    
    The best choice of $M$ is $A^{-1}$, because $\kappa (A^{-1} A) = \kappa (I) = 1$. However, if we know $A^{-1}$, the original problem is already solved, that is why we need to find some trade-off between enhanced convergence, and extra cost of working with $M$. The goal is to find $M$ that is cheap to multiply, and approximate inverse of $A$ (or at least has a more clustered spectrum than $A$). 

    Note, that for the linear least squares problem the matrix of quadratic form is $A = \hat{A}^T\hat{A} \in \mathbb{R}^{n \times n}$ and the rhs vector is $b = \hat{A}^T\hat{b} \in \mathbb{R}^n$. Below you can find Vanilla CG algorithm (on the left) and preconditioned CG algorithm (on the right):

    $$
    \begin{aligned}
    & \mathbf{r}_0 := \mathbf{b} - \mathbf{A x}_0 \\
    & \hbox{if } \mathbf{r}_{0} \text{ is sufficiently small, then return } \mathbf{x}_{0} \text{ as the result}\\
    & \mathbf{d}_0 := \mathbf{r}_0 \\
    & k := 0 \\
    & \text{repeat} \\
    & \qquad \alpha_k := \frac{\mathbf{r}_k^\mathsf{T} \mathbf{r}_k}{\mathbf{d}_k^\mathsf{T} \mathbf{A d}_k}  \\
    & \qquad \mathbf{x}_{k+1} := \mathbf{x}_k + \alpha_k \mathbf{d}_k \\
    & \qquad \mathbf{r}_{k+1} := \mathbf{r}_k - \alpha_k \mathbf{A d}_k \\
    & \qquad \hbox{if } \mathbf{r}_{k+1} \text{ is sufficiently small, then exit loop} \\
    & \qquad \beta_k := \frac{\mathbf{r}_{k+1}^\mathsf{T} \mathbf{r}_{k+1}}{\mathbf{r}_k^\mathsf{T} \mathbf{r}_k} \\
    & \qquad \mathbf{d}_{k+1} := \mathbf{r}_{k+1} + \beta_k \mathbf{d}_k \\
    & \qquad k := k + 1 \\
    & \text{end repeat} \\
    & \text{return } \mathbf{x}_{k+1} \text{ as the result}
    \end{aligned} \qquad 
    \begin{aligned}
    & \mathbf{r}_0 := \mathbf{b} - \mathbf{A x}_0 \\
    & \text{if } \mathbf{r}_0 \text{ is sufficiently small, then return } \mathbf{x}_0 \text{ as the result} \\
    & \mathbf{z}_0 := \mathbf{M} \mathbf{r}_0 \\
    & \mathbf{d}_0 := \mathbf{z}_0 \\
    & k := 0 \\
    & \text{repeat} \\
    & \qquad \alpha_k := \frac{\mathbf{r}_k^\mathsf{T} \mathbf{z}_k}{\mathbf{d}_k^\mathsf{T} \mathbf{A d}_k} \\
    & \qquad \mathbf{x}_{k+1} := \mathbf{x}_k + \alpha_k \mathbf{d}_k \\
    & \qquad \mathbf{r}_{k+1} := \mathbf{r}_k - \alpha_k \mathbf{A d}_k \\
    & \qquad \text{if } \mathbf{r}_{k+1} \text{ is sufficiently small, then exit loop} \\
    & \qquad \mathbf{z}_{k+1} := \mathbf{M} \mathbf{r}_{k+1} \\
    & \qquad \beta_k := \frac{\mathbf{r}_{k+1}^\mathsf{T} \mathbf{z}_{k+1}}{\mathbf{r}_k^\mathsf{T} \mathbf{z}_k} \\
    & \qquad \mathbf{d}_{k+1} := \mathbf{z}_{k+1} + \beta_k \mathbf{d}_k \\
    & \qquad k := k + 1 \\
    & \text{end repeat} \\
    & \text{return } \mathbf{x}_{k+1} \text{ as the result}
    \end{aligned}
    $$

    *Hadamard matrix*

    Given $m \in \{2^i, i = 1, 2, \ldots\}$, the (unnormalized) Hadamard matrix of order $m$ is defined recursively as

    $$
    H_2 = \begin{bmatrix} 1 & 1 \\ 1 & -1 \end{bmatrix}, \quad \text{and} \quad H_m = \begin{bmatrix} H_{m/2} & H_{m/2} \\ H_{m/2} & -H_{m/2} \end{bmatrix}.
    $$

    The associated normalized Hadamard matrix is given by $H^{(\text{norm})}_m = \frac{1}{\sqrt{m}} H_m$, which evidently satisfies $H^{(\text{norm})T}_m H^{(\text{norm})}_m = I_{m \times m}$. Moreover, via a recursive algorithm, it is possible to compute matvec $H_m x$ in time $O(m \log m)$, which is much faster than $m^2$ for a general matrix.

    To solve the least squares minimization problem using conjugate gradients, we must solve $\hat{A}^T \hat{A} x = \hat{A}^T b$. Using a preconditioner $M$ such that $M \approx A^{-1}$ can give substantial speedup in computing solutions to large problems.

    Consider the following scheme to generate a randomized preconditioner, assuming that $m = 2^i$ for some $i$:

    1. Let $S = \text{diag}(S_{11}, \ldots, S_{mm})$, where $S_{jj}$ are random $\{-1,+1\}$ signs
    2. Let $p \in \mathbb{Z}^+$ be a small positive integer, say $20$ for this problem.
    3. Let $R \in \{0, 1\}^{n+p \times m}$ be a *row selection matrix*, meaning that each row of $R$ has only 1 non-zero entry, chosen uniformly at random. (The location of these non-zero columns is distinct.)

        ```python
        import jax.numpy as jnp
        from jax import random

        def create_row_selection_matrix_jax(m, n, p, key):
            # m is the number of columns in the original matrix A
            # n+p is the number of rows in the row selection matrix R
            # key is a PRNGKey needed for randomness in JAX
            inds = random.permutation(key, m)[:n+p]  # Generate a random permutation and select the first n+p indices
            R = jnp.zeros((n+p, m), dtype=jnp.int32)  # Create a zero matrix of shape (n+p, m)
            R = R.at[np.arange(n+p), inds].set(1)     # Use JAX's indexed update to set the entries corresponding to inds to 1
            return R
        ```

    4. Define $\Phi = R H^{(\text{norm})}_m S \in \mathbb{R}^{n+p \times m}$

    We then define the matrix $M$ via its inverse $M^{-1} = \hat{A}^T \Phi^T \Phi \hat{A} \in \mathbb{R}^{n \times n}$.

    *Questions*

    1. **(2 point)** How many FLOPs (floating point operations, i.e. multiplication and additions) are required to compute the matrices $M^{-1}$ and $M$, respectively, assuming that you can compute the matrix-vector product $H_mv$ in time $m \log m$ for any vector $v \in \mathbb{R}^m$?

    1. **(2 point)** How many FLOPs are required to naively compute $\hat{A}^T \hat{A}$, assuming $\hat{A}$ is dense (using standard matrix algorithms)?
    
    1. **(2 point)** How many FLOPs are required to compute $\hat{A}^T \hat{A} v$ for a vector $v \in \mathbb{R}^n$ by first computing $u = \hat{A}v$ and then computing $\hat{A}^T u$?
    
    1. **(4 poins)** Suppose that conjugate gradients runs for $k$ iterations. Using the preconditioned conjugate gradient algorithm with $M = (\hat{A}^T \Phi^T \Phi \hat{A})^{-1}$, how many total floating point operations have been performed? How many would be required to directly solve $\hat{A}^T \hat{A} x = \hat{A}^T b$? How large must $k$ be to make the conjugate gradient method slower?
    
    1. **(10 points)** Implement the conjugate gradient algorithm for solving the positive definite linear system $\hat{A}^T \hat{A} x = \hat{A}^T b$ both with and without the preconditioner $M$. To generate data for your problem, set $m = 2^{12}$ and $n = 400$, then generate the matrix $A$ and the vector $b$. For simplicity in implementation, you may directly pass $\hat{A}^T \hat{A}$ and $\hat{A}^T b$ into your conjugate gradient solver, as we only wish to explore how the methods work.

    ```python
    import numpy as np
    from scipy.sparse import diags

    m = 2**12  # 4096
    n = 400
    # Create a linear space of values from 0.001 to 100
    values = np.linspace(0.001, 100, n)
    # Generate the matrix A
    A = np.random.randn(m, n) * diags(values).toarray()
    b = np.random.randn(m, 1)
    ```

    Plot the norm of the residual $r_k = \hat{A}^T b - \hat{A}^T \hat{A} x_k$ (relative to $\|\hat{A}^T b\|_2$) as a function of iteration $k$ for each of your conjugate gradient procedures. Additionally, compute and print the condition numbers $\kappa(\hat{A}^T \hat{A})$ and $\kappa(M^{1/2} \hat{A}^T \hat{A} M^{1/2})$.

### Newton and quasinewton methods

1. **😱 Newton convergence issue** (10 points) 

    Consider the following function: 

    $$
    f(x,y) = \dfrac{x^4}{4} - x^2 + 2x + (y-1)^2
    $$
    
    And the starting point is $x_0 = (0,2)^\top$. How does Newton's method behave when started from this point? How can this be explained? How does the gradient descent with fixed step $\alpha = 0.01$ and the steepest descent method behave under the same conditions? (It is not necessary to show numerical simulations in this problem).

1. **Hessian-Free Newton method** (20 points) In this exercise, we'll explore the optimization of a binary logistic regression problem using various methods. Don't worry about the size of the problem description, first 5 bullets out of 7 could be done pretty quickly. In this problem you should start with this [\faPython colab notebook](https://colab.research.google.com/github/MerkulovDaniil/optim/blob/master/assets/Notebooks/Hessian_free_Newton.ipynb)

    Given a dataset with $n$ observations, where each observation consists of a feature vector $x_i$ and an associated binary target variable $y_i \in \{0,1\}$, the logistic regression model predicts the probability that $y_i = 1$ given $x_i$ using the logistic function. The loss function to be minimized is the negative log-likelihood of the observed outcomes under this model, summed across all observations. It has a high value when the model outputs differ significantly from the data $y$.

    The binary cross-entropy loss function for a single observation $(x_i, y_i)$ is given by:
    $$
    \text{Loss}(w; x_i, y_i) = -\left[ y_i \log(p(y_i=1 | x_i; w)) + (1-y_i) \log(1-p(y_i=1 | x_i; w)) \right]
    $$

    Here, $p(y=1 | x;w)$ is defined as:
    $$
    p(y=1 | x;w) = \frac{1}{1 + e^{-w^T x}}
    $$

    To define the total loss over the dataset, we sum up the individual losses:
    $$
    f(w) = -\sum_{i=1}^n \left[ y_i \log(p(y_i=1 | x_i; w)) + (1-y_i) \log(1-p(y_i=1 | x_i; w)) \right]
    $$

    Therefore, the optimization problem in logistic regression is:
    $$
    \min_w f(w) = \min_w -\sum_{i=1}^n \left[ y_i \log\left(p\left(y_i=1 | x_i; w\right)\right) + \left(1-y_i\right) \log\left(1-p(y_i=1 | x_i; w)\right) \right]
    $$

    This is a convex optimization problem and can be solved using gradient-based methods such as gradient descent, Newton's method, or more sophisticated optimization algorithms often available in machine learning libraries. However, it is the problem is often together with $l_2$ regularization:

    $$
    \min_w f(w) = \min_w -\sum_{i=1}^n \left[ y_i \log\left(p\left(y_i=1 | x_i; w\right)\right) + \left(1-y_i\right) \log\left(1-p(y_i=1 | x_i; w)\right) \right] + \frac{\mu}{2} \|w\|_2^2
    $$

    1. (2 points) Firstly, we address the optimization with Gradient Descent (GD) in a strongly convex setting, with $\mu = 1$. Use a constant learning rate $\alpha$. Run the gradient descent algorithm. Report the highest learning rate that ensures convergence of the algorithm. Plot the convergence graph in terms of both domain (parameter values) and function value (loss). Describe the type of convergence observed.

        ```python
        params = {
            "mu": 1,
            "m": 1000,
            "n": 100,
            "methods": [
                {
                    "method": "GD",
                    "learning_rate": 3e-2,
                    "iterations": 550,
                },
            ]
        }

        results, params = run_experiments(params)
        ```
    
    2. (2 points) Run Newton's method under the same conditions, using the second derivatives to guide the optimization. Describe and analyze the convergence properties observed.

        ```python
        params = {
            "mu": 1,
            "m": 1000,
            "n": 100,
            "methods": [
                {
                    "method": "GD",
                    "learning_rate": 3e-2,
                    "iterations": 550,
                },
                {
                    "method": "Newton",
                    "iterations": 20,
                },
            ]
        }

        results, params = run_experiments(params)
        ```
        
    3. (2 points) In cases where Newton's method may converge too rapidly or overshoot, a damped version can be more stable. Run the damped Newton method. Adjust the damping factor as a learning rate. Report the highest learning rate ensuring stability and convergence. Plot the convergence graph.

        ```python
        params = {
            "mu": 1,
            "m": 1000,
            "n": 100,
            "methods": [
                {
                    "method": "GD",
                    "learning_rate": 3e-2,
                    "iterations": 550,
                },
                {
                    "method": "Newton",
                    "iterations": 20,
                },
                {
                    "method": "Newton",
                    "learning_rate": 5e-1,
                    "iterations": 50,
                },
            ]
        }

        results, params = run_experiments(params)
        ```
    
    4. (2 points) Now turn off the regularization by setting $\mu=0$. Try to find the largest learning rate, which ensures convergence of the Gradient Descent. Use a constant learning rate $\alpha$. Run the gradient descent algorithm. Report the highest learning rate that ensures convergence of the algorithm. Plot the convergence graph in terms of both domain (parameter values) and function value (loss). Describe the type of convergence observed. How can you describe an idea to run this method for the problem to reach tight primal gap $f(x_k) - f^* \approx 10^{-2}$ or $10^{-3}$, $10^{-4}$?

        ```python
        params = {
            "mu": 0,
            "m": 1000,
            "n": 100,
            "methods": [
                {
                    "method": "GD",
                    "learning_rate": 3e-2,
                    "iterations": 200,
                },
                {
                    "method": "GD",
                    "learning_rate": 7e-2,
                    "iterations": 200,
                },
            ]
        }

        results, params = run_experiments(params)
        ```
    
    5. (2 points) What can you say about Newton's method convergence in the same setting $\mu=0$? Try several learning rates smaller, than $1$ for the damped Newton method. Does it work? Write your conclusions about the second-order method convergence for a binary logistic regression problem.
    
    6. (5 points) Now switch back to the strongly convex setting $\mu=1$. To avoid directly computing the Hessian matrix in Newton's method, use the Conjugate Gradient (CG) method to solve the linear system in the Newton step. Develop the `newton_method_cg` function, which computes the Newton step using CG to solve the system $\nabla^2 f(x_k) d_k = - \nabla f(x_k), \; x_{k+1} = x_k + \alpha d_k$ defined by the Hessian. You have to use [`jax.scipy.sparse.linalg.cg`](https://jax.readthedocs.io/en/latest/_autosummary/jax.scipy.sparse.linalg.cg.html) function here. So, firstly compute the hessian as it was done in the code, then put it into this linear solver. Compare its performance in terms of computational efficiency and convergence rate to the standard Newton method.
    
    7. (5 points) Finally, implement a Hessian-free version of Newton's method (HFN) which utilizes Hessian-vector products derived via automatic differentiation. Note, that `jax.scipy.sparse.linalg.cg` function can take the matvec function, which directly produces the multiplication of any input vector $x$. Implement the HFN method without explicitly forming or storing the Hessian matrix in function `newton_method_hfn`. Use autograd to compute Hessian-vector products as it is described [here](https://iclr-blogposts.github.io/2024/blog/bench-hvp/). Compare this method's time complexity and memory requirements against previous implementations.

### Conditional gradient methods

1.  **Projection onto the Birkhoff Polytope using Frank-Wolfe** [20 points]

    In a recent [book](https://arxiv.org/pdf/2211.14103) authors presented the following comparison table with complexities of linear minimizations and projections on some convex sets up to an additive error $\epsilon$ in the Euclidean norm. When $\epsilon$ is missing, there is no additive error. The $\tilde{\mathcal{O}}$ hides polylogarithmic factors in the dimensions and polynomial factors in constants related to thedistancetothe optimum. For the nuclear norm ball, i.e., the spectrahedron, $\nu$ denotes the number of non-zero entries and $\sigma_1$ denotes the top singular value of the projected matrix.

    | **Set**   | **Linear minimization**  | **Projection**    |
    |------------------------|--------------------|---------|
    | $n$-dimensional $\ell_p$-ball, $p \neq 1,2,\infty$ | $\mathcal{O}(n)$  | $\tilde{\mathcal{O}}\!\bigl(\tfrac{n}{\epsilon^2}\bigr)$|
    | Nuclear norm ball of $n\times m$ matrices | $\mathcal{O}\!\Bigl(\nu\,\ln(m + n)\,\tfrac{\sqrt{\sigma_1}}{\sqrt{\epsilon}}\Bigr)$    | $\mathcal{O}\!\bigl(m\,n\,\min\{m,n\}\bigr)$   |
    | Flow polytope on a graph with $m$ vertices and $n$ edges (capacity bound on edges) | $\mathcal{O}\!\Bigl((n \log m)\bigl(n + m\,\log m\bigr)\Bigr)$ | $\tilde{\mathcal{O}}\!\bigl(\tfrac{n}{\epsilon^2}\bigr)\ \text{or}\ \mathcal{O}(n^4\,\log n)$    |
    | Birkhoff polytope ($n \times n$ doubly stochastic matrices)   | $\mathcal{O}(n^3)$| $\tilde{\mathcal{O}}\!\bigl(\tfrac{n^2}{\epsilon^2}\bigr)$   |

    The Birkhoff polytope, denoted as $B_n$, is the set of $n \times n$ doubly stochastic matrices:
    $$
    B_n = \{ X \in \mathbb{R}^{n \times n} \mid X_{ij} \ge 0 \;\forall i,j, \quad X \mathbf{1} = \mathbf{1}, \quad X^T \mathbf{1} = \mathbf{1} \}
    $$
    where $\mathbf{1}$ is the vector of all ones. This set is convex and compact. Its extreme points are the permutation matrices.

    Given an arbitrary matrix $Y \in \mathbb{R}^{n \times n}$, we want to find its projection onto $B_n$, which is the solution to the optimization problem:
    $$
    \min_{X \in B_n} f(X) = \frac{1}{2} \| X - Y \|_F^2
    $$
    where $\| \cdot \|_F$ is the Frobenius norm.

    We will use the Frank-Wolfe (Conditional Gradient) algorithm to solve this problem. Recall the steps of the Frank-Wolfe algorithm:
    *   Initialize $X_0 \in B_n$.
    *   For $k = 0, 1, 2, \ldots$:
        *   Compute the gradient $\nabla f(X_k)$.
        *   Solve the Linear Minimization Oracle (LMO): $S_k = \arg\min_{S \in B_n} \langle \nabla f(X_k), S \rangle$.
        *   Determine the step size $\gamma_k \in [0, 1]$.
        *   Update $X_{k+1} = (1-\gamma_k) X_k + \gamma_k S_k$.

    **Tasks:**

    1.  [5 points] Explicitly write down the gradient $\nabla f(X_k)$. Explain how to solve the LMO step $\min_{S \in B_n} \langle \nabla f(X_k), S \rangle$. What kind of matrix is the solution $S_k$? *Hint: Consider the connection to the linear assignment problem (Hungarian algorithm).*

    2.  [10 points] Implement the Frank-Wolfe algorithm in Python to solve the projection problem. Use `scipy.optimize.linear_sum_assignment` to solve the LMO. For the step size, you can use the optimal closed-form solution for projection: $\gamma_k = \frac{\langle X_k - Y, X_k - S_k \rangle}{\| X_k - S_k \|_F^2}$, clipped to $[0, 1]$.

        ```python
        import numpy as np
        from scipy.optimize import linear_sum_assignment
        import matplotlib.pyplot as plt

        def project_to_birkhoff_frank_wolfe(Y, max_iter=100, tol=1e-6):
            """
            Projects matrix Y onto the Birkhoff polytope using the Frank-Wolfe algorithm.

            Args:
                Y (np.ndarray): The matrix to project.
                max_iter (int): Maximum number of iterations.
                tol (float): Tolerance for convergence (change in objective value).

            Returns:
                np.ndarray: The projection of Y onto the Birkhoff polytope.
                list: History of objective function values.
            """
            n = Y.shape[0]
            assert Y.shape[0] == Y.shape[1], "Input matrix must be square"

            # Initialize with a feasible point (e.g., uniform matrix)
            Xk = np.ones((n, n)) / n
            
            objective_history = []

            for k in range(max_iter):
                # Objective function value
                obj_val = 0.5 * np.linalg.norm(Xk - Y, 'fro')**2
                objective_history.append(obj_val)
                
                if k > 0 and abs(objective_history[-1] - objective_history[-2]) < tol:
                    print(f"Converged after {k} iterations.")
                    break

                # 1. Compute gradient
                grad_fk = ... # YOUR CODE HERE 

                # 2. Solve the LMO: S_k = argmin_{S in Birkhoff} <grad_fk, S>
                # Use linear_sum_assignment on the cost matrix grad_fk
                row_ind, col_ind = ... # YOUR CODE HERE using linear_sum_assignment
                Sk = np.zeros((n, n))
                # Construct permutation matrix Sk based on row_ind, col_ind
                ... # YOUR CODE HERE 

                # 3. Compute step size gamma_k 
                # Optimal step size for projection, clipped to [0, 1]
                delta_k = Xk - Sk
                denom = np.linalg.norm(delta_k, 'fro')**2
                if denom < 1e-12: # Avoid division by zero if Xk is already the vertex Sk
                    gamma_k = 0.0
                else:
                    gamma_k = ... # YOUR CODE HERE for optimal step size
                    gamma_k = np.clip(gamma_k, 0.0, 1.0) 

                # 4. Update
                Xk = ... # YOUR CODE HERE 

            else: # If loop finishes without breaking
                 print(f"Reached max iterations ({max_iter}).")

            return Xk, objective_history
        ```

    3.  [5 points] Test your implementation with $n=5$ and a randomly generated matrix $Y = \text{np.random.rand}(5, 5)$. Run the algorithm for 200 iterations. Plot the objective function value $f(X_k)$ versus the iteration number $k$. Verify numerically that the final matrix $X_{200}$ approximately satisfies the conditions for being in $B_5$ (non-negative entries, row sums equal to 1, column sums equal to 1).

1.  **[Minimizing a Quadratic over the Simplex]** [20 points]
    Consider the problem of minimizing a quadratic function over the standard probability simplex:
    $$
    \min_{x \in \Delta_n} f(x) = \frac{1}{2} x^T Q x + c^T x
    $$
    where $\Delta_n = \{x \in \mathbb{R}^n \mid \sum_{i=1}^n x_i = 1, x_i \ge 0\}$ is the standard simplex in $\mathbb{R}^n$, $Q \in \mathbb{S}^n_{++}$ is a positive definite matrix, and $c \in \mathbb{R}^n$.

    *   [5 points] Generate the problem data: Choose a dimension $n$ (e.g., $n=20$). Create a random positive definite matrix $Q$ with a given spectrum $[\mu; L]$ and a random vector $x^* \in \Delta_n$, so $c = -Q x^*$ (e.g., with standard normal entries).
        * Specify and consider $2$ different starting points (you will use them for another algorithm as well)
        * Calculate $f(x^*)$ and $f(x_0)$, you will have to track $|f(x_k) - f(x^*)|$ for both algorithms
    *   [7 points] Implement the Frank-Wolfe (Conditional Gradient) algorithm to solve this problem. Do not forget to start from a feasible point. 
    *   [8 points] Implement the Projected Gradient Descent algorithm.
        *   Use the same starting points $x_0$. 
        *   Justify learning rate selection.
        *   We do not have an explicit formula for Euclidean projection onto the standard simplex. You will need an algorithm for projection onto the standard simplex (e.g., see [Duchi et al., 2008](https://stanford.edu/~jduchi/projects/DuchiShSiCh08.pdf) or use available implementations).
    *   Plot the objective function value $f(x_k)$ versus the iteration number $k$ for both Frank-Wolfe and Projected Gradient Descent on the same graph. Compare their convergence behavior. Discuss which method appears to converge faster in terms of iterations for this problem.

### Subgradient method

1. **Finding a point in the intersection of convex sets.** [30 points] Let $A \in \mathbb{R}^{n \times n}$ be some non-degenerate matrix and let $\Sigma$ be an $n \times n$ diagonal matrix with diagonal entries $\sigma_1,...,\sigma_n > 0$, and $y$ a given vector in $\mathbb{R}^n$. Consider the compact convex sets $U = \{x \in \mathbb{R}^n \mid \|A(x-y)\|_2 \leq 1\}$ and $V = \{x \in \mathbb{R}^n \mid \|\Sigma x\|_\infty \leq 1\}$.

    * [10 points] Minimize maximum distance from the current point to the convex sets. 

        $$
        \min_{x\in\mathbb{R}^n} f(x) =  \min_{x\in\mathbb{R}^n} \max\{\mathbf{dist}(x, U), \mathbf{dist}(x, V)\}
        $$

        propose an algorithm to find a point $x \in U \cap V$. You can assume that $U \cap V$ is not empty. Your algorithm must be specific and provably converging (although you do not need to prove it and you can simply refer to the lecture slides).

    * [15 points] Implement your algorithm with the following data: $n = 2$, $y = (3, 2)$, $\sigma_1 = 0.5$, $\sigma_2 = 1$, 

        $$
        A = \begin{bmatrix} 
        1 & 0 \\
        -1 & 1 
        \end{bmatrix},
        $$

        Plot the objective value of your optimization problem versus the number of iterations. Choose the following initial points $x_0 = [(2, -1), (0, 0), (1, 2)]$. 

    * [5 points] Discussion: compare the three curves. Describe the properties of this optimization problem. 
    
        * Is it convex/strongly convex? 
        * Is it smooth? 
        * Do we have a unique solution here? 
        * Which start converges fastest / slowest and why? Relate your observations to the initial distance to $U \cap V$ and to the contact angle between the two sets at the solution.

    ![Illustration of the problem](convex_intersection.png)

1. **Subgradient methods for Lasso.**  (10 points)

    Consider the optimization problem 

    $$
    \min_{x \in \mathbb{R}^n} f(x) := \frac12 \|Ax - b\|^2 + \lambda \|x\|_1,
    $$

    with variables $x \in \mathbb{R}^n$ and problem data $A \in \mathbb{R}^{m \times n}$, $b \in \mathbb{R}^m$ and $\lambda > 0$. This model is known as Lasso, or Least Squares with $l_1$ regularization, which encourages sparsity in the solution via the non-smooth penalty $\|x\|_1 := \sum_{j=1}^n |x_j|$. In this problem, we will explore various subgradient methods for fitting this model.

    * Derive the subdifferential $\partial f(x)$ of the objective.

    * Find the update rule of the subgradient method and state the computational complexity of applying one update using big O notation in terms of the dimensions.

    * Let $n = 1000$, $m = 200$ and $\lambda = 0.01$. Generate a random matrix $A \in \mathbb{R}^{m \times n}$ with independent Gaussian entries with mean 0 and variance $1/m$, and a fixed vector $x^* = {\underbrace{[1, \ldots, 1}_{\text{k times}}, \underbrace{0, \ldots, 0]}_{\text{n-k times}}}^T \in \mathbb{R}^n$. Let $k = 5$ and then set $b = Ax^*$. Implement the subgradient method to minimize $f(x)$, initialized at the all-zeros vector. Try different step size rules, including:
        * constant step size $\alpha_k = \alpha$
        * constant step length $\alpha_k = \frac{\gamma}{\|g_k\|_2}$ (so $\|x^{k+1} - x^k\|_2 = \gamma$)
        * Inverse square root $\frac{1}{\sqrt{k}}$
        * Inverse $\frac1k$
        * Polyak's step length with estimated objective value:

            $$
            \alpha_k = \frac{f(x_k) - f_k^{\text{best}} + \gamma_k}{\|g_k\|_2^2}, \quad \text{ with} \sum_{k=1}^\infty \gamma_k = \infty, \quad \sum_{k=1}^\infty \gamma_k^2 < \infty
            $$

            For example, one can use $\gamma_k = \frac{10}{10 + k}$. Here  $f_k^{\text{best}} - \gamma_k$ serves as estimate of $f^*$. It is better to take $\gamma_k$ in the same scale as the objective value. One can show, that $f_k^{\text{best}} \to f^*$.

    
        Plot objective value versus iteration curves of different step size rules on the same figure.

    * Repeat previous part using a heavy ball term, $\beta_k(x^k - x^{k-1})$, added to the subgradient. Try different step size rules as in the previous part and tune the heavy ball parameter $\beta_k = \beta$ for faster convergence.

### Proximal gradient method

1. [20 points] **Proximal Method for Sparse Softmax Regression** Softmax regression, also known as multinomial logistic regression, is a generalization of logistic regression to multiple classes. It is used to model categorical outcome variables where each category is mutually exclusive. The softmax function transforms any input vector to the probability-like vector as follows:

    $$
    P(y = j | x; W) = \frac{e^{W_j^T x}}{\sum\limits_{i=1}^{c} e^{W_i^T x}}
    $$

    ![Scheme of softmax regression](Softmax.svg)

    where $x$ is the input vector, $W$ is the weight matrix, $c$ is the number of classes, and $P(y = j | x; W)$ is the probability that the input $x$ belongs to class $j$.

    The optimization problem for softmax regression is to minimize the negative log-likelihood:

    $$
    \min_{W \in \mathbb{R}^{c \times d}} -\sum_{i=1}^{N} \log P(y_i | x_i; W) + \lambda \| W \|_1
    $$

    where $N$ is the number of training examples, $\lambda$ is the regularization parameter, and $\| W \|_1$ is the L1 norm of the weight matrix, which promotes sparsity in the solution. I suggest you to vectorize matrix and add $1$-vector norm.

    We will solve the sparse softmax regression problem using the subgradient method and the proximal gradient method, both incorporating L1 regularization. The proximal gradient method is particularly useful for optimization problems involving non-smooth regularizers like the L1 norm. We will use 3 class classification problem of [Predicting Students' Dropout and Academic Success](https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success). In this problem you should start with this [\faPython colab notebook](https://colab.research.google.com/github/MerkulovDaniil/optim/blob/master/assets/Notebooks/Proximal_softmax_regression.ipynb)

    1. [4 points] Write down exact formulation of subgradient method and proximal gradient method here (you can not use any optimization problems in this formulation).
    1. [6 points] Choose $\lambda = 0$. Solve the softmax regression problem using subgradient method and proximal gradient descent. Find the highest learning (individually), acceptable for both methods to converge. Report convergence curves and report final sparsity of both methods. Draw you conclusions.
    1. [10 points] Solve non-smooth problem and fill the following table. For each value of $\lambda$ provide convergence curves.

    Report the number of iterations needed to reach specified primal gaps for each method. Present the results in the following markdown table:

    | Method                     | Learning Rate ($\eta$)   | Tolerance ($\epsilon$) | Number of Iterations | Comment(if any)          | Final Sparsity of the solution | $\lambda$ | Final test accuracy |
    |:--------------------------:|:------------------------:|-----------------------:|:--------------------:|:------------------------:|:------------------------------:|:----------|:-------------------:|
    | subgradient method        |                          | $10^{-1}$              |                      |                          |                                |  `1e-2`   |                     |
    | subgradient method        |                          | $10^{-2}$              |                      |                          |                                |  `1e-2`   |                     |
    | subgradient method        |                          | $10^{-3}$              |                      |                          |                                |  `1e-2`   |                     |
    | subgradient method        |                          | $10^{-4}$              |                      |                          |                                |  `1e-2`   |                     |
    | subgradient method        |                          | $10^{-5}$              |                      |                          |                                |  `1e-2`   |                     |
    | Proximal Gradient Descent  |                          | $10^{-1}$              |                      |                          |                                |  `1e-2`   |                     |
    | Proximal Gradient Descent  |                          | $10^{-2}$              |                      |                          |                                |  `1e-2`   |                     |
    | Proximal Gradient Descent  |                          | $10^{-3}$              |                      |                          |                                |  `1e-2`   |                     |
    | Proximal Gradient Descent  |                          | $10^{-4}$              |                      |                          |                                |  `1e-2`   |                     |
    | Proximal Gradient Descent  |                          | $10^{-5}$              |                      |                          |                                |  `1e-2`   |                     |
    | subgradient method        |                          | $10^{-2}$              |                      |                          |                                |  `1e-3`   |                     |
    | Proximal Gradient Descent  |                          | $10^{-2}$              |                      |                          |                                |  `1e-3`   |                     |
    | subgradient method        |                          | $10^{-2}$              |                      |                          |                                |  `1e-1`   |                     |
    | Proximal Gradient Descent  |                          | $10^{-2}$              |                      |                          |                                |  `1e-1`   |                     |
    | subgradient method        |                          | $10^{-2}$              |                      |                          |                                |  `1`      |                     |
    | Proximal Gradient Descent  |                          | $10^{-2}$              |                      |                          |                                |  `1`      |                     |


### Stochastic gradient methods

1. **Variance reduction for stochastic gradient methods**. [20 points]

    [5 points]Open [\faPython colab notebook](https://colab.research.google.com/github/MerkulovDaniil/optim/blob/master/assets/Notebooks/VR_exercise.ipynb). Implement SAG and SVRG method. Consider Linear least squares problem with the following setup

    ```python
    params = {
        "mu": 0,
        "m": 50,
        "n": 100,
        "methods": [
            {
                "method": "SGD",
                "learning_rate": 1e-2,
                "batch_size": 2,
                "iterations": 1000,
            },
            {
                "method": "SGD",
                "learning_rate": 1e-2,
                "batch_size": 50,
                "iterations": 1000,
            },
            {
                "method": "SAG",
                "learning_rate": 1e-2,
                "batch_size": 2,
                "iterations": 1000,
            },
            {
                "method": "SVRG",
                "learning_rate": 1e-2,
                "epoch_length": 2,
                "batch_size": 2,
                "iterations": 1000,
            },
        ]
    }

    results = run_experiments(params)
    ```

    [5 points] Then, consider strongly convex case with:

    ```python
    params = {
        "mu": 1e-1,
        "m": 50,
        "n": 100,
        "methods": [
            {
                "method": "SGD",
                "learning_rate": 1e-2,
                "batch_size": 2,
                "iterations": 2000,
            },
            {
                "method": "SGD",
                "learning_rate": 1e-2,
                "batch_size": 50,
                "iterations": 2000,
            },
            {
                "method": "SAG",
                "learning_rate": 1e-2,
                "batch_size": 2,
                "iterations": 2000,
            },
            {
                "method": "SVRG",
                "learning_rate": 1e-2,
                "epoch_length": 2,
                "batch_size": 2,
                "iterations": 2000,
            },
        ]
    }
    ```

    [5 points] And for the convex binary logistic regression:

    ```python
    params = {
        "mu": 0,
        "m": 100,
        "n": 200,
        "methods": [
            {
                "method": "SGD",
                "learning_rate": 1e-2,
                "batch_size": 2,
                "iterations": 2000,
            },
            {
                "method": "SAG",
                "learning_rate": 1e-2,
                "batch_size": 2,
                "iterations": 2000,
            },
            {
                "method": "SVRG",
                "learning_rate": 1e-2,
                "epoch_length": 3,
                "batch_size": 2,
                "iterations": 2000,
            },
            {
                "method": "SGD",
                "learning_rate": 1e-2,
                "batch_size": 100,
                "iterations": 2000,
            },
        ]
    }
    ```

    [5 points] and strongly convex case

    ```python
    params = {
        "mu": 1e-1,
        "m": 100,
        "n": 200,
        "methods": [
            {
                "method": "SGD",
                "learning_rate": 2e-2,
                "batch_size": 2,
                "iterations": 3000,
            },
            {
                "method": "SAG",
                "learning_rate": 2e-2,
                "batch_size": 2,
                "iterations": 3000,
            },
            {
                "method": "SVRG",
                "learning_rate": 2e-2,
                "epoch_length": 3,
                "batch_size": 2,
                "iterations": 3000,
            },
            {
                "method": "SGD",
                "learning_rate": 2e-2,
                "batch_size": 100,
                "iterations": 3000,
            },
        ]
    }
    ```

    Describe the obtained convergence and compare methods.


    ![](lls_VR.svg)

    ![](logreg_VR.svg)

<!-- 1. **Do we need to tune hyperparameters for stochastic gradient methods?** [15 points]

    The performance of stochastic gradient-based optimizers, such as Adam and AdamW, is highly dependent on their hyperparameters. While the learning rate together with the momentum term are often the most tuned parameters, other hyperparameters like the exponential decay rates for the moment estimates, `beta1` and `beta2`, can also have an impact on training dynamics and final model performance. In this problem, you will investigate the sensitivity of Adam and AdamW to these beta parameters when training a small Transformer model.

    Your task is to train a small Transformer on the TinyStories dataset and perform a grid search over `beta1` and `beta2` for both `optax.adam` and `optax.adamw`. You will then visualize the results as heatmaps to analyze the performance landscape of these hyperparameters.

    **Tasks:**

    1.  **[5 points] Implement the basic training loop:**
        *   Define a small Transformer model. You can use the tutorial [here](https://docs.jaxstack.ai/en/latest/JAX_for_LLM_pretraining.html).
        *   Load and preprocess a subset of the TinyStories dataset.
        *   Create a training loop that, for a given set of hyperparameters (`beta1`, `beta2`), trains the model for a fixed number of epochs (e.g., 1).
        *   For each method find a set of hyperparameters (try the default values) and training length, that satisfies the loss value on the test set to be less than $3.14$.

    2.  **[10 points] Visualize and Analyze:**
        *   Perform a grid search for `beta1` and `beta2` for both `optax.adam` and `optax.adamw`. Suggested grid:
            *   `beta1_values = jnp.linspace(0.8, 0.99, 10)`
            *   `beta2_values = jnp.linspace(0.8, 0.9999, 10)`
        *   For each pair of (`beta1`, `beta2`), record the final training loss and test loss.
        *   For each optimizer (Adam and AdamW), create two 2D heatmaps (4 in total):
            1.  A heatmap showing the final **training loss** for each (`beta1`, `beta2`) pair.
            2.  A heatmap showing the final **test loss** for each (`beta1`, `beta2`) pair.
        *   The axes of the heatmaps should correspond to `beta1` and `beta2` values. The color of each cell should represent the loss.
        *   Analyze the heatmaps. Which hyperparameter settings yield the best performance for each optimizer? Are the optimizers sensitive to changes in `beta1` and `beta2`? Do you observe significant differences between Adam and AdamW in terms of their sensitivity or optimal beta values?

    You can use the following code as a starting point. You can use any other libraries you want, i.e. you are not restricted to JAX. 

    ```python
    import jax
    import jax.numpy as jnp
    import optax
    import flax.nnx as nnx
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from tqdm.auto import tqdm
    import requests
    import tiktoken
    from sklearn.model_selection import train_test_split

    # --- Model Definition ---
    def causal_attention_mask(seq_len):
        return jnp.tril(jnp.ones((seq_len, seq_len)))

    class TransformerBlock(nnx.Module):
        def __init__(self, embed_dim, num_heads, ff_dim, rngs: nnx.Rngs, rate: float = 0.1):
            self.mha = nnx.MultiHeadAttention(num_heads=num_heads, in_features=embed_dim, rngs=rngs)
            self.dropout1 = nnx.Dropout(rate=rate, rngs=rngs)
            self.layer_norm1 = nnx.LayerNorm(num_features=embed_dim, rngs=rngs)
            self.linear1 = nnx.Linear(in_features=embed_dim, out_features=ff_dim, rngs=rngs)
            self.linear2 = nnx.Linear(in_features=ff_dim, out_features=embed_dim, rngs=rngs)
            self.dropout2 = nnx.Dropout(rate=rate, rngs=rngs)
            self.layer_norm2 = nnx.LayerNorm(num_features=embed_dim, rngs=rngs)

        def __call__(self, inputs, training: bool = False):
            mask = causal_attention_mask(inputs.shape[1])
            attention_output = self.mha(inputs_q=inputs, mask=mask, decode=False)
            attention_output = self.dropout1(attention_output, deterministic=not training)
            out1 = self.layer_norm1(inputs + attention_output)
            ffn_output = self.linear2(nnx.relu(self.linear1(out1)))
            ffn_output = self.dropout2(ffn_output, deterministic=not training)
            return self.layer_norm2(out1 + ffn_output)

    class TokenAndPositionEmbedding(nnx.Module):
        def __init__(self, maxlen, vocab_size, embed_dim, rngs: nnx.Rngs):
            self.token_emb = nnx.Embed(num_embeddings=vocab_size, features=embed_dim, rngs=rngs)
            self.pos_emb = nnx.Embed(num_embeddings=maxlen, features=embed_dim, rngs=rngs)

        def __call__(self, x):
            positions = jnp.arange(0, x.shape[1])[None, :]
            return self.token_emb(x) + self.pos_emb(positions)

    class MiniGPT(nnx.Module):
        def __init__(self, maxlen, vocab_size, embed_dim, num_heads, ff_dim, num_blocks, rngs: nnx.Rngs):
            self.embedding_layer = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim, rngs=rngs)
            self.transformer_blocks = [TransformerBlock(embed_dim, num_heads, ff_dim, rngs=rngs) for _ in range(num_blocks)]
            self.output_layer = nnx.Linear(in_features=embed_dim, out_features=vocab_size, rngs=rngs)

        def __call__(self, inputs, training: bool = False):
            x = self.embedding_layer(inputs)
            for block in self.transformer_blocks:
                x = block(x, training=training)
            return self.output_layer(x)

    # --- Data Loading ---
    def load_and_preprocess_data(maxlen, num_samples=5000):
        url = 'https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-train.txt?download=true'
        response = requests.get(url)
        text = response.text
        stories = text.split('<|endoftext|>')
        stories = [story.strip() for story in stories if story.strip()]
        
        tokenizer = tiktoken.get_encoding("gpt2")
        
        # Use a subset of data for speed
        data = [tokenizer.encode(s, allowed_special={'<|endoftext|>'}) for s in stories[:num_samples]]
        
        # Pad sequences
        padded_data = np.zeros((len(data), maxlen), dtype=int)
        for i, d in enumerate(data):
            seq_len = min(len(d), maxlen)
            padded_data[i, :seq_len] = d[:seq_len]

        train_data, test_data = train_test_split(padded_data, test_size=0.2, random_state=42)
        return train_data, test_data, tokenizer

    # --- Training and Evaluation ---
    def loss_fn(model, batch):
        inputs, targets = batch
        logits = model(inputs, training=True)
        loss = optax.softmax_cross_entropy_with_integer_labels(
            logits=logits, labels=targets).mean()
        return loss

    @nnx.jit
    def train_step(model: MiniGPT, optimizer: nnx.Optimizer, batch):
        grad_fn = nnx.value_and_grad(loss_fn)
        loss, grads = grad_fn(model, batch)
        optimizer.update(grads)
        return loss

    @nnx.jit
    def eval_step(model: MiniGPT, batch):
        inputs, targets = batch
        logits = model(inputs, training=False)
        return optax.softmax_cross_entropy_with_integer_labels(logits=logits, labels=targets).mean()

    def create_batches(data, batch_size):
        num_batches = len(data) // batch_size
        for i in range(num_batches):
            batch_data = data[i*batch_size:(i+1)*batch_size]
            inputs = batch_data[:, :-1]
            targets = batch_data[:, 1:]
            yield inputs, targets
    
    def get_full_ds_loss(model, data, batch_size):
        losses = []
        for batch in create_batches(data, batch_size):
            losses.append(eval_step(model, batch))
        return jnp.mean(jnp.array(losses))


    def train_and_evaluate(optimizer_name, beta1, beta2, num_epochs, model_params, data, batch_size):
        rngs = nnx.Rngs(0)
        model = MiniGPT(**model_params, rngs=rngs)
        
        if optimizer_name == 'adam':
            optimizer = nnx.Optimizer(model, optax.adam(learning_rate=1e-3, b1=beta1, b2=beta2))
        else:
            optimizer = nnx.Optimizer(model, optax.adamw(learning_rate=1e-3, b1=beta1, b2=beta2))
        
        train_data, test_data = data

        for epoch in range(num_epochs):
            # Simple shuffling
            key = jax.random.PRNGKey(epoch)
            perm = jax.random.permutation(key, len(train_data))
            train_data_shuffled = train_data[perm]
            
            for batch in create_batches(train_data_shuffled, batch_size):
                train_step(model, optimizer, batch)

        final_train_loss = get_full_ds_loss(model, train_data, batch_size)
        final_test_loss = get_full_ds_loss(model, test_data, batch_size)
        
        return final_train_loss, final_test_loss

    def plot_heatmap(data, title, xlabel, ylabel, xticklabels, yticklabels):
        ### YOUR CODE HERE ###
        plt.show()

    # --- Main Execution ---
    if __name__ == '__main__':
        # Model and training parameters
        maxlen = 128
        embed_dim = 128
        num_heads = 4
        ff_dim = 128
        num_blocks = 2
        num_epochs = 3
        batch_size = 64
        
        train_data, test_data, tokenizer = load_and_preprocess_data(maxlen)
        vocab_size = tokenizer.n_vocab

        model_params = {
            "maxlen": maxlen, "vocab_size": vocab_size, "embed_dim": embed_dim,
            "num_heads": num_heads, "ff_dim": ff_dim, "num_blocks": num_blocks
        }

        beta1_values = jnp.linspace(0.8, 0.99, 10)
        beta2_values = jnp.linspace(0.8, 0.9999, 10)

        for optimizer_name in ['adam', 'adamw']:
            print(f"--- Tuning {optimizer_name.upper()} ---")
            train_losses = np.zeros((len(beta2_values), len(beta1_values)))
            test_losses = np.zeros((len(beta2_values), len(beta1_values)))

            for i, beta2 in enumerate(tqdm(beta2_values, desc=f'{optimizer_name} beta2')):
                for j, beta1 in enumerate(tqdm(beta1_values, desc=f'{optimizer_name} beta1')):
                    train_loss, test_loss = train_and_evaluate(
                        optimizer_name, beta1, beta2, num_epochs, 
                        model_params, (train_data, test_data), batch_size
                    )
                    train_losses[i, j] = train_loss
                    test_losses[i, j] = test_loss

            plot_heatmap(train_losses, f'{optimizer_name.upper()} - Final Train Loss',
                         'beta1', 'beta2', beta1_values, beta2_values)
            plot_heatmap(test_losses, f'{optimizer_name.upper()} - Final Test Loss',
                         'beta1', 'beta2', beta1_values, beta2_values)
    ``` -->

### Neural network training

1.  **Anomaly detection with neural network.** [30 points] 

    In this problem we will try to detect anomalies in time series with neural network. 
    
    :::{.plotly}
    anomaly_detection.html
    :::

    We will train the model to reconstruct normal data and when the reconstruction error for the actual data on trained model is high, we report an anomaly. Start with this notebook [\faPython colab notebook](https://colab.research.google.com/github/MerkulovDaniil/optim/blob/master/assets/Notebooks/time_series_anomaly.ipynb). The default solution is adam and after training it can detect 4 out of 5 anomalies. Train and compare several methods on the same problem. For each method try to find hyperparameters, which ensures at least 3 out of 5 anomalies detection. Present learning curves and anomaly predictions for each method.

    * SGD with momentum [5 points] from optax
    * Adadelta [5 points] from optax
    * BFGS [10 points] implemented manually
    * [Muon](https://github.com/KellerJordan/Muon) [optimizer](https://arxiv.org/pdf/2502.16982) [10 points] implemented manually

### Big models

1. **Fit the largest model you can on a single GPU.** [15 points]

    In this assignment, you will train a language model (LM) using the TinyStories dataset, focusing on optimizing model performance within the constraints of Google Colab's hardware. For the sake of speed, we will do it on the part of the dataset.
    
    ```Tiny_Stories
    Once upon a time, there was a little car named Beep. Beep loved to go fast and play in the sun. 
    Beep was a healthy car because he always had good fuel....
    ```

    Your objective is to maximize the size of the model without exceeding the available computational resources (~ 16GB VRAM). You could start with the Hugging Face Transformers library and experiment with various memory optimization techniques, such as (but not limited to):

    * Different batch size
    * Different optimizer
    * Gradient accumulation
    * Activation checkpointing
    * CPU offloading
    * 8bit optimizers

    You have a baseline of training `gpt-2` model prepared at the following [\faPython colab notebook](https://colab.research.google.com/github/MerkulovDaniil/optim/blob/master/assets/Notebooks/TinyStories_baseline.ipynb). You can easily switch it to `opt-350m`, `opt-1.3b`, `gpt2` etc. You can find a great beginner-level guide on the topic [here](https://huggingface.co/docs/transformers/v4.18.0/en/performance).

    ```GPT2
    A long time ago in a galaxy far far away... a little girl named Lily was playing in the garden. She was so excited! She wanted to explore the garden and see what was around her.
    Suddenly, she heard a loud noise. Lily looked up and saw a big, hairy creature. Lily was so excited! She ran to the creature and grabbed it by the arm. The creature was so big and hairy that Lily couldn't help but laugh. 
    ```

    ![](gpt2_generation.jpeg)

    You have to fill this table with your description/observations.

    | Setup | # of parameters | GPU peak memory, MB | Final eval loss | Batch Size | Time to run 5 epochs, s | Generation example | Comment |
    |:---:|:---:|:---:|:---:|:---:|:---:|:---------:|:---------:|
    | Baseline (OPT-125M) | 125 M | 9044 | 1.928 | 8 | 442.34 | `A long time ago in a galaxy far far away... there was a little girl named Lily. She was three years old and loved to explore. One day, she decided to go for a walk in the park. Lily was so excited to go for a walk. She asked her mom, "What do you want to do?" Her mom smiled and said, "I want to explore the galaxy." Lily was so excited to explore the galaxy.` |  |
    | Baseline (GPT2-S) | 124 M | 13016 | 2.001 | 8 | 487.75 | `A long time ago in a galaxy far far away... a little girl named Lily was playing in the garden. She was so excited! She wanted to explore the garden and see what was around her. Suddenly, she heard a loud noise. Lily looked up and saw a big, hairy creature. Lily was so excited! She ran to the creature and grabbed it by the arm. The creature was so big and hairy that Lily couldn't help but laugh.` | The generation seems more interesting, despite the fact, that eval loss is higher. |
    |  |  |  |  |  |  |  |  |
    |  |  |  |  |  |  |  |  |
    |  |  |  |  |  |  |  |  |
    |  |  |  |  |  |  |  |  |
     
    For each unique trick for memory optimization, you will get 3 points (maximum 15 points). A combination of tricks is not counted as a unique trick, but will, probably, be necessary to train big models. The maximum grade is bounded with the size of the trained model:

    * If the model size you train is <= 125M - you can get a maximum of 6 points.
    * If the model size you train is 126M <= 350M - you can get a maximum of 8 points.
    * If the model size you train is 350M <= 1B - you can get a maximum of 12 points.
    * If you fit 1B model or more - you can get a maximum 15 points.

### ADMM (Dual methods)

1. **Low‑Rank Matrix Completion via ADMM**  [25 points]

   **Background.** In many applications such as recommender systems, computer vision and system identification, the data matrix is approximately low‑rank but only a subset of its entries are observed. Recovering the missing entries can be posed as a convex program that combines a data‑fitting term with the nuclear norm, a convex surrogate for rank.

   We are given a partially observed matrix $M \in \mathbb{R}^{m\times n}$ and the index set of observed entries $\Omega \subseteq \{1,\dots,m\} \times \{1,\dots,n\}$. Define the sampling operator $P_\Omega : \mathbb{R}^{m\times n}\to\mathbb{R}^{m\times n}$ by $(P_\Omega(X))_{ij}= X_{ij}$ if $(i,j)\in\Omega$ and $0$ otherwise.

   We consider the optimization problem
   $$
   \min_{X\in\mathbb{R}^{m\times n}}\;\frac12\|P_\Omega(X-M)\|_F^2\; + \;\lambda\|X\|_*,
   $$
   where $\|X\|_* = \sum_k \sigma_k(X)$ is the nuclear norm.

   * **(a) [10 points] Derive a two‑block ADMM algorithm.**  
     Introduce an auxiliary variable $Z$ and rewrite the problem in the form  
     $$
     \min_{X,Z}\; \frac12\|P_\Omega(Z-M)\|_F^2 + \lambda\|X\|_* \quad\text{s.t. } X-Z = 0.
     $$
     Derive explicit closed‑form expressions for each ADMM update:

     * **$X$‑update:** singular‑value soft‑thresholding (SVT);
     * **$Z$‑update:** projection onto the observed entries (keep $M$ on $\Omega$, average with $X$ elsewhere);
     * **dual‑variable update.**

     State a practical stopping rule based on the primal and dual residuals.

   * **(b) [10 points] Implement the algorithm on synthetic data.**  
     Use the following set‑up (in Python):

     ```python
     import numpy as np
     np.random.seed(0)
     m, n, r = 50, 40, 3
     U = np.random.randn(m, r)
     V = np.random.randn(n, r)
     M_star = U @ V.T                      # ground‑truth low‑rank matrix
     mask = np.random.rand(m, n) < 0.3     # 30 % observations
     noise = 0.01 * np.random.randn(m, n)
     M = mask * (M_star + noise)           # observed matrix (zeros elsewhere)
     lambda_ = 1 / np.sqrt(max(m, n))
     ```

     1. Implement the ADMM algorithm derived in part (a).
     2. Run it from $X^0 = 0$ for three penalty parameters $\rho \in \{0.1, 1, 10\}$.
     3. For each $\rho$:
        * plot **(i)** the objective value and **(ii)** the relative reconstruction error $\frac{\|X^k - M_\star\|_F}{\|M_\star\|_F}$ versus iteration number;
        * report the number of iterations required until $\max(\|r_{\mathrm p}^k\|_F,\|r_{\mathrm d}^k\|_F) \le 10^{-3}$.

   * **(c) [5 points] Discussion.**  
     Compare the convergence behaviour across the three values of $\rho$. How does $\rho$ influence the rate at which the primal and dual residuals decrease? Comment on

     * the rank of the iterates (after SVT);
     * the trade‑off between data‑fit and nuclear‑norm penalty as $\lambda$ varies;
     * the quality of the reconstruction once the stopping criterion is met.

     Relate your observations to the theory of ADMM and to the sensitivity of singular‑value thresholding to the choice of $\rho$.  



### Bonus: Continuous time methods

1.  **SGD as a splitting scheme and the importance of batches order** [30 points]

    **Background: (to be honest you can do the task without reading it)**

    The standard Gradient Descent (GD) method for minimizing $f(x) = \frac{1}{n} \sum_{i=1}^n f_i(x)$ can be viewed as an Euler discretization of the gradient flow Ordinary Differential Equation (ODE):
    $$
    \frac{d x}{d t} = -\nabla f(x) = -\frac{1}{n} \sum_{i=1}^n \nabla f_i(x)
    $$
    Stochastic Gradient Descent (SGD), particularly with cycling through mini-batches without replacement, can be interpreted as a *splitting scheme* applied to this ODE. In a first-order splitting scheme for $\frac{dx}{dt} = A x = \sum_{i=1}^m A_i x$, we approximate the solution $x(h)$ by sequentially applying the flows corresponding to each $A_i$: $x(h) \approx e^{A_{\sigma(m)} h} \ldots e^{A_{\sigma(1)} h} x_0$ for some permutation $\sigma$. 
    
    In the [paper](https://arxiv.org/abs/2004.08981) authors show that for the linear least squares problem $f(x) = \frac{1}{2n}\|X x - y\|^2$, where $X$ is split into $m$ row blocks $X_i$, the corresponding ODE involves matrices $A_i = -\frac{1}{n} X_i^T X_i$. If $X_i^T = Q_i R_i$ is the QR decomposition ($Q_i$ has orthonormal columns), let $\Pi_i = I - Q_i Q_i^*$ be the projector onto the null space of $X_i$. The paper presents the following result for the asymptotic global error of the splitting scheme:
    
    **Theorem:** Let $A_i = -\frac{1}{n} X_i^T X_i$ for $i=1,\dots,m$. Assume each $A_i$ is negative semidefinite and does not have full rank, but their sum $A = \sum A_i$ does have full rank. Then, for any permutation $\sigma$ of $\{1, \dots, m\}$:
    $$
    \lim_{t \to \infty}\| e^{A_{\sigma(m)}t} \cdots e^{A_{\sigma(1)}t} - e^{At}\| = \left\|\prod_{i=1}^m \Pi_{\sigma(i)}\right\|
    $$

    This error bound depends on the product of projectors $\Pi_i$ and thus *on the order* specified by the permutation $\sigma$. Since one epoch of SGD corresponds to applying the Euler discretization of each local problem $\frac{dx}{dt} = A_i x$ sequentially, this suggests that the order in which batches are processed in SGD might affect convergence, especially over many epochs.

    **Tasks:**

    1.  **Investigating the Bound Distribution** [5 points]
        *   Consider a simple linear least squares problem.
            $$
            \frac{1}{2n}\|X \theta - y\|^2 \to \min_{\theta \in \mathbb{R}^{d}}, X \in \mathbb{R}^{n \times d}, y \in \mathbb{R}^n
            $$
            For example, generate a random matrix $X \in \mathbb{R}^{80 \times 20}$ and a random vector $y \in \mathbb{R}^{80}$.
        *   Split $X$ into $m=8$ batches (row blocks) sequentially $X_1, \ldots, X_8$, where each $X_i \in \mathbb{R}^{10 \times 20}$.
        *   For each batch $X_i \in \mathbb{R}^{10 \times 20}$, you have to compute the projector matrix $\Pi_i = I - Q_i Q_i^* \in \mathbb{R}^{20 \times 20}$, where $X_i^T = Q_i R_i$ is the (thin) QR decomposition of $X_i^T \in \mathbb{R}^{20 \times 10}$ ($Q_i \in \mathbb{R}^{20 \times r_i}, R_i \in \mathbb{R}^{r_i \times 10}$, with $r_i = \text{rank}(X_i) \le 10$).
        *   Calculate the error bound norm $E(\sigma) = \|\prod_{j=1}^m \Pi_{\sigma(j)}\|_2$ (where the product is a $20 \times 20$ matrix) for *all* $m! = 8! = 40320$ possible permutations $\sigma$. Note, that this quantity is a scalar and depends on the order of batches in multiplication (permutation $\sigma$), i.e. $\|\Pi_1 \Pi_2\| \neq \|\Pi_2 \Pi_1\|$.
        *   Plot a histogram of the distribution of these scalar $E(\sigma)$ values. Does the order seem to matter significantly in this random case?

    2.  **Maximizing Order Dependence with adversarial dataset construction** [20 points]
        *   Modify the structure of the matrix $X$ (or the way it is split into $X_i$, but you cannot change the number of batches and their size) from Task 1 to create a scenario where the distribution of the error bounds $E(\sigma)$ has a significantly larger variance (to be precise, the ratio of the maximum to minimum values for different permutations should be maximized). *Hint: Think about how the projectors $\Pi_i$ interact. How could you make the product $\Pi_{\sigma(m)} \cdots \Pi_{\sigma(1)}$ very different for different orders $\sigma$? Consider cases where the null spaces have specific overlaps or orthogonality properties.*
        *   Explain your reasoning for the modification.
        *   Repeat the calculation and plotting from Task 1 for your modified problem to demonstrate the increased variance in the error bounds. Report the ratio of the maximum to minimum values for different permutations before and after adversarial dataset construction.

    3.  **Testing SGD Convergence** [5 points]
        *   Using the adversarial dataset from Task 2, identify two specific permutations: $\sigma_{\text{low}}$ with a low error bound $E(\sigma_{\text{low}})$ and $\sigma_{\text{high}}$ with a high error bound $E(\sigma_{\text{high}})$.
        *   Implement SGD for the linear least squares problem $\min_x \frac{1}{2n} \|Xx - y\|^2$. Use a fixed, small learning rate (e.g., $\alpha = 0.01/L$ where $L$ is the Lipschitz constant of the full gradient).
        *   Run SGD for a sufficient number of epochs (e.g., 50-100), applying the batches *deterministically* according to the order defined by $\sigma_{\text{low}}$ in each epoch. Record the squared error $\|X x_k - y\|^2$ at the end of each epoch $k$.
        *   Repeat the SGD run using the fixed batch order defined by $\sigma_{\text{high}}$.
        *   Plot the convergence curves (squared error vs. epoch number) for both $\sigma_{\text{low}}$ and $\sigma_{\text{high}}$ on the same graph.
        *   Discuss your results. Does the observed convergence speed of SGD correlate with the theoretical asymptotic error bound $E(\sigma)$? Does the order of batches appear to matter more in your modified problem compared to the random one?