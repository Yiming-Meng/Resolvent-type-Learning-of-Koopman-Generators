import sympy as sp
import numpy as np

# define a function to extract the coefficients of the exact monomials
def extract_coefficients(m_monomial, n_monomial, p_monomial):
    # Define the number of monomials per dimension
    dim = 3
    X = np.array([sp.symbols(f'X{i+1}') for i in range(dim)])

    # Define the ODE system symbolically
    dXdt = [None] * dim
    sigma = 10.
    beta = 8/30
    rho = 0.28
    dXdt[0] = sigma*X[1] - X[0]
    dXdt[1] = X[0] * (rho - X[2])-0.1*X[1]
    dXdt[2] = X[0] * X[1] - beta*X[2]

    # Generate all basis functions using itertools.product
    basis_functions = []
    # Iterate over the combinations of m, n, p
    for p in range (p_monomial):
        for n in range (n_monomial):
            for m in range(m_monomial):
            # Define the symbolic expression for the basis function
                basis_function = X[0]**m * X[1]**n * X[2]**p
                # Append the basis function to the list
                basis_functions.append(basis_function)
        
    # Initialize a matrix to store coefficients
    coeff_matrix = sp.zeros(dim, m_monomial*n_monomial*p_monomial)

    # Extract coefficients
    for i in range(dim):
        expanded_expr = sp.expand(dXdt[i])
        for j, basis in enumerate(basis_functions):
            # Get the coefficient of the exact monomial (basis function) in the expression
            coeff = expanded_expr.as_coefficients_dict().get(basis, 0)
            coeff_matrix[i, j] = coeff

    # Convert to a numpy array if needed
    coeff_matrix_np = np.array(coeff_matrix).astype(float)
    return coeff_matrix_np

# a function to extract the expressions of the bases
def extract_bases(m_monomial, n_monomial, p_monomial):
    # Define the number of monomials per dimension
    dim = 3
    X = np.array([sp.symbols(f'X{i+1}') for i in range(dim)])

    # Generate all basis functions using itertools.product
    basis_functions = []
    # Iterate over the combinations of m, n, p
    for p in range (p_monomial):
        for n in range (n_monomial):
            for m in range(m_monomial):
            # Define the symbolic expression for the basis function
                basis_function = X[0]**m * X[1]**n * X[2]**p
                basis_functions.append(basis_function)
    
    return basis_functions


if __name__ == "__main__":

    # Define the number of monomials per dimension
    m_monomial = 2
    n_monomial = 2
    p_monomial = 2
    
    # Extract the coefficients of the exact monomials
    coeff_matrix_np = extract_coefficients(m_monomial, n_monomial, p_monomial)
    # print(coeff_matrix_np)
    # Save the coefficients to a file
    np.save(f'coeff_exact_m_{m_monomial}_n_{n_monomial}_p_{p_monomial}.npy', coeff_matrix_np)

    # get the multiplication of size 0 * size 1 of the matrix
    print(coeff_matrix_np.shape[0]*coeff_matrix_np.shape[1])
    bases = extract_bases(m_monomial, n_monomial, p_monomial)
    print(bases)