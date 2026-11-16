import numpy as np

def _setup(a, b, c):
    """1-based padded copies so indices match the paper."""
    n = len(a)
    A = np.zeros(n + 3); B = np.zeros(n + 3); C = np.zeros(n + 3)
    A[1:n + 1] = a; B[1:n] = b; C[1:n] = c
    return n, A, B, C

def _factor(n, A, B, C, check=True):
    """Fig. 3: J = L+D+U+ (increasing rows) and J = U-D-L- (decreasing rows)"""
    Dp = np.zeros(n + 3); Dm = np.zeros(n + 3)
    Lp = np.zeros(n + 3); Um = np.zeros(n + 3)
    Dp[1] = A[1]
    for i in range(1, n):
        if check and Dp[i] == 0 and B[i] * C[i] == 0:
            return None
        Lp[i] = B[i] / Dp[i]
        Dp[i + 1] = A[i + 1] - C[i] * Lp[i]
    Dm[n] = A[n]
    for i in range(n - 1, 0, -1):
        if check and Dm[i + 1] == 0 and B[i] * C[i] == 0:
            return None
        Um[i] = C[i] / Dm[i + 1]
        Dm[i] = A[i] - B[i] * Um[i]
    return Dp, Dm, Lp, Um

def nrminv_final1(a, b, c):
    """Algorithm Nrminv Final1 (Fig. 7): NaN-free, handles zero / infinite pivots."""
    INF = np.inf
    n, A, B, C = _setup(a, b, c)
    
    with np.errstate(all="ignore"):
        f = _factor(n, A, B, C)
        if f is None:
            return INF
        Dp, Dm, Lp, Um = f
        
        if Dp[n] == 0 or Dm[1] == 0:
            return INF
            
        for i in range(2, n):
            if Dp[i-1] == 0 and Dm[i+1] == 0:
                return INF
                
        delta = np.zeros(n + 3)
        su = np.zeros(n + 3)
        sl = np.zeros(n + 3)
        
        delta[1] = 1.0 / Dm[1]
        
        for i in range(1, n):
            if Dp[i] == 0 or 1.0 / Dm[i+1] == 0:
                delta[i+1] = 0.0
            elif Dm[i+1] == 0:
                delta[i+1] = -Dp[i] / (B[i] * C[i])
            elif 1.0 / Dp[i] == 0:
                delta[i+1] = 1.0 / Dm[i+1]
            elif delta[i] == 0:
                delta[i+1] = 1.0 / (Dm[i+1] - B[i] * C[i] / Dp[i])
            else:
                delta[i+1] = delta[i] / Dm[i+1] * Dp[i]
                
            if 1.0 / delta[i+1] == 0:
                return INF
                
        for i in range(1, n):
            if su[i] + abs(delta[i]) == 0:
                su[i+1] = (su[i-1] + abs(delta[i-1])) * abs(C[i-1] / B[i]) + abs(1.0 / B[i])
            else:
                su[i+1] = (su[i] + abs(delta[i])) * abs(Um[i])
                
            if 1.0 / su[i+1] == 0:
                return INF
                
        for i in range(n, 1, -1):
            if sl[i] + abs(delta[i]) == 0:
                sl[i-1] = (sl[i+1] + abs(delta[i+1])) * abs(B[i] / C[i-1]) + abs(1.0 / C[i-1])
            else:
                sl[i-1] = (sl[i] + abs(delta[i])) * abs(Lp[i-1])
                
            if 1.0 / sl[i-1] == 0:
                return INF
                
        gamma = 0.0
        for i in range(1, n + 1):
            if su[i] + sl[i] + abs(delta[i]) > gamma:
                gamma = su[i] + sl[i] + abs(delta[i])
                
        return gamma

def nrminv_final2(a, b, c):
    """Algorithm Nrminv Final2 (Fig. 8): 2x2 block pivoting, no division by tiny pivots."""
    INF = np.inf
    n, A, B, C = _setup(a, b, c)
    
    with np.errstate(all="ignore"):
        f = _factor(n, A, B, C)
        if f is None:
            return INF
        Dp, Dm, Lp, Um = f
        
        if Dp[n] == 0 or Dm[1] == 0:
            return INF
            
        for i in range(2, n):
            if Dp[i-1] == 0 and Dm[i+1] == 0:
                return INF
                
        delta = np.zeros(n + 3)
        su = np.zeros(n + 3)
        sl = np.zeros(n + 3)
        
        delta[1] = 1.0 / Dm[1]
        
        for i in range(1, n):
            delta[i + 1] = 1.0 / (Dm[i + 1] - B[i] * C[i] / Dp[i])
            
            if 1.0 / delta[i+1] == 0:
                return INF
                
        for i in range(1, n):
            DET = Dm[i+1] * A[i] - B[i] * C[i]
            
            if 1.0 / Dm[i+1] == 0 or abs(Dm[i+1] * A[i]) >= abs(DET):
                su[i+1] = (su[i] + abs(delta[i])) * abs(Um[i])
            else:
                su[i+1] = ((su[i-1] + abs(delta[i-1])) * abs(C[i-1] * C[i] / DET) 
                           + abs(C[i] / (Dm[i+1] * Dp[i] - B[i] * C[i])))
                           
            if 1.0 / su[i+1] == 0:
                return INF
                
        for i in range(n, 1, -1):
            DET = Dp[i-1] * A[i] - B[i-1] * C[i-1]
            
            if 1.0 / Dp[i-1] == 0 or abs(Dp[i-1] * A[i]) >= abs(DET):
                sl[i-1] = (sl[i] + abs(delta[i])) * abs(Lp[i-1])
            else:
                sl[i-1] = ((sl[i+1] + abs(delta[i+1])) * abs(B[i-1] * B[i] / DET) 
                           + abs(B[i-1] / (Dp[i-1] * Dm[i] - B[i-1] * C[i-1])))
                           
            if 1.0 / sl[i-1] == 0:
                return INF
                
        gamma = 0.0
        for i in range(1, n + 1):
            if su[i] + sl[i] + abs(delta[i]) > gamma:
                gamma = su[i] + sl[i] + abs(delta[i])
                
        return gamma

if __name__ == "__main__":
    # Define a test matrix (Toeplitz a=64, b=1, c=1, n=300)
    n = 300
    a = np.full(n, 64.0)
    b = np.ones(n - 1)
    c = np.ones(n - 1)
    
    # Calculate results
    res1 = nrminv_final1(a, b, c)
    res2 = nrminv_final2(a, b, c)
    
    # Write to a Markdown file
    with open("RESULTS.md", "w") as file:
        file.write("# NRMINV Algorithm Results\n\n")
        file.write("Execution of the `O(n)` tridiagonal matrix condition number algorithms.\n\n")
        file.write("## Test Case: Strongly Diagonally Dominant Toeplitz\n")
        file.write(f"- **Matrix size (n):** {n}\n")
        file.write("- **Diagonals:** a=64.0, b=1.0, c=1.0\n\n")
        file.write("### Outputs\n")
        file.write(f"- **Nrminv Final1 Result:** `{res1}`\n")
        file.write(f"- **Nrminv Final2 Result:** `{res2}`\n")
        
    print("Results successfully written to RESULTS.md")