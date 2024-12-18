### Heston Model Derivation of the PDE

The Heston model describes the evolution of two stochastic processes:  
1. **The asset price \( S_t \)**  
2. **The variance \( v_t \) (stochastic volatility)**  

---

### Stochastic Differential Equations (SDEs)

1. **Asset Price Process** \( S_t \) (under the risk-neutral measure \( \mathbb{Q} \)):

$$
dS_t = r S_t \, dt + \sqrt{v_t} S_t \, dW_t^S
$$

2. **Variance Process** \( v_t \) (CIR-type process):

$$
dv_t = \kappa (\theta - v_t) \, dt + \sigma_v \sqrt{v_t} \, dW_t^v
$$

3. **Correlation Between the Brownian Motions**:

$$
dW_t^S \, dW_t^v = \rho \, dt
$$

---

### Feynman-Kac Theorem

The Feynman-Kac theorem relates a stochastic process to a partial differential equation (PDE). If \( V = V(S, v, t) \) represents the option price, then the following relationships hold:

- The **total derivative** of \( V \) under Itô’s lemma is:

$$
dV = \frac{\partial V}{\partial t} \, dt 
    + \frac{\partial V}{\partial S} \, dS_t 
    + \frac{\partial V}{\partial v} \, dv_t 
    + \frac{1}{2} \frac{\partial^2 V}{\partial S^2} \, (dS_t)^2
    + \frac{1}{2} \frac{\partial^2 V}{\partial v^2} \, (dv_t)^2
    + \frac{\partial^2 V}{\partial S \partial v} \, (dS_t \, dv_t).
$$

Substitute \( dS_t \) and \( dv_t \) from the SDEs:

- \( dS_t = r S_t \, dt + \sqrt{v_t} S_t \, dW_t^S \)
- \( dv_t = \kappa (\theta - v_t) \, dt + \sigma_v \sqrt{v_t} \, dW_t^v \).

---

### Step 1: Calculating the Terms

1. **First-order terms**:

- From \( dS_t \):
  
$$
\frac{\partial V}{\partial S} \, dS_t = \frac{\partial V}{\partial S} \left( r S_t \, dt + \sqrt{v_t} S_t \, dW_t^S \right).
$$

- From \( dv_t \):

$$
\frac{\partial V}{\partial v} \, dv_t = \frac{\partial V}{\partial v} \left( \kappa (\theta - v_t) \, dt + \sigma_v \sqrt{v_t} \, dW_t^v \right).
$$

2. **Second-order terms**:

- \( (dS_t)^2 \):

$$
(dS_t)^2 = (\sqrt{v_t} S_t)^2 (dW_t^S)^2 = v_t S_t^2 \, dt.
$$

- \( (dv_t)^2 \):

$$
(dv_t)^2 = (\sigma_v \sqrt{v_t})^2 (dW_t^v)^2 = \sigma_v^2 v_t \, dt.
$$

- \( dS_t \, dv_t \) (correlation term):

$$
dS_t \, dv_t = \sqrt{v_t} S_t \, \sigma_v \sqrt{v_t} \, \rho \, dt = \rho \sigma_v S_t v_t \, dt.
$$

---

### Step 2: Substitute into Itô's Lemma

Combining all the terms, we get:

$$
dV = \frac{\partial V}{\partial t} \, dt 
    + \frac{\partial V}{\partial S} \, r S_t \, dt 
    + \frac{\partial V}{\partial v} \, \kappa (\theta - v_t) \, dt
    + \frac{1}{2} \frac{\partial^2 V}{\partial S^2} v_t S_t^2 \, dt
    + \frac{1}{2} \frac{\partial^2 V}{\partial v^2} \sigma_v^2 v_t \, dt
    + \frac{\partial^2 V}{\partial S \partial v} \rho \sigma_v S_t v_t \, dt
    + \text{stochastic terms}.
$$

The stochastic terms (containing \( dW_t \)) are eliminated under the risk-neutral measure since the expectation of \( dW_t \) is zero.

---

### Step 3: Heston PDE

Grouping the coefficients of \( dt \), the Heston PDE for the option price \( V(S, v, t) \) is:

$$
\frac{\partial V}{\partial t} 
+ r S \frac{\partial V}{\partial S} 
+ \kappa (\theta - v) \frac{\partial V}{\partial v} 
- r V 
+ \frac{1}{2} v S^2 \frac{\partial^2 V}{\partial S^2} 
+ \rho \sigma_v S v^{1/2} \frac{\partial^2 V}{\partial S \partial v} 
+ \frac{1}{2} \sigma_v^2 v \frac{\partial^2 V}{\partial v^2} 
= 0.
$$

---

### Conclusion

The Heston PDE is derived by applying **Itô's lemma** to the option price \( V(S, v, t) \) under the risk-neutral measure and combining the terms for \( dt \).  
This PDE governs the evolution of the option price in the Heston stochastic volatility model.
