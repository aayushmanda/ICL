
# In-Context Learning

| Title                                                       | PDF                                                         |
|-------------------------------------------------------------|-------------------------------------------------------------|
| **Trained Transformers Learn Linear Models In-Context**    | [PDF](https://www.jmlr.org/papers/volume25/23-1042/23-1042.pdf)    |
| **Learning without training - The implicit dynamics of in-context learning** | [PDF](https://arxiv.org/pdf/2507.16003) 

### Output To Compare

| Task      |     $y$     |  OLS Pred   |  LSA Pred   |
|:---------:|:-----------:|:-----------:|:-----------:|
| 1 of 50   |  -1.1117    |  -1.1117    |  -1.3902    |
| 2 of 50   |   1.7026    |   1.7026    |   1.7500    |
| 3 of 50   |   0.5252    |   0.5252    |   0.6204    |
| 4 of 50   |   2.0304    |   2.0304    |   1.8016    |
| 5 of 50   |   0.3628    |   0.3628    |   0.4924    |
| 6 of 50   |   3.2344    |   3.2344    |   2.8936    |
| 7 of 50   |   0.1720    |   0.1720    |   0.1944    |
| 8 of 50   |  -0.8624    |  -0.8624    |  -0.7823    |
| 9 of 50   |   0.5004    |   0.5004    |   0.5873    |
| 10 of 50  |   4.5377    |   4.5377    |   4.3291    |

---

### Theorem 4
**using initialization from assumpsion 3**

Let $\sigma > 0$ be a parameter, and let $\Theta \in \mathbb{R}^{d\times d}$ be any matrix satisfying $\|\Theta \Theta^\top\|_F = 1$ and $\Theta \Lambda \neq 0_{d \times d}$. We assume
$$
W_{PV}(0) = \sigma
\begin{bmatrix}
0_{d \times d} & 0_d \\
0_d^\top & 1
\end{bmatrix},
\qquad
W_{KQ}(0) = \sigma
\begin{bmatrix}
\Theta \Theta^\top & 0_d \\
0_d^\top & 0
\end{bmatrix}.
$$


**and check and checking if theorem 4 if really holds.**

Consider gradient flow of a linear self-attention network $f_{\mathrm{LSA}}$ defined in over the population loss. Suppose the initialization satisfies  with initialization scale $\sigma > 0$ satisfying $\sigma^2 \|\Gamma\|_{\mathrm{op}} \sqrt{d} < 2$, where
$$
\Gamma := \left(1 + \frac{1}{N}\right) \Lambda + \frac{1}{N} \operatorname{tr}(\Lambda) I_d \in \mathbb{R}^{d\times d}.
$$
Then gradient flow converges to a global minimum of the population loss~(8). Moreover, $W_{PV}$ and $W_{KQ}$ converge to $W_{PV}^*$ and $W_{KQ}^*$, respectively, where


$W_{KQ}^* = 
\begin{bmatrix}
{(c\Gamma)}^{-1} & 0_{d} \\
0_{d}^\top & 0
\end{bmatrix}$



$W_{PV}^* = \begin{bmatrix}
0_{d\times d} & 0_{d} \\
0_{d}^\top & c
\end{bmatrix}.$

here $c = [\operatorname{tr}(Γ^{−2})^{1/4}]$



![alt text](image.png)


[1] https://www.jmlr.org/papers/volume25/23-1042/23-1042.pdf