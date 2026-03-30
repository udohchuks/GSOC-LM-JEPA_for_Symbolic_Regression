# 📊 Synthetic vs AI Feynman Dataset Comparison Report
**Generated:** 2026-03-30T12:12:32.208795
**Synthetic Equations:** 152
**AIF Equations:** 99

## Executive Summary
- **Variable Count:** ✓ Synthetic 3.36 vs AIF 4.09 (-18.0%)- **Node Count:** ✓ Synthetic 11.10 vs AIF 12.47 (-11.0%)- **Tree Depth:** ✓ Synthetic 2.87 vs AIF 2.92 (-1.7%)
---
## Detailed Statistics

### SYNTHETIC Dataset

#### Complexity Metrics
| Metric | Mean | Std | Min | Max | Median |
|--------|------|-----|-----|-----|--------|
| Variable Count | 3.36 | 1.27 | 1.00 | 8.00 | 3.00 |
| Node Count | 11.10 | 4.40 | 3.00 | 26.00 | 10.00 |
| Tree Depth | 2.87 | 0.81 | 2.00 | 5.00 | 3.00 |
| Dimensionless Ratio | 0.06 | 0.11 | 0.00 | 0.50 | 0.00 |

#### Top 10 Operators
| Operator | Count |
|----------|-------|
| `*` | 688 |
| `inv` | 172 |
| `sq` | 80 |
| `+` | 41 |
| `sqrt` | 19 |
| `exp` | 15 |
| `neg` | 3 |
| `/` | 1 |

#### Sample Equations
| # | Formula | Vars | Nodes |
|---|---------|------|-------|
| 1 | `$c1*sqrt(c2*x2**2*x4*(c3 + c4*x3)**2)$` | 3 | 17 |
| 2 | `$c1*c2*c3*x1*x2$` | 2 | 6 |
| 3 | `$c1*x2$` | 1 | 3 |
| 4 | `$3*c1*x3*x4/x2$` | 3 | 8 |
| 5 | `$c1*x4/(x1*x3)$` | 3 | 9 |

### AIF Dataset

#### Complexity Metrics
| Metric | Mean | Std | Min | Max | Median |
|--------|------|-----|-----|-----|--------|
| Variable Count | 4.09 | 1.66 | 1.00 | 10.00 | 4.00 |
| Node Count | 12.47 | 5.95 | 3.00 | 28.00 | 13.00 |
| Tree Depth | 2.92 | 0.82 | 2.00 | 5.00 | 3.00 |
| Dimensionless Ratio | 0.16 | 0.27 | 0.00 | 1.00 | 0.00 |

#### Top 10 Operators
| Operator | Count |
|----------|-------|
| `*` | 389 |
| `inv` | 115 |
| `+` | 65 |
| `sq` | 57 |
| `/` | 41 |
| `neg` | 33 |
| `sqrt` | 21 |
| `cos` | 10 |
| `exp` | 8 |
| `sin` | 8 |

#### Sample Equations
| # | Formula | Vars | Nodes |
|---|---------|------|-------|
| 1 | `$sqrt(2)*exp(c1*x1**2/2)/(2*sqrt(pi))$` | 1 | 15 |
| 2 | `$sqrt(2)*exp(c1*c2*x1*x2**2/2)/(2*sqrt(pi)*x1)$` | 3 | 20 |
| 3 | `$sqrt(2)*exp(c1*c2*x1*(x2 - x3)**2/2)/(2*sqrt(pi)*x1)$` | 4 | 24 |
| 4 | `$sqrt((x1 - x2)**2 + (x3 - x4)**2)$` | 4 | 17 |
| 5 | `$x1*x2*x3/((x4 - x5)**2 + (x6 - x7)**2 + (x8 - x9)**2)$` | 9 | 28 |

---
## Distribution Comparison

### Variable Count Distribution
| N Vars | Synthetic | AIF |
|--------|-----------|-----|
| 1 | 9 | 1 |
| 2 | 24 | 13 |
| 3 | 55 | 27 |
| 4 | 46 | 27 |
| 5 | 10 | 12 |
| 6 | 4 | 11 |
| 7 | 2 | 5 |
| 8 | 2 | 0 |
| 9 | 0 | 2 |
| 10 | 0 | 1 |

---
## Recommendations
✅ All metrics show good alignment between synthetic and AIF data.