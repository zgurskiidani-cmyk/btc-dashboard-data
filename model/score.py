import math, sys, datetime
# usage: python3 score.py DATE PRICE REALIZED MA200D MA200W ATH
d, price, rp, d200, w200, ath = sys.argv[1], *map(float, sys.argv[2:7])
H = [datetime.date(2012,11,28), datetime.date(2016,7,9), datetime.date(2020,5,11), datetime.date(2024,4,20)]
t = datetime.date.fromisoformat(d); dsh = (t - max(h for h in H if h <= t)).days; ph = 2*math.pi*dsh/1458
x = {"ln_mvrv": math.log(price/rp), "ln_p_200d": math.log(price/d200), "ln_p_200w": math.log(price/w200),
     "ln_p_ath": math.log(price/max(ath, price)), "cyc_sin": math.sin(ph), "cyc_cos": math.cos(ph)}
M = {  # target: (intercept, {feature: (coef, mean, sd)}), primary model C = 0.1, trained 25.11.2026 on data to 24.11.2026
 "p180": (0.7952, {"ln_mvrv": (-0.7046, 0.4778, 0.3730), "ln_p_200d": (0.7441, 0.0800, 0.3289), "ln_p_200w": (-0.8829, 0.6645, 0.5378), "ln_p_ath": (-0.7573, -0.6731, 0.4832), "cyc_sin": (0.9644, 0.0286, 0.7086), "cyc_cos": (1.6501, -0.0945, 0.6987)}),
 "p365": (1.6354, {"ln_mvrv": (-0.6923, 0.4833, 0.3794), "ln_p_200d": (0.7962, 0.0926, 0.3293), "ln_p_200w": (-0.4457, 0.6806, 0.5428), "ln_p_ath": (-0.8940, -0.6819, 0.4910), "cyc_sin": (-0.6130, 0.0202, 0.7212), "cyc_cos": (1.3339, -0.0576, 0.6901)}),
 "pdd90": (-0.7889, {"ln_mvrv": (0.3865, 0.4722, 0.3714), "ln_p_200d": (-0.4459, 0.0757, 0.3271), "ln_p_200w": (1.0481, 0.6517, 0.5397), "ln_p_ath": (0.5196, -0.6726, 0.4784), "cyc_sin": (-0.8399, 0.0210, 0.7035), "cyc_cos": (-0.9277, -0.1115, 0.7015)}),
}
for k, (b0, cs) in M.items():
    z = b0 + sum(c * (x[f] - m) / s for f, (c, m, s) in cs.items())
    print(k, round(1 / (1 + math.exp(-z)), 3))
print("MVRV", round(price/rp, 3), "days_since_halving", dsh)
