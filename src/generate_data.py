import numpy as np
import pandas as pd


def generate(n=2000, seed=42):
    rng = np.random.default_rng(seed)
    age = rng.integers(21, 61, n)
    income = np.clip(rng.lognormal(2.1, 0.45, n), 2, 60)      # juta Rp / bulan
    credit_score = np.clip(rng.normal(620, 90, n), 300, 850).round()
    debt_ratio = np.clip(rng.beta(2, 4, n), 0, 0.9).round(3)  # cicilan / pendapatan
    employment_years = np.clip(age - 21 - rng.integers(0, 8, n), 0, None)
    gender = rng.integers(0, 2, n)                            # 0=perempuan, 1=laki-laki

    z = (0.012 * (credit_score - 600)
         + 0.08 * (income - 8)
         - 4.0 * (debt_ratio - 0.35)
         + 0.05 * employment_years
         + 0.8 * gender
         - 0.4
         + rng.normal(0, 1.0, n))
    approved = (z > 0).astype(int)

    df = pd.DataFrame({
        "age": age, "income": income.round(2), "credit_score": credit_score,
        "debt_ratio": debt_ratio, "employment_years": employment_years,
        "gender": gender, "approved": approved,
    })
    # Sisipkan ~5% data kosong
    for col in ["income", "credit_score"]:
        df.loc[rng.random(n) < 0.05, col] = np.nan
    return df


if __name__ == "__main__":
    df = generate()
    df.to_csv("data/loan_data.csv", index=False)
    print(f"Dataset tersimpan: {df.shape[0]} baris, {df.shape[1]} kolom")
