import os
import joblib
import pandas as pd
from tqdm import tqdm


def calc_adj_r2():
    final_csv_path = "tables/models_r2.csv"

    def generate_table():
        data = []
        t_bat = tqdm(total=12)
        for root, dirs, files in os.walk("model"):
            for file in files:
                if file.endswith("model.joblib"):
                    with open(os.path.join(root, file), "rb") as f:
                        _, family, metric = root.split("/")
                        model = joblib.load(f)
                        player, metric = metric.split("_")
                        if "Gain" in metric:
                            metric = f"{player} gain"
                        r2_adj = model.rsquared_adj
                        data += [{"family": family, "metric": metric, "r2_adj": r2_adj}]
                        t_bat.update(1)

        df = pd.DataFrame(data)
        df = df.drop_duplicates()
        df = df.pivot(index="metric", columns="family", values="r2_adj")
        df.to_csv(final_csv_path)

    generate_table()
    df = pd.read_csv(final_csv_path, index_col=0).T
    print(df.to_latex(float_format="%.2f"))


if __name__ == "__main__":
    calc_adj_r2()
