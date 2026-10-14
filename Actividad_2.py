import pandas as pd
import constants.parameters as proconst
import matplotlib.pyplot as plt
from scipy.stats import trim_mean

def scatter_plots(raw_data : pd.DataFrame) -> None :

    #If you comment xlabel and ylabel it puts the defaults names of the columns.

    raw_data.plot.scatter(
        x = proconst.IRIS_CSV_SEPAL_LENGTH,
        y = proconst.IRIS_CSV_SEPAL_WIDTH,
        title="Sépalo - Largo vs Ancho",
        xlabel="Sépalo Largo (cm)",
        ylabel="Sépalo Ancho (cm)"
    )
    plt.show()

    raw_data.plot.scatter(
        x = proconst.IRIS_CSV_PETAL_LENGTH,
        y = proconst.IRIS_CSV_PETAL_WIDTH,
        title="Pétalo - Largo vs Ancho",
        xlabel = "Pétalo Largo (cm)",
        ylabel = "Pétalo Ancho (cm)"
    )
    plt.show()

    fig, ax = plt.subplots()

    species_colors = {
        "Virginica": "green",
        "Versicolor": "orange",
        "Setosa": "purple"
    }

    for species, color in species_colors.items():
        species_data = raw_data[raw_data[proconst.IRIS_CSV_VARIETY] == species]

        ax.scatter(
            species_data[proconst.IRIS_CSV_SEPAL_LENGTH],
            species_data[proconst.IRIS_CSV_SEPAL_WIDTH],
            color=color,
            marker="o",
            label=f"{species} - Sépalo"
        )

    for species, color in species_colors.items():
        species_data = raw_data[raw_data[proconst.IRIS_CSV_VARIETY] == species]

        ax.scatter(
            species_data[proconst.IRIS_CSV_PETAL_LENGTH],
            species_data[proconst.IRIS_CSV_PETAL_WIDTH],
            color=color,
            marker="^",
            label=f"{species} - Pétalo"
        )

    ax.set_title("Iris: Sépalo vs Pétalo")
    ax.set_xlabel("Largo (cm)")
    ax.set_ylabel("Ancho (cm)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.show()
    return

def histogram(raw_data : pd.DataFrame) -> None :

    #If you comment xlabel and ylabel it puts the defaults names of the columns.

    raw_data.plot.hist(
        x = proconst.IRIS_CSV_SEPAL_LENGTH,
        y = proconst.IRIS_CSV_SEPAL_WIDTH,
        title="Sépalo - Largo vs Ancho",
        xlabel="Sépalo Largo (cm)",
        ylabel="Sépalo Ancho (cm)"
    )
    plt.show()

    raw_data.plot.hist(
        x = proconst.IRIS_CSV_PETAL_LENGTH,
        y = proconst.IRIS_CSV_PETAL_WIDTH,
        title="Pétalo - Largo vs Ancho",
        xlabel = "Pétalo Largo (cm)",
        ylabel = "Pétalo Ancho (cm)"
    )


    plt.show()

    fig, ax = plt.subplots()

    raw_data.plot.hist(
        x=proconst.IRIS_CSV_SEPAL_LENGTH,
        y=proconst.IRIS_CSV_SEPAL_WIDTH,
        color="blue",
        label="Sépalo",
        ax=ax
    )

    raw_data.plot.hist(
        x=proconst.IRIS_CSV_PETAL_LENGTH,
        y=proconst.IRIS_CSV_PETAL_WIDTH,
        color="red",
        label="Pétalo",
        ax=ax
    )

    ax.set_title("Iris : Sépalo vs Pétalo")
    ax.set_xlabel("Largo")
    ax.set_ylabel("Ancho")
    ax.legend()
    plt.show()

    return

def boxplot(raw_data: pd.DataFrame) -> None:

    columns = [
        proconst.IRIS_CSV_SEPAL_LENGTH,
        proconst.IRIS_CSV_SEPAL_WIDTH,
        proconst.IRIS_CSV_PETAL_LENGTH,
        proconst.IRIS_CSV_PETAL_WIDTH
    ]

    titles = [
        "Sépalo - Largo",
        "Sépalo - Ancho",
        "Pétalo - Largo",
        "Pétalo - Ancho"
    ]

    for column, title in zip(columns, titles):

        fig, ax = plt.subplots()

        raw_data.boxplot(
            column=column,
            ax=ax,
            grid=False
        )

        ax.set_title(title)
        ax.set_ylabel("Medida (cm)")

        plt.tight_layout()
        plt.show()

    fig, ax = plt.subplots(figsize=(12, 6))

    raw_data.boxplot(
        column=columns,
        ax=ax,
        grid=False
    )

    ax.set_title("Iris: Cajas de las cuatro características")
    ax.set_ylabel("Medida (cm)")
    ax.set_xlabel("Características")

    plt.tight_layout()
    plt.show()

    return

def boxplot_by_quartiles(raw_data: pd.DataFrame) -> None:

    numeric_data = raw_data.select_dtypes(include="number")

    for column in numeric_data.columns:

        quartiles = pd.qcut(
            numeric_data[column],
            q=4,
            labels=["Q1", "Q2", "Q3", "Q4"]
        )

        quartile_data = [
            numeric_data.loc[quartiles == quartile, column].dropna()
            for quartile in ["Q1", "Q2", "Q3", "Q4"]
        ]

        fig, ax = plt.subplots(figsize=(8, 5))

        ax.boxplot(
            quartile_data,
            tick_labels=["Q1", "Q2", "Q3", "Q4"]
        )

        ax.set_title(f"{column} — Distribution by Quartile")
        ax.set_xlabel("Quartile")
        ax.set_ylabel("Value")

        plt.tight_layout()
        plt.show()

    return


def calculate_mode(x : pd.Series) -> float:
    return x.mode().iloc[0]


def array_trim_mean(x : pd.Series) -> float:
    return float(trim_mean(x.dropna(), proportiontocut=0.1))

def measures_of_central_tendencies(raw_data : pd.DataFrame) -> None :
    numeric_data = raw_data.select_dtypes(include="number")
    statistics = numeric_data.agg(["mean", array_trim_mean, "median", calculate_mode, "var", "std"])
    statistics.index = ["Media","Media Recortada", "Mediana","Moda","Varianza","Desviación Estandar"]
    print(statistics)
    return

def main():
    raw_data : pd.DataFrame = pd.read_csv(proconst.IRIS_CSV_LOCAL_PATH)
    scatter_plots(raw_data)
    histogram(raw_data)
    boxplot(raw_data)
    print(f"\n--- General ---")
    measures_of_central_tendencies(raw_data)
    print(f"\n--- Setosa ---")
    setosa_data = raw_data[raw_data[proconst.IRIS_CSV_VARIETY] == "Setosa"]
    measures_of_central_tendencies(setosa_data)
    print(f"\n--- Virginica ---")
    virginica_data = raw_data[raw_data[proconst.IRIS_CSV_VARIETY] == "Virginica"]
    measures_of_central_tendencies(virginica_data)
    print(f"\n--- Versicolor ---")
    versicolor_data = raw_data[raw_data[proconst.IRIS_CSV_VARIETY] == "Versicolor"]
    measures_of_central_tendencies(versicolor_data)

main()