import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn


def select_features(df):
    X = df[
        [
            "age",
            "workclass",
            "fnlwgt",
            "education",
            "marital-status",
            "occupation",
            "relationship",
            "race",
            "sex",
            "hours-per-week",
            "native-country",
        ]
    ]
    y = df["income"]
    return X, y


def transform_cols_astype_str(X, cols):
    for c in cols:
        X[c] = X[c].astype(str)
    return X


def transform_native_country(X):
    # transformamos la categoría native-country en dos, para que sea más representativo
    filter_by_not_eeuu = ~X["native-country"].str.contains("United-States")
    X.loc[filter_by_not_eeuu, "native-country"] = "EXTRANJEROS"
    return X


def strip_and_lower_categoricals(X, categorical_columns):
    return (
        X.loc[:, categorical_columns]
        .apply(lambda df: df.str.lower(), axis=1)
        .apply(lambda df: df.str.strip(), axis=1)
    )


def transform_with_encoders(X, encoders):
    for col, encoder in encoders.items():
        if encoder.__class__.__name__ == "OneHotEncoder":
            feature_names = list(
                map(lambda cat: col + "_" + cat, encoder.categories_[0])
            )
            data = encoder.transform(X.loc[:, [col]])
            nx = pd.DataFrame(data, columns=feature_names).set_index(X.index)
            X = pd.concat([X, nx], axis=1).drop(col, axis=1)
        else:
            X[[col]] = encoder.transform(X[[col]])

    return X


def plot_regresion_lineal(w, b, x, y, title=""):
    # genero una ventana de dibujo con una sola zona de dibujo (1,1)
    # que permita graficos en 3D
    figure = plt.figure(figsize=(10, 10), dpi=100)
    ax_data = figure.add_subplot(1, 1, 1, projection="3d")

    # dibujo el dataset en 3D (x1,x2,y)
    x1 = x[:, 0]
    x2 = x[:, 1]
    ax_data.scatter(x1, x2, y, color="blue")
    figure.suptitle(title)

    # Dibujo el plano dado por los parametros del modelo (w,b)
    # Este codigo probablemente no sea facil de entender
    # si no tenes experiencia con calculos en 3D
    detail = 0.05
    # genero coordenadas x,y de a pares, las llamo xx e yy
    xr = np.arange(x.min(), x.max(), detail)
    yr = np.arange(y.min(), 10, detail)
    xx, yy = np.meshgrid(xr, yr)
    # calculo las coordenadas z en base a xx, yy, y el modelo (w,b)
    zz = xx * w[0] + yy * w[1] + b
    # dibujo la superficie dada por los puntos (xx,yy,zz)
    surf = ax_data.plot_surface(
        xx, yy, zz, cmap="Reds", alpha=0.5, linewidth=0, antialiased=True
    )

    # Establezco las etiquetas de los ejes
    ax_data.set_xlabel("x1 (Horas estudiadas)")
    ax_data.set_ylabel("x2 (Promedio)")
    ax_data.set_zlabel("y (Nota)")
    # Establezco el titulo del grafico
    ax_data.set_title("(Horas estudiadas x Promedio) vs Nota")


def calculate_class_weight(y):
    weights = {}
    for i in np.unique(y):
        weights[i] = ((y == i).sum()) / y.size
    return weights


def print_report(y_real, y_pred):
    print(80 * "=")
    print("Acc: ", sklearn.metrics.accuracy_score(y_real, y_pred))
    print("Roc_Auc: ", sklearn.metrics.roc_auc_score(y_real, y_pred))
    print(80 * "_")
    print(sklearn.metrics.classification_report(y_real, y_pred))
    print(80 * "=")


def print_learning_curve(model, history):
    for k, i in zip(list(range(len(model.metrics_names))), model.metrics_names):
        plt.figure(figsize=(8, 8))
        # mean_ind = np.mean(history.history[i])
        # std_ind = np.std(history.history[i]) + 0.2
        # plt.ylim(mean_ind - std_ind,mean_ind + std_ind)
        plt.ylim(0, 1)
        plot_curve(history, i)


def plot_curve(history, ind):
    # summarize history for loss
    plt.plot(history.history[ind])
    if "val_" + ind in history.history.keys():
        plt.plot(history.history["val_" + ind])
    # plt.plot(history.history['val_loss'])
    plt.title(f"model {ind}")
    plt.ylabel(ind)
    plt.xlabel("epoch")
    plt.legend(["train", "test"], loc="upper left")
    plt.show()
