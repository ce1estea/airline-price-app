import os
import io
import joblib

from threading import Timer

from flask import (
    Flask, render_template, request, redirect,
    url_for, flash, send_file
)

import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-GUI backend
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

app = Flask(__name__)
app.secret_key = "super_secret_key_change_me"

# --------- Global state (simple single-user style) ---------
DATAFRAME = None
CLEANED_DF = None
TRAIN_RESULTS = {}
TRAINED_MODELS = {}
DEPLOYED_MODEL_PATH = "deployed_model.pkl"

REQUIRED_COLUMNS = [
    "airline", "flight", "source_city", "departure_time", "stops",
    "arrival_time", "destination_city", "class", "duration",
    "days_left", "price"
]


def wipe_persistent_data():
    # Delete deployed model file if it exists
    if os.path.exists(DEPLOYED_MODEL_PATH):
        try:
            os.remove(DEPLOYED_MODEL_PATH)
        except Exception as e:
            print(f"Could not delete model file: {e}")

    # Delete generated plot files
    plots_dir = os.path.join(app.static_folder, "plots")
    if os.path.isdir(plots_dir):
        for fname in os.listdir(plots_dir):
            fpath = os.path.join(plots_dir, fname)
            try:
                if os.path.isfile(fpath):
                    os.remove(fpath)
            except Exception as e:
                print(f"Could not delete plot file {fpath}: {e}")


@app.context_processor
def inject_status_flags():
    """Make status flags available in all templates."""
    data_loaded = DATAFRAME is not None
    cleaned = CLEANED_DF is not None
    models_trained = bool(TRAIN_RESULTS)  # non-empty dict = trained
    model_deployed = os.path.exists(DEPLOYED_MODEL_PATH)

    return dict(
        status_data_loaded=data_loaded,
        status_cleaned=cleaned,
        status_models_trained=models_trained,
        status_model_deployed=model_deployed,
    )




# --------- Helpers ---------
def validate_columns(df):
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    return missing


def get_feature_target(df):
    # Drop obvious non-feature columns if present
    df = df.copy()
    if "index" in df.columns:
        df = df.drop(columns=["index"])

    X = df.drop(columns=["price"])
    y = df["price"]

    # Simple heuristic: numeric vs categorical
    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = [c for c in X.columns if c not in numeric_features]

    return X, y, numeric_features, categorical_features


def build_and_train_models(df, config):
    global TRAINED_MODELS, TRAIN_RESULTS

    X, y, numeric_features, categorical_features = get_feature_target(df)

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            ("num", "passthrough", numeric_features)
        ]
    )

    TRAINED_MODELS = {}
    TRAIN_RESULTS = {}

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Build model dict based on user config
    models = {}

    if config.get("train_linear", True):
        models["Linear Regression"] = LinearRegression()

    if config.get("train_rf", True):
        rf_estimators = config.get("rf_estimators", 30)      # small default for demo
        rf_max_depth = config.get("rf_max_depth", None)      # None = no limit
        models["Random Forest"] = RandomForestRegressor(
            n_estimators=rf_estimators,
            max_depth=rf_max_depth,
            random_state=42,
            n_jobs=-1  # use all cores to speed up
        )

    if config.get("train_gb", True):
        gb_estimators = config.get("gb_estimators", 50)      # small default
        gb_lr = config.get("gb_learning_rate", 0.1)
        gb_max_depth = config.get("gb_max_depth", 3)
        models["Gradient Boosting"] = GradientBoostingRegressor(
            n_estimators=gb_estimators,
            learning_rate=gb_lr,
            max_depth=gb_max_depth,
            random_state=42
        )

    for name, model in models.items():
        print(f"Training {name}...")  # log for debugging

        try:
            pipe = Pipeline(steps=[
                ("preprocess", preprocessor),
                ("model", model)
            ])

            pipe.fit(X_train, y_train)
            y_pred = pipe.predict(X_test)

            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            rmse = mse ** 0.5
            r2 = r2_score(y_test, y_pred)

            TRAINED_MODELS[name] = pipe
            TRAIN_RESULTS[name] = {
                "mae": mae,
                "rmse": rmse,
                "r2": r2,
                "error": None
            }
        except Exception as e:
            print(f"Error training {name}: {e}")
            TRAIN_RESULTS[name] = {
                "mae": None,
                "rmse": None,
                "r2": None,
                "error": str(e)
            }



def save_plot(fig, filename):
    plots_dir = os.path.join(app.static_folder, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    path = os.path.join(plots_dir, filename)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return f"plots/{filename}"  # relative path from /static


# --------- Routes ---------
@app.route("/")
def index():
    global DATAFRAME, CLEANED_DF, TRAIN_RESULTS
    has_data = DATAFRAME is not None
    has_cleaned = CLEANED_DF is not None
    has_results = len(TRAIN_RESULTS) > 0

    deployed_exists = os.path.exists(DEPLOYED_MODEL_PATH)

    return render_template(
        "index.html",
        has_data=has_data,
        has_cleaned=has_cleaned,
        has_results=has_results,
        deployed_exists=deployed_exists
    )


@app.route("/upload", methods=["POST"])
def upload():
    global DATAFRAME, CLEANED_DF, TRAINED_MODELS, TRAIN_RESULTS

    file = request.files.get("file")
    if not file:
        flash("Please select a CSV file.", "error")
        return redirect(url_for("index"))

    try:
        df = pd.read_csv(file)
    except Exception as e:
        flash(f"Error reading CSV: {e}", "error")
        return redirect(url_for("index"))

    missing = validate_columns(df)
    if missing:
        flash(f"Missing required columns: {', '.join(missing)}", "error")
        return redirect(url_for("index"))

    DATAFRAME = df
    CLEANED_DF = None   # freshly loaded → not cleaned yet

    # Reset any previous training/deployment when new data is loaded
    TRAINED_MODELS = {}
    TRAIN_RESULTS = {}
    if os.path.exists(DEPLOYED_MODEL_PATH):
        try:
            os.remove(DEPLOYED_MODEL_PATH)
        except Exception as e:
            print(f"Could not delete old deployed model: {e}")

    flash(f"Data loaded successfully with {len(df)} rows. Ready to clean.", "success")
    return redirect(url_for("index"))



@app.route("/preview")
def preview():
    global DATAFRAME
    if DATAFRAME is None:
        flash("No data loaded yet.", "error")
        return redirect(url_for("index"))

    preview_html = DATAFRAME.head(20).to_html(classes="table table-striped", index=False)
    return render_template("preview.html", table=preview_html)


@app.route("/clean", methods=["GET", "POST"])
def clean():
    global CLEANED_DF, DATAFRAME
    if DATAFRAME is None:
        flash("Load data first.", "error")
        return redirect(url_for("index"))

    if request.method == "POST":
        df = DATAFRAME.copy()

        # Options from form
        drop_index = request.form.get("drop_index") == "on"
        drop_duplicates = request.form.get("drop_duplicates") == "on"
        remove_outliers = request.form.get("remove_outliers") == "on"

        if drop_index and "index" in df.columns:
            df = df.drop(columns=["index"])

        if drop_duplicates:
            before = len(df)
            df = df.drop_duplicates()
            after = len(df)
            flash(f"Removed {before - after} duplicate rows.", "info")

        if remove_outliers:
            if "price" in df.columns:
                low_q = df["price"].quantile(0.01)
                high_q = df["price"].quantile(0.99)
                before = len(df)
                df = df[(df["price"] >= low_q) & (df["price"] <= high_q)]
                after = len(df)
                flash(f"Removed {before - after} outlier rows based on price.", "info")

        CLEANED_DF = df
        flash(f"Cleaning applied. Current rows: {len(df)}", "success")
        return redirect(url_for("clean"))

    # GET
    summary = None
    if CLEANED_DF is not None:
        summary = {
            "rows": len(CLEANED_DF),
            "columns": len(CLEANED_DF.columns),
            "avg_price": float(CLEANED_DF["price"].mean()),
            "min_price": int(CLEANED_DF["price"].min()),
            "max_price": int(CLEANED_DF["price"].max())
        }

    return render_template("clean.html", summary=summary)


@app.route("/visualize", methods=["GET", "POST"])
def visualize():
    global CLEANED_DF

    if CLEANED_DF is None:
        flash("Load and clean data first.", "error")
        return redirect(url_for("index"))

    df = CLEANED_DF.copy()

    # Numeric vs categorical columns
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = [c for c in df.columns if c not in numeric_cols]

    # Filter options
    airlines = ["ALL"]
    classes_list = ["ALL"]
    if "airline" in df.columns:
        airlines += sorted(df["airline"].unique())
    if "class" in df.columns:
        classes_list += sorted(df["class"].unique())

    plot_url = None

    # -------- defaults for selections (will be overridden on POST) --------
    selected_plot_type = "hist"
    selected_x_col = numeric_cols[0] if numeric_cols else None
    selected_y_col = numeric_cols[0] if numeric_cols else None
    selected_cat_col = categorical_cols[0] if categorical_cols else None
    selected_agg_func = "mean"
    selected_filter_airline = "ALL"
    selected_filter_class = "ALL"

    if request.method == "POST":
        # Read current selections from form
        selected_plot_type = request.form.get("plot_type") or selected_plot_type
        selected_x_col = request.form.get("x_col") or selected_x_col
        selected_y_col = request.form.get("y_col") or selected_y_col
        selected_cat_col = request.form.get("cat_col") or selected_cat_col
        selected_agg_func = request.form.get("agg_func") or selected_agg_func
        selected_filter_airline = request.form.get("filter_airline") or selected_filter_airline
        selected_filter_class = request.form.get("filter_class") or selected_filter_class

        # Apply filters
        if selected_filter_airline != "ALL" and "airline" in df.columns:
            df = df[df["airline"] == selected_filter_airline]

        if selected_filter_class != "ALL" and "class" in df.columns:
            df = df[df["class"] == selected_filter_class]

        try:
            if selected_plot_type == "hist":
                if selected_x_col in numeric_cols:
                    fig, ax = plt.subplots()
                    ax.hist(df[selected_x_col].dropna(), bins=40)
                    ax.set_title(f"Histogram of {selected_x_col}")
                    ax.set_xlabel(selected_x_col)
                    ax.set_ylabel("Frequency")
                    plot_url = save_plot(fig, "viz_hist.png")

            elif selected_plot_type == "scatter":
                if selected_x_col in numeric_cols and selected_y_col in numeric_cols:
                    fig, ax = plt.subplots()
                    ax.scatter(df[selected_x_col], df[selected_y_col], alpha=0.3)
                    ax.set_title(f"{selected_y_col} vs {selected_x_col}")
                    ax.set_xlabel(selected_x_col)
                    ax.set_ylabel(selected_y_col)
                    plot_url = save_plot(fig, "viz_scatter.png")

            elif selected_plot_type == "bar":
                if selected_cat_col in categorical_cols and selected_y_col in numeric_cols:
                    if selected_agg_func == "mean":
                        grouped = df.groupby(selected_cat_col)[selected_y_col].mean().sort_values()
                        title = f"Average {selected_y_col} by {selected_cat_col}"
                    elif selected_agg_func == "median":
                        grouped = df.groupby(selected_cat_col)[selected_y_col].median().sort_values()
                        title = f"Median {selected_y_col} by {selected_cat_col}"
                    else:  # count
                        grouped = df.groupby(selected_cat_col)[selected_y_col].count().sort_values()
                        title = f"Count of {selected_y_col} by {selected_cat_col}"

                    fig, ax = plt.subplots()
                    grouped.plot(kind="bar", ax=ax)
                    ax.set_title(title)
                    ax.set_ylabel(selected_y_col if selected_agg_func != "count" else "Count")
                    plt.xticks(rotation=45, ha="right")
                    plot_url = save_plot(fig, "viz_bar.png")

            elif selected_plot_type == "box":
                if selected_cat_col in categorical_cols and selected_y_col in numeric_cols:
                    fig, ax = plt.subplots()
                    df.boxplot(column=selected_y_col, by=selected_cat_col, ax=ax)
                    ax.set_title(f"{selected_y_col} by {selected_cat_col}")
                    ax.set_xlabel(selected_cat_col)
                    ax.set_ylabel(selected_y_col)
                    plt.suptitle("")
                    plt.xticks(rotation=45, ha="right")
                    plot_url = save_plot(fig, "viz_box.png")

        except Exception as e:
            print(f"Error building visualization: {e}")
            flash(f"Could not generate plot: {e}", "error")

    return render_template(
        "visualize.html",
        plot_url=plot_url,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        airlines=airlines,
        classes_list=classes_list,
        selected_plot_type=selected_plot_type,
        selected_x_col=selected_x_col,
        selected_y_col=selected_y_col,
        selected_cat_col=selected_cat_col,
        selected_agg_func=selected_agg_func,
        selected_filter_airline=selected_filter_airline,
        selected_filter_class=selected_filter_class,
    )



@app.route("/train", methods=["GET", "POST"])
def train():
    global CLEANED_DF, TRAIN_RESULTS

    if CLEANED_DF is None:
        flash("Load and clean data first.", "error")
        return redirect(url_for("index"))

    if request.method == "POST":
        # Read options from form
        train_linear = request.form.get("train_linear") == "on"
        train_rf = request.form.get("train_rf") == "on"
        train_gb = request.form.get("train_gb") == "on"

        # Random Forest params
        try:
            rf_estimators = int(request.form.get("rf_estimators") or 30)
        except ValueError:
            rf_estimators = 30

        rf_max_depth_raw = request.form.get("rf_max_depth") or ""
        try:
            rf_max_depth = int(rf_max_depth_raw) if rf_max_depth_raw.strip() != "" else None
        except ValueError:
            rf_max_depth = None

        # Gradient Boosting params
        try:
            gb_estimators = int(request.form.get("gb_estimators") or 50)
        except ValueError:
            gb_estimators = 50

        try:
            gb_learning_rate = float(request.form.get("gb_learning_rate") or 0.1)
        except ValueError:
            gb_learning_rate = 0.1

        gb_max_depth_raw = request.form.get("gb_max_depth") or ""
        try:
            gb_max_depth = int(gb_max_depth_raw) if gb_max_depth_raw.strip() != "" else 3
        except ValueError:
            gb_max_depth = 3

        config = {
            "train_linear": train_linear,
            "train_rf": train_rf,
            "train_gb": train_gb,
            "rf_estimators": rf_estimators,
            "rf_max_depth": rf_max_depth,
            "gb_estimators": gb_estimators,
            "gb_learning_rate": gb_learning_rate,
            "gb_max_depth": gb_max_depth,
        }

        build_and_train_models(CLEANED_DF, config)
        flash("Models trained successfully!", "success")
        # Immediately show updated results
        return render_template("train.html", results=TRAIN_RESULTS)

    # GET
    return render_template("train.html", results=TRAIN_RESULTS)



@app.route("/deploy", methods=["POST"])
def deploy():
    global TRAINED_MODELS

    model_name = request.form.get("model_name")
    if model_name not in TRAINED_MODELS:
        flash("Please select a valid model to deploy.", "error")
        return redirect(url_for("train"))

    model = TRAINED_MODELS[model_name]
    joblib.dump(model, DEPLOYED_MODEL_PATH)
    flash(f"Model '{model_name}' deployed successfully!", "success")
    return redirect(url_for("predict"))


@app.route("/predict", methods=["GET", "POST"])
def predict():
    # Make sure a model has been deployed
    if not os.path.exists(DEPLOYED_MODEL_PATH):
        flash("No deployed model found. Train and deploy a model first.", "error")
        return redirect(url_for("index"))

    # We need data to build dropdown options & reference distributions
    global DATAFRAME, CLEANED_DF
    df = CLEANED_DF if CLEANED_DF is not None else DATAFRAME

    if df is None:
        flash("No data available to build dropdowns. Please upload a CSV and (optionally) clean/train first.", "error")
        return redirect(url_for("index"))

    # Build dropdown options from the current data
    airlines = sorted(df["airline"].unique())
    flights = sorted(df["flight"].unique())
    source_cities = sorted(df["source_city"].unique())
    destination_cities = sorted(df["destination_city"].unique())
    departure_times = sorted(df["departure_time"].unique())
    arrival_times = sorted(df["arrival_time"].unique())
    stops_list = sorted(df["stops"].unique())
    classes = sorted(df["class"].unique())

    model = joblib.load(DEPLOYED_MODEL_PATH)

    # Defaults for selections (GET) – will be overridden on POST
    selected_airline = airlines[0] if airlines else None
    selected_flight = flights[0] if flights else None
    selected_source = source_cities[0] if source_cities else None
    selected_dest = destination_cities[0] if destination_cities else None
    selected_dep_time = departure_times[0] if departure_times else None
    selected_arr_time = arrival_times[0] if arrival_times else None
    selected_stops = stops_list[0] if stops_list else None
    selected_class = classes[0] if classes else None
    selected_duration = ""
    selected_days_left = ""

    pred_hist_url = None
    pred_compare_url = None
    single_pred = None

    if request.method == "POST":
        # Get raw form values (so we can repopulate even if parsing fails)
        selected_airline = request.form.get("airline")
        selected_flight = request.form.get("flight")
        selected_source = request.form.get("source_city")
        selected_dest = request.form.get("destination_city")
        selected_dep_time = request.form.get("departure_time")
        selected_arr_time = request.form.get("arrival_time")
        selected_stops = request.form.get("stops")
        selected_class = request.form.get("class")
        selected_duration = request.form.get("duration") or ""
        selected_days_left = request.form.get("days_left") or ""

        try:
            sel_duration = float(selected_duration)
            sel_days_left = int(selected_days_left)

            data = {
                "airline": [selected_airline],
                "flight": [selected_flight],
                "source_city": [selected_source],
                "departure_time": [selected_dep_time],
                "stops": [selected_stops],
                "arrival_time": [selected_arr_time],
                "destination_city": [selected_dest],
                "class": [selected_class],
                "duration": [sel_duration],
                "days_left": [sel_days_left]
            }
        except Exception as e:
            flash(f"Invalid input: {e}", "error")
            return render_template(
                "predict.html",
                single_pred=None,
                airlines=airlines,
                flights=flights,
                source_cities=source_cities,
                destination_cities=destination_cities,
                departure_times=departure_times,
                arrival_times=arrival_times,
                stops_list=stops_list,
                classes=classes,
                pred_hist_url=None,
                pred_compare_url=None,
                selected_airline=selected_airline,
                selected_flight=selected_flight,
                selected_source=selected_source,
                selected_dest=selected_dest,
                selected_dep_time=selected_dep_time,
                selected_arr_time=selected_arr_time,
                selected_stops=selected_stops,
                selected_class=selected_class,
                selected_duration=selected_duration,
                selected_days_left=selected_days_left,
            )

        df_new = pd.DataFrame(data)
        pred = model.predict(df_new)[0]
        single_pred = round(float(pred), 2)

        # ---------- VISUALIZATION 1: Global price distribution + predicted price ----------
        try:
            fig, ax = plt.subplots()
            ax.hist(df["price"], bins=50, alpha=0.7)
            ax.axvline(single_pred, color="red", linestyle="--", linewidth=2)
            ax.set_title("Predicted Price vs Overall Distribution")
            ax.set_xlabel("Price")
            ax.set_ylabel("Frequency")
            pred_hist_url = save_plot(fig, "predicted_price_hist.png")
        except Exception as e:
            print(f"Error creating prediction histogram: {e}")
            pred_hist_url = None

        # ---------- VISUALIZATION 2: Average similar flights vs predicted price ----------
        try:
            similar_mask = (
                (df["source_city"] == selected_source) &
                (df["destination_city"] == selected_dest) &
                (df["class"] == selected_class)
            )
            similar_df = df[similar_mask]

            if not similar_df.empty:
                avg_similar_price = float(similar_df["price"].mean())

                fig2, ax2 = plt.subplots()
                labels = ["Avg Similar Flights", "Predicted Price"]
                values = [avg_similar_price, single_pred]
                ax2.bar(labels, values)
                ax2.set_ylabel("Price")
                ax2.set_title("Prediction vs Average for Similar Flights")
                pred_compare_url = save_plot(fig2, "predicted_vs_avg_similar.png")
            else:
                pred_compare_url = None
        except Exception as e:
            print(f"Error creating comparison plot: {e}")
            pred_compare_url = None

        return render_template(
            "predict.html",
            single_pred=single_pred,
            airlines=airlines,
            flights=flights,
            source_cities=source_cities,
            destination_cities=destination_cities,
            departure_times=departure_times,
            arrival_times=arrival_times,
            stops_list=stops_list,
            classes=classes,
            pred_hist_url=pred_hist_url,
            pred_compare_url=pred_compare_url,
            selected_airline=selected_airline,
            selected_flight=selected_flight,
            selected_source=selected_source,
            selected_dest=selected_dest,
            selected_dep_time=selected_dep_time,
            selected_arr_time=selected_arr_time,
            selected_stops=selected_stops,
            selected_class=selected_class,
            selected_duration=selected_duration,
            selected_days_left=selected_days_left,
        )

    # GET: just show the form with dropdowns, using defaults
    return render_template(
        "predict.html",
        single_pred=None,
        airlines=airlines,
        flights=flights,
        source_cities=source_cities,
        destination_cities=destination_cities,
        departure_times=departure_times,
        arrival_times=arrival_times,
        stops_list=stops_list,
        classes=classes,
        pred_hist_url=None,
        pred_compare_url=None,
        selected_airline=selected_airline,
        selected_flight=selected_flight,
        selected_source=selected_source,
        selected_dest=selected_dest,
        selected_dep_time=selected_dep_time,
        selected_arr_time=selected_arr_time,
        selected_stops=selected_stops,
        selected_class=selected_class,
        selected_duration=selected_duration,
        selected_days_left=selected_days_left,
    )



@app.route("/shutdown", methods=["POST"])
def shutdown():
    # Get the Werkzeug shutdown function while we're still in the request context
    func = request.environ.get("werkzeug.server.shutdown")

    def shutdown_server(shutdown_func):
        # Wipe any persisted files (model, plots, etc.)
        wipe_persistent_data()

        # Stop the Flask dev server if possible
        if shutdown_func is not None:
            shutdown_func()

        # Hard-exit the Python process just in case
        os._exit(0)

    # Schedule shutdown so this response returns cleanly
    Timer(1, shutdown_server, args=(func,)).start()

    # Return a centered message instead of trying to auto-close the tab
    return """
    <html>
        <head>
            <style>
                body {
                    background-color: #f8f9fa;
                    font-family: Arial, Helvetica, sans-serif;
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    height: 100vh;
                    text-align: center;
                    color: #333;
                }
                .msg-box {
                    padding: 30px;
                    border-radius: 12px;
                    background: white;
                    box-shadow: 0 0 20px rgba(0,0,0,0.1);
                    width: 60%;
                    max-width: 500px;
                }
                h2 {
                    margin-bottom: 10px;
                    color: #007bff;
                }
                p {
                    font-size: 18px;
                }
            </style>
        </head>
        <body>
            <div class="msg-box">
                <h2>Application Closed</h2>
                <p>The backend has shut down successfully.<br><br>
                <strong>Please manually close this tab.</strong></p>
            </div>
        </body>
    </html>
    """



if __name__ == "__main__":
    # Ensure any leftover model/plots from a previous run are cleaned
    wipe_persistent_data()
    app.run(debug=True)
