import logging
# Import necessary libraries
# custom function
from download_data import download_files
from load_and_preprocess_data import load_data, preprocess_data
from train_and_evaluate_model import train_and_evaluate
from save_model_and_data import save_model_and_data
from generate_report import generate_report, add_report_to_workspace
from evidently.ui.workspace import Workspace

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

if __name__ == "__main__":
    # Defining workspace and project details
    WORKSPACE_NAME = "datascientest-workspace"
    PROJECT_NAME = "taxi_monitoring"
    PROJECT_DESCRIPTION = "Evidently Dashboards"
    
    try:
        # load january data & preprocess data
        logging.info("Loading and preprocessing data...")
        jan_data = load_data('./data/green_tripdata_2022-01.parquet')
        jan_preprocessed_data = preprocess_data(jan_data)
        logging.info("Data loaded and preprocessed successfully.")

        # Define feature and target
        target = "duration_min"
        num_features = ["passenger_count", "trip_distance", "fare_amount", "total_amount"]
        cat_features = ["PULocationID", "DOLocationID"]

        # TODO : Complete the Train and evaluate model (replace the None)
        logging.info("Training and evaluating model...")
        model, train_data, val_data, _, _ = train_and_evaluate(jan_preprocessed_data, num_features, cat_features, target)
        logging.info("Model trained and evaluated successfully.")

        # TODO : save model and val_data (Replace the None)
        logging.info("Saving model and data...")
        save_model_and_data(model, val_data)
        logging.info("Model and data saved successfully.")

        # generate report
        print(train_data.head(3))
        logging.info("generating report...")
        report = generate_report(train_data, val_data, num_features, cat_features)
        logging.info("Report generated successfully.")

        # Extract key metrics
        # Pay attention to this part here since this is how
        # you will proceed in real situation
        result = report.as_dict()
        print("Drift score of the prediction column: ", result['metrics'][0]['result']['drift_score'])
        print("Number of drifted columns: ", result['metrics'][1]['result']['number_of_drifted_columns'])
        print("Share of missing values: ", result['metrics'][2]['result']['current']['share_of_missing_values'])
        #report.show(mode='inline')
        report.save_html("Drift_Report.html")

        # Set workspace
        workspace = Workspace(WORKSPACE_NAME)

        # Add report to workspace
        add_report_to_workspace(workspace, PROJECT_NAME, PROJECT_DESCRIPTION, report)

    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")

# run the evidently report UI
#evidently ui --workspace ./datascientest-workspace/