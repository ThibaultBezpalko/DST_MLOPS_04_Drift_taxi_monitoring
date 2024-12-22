# Import Evidently for data drift and model performance monitoring
from evidently import ColumnMapping
from evidently.report import Report
from evidently.metrics import ColumnDriftMetric, DatasetDriftMetric, DatasetMissingValuesMetric
from evidently.ui.workspace import Workspace

def generate_report(train_data, val_data, num_features, cat_features):
    # Define the column mapping for the Evidently report
    # This includes the prediction column, numerical features, and categorical features
    column_mapping = ColumnMapping(
        target=None,
        prediction='prediction',
        numerical_features=num_features,
        categorical_features=cat_features
    )
    # Initialize the Evidently report with the desired metrics
    # In this case, we're using the ColumnDriftMetric for the 'prediction' column,
    # the DatasetDriftMetric to measure drift across the entire dataset,
    # and the DatasetMissingValuesMetric to measure the proportion of missing values
    report = Report(metrics=[
        ColumnDriftMetric(column_name='prediction'),
        DatasetDriftMetric(),
        DatasetMissingValuesMetric()
    ])
    # Run the report on the training and validation data
    # The training data is used as the reference data, and the validation data is the current data
    report.run(reference_data=train_data, current_data=val_data, column_mapping=column_mapping)
    # Return the generated report
    return report


def add_report_to_workspace(workspace, project_name, project_description, report):
    """
    Adds a report to an existing or new project in a workspace.
    """
    # Check if project already exists
    project = None
    for p in workspace.list_projects():
        if p.name == project_name:
            project = p
            break

    # Create a new project if it doesn't exist
    if project is None:
        project = workspace.create_project(project_name)
        project.description = project_description

    # Add report to the project
    workspace.add_report(project.id, report)
    print(f"New report added to project {project_name}")


if __name__ == "__main__":
    pass