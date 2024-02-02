import os


def resource_path(resource: str):
    return os.path.join(os.path.dirname(__file__), "resources", resource)


def run_result_path(result_file: str):
    run_results_dir = os.path.join(os.path.dirname(__file__), 'run-results')
    os.makedirs(run_results_dir, exist_ok=True)
    return os.path.join(run_results_dir, result_file)
