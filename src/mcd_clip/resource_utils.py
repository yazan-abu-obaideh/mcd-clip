import os


def resource_path(resource: str):
    return os.path.join(os.path.dirname(__file__),
                        "resources",
                        resource)


def run_result_path(result_file: str):
    return os.path.join(os.path.dirname(__file__),
                        "run-results",
                        result_file)
