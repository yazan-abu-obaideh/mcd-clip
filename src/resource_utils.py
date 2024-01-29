import os


def resource_path(resource: str):
    return os.path.join(os.path.dirname(__file__),
                        "resources",
                        resource)
