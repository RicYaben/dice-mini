# NOTE: not sure this is needed
class Monitor:
    def __init__(self):
        ...

    def summary(self) -> str:
        return ""

    def log(self, message: str):
        ...

def monitor() -> Monitor:
    return Monitor()
