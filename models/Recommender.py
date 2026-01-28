
class Recommender:

    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path

    def run_evaluation(self):
        raise NotImplementedError("Subclasses should implement this method.")