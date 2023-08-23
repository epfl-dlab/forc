class NoCollationCollator:
    def __init__(self):
        pass

    def collate_fn(self, batch):
        return batch
