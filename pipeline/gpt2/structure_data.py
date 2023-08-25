import torch

class GenerateTraindata():
    def __init__(self, total_sample: int = 10000, batch_size: int = 2, seq_length: int = 2048) -> None:

        self.total_sample = total_sample
        self.batch_size = batch_size
        self.seq_length = seq_length
        # self.input_ids = torch.randint(
        #     10, 4000, (self.batch_size, self.seq_length))
        # self.labels = input_ids.clone()

    def generate_data(self) -> (torch.tensor, torch.Tensor):
        input_ids = torch.randint(
            10, 4000, (self.batch_size, self.seq_length)).cuda()
        labels = input_ids.clone()
        # torch.randint(low=0, high=1, size=(self.batch_size, self.seq_length))
        atten_mask = torch.ones_like(labels)
        return input_ids, labels, atten_mask