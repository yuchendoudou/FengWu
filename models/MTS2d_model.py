import torch
from models.model import basemodel


class MTS2d_model(basemodel):
    def __init__(self, logger, **params) -> None:
        super().__init__(logger, **params)

    def data_preprocess(self, data):
        # print(len(data))
        # begin_time = time.time()
        if isinstance(data[0], list):
            data = data[0]
        inp = [data[0]]
        for i in range(1, len(data)-1):
            inp.append(data[i])
        inp = torch.cat(inp, dim=1).float().to(self.device, non_blocking=True)

        tar_step1 = data[-1].float().to(self.device, non_blocking=True)

        return inp, tar_step1
    


    def train_one_step(self, batch_data, step):
        inp, tar_step1 = self.data_preprocess(batch_data)

            
        self.optimizer[list(self.model.keys())[0]].zero_grad()

        predict = self.model[list(self.model.keys())[0]](inp)
 
        step_one_loss = self.loss(predict, tar_step1)
        step_one_loss.backward()

        self.optimizer[list(self.model.keys())[0]].step()
        loss = step_one_loss


        return {self.loss_type: loss.item()}



    def test_one_step(self, batch_data):
        inp, tar_step1,  = self.data_preprocess(batch_data)

        predict = self.model[list(self.model.keys())[0]](inp)

        step_one_loss = self.loss(predict, tar_step1)

        loss = step_one_loss
        return {self.loss_type: loss.item()}

