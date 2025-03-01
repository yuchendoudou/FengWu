import torch
from models.model import basemodel

class MTS2d_finetune(basemodel):
    def __init__(self, logger, **params) -> None:
        super().__init__(logger, **params)
        self.finetune_cycle_num = self.extra_params.get("finetune_cycle_num", 12)
        self.use_harmonic = self.extra_params.get("use_harmonic", False)
        self.use_harmonic_weight = self.extra_params.get("use_harmonic_weight", 1.0)
    def data_preprocess(self, data):
        # print(len(data))
        # begin_time = time.time()
        data_list, tar_idx = data
        inp = [data_list[0]]
        
        for i in range(1, len(data_list)-1):
            inp.append(data_list[i])
        inp_tensor = torch.cat(inp, dim=1).float().to(self.device, non_blocking=True)

        tar_step = data_list[-1][:,self.constants_len:].float().to(self.device, non_blocking=True)
        del inp
        del data_list
        return inp_tensor, tar_step, tar_idx.numpy()
    


    def train_one_step(self, batch_data, step):
        inp, tar_step, tar_idx = self.data_preprocess(batch_data)

        batch_size = inp.shape[0]

        predict = self.model[list(self.model.keys())[0]](inp)
        if self.loss_type == "POSSFFT_Loss":
            step_one_loss, amp_loss, frep_loss = self.loss(predict, tar_step)
        else:
            step_one_loss = self.loss(predict, tar_step)
        self.optimizer[list(self.model.keys())[0]].zero_grad()
        step_one_loss.backward()
        self.optimizer[list(self.model.keys())[0]].step()

        predict_mean, _ = predict.chunk(2, dim = 1)

        step_inp = torch.cat((inp[:, :self.constants_len], inp[:, self.constants_len+tar_step.shape[1]:], predict_mean.detach()), dim=1).cpu().numpy()
        loss = step_one_loss.detach()
        self.replay_buff.store(step_inp, tar_idx)
        del step_inp
        del predict_mean
        del predict
        del inp
        del tar_step
        if (self.replay_buff.size > 50) or (step >=50):
            for i in range(self.finetune_cycle_num):
                inp_npy, tar_npy, tar_idx = self.replay_buff.sample(batch_size)
                inp = torch.Tensor(inp_npy).float().to(self.device, non_blocking=True)
                tar = torch.Tensor(tar_npy)[:,self.constants_len:].float().to(self.device, non_blocking=True)

                del inp_npy, tar_npy
  
                predict = self.model[list(self.model.keys())[0]](inp)
                if self.loss_type == "POSSFFT_Loss":
                    step_two_loss, amp_loss, frep_loss = self.loss(predict, tar)
                else:
                    step_two_loss = self.loss(predict, tar)
                self.optimizer[list(self.model.keys())[0]].zero_grad()
                step_two_loss.backward()
                self.optimizer[list(self.model.keys())[0]].step()
          
                predict_mean, _ = predict.chunk(2, dim = 1)

                step_inp = torch.cat((inp[:, :self.constants_len], inp[:, self.constants_len+tar.shape[1]:], predict_mean.detach()), dim=1).cpu().numpy()
                loss = loss + step_two_loss.detach()

                self.replay_buff.store(step_inp, tar_idx)
                del step_inp
                del inp
                del predict_mean
                del predict
                del tar

        return {self.loss_type: loss.item(), "step_one_loss": step_one_loss.item()}


    def test_one_step(self, batch_data):
        inp, tar_step, tar_idx  = self.data_preprocess(batch_data)
        predict = self.model[list(self.model.keys())[0]](inp)
        step_one_pred = predict

        step_one_loss = self.loss(predict, tar_step)
        loss = step_one_loss

        data_dict = {}
        data_dict['gt'] = tar_step
        data_dict['pred'] = step_one_pred[:,:tar_step.shape[1]]
        metrics_loss = self.eval_metrics.evaluate_batch(data_dict)

  
        metrics_loss.update({self.loss_type: loss.item()})
        
        return metrics_loss

