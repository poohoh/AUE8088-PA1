from torchmetrics import Metric
import torch

# [TODO] Implement this!
# class MyF1Score(Metric):
#     def __init__(self, num_classes: int=200):
#         super().__init__()
#         self.add_state('tps', default=torch.zeros(num_classes), dist_reduce_fx='sum')
#         self.add_state('fps', default=torch.zeros(num_classes), dist_reduce_fx='sum')
#         self.add_state('fns', default=torch.zeros(num_classes), dist_reduce_fx='sum')

#         self.eps = 1e-6

#     def update(self, preds, target):
#         # preds: B X C
#         preds = torch.argmax(preds, dim=1)
        
#         for i in range(preds.shape[0]):
#             pred_idx = preds[i]  # predicted class idx
#             target_idx = target[i]  # target class idx

#             if pred_idx == target_idx:
#                 self.tps[pred_idx] += 1
#             else:
#                 self.fps[pred_idx] += 1
#                 self.fns[target_idx] += 1
    
#     def compute(self):
#         precision = self.tps / (self.tps + self.fps + self.eps)  # TP / (TP + FP)
#         recall = self.tps / (self.tps + self.fns + self.eps)  # TP / (TP + FN)
#         f1 = 2 * (precision * recall) / (precision + recall + self.eps)  # B x 1

#         return f1.mean().item()  # average f1 score

class MyF1Score(Metric):
    def __init__(self, num_classes: int=200):
        super().__init__()
        self.add_state('tps', default=torch.zeros(num_classes), dist_reduce_fx='sum')
        self.add_state('fps', default=torch.zeros(num_classes), dist_reduce_fx='sum')
        self.add_state('fns', default=torch.zeros(num_classes), dist_reduce_fx='sum')

        self.eps = 1e-6
        self.num_classes = num_classes

    def update(self, preds, target):
        # preds: B X C
        preds = torch.argmax(preds, dim=1)

        preds = preds.flatten()
        target = target.flatten()

        if preds.shape != target.shape:
            raise ValueError(f"shape mismatch")

        tp_mask = (preds == target)
        fp_mask = (preds != target)

        tp_indices = target[tp_mask]
        tps_batch = torch.bincount(tp_indices, minlength=self.num_classes)

        fp_indices = preds[fp_mask]
        fps_batch = torch.bincount(fp_indices, minlength=self.num_classes)

        fn_indices = target[fp_mask]
        fns_batch = torch.bincount(fn_indices, minlength=self.num_classes)

        self.tps += tps_batch.to(self.tps.device)
        self.fps += fps_batch.to(self.fps.device)
        self.fns += fns_batch.to(self.fns.device)
    
    def compute(self):
        precision = self.tps / (self.tps + self.fps + self.eps)
        recall = self.tps / (self.tps + self.fns + self.eps)
        f1_per_class = 2 * (precision * recall) / (precision + recall + self.eps)

        return f1_per_class.mean().item()  # average f1 score

class MyAccuracy(Metric):
    def __init__(self):
        super().__init__()
        self.add_state('total', default=torch.tensor(0), dist_reduce_fx='sum')
        self.add_state('correct', default=torch.tensor(0), dist_reduce_fx='sum')

    def update(self, preds, target):
        # [TODO] The preds (B x C tensor), so take argmax to get index with highest confidence
        preds = torch.argmax(preds, dim=1)

        # [TODO] check if preds and target have equal shape
        assert preds.shape == target.shape, f'shape not equal'

        # [TODO] Cound the number of correct prediction
        correct = torch.sum(preds == target)

        # Accumulate to self.correct
        self.correct += correct

        # Count the number of elements in target
        self.total += target.numel()

    def compute(self):
        return self.correct.float() / self.total.float()
