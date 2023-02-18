import os, logging, json
from tqdm import tqdm
import torch
from torch import nn, optim
from .data_loader import SentenceRELoader
from .utils import AverageMeter
import torch.nn.functional as F
from collections import Counter
import time
import random

class SentenceRE(nn.Module):

    def __init__(self, 
                 model,
                 train_path, 
                 val_path, 
                 test_path,
                 ckpt,
                 pre_ckpt,
                 metric,
                 num_class,
                 pretrained,
                 max_length,
                 batch_size=32, 
                 max_epoch=100,
                 pretrain_epoch=3,
                 lr=0.1, 
                 weight_decay=1e-5, 
                 warmup_step=300,
                 opt='sgd'):
    
        super().__init__()
        self.max_epoch = max_epoch
        self.metric = metric
        self.pretrain_epoch = pretrain_epoch
        self.num_class = num_class
        self.pretrained = pretrained
        self.pre_ckpt = pre_ckpt
        self.max_length=max_length
        self.cls = torch.Tensor([101]).long().cuda()
        self.sep = torch.Tensor([102]).long().cuda()
        # Load data
        if train_path != None:
            self.train_loader, self.train_len = SentenceRELoader(
                train_path,
                model.rel2id,
                model.sentence_encoder.tokenize,
                batch_size,
                True
            )
        self.id2reltoken=self.train_loader.dataset.id2reltoken

        if val_path != None:
            self.val_loader, _ = SentenceRELoader(
                val_path,
                model.rel2id,
                model.sentence_encoder.tokenize,
                batch_size,
                False
            )
        
        if test_path != None:
            self.test_loader, _ = SentenceRELoader(
                test_path,
                model.rel2id,
                model.sentence_encoder.tokenize,
                batch_size,
                False
            )

        # Model
        self.model = model
        self.parallel_model = nn.DataParallel(self.model)
        # Criterion
        # self.criterion = nn.CrossEntropyLoss()
        self.criterion = nn.KLDivLoss(reduction = 'batchmean')
        self.KL = nn.KLDivLoss(reduction = 'batchmean')
        self.CE = nn.CrossEntropyLoss()
        self.softmax = nn.Softmax(dim=1)
        # Params and optimizer
        params = self.parameters()
        self.lr = lr
        self.neg_id = 12
        if opt == 'sgd':
            self.optimizer = optim.SGD(params, lr, weight_decay=weight_decay)
        elif opt == 'adam':
            self.optimizer = optim.Adam(params, lr, weight_decay=weight_decay)
        elif opt == 'adamw': # Optimizer for BERT
            from transformers import AdamW
            params = list(self.named_parameters())
            no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
            grouped_params = [
                {
                    'params': [p for n, p in params if not any(nd in n for nd in no_decay)], 
                    'weight_decay': 0.01,
                    'lr': lr,
                    'ori_lr': lr
                },
                {
                    'params': [p for n, p in params if any(nd in n for nd in no_decay)], 
                    'weight_decay': 0.0,
                    'lr': lr,
                    'ori_lr': lr
                }
            ]
            self.optimizer = AdamW(grouped_params, correct_bias=False)
        else:
            raise Exception("Invalid optimizer. Must be 'sgd' or 'adam' or 'adamw'.")
        # Warmup
        if warmup_step > 0:
            from transformers import get_linear_schedule_with_warmup
            training_steps = self.train_loader.dataset.__len__() // batch_size * self.max_epoch
            self.scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=warmup_step, num_training_steps=training_steps)
        else:
            self.scheduler = None
        # Cuda
        if torch.cuda.is_available():
            self.cuda()
        # Ckpt
        self.ckpt = ckpt

    def train_model(self):
        best_metric = 0
        scale = 0.1
        noise_id, pos_id = self.denoise()
        not_better=0

        for i in range(100):
            print()
            print("=== Epoch %d pretrain ===" % i)
            self.train_batch_np_1(self.train_loader, None, pos_id)
            print("=== Epoch %d pretrain test ===" % i)
            print()
            result = self.eval_model_np(self.test_loader)
            print(result)
            if result[self.metric] > best_metric:
                print("Best ckpt and saved.")
                folder_path = '/'.join(self.pre_ckpt.split('/')[:-1])
                if not os.path.exists(folder_path):
                    os.mkdir(folder_path)
                torch.save({'state_dict': self.model.state_dict()}, self.pre_ckpt)
                best_metric = result[self.metric]
            else:
                not_better+=1
                if not_better==3:
                    self.load_state_dict(torch.load(self.pre_ckpt)['state_dict'])
                    break
        print("Denoise")
        not_better = 0
        last_result = 0.
        for epoch in range(self.max_epoch):
            print()
            print("relabel %f " % scale)
            cur_label, noise_id_now = self.relabel(noise_id, scale)
            print("=== Epoch %d train ===" % epoch)
            self.train_batch(self.train_loader, cur_label, noise_id_now)
            print("=== Epoch %d test ===" % epoch)
            result = self.eval_model_np(self.test_loader)
            if result[self.metric] > best_metric:
                print("Best ckpt and saved.")
                folder_path = '/'.join(self.pre_ckpt.split('/')[:-1])
                if not os.path.exists(folder_path):
                    os.mkdir(folder_path)
                torch.save({'state_dict': self.model.state_dict()}, self.pre_ckpt)
                best_metric = result[self.metric]
            if result[self.metric] > last_result:
                not_better = 0
            else:
                not_better += 1
                if not_better==2:
                    not_better=0
                    self.load_state_dict(torch.load(self.pre_ckpt)['state_dict'])
                    if scale < 1.0:
                        scale += 0.1
            last_result = result[self.metric]

        print("Best %s on val set: %f" % (self.metric, best_metric))

    def relabel(self, noise_id, scale):
        self.eval()
        with torch.no_grad():
            new_label = [self.neg_id] * self.train_loader.dataset.__len__()
            tmp_label = [self.neg_id] * self.train_loader.dataset.__len__()
            all_score = torch.Tensor([0] * self.train_loader.dataset.__len__())
            all_var = torch.Tensor([1] * self.train_loader.dataset.__len__())
            t = tqdm(self.train_loader)
            for iter, data in enumerate(t):
                label = data[0].numpy().tolist()
                id = data[1].numpy().tolist()
                b_noised = []
                for i in range(len(id)):
                    if id[i] not in noise_id:
                        new_label[id[i]] = label[i]
                    else:
                        b_noised.append(i)
                if len(b_noised)>0:
                    if torch.cuda.is_available():
                        try:
                            index = data[2][b_noised].cuda()
                            att_mask = data[3][b_noised].cuda()
                            pos1 = data[4][b_noised].cuda()
                            pos2 = data[5][b_noised].cuda()
                        except:
                            pass
                    logits, _ = self.parallel_model(index, att_mask, pos1, pos2)
                    logits = F.log_softmax(logits, dim=-1)
                    var = torch.var(logits, dim=1)
                    score, pred = logits.max(-1)  # (B)
                    for i in range(len(b_noised)):
                        all_var[id[b_noised[i]]]=var[i]
                        all_score[id[b_noised[i]]]=score[i]
                        tmp_label[id[b_noised[i]]]=pred[i]
        values, _ = torch.sort(all_var)
        na_th = values[int(self.train_loader.dataset.__len__() * 0.13 * scale)]
        na = set(torch.where(all_var < na_th)[0].cpu().numpy())
        values, _ = torch.sort(all_score, descending=True)
        noise_th = values[int((len(noise_id) - int(self.train_loader.dataset.__len__() * 0.3))*scale)]
        for i in set(torch.where(all_score > noise_th)[0].cpu().numpy()):
            if i not in na:
                new_label[i] = tmp_label[i]
        noise_id=noise_id-na
        noise_id=noise_id-set(torch.where(all_score > noise_th)[0].cpu().numpy())
        return new_label, noise_id

    def denoise(self):
        self.eval()
        all_sim = torch.Tensor([-1] * self.train_loader.dataset.__len__())
        pos_num = 0
        # all_embs={}
        all_label={}
        with torch.no_grad():
            cls = torch.Tensor([101]).long().cuda()
            sep = torch.Tensor([102]).long().cuda()
            t = tqdm(self.train_loader)
            for iter, data in enumerate(t):
                label = data[0].numpy().tolist()
                b_pos=[]
                for i in range(len(label)):
                    if label[i] != self.neg_id:
                        b_pos.append(i)
                        pos_num +=1
                if len(b_pos)>0:
                    indexed_tokens = data[2][b_pos].cuda()
                    att_mask = data[3][b_pos].cuda()
                    emb = self.model.sentence_encoder.get_cls_emb(indexed_tokens, att_mask)
                    pos1_mask = data[6][b_pos].cuda()
                    pos2_mask = data[7][b_pos].cuda()
                    id = data[1][b_pos].numpy().tolist()
                    entites_1 = pos1_mask * indexed_tokens
                    entites_2 = pos2_mask * indexed_tokens
                    props = []
                    atts = []
                    for i in range(len(b_pos)):
                        # all_embs[id[i]] = emb[i].clone().detach().cpu()
                        all_label[id[i]] = label[b_pos[i]]
                        entity1 = entites_1[i][torch.where(entites_1[i] != 0)]
                        entity2 = entites_2[i][torch.where(entites_2[i] != 0)]
                        rel_tokens = self.id2reltoken[label[b_pos[i]]].long().cuda()
                        prop = torch.cat((cls, entity1, rel_tokens, entity2, sep))
                        pad = torch.Tensor([0] * (self.max_length - len(prop))).long().cuda()
                        att = torch.cat((torch.Tensor([1] * (len(prop))).long().cuda(),
                                         torch.Tensor([0] * (self.max_length - len(prop))).long().cuda()))
                        prop = torch.cat((prop, pad))
                        props.append(prop)
                        atts.append(att)
                    props = torch.stack(props)
                    atts = torch.stack(atts)
                    prop_emb = self.model.sentence_encoder.get_cls_emb(props, atts)
                    sim = (emb * prop_emb).sum(-1) / ((emb ** 2).sum(-1) ** 0.5) / (
                                (prop_emb ** 2).sum(-1) ** 0.5)
                    for i in range(len(b_pos)):
                        all_sim[id[i]]=sim[i]
            values, _ = torch.sort(all_sim, descending=True)
            pos_id = set(torch.where(all_sim > -1)[0].cpu().numpy())
            th = values[int(pos_num * 0.5)]
            noise_id = set(torch.where(all_sim < th)[0].cpu().numpy())
            print()
            print("Noise Id Number")
            print(len(noise_id))
            return noise_id, pos_id

    def train_batch_np(self,train_loader,cur_label, noise_id):
        if cur_label==None:
            self.CE = nn.CrossEntropyLoss(weight=train_loader.dataset.loss_weight_np())
        else:
            self.CE = nn.CrossEntropyLoss(weight=self.loss_weight_np(cur_label))
        self.train()
        avg_loss = AverageMeter()
        t = tqdm(train_loader)
        for iter, data in enumerate(t):
            id = data[1].numpy().tolist()
            clear=[]
            for i in range(len(id)):
                if id[i] not in noise_id:
                    clear.append(i)
            if len(clear)==0:
                continue
            if cur_label == None:
                label = data[0][clear].cuda()
            else:
                label = torch.Tensor([cur_label[id[i]] for i in clear]).long().cuda()
            if torch.cuda.is_available():
                try:
                    index = data[2][clear].cuda()
                    att_mask = data[3][clear].cuda()
                    pos1 = data[4][clear].cuda()
                    pos2 = data[5][clear].cuda()
                except:
                    pass
            logits, _ = self.parallel_model(index, att_mask, pos1, pos2)
            neg = torch.where(label == self.neg_id)[0]
            pos = torch.where(label != self.neg_id)[0]
            n_loss = 0.0
            p_loss = 0.0
            if len(neg) > 0:
                n_logit = logits[neg]
                n_logit = self.softmax(n_logit)
                n_logit = torch.log(n_logit)
                n_label = (torch.ones(len(neg), self.num_class-1)/(self.num_class-1)).cuda()
                n_loss = self.KL(n_logit, n_label)
            if len(pos) > 0:
                p_logit = logits[pos]
                p_label = label[pos]
                p_loss = self.CE(p_logit, p_label)
            loss = p_loss + n_loss
            avg_loss.update(loss.item(), 1)
            t.set_postfix(loss=avg_loss.avg)
            loss.backward()
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()
            self.optimizer.zero_grad()

    def eval_model_np(self, eval_loader):
        self.eval()
        avg_acc = AverageMeter()
        pred_result = []
        var_result = []
        with torch.no_grad():
            t = tqdm(eval_loader)
            for iter, data in enumerate(t):
                if torch.cuda.is_available():
                    for i in range(len(data)):
                        try:
                            data[i] = data[i].cuda()
                        except:
                            pass
                args = data[2:6]
                logits, _ = self.parallel_model(*args)
                logits = F.log_softmax(logits, dim=-1)
                var = torch.var(logits, dim=1)
                score, pred = logits.max(-1) # (B)
                for i in range(pred.size(0)):
                    pred_result.append(pred[i].item())
                    var_result.append(var[i].item())
        var_result_cp = var_result.copy()
        var_result_cp.sort()
        th = var_result_cp[4]
        for i in range(len(pred_result)):
            if var_result[i] < th:
                pred_result[i] = self.neg_id
        result = eval_loader.dataset.eval(pred_result)
        return result


    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)





