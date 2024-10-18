# coding=utf-8

import os
import cv2
import json, re, en_vectors_web_lg, random
import numpy as np

import torch
import torch.utils.data as Data

from .utils import label2yolobox
from crec.utils.distributed import is_main_process


class RefCOCODataSet(Data.Dataset):
    def __init__(self, 
                 ann_path, 
                 image_path, 
                 mask_path, 
                 input_shape, 
                 flip_lr, 
                 transforms, 
                 candidate_transforms, 
                 max_token_length,
                 use_glove=True, 
                 split="train", 
                 dataset="refcoco"
        ):
        super(RefCOCODataSet, self).__init__()
        self.split=split

        assert dataset in ['refcoco', 'refcoco+', 'refcocog', 'c-refcoco', 'c-refcoco+', 'c-refcocog']
        self.dataset = dataset
        self.vocabulary_path = ann_path['vocabulary']
        self.pretrained_emb_path = ann_path['pretrained_emb']
        # --------------------------
        # ---- Raw data loading ---
        # --------------------------
        stat_refs_list=json.load(open(ann_path[dataset], 'r'))
        

        self.ques_list = []
        splits=split.split('+')
        self.refs_anno=[]
        
        for split_ in splits:
            self.refs_anno+= stat_refs_list[split_]     #annos

        self.image_path=image_path[dataset]
        # self.mask_path=mask_path[dataset]
        self.input_shape=input_shape

        self.flip_lr = flip_lr if split=='train' else False
        
        # Define run data size
        self.data_size = len(self.refs_anno)

        if is_main_process():
            print(' ========== Dataset size:', self.data_size)
        # ------------------------
        # ---- Data statistic ----
        # ------------------------
        # Tokenize
        self.token_to_ix,self.ix_to_token, self.pretrained_emb, max_token = self.tokenize(stat_refs_list, use_glove)
        self.token_size = self.token_to_ix.__len__()

        if is_main_process():
            print(' ========== Question token vocab size:', self.token_size)

        self.max_token = max_token_length
        if self.max_token == -1:
            self.max_token = max_token
        
        if is_main_process():
            print('Max token length:', max_token, 'Trimmed to:', self.max_token)
            print('Finished!')
            print('')

        if split == 'train':
            self.candidate_transforms = candidate_transforms
        else:
            self.candidate_transforms = {}

        self.transforms = transforms


    def tokenize(self, stat_refs_list, use_glove):
        if os.path.exists(self.vocabulary_path):
            fin = json.load(open(self.vocabulary_path, 'r'))
            token_to_ix = fin['token_to_ix']
            max_token = fin['max_token']
            pretrained_emb = np.load(self.pretrained_emb_path)
        else:
            token_to_ix = {
                'PAD': 0,
                'UNK': 1,
                'CLS': 2,
            }

            spacy_tool = None
            pretrained_emb = []
            if use_glove:
                spacy_tool = en_vectors_web_lg.load()
                pretrained_emb.append(spacy_tool('PAD').vector)
                pretrained_emb.append(spacy_tool('UNK').vector)
                pretrained_emb.append(spacy_tool('CLS').vector)

            max_token = 0
            for split in stat_refs_list:
                for ann in stat_refs_list[split]:
                    for ref in ann['refs']:
                        words = re.sub(
                            r"([.,'!?\"()*#:;])",
                            '',
                            ref.lower()
                        ).replace('-', ' ').replace('/', ' ').split()
                        
                        if len(words) > max_token:
                            max_token = len(words)

                        for word in words:
                            if word not in token_to_ix:
                                token_to_ix[word] = len(token_to_ix)
                                if use_glove:
                                    pretrained_emb.append(spacy_tool(word).vector)

            # save new dictionary
            with open(self.vocabulary_path, 'w') as fout:
                voc = {'token_to_ix':token_to_ix, 'max_token':max_token}
                json.dump(voc, fout)
            np.save(self.pretrained_emb_path, pretrained_emb)
        
        pretrained_emb = np.array(pretrained_emb)
        ix_to_token={}
        for item in token_to_ix:
            ix_to_token[token_to_ix[item]]=item

        return token_to_ix, ix_to_token,pretrained_emb, max_token


    def proc_ref(self, words, token_to_ix, max_token):
        ques_ix = np.zeros(max_token, np.int64)
        
        for ix, word in enumerate(words):
            if word in token_to_ix:
                ques_ix[ix] = token_to_ix[word]
            else:
                ques_ix[ix] = token_to_ix['UNK']

            if ix + 1 == max_token:
                break

        return ques_ix

    # ----------------------------------------------
    # ---- Real-Time Processing Implementations ----
    # ----------------------------------------------
    
    def load_refs(self, idx):
        try:
            ref = self.refs_anno[idx]['query']
            ref = self.proc_ref(ref,self.token_to_ix,self.max_token)
        except:
            ref = np.random.choice(self.refs_anno[idx]['refs'])
            ref = re.sub(
                r"([.,'!?\"()*#:;])",
                '',
                ref.lower()
                ).replace('-', ' ').replace('/', ' ').split()
            ref = self.proc_ref(ref,self.token_to_ix,self.max_token)
        return ref
    
    def load_r(self, idx):
        try:
            r = self.refs_anno[idx]['att_id']
        except:
            r = None
        return r
    
    def load_aw(self, idx):
        try:
            aw = self.refs_anno[idx]['atts']
            aw = self.proc_ref(aw,self.token_to_ix,self.max_token)
        except:
            aw = np.zeros((1,1))
        return aw
    
    def load_neg_refs(self, idx):
        try:
            neg = self.refs_anno[idx]['neg']
            neg = self.proc_ref(neg,self.token_to_ix,self.max_token)
        except:
            neg = np.zeros((1,1))
        return neg
    
    def load_cf_label(self, idx):
        try:
            cf_label = self.refs_anno[idx]['cf_id']
        except:
            cf_label = 0
        return cf_label

    def preprocess_info(self,img,box,iid,lr_flip=False):
        h, w, _ = img.shape
        imgsize=self.input_shape[0]
        new_ar = w / h
        if new_ar < 1:
            nh = imgsize
            nw = nh * new_ar
        else:
            nw = imgsize
            nh = nw / new_ar
        nw, nh = int(nw), int(nh)

        dx = (imgsize - nw) // 2
        dy = (imgsize - nh) // 2
        
        img = cv2.resize(img, (nw, nh))
        sized = np.ones((imgsize, imgsize, 3), dtype=np.uint8) * 127
        sized[dy:dy + nh, dx:dx + nw, :] = img
        
        info_img = (h, w, nh, nw, dx, dy, iid)
        sized_box=label2yolobox(box,info_img,self.input_shape[0],lrflip=lr_flip)
        return sized, sized_box, info_img

    def load_img_feats(self, idx):
        img_path=None
        if self.dataset in ['refcoco','refcoco+','refcocog', 'c-refcoco', 'c-refcoco+', 'c-refcocog']:
            img_path=os.path.join(self.image_path,'COCO_train2014_%012d.jpg'%self.refs_anno[idx]['iid'])
        else:
            assert NotImplementedError
        
        image= cv2.imread(img_path)
        box=np.array([self.refs_anno[idx]['bbox']])
        
        return image, box, self.refs_anno[idx]['iid']

    def __getitem__(self, idx):
        ref_iter = self.load_refs(idx)
        aw_iter = self.load_aw(idx)
        #r_iter = self.load_r(idx)
        negs_iter = self.load_neg_refs(idx)
        image_iter,gt_box_iter,iid= self.load_img_feats(idx)
        image_iter = cv2.cvtColor(image_iter, cv2.COLOR_BGR2RGB)
        cf_label = self.load_cf_label(idx) 
        ops=None

        if len(list(self.candidate_transforms.keys()))>0:
            ops = random.choices(list(self.candidate_transforms.keys()), k=1)[0]
        
        if ops is not None and ops!='RandomErasing':
            image_iter = self.candidate_transforms[ops](image=image_iter)['image']

        flip_box=False
        if self.flip_lr and random.random() < 0.5:
            image_iter=image_iter[::-1]
            flip_box=True

        image_iter, box_iter, info_iter=self.preprocess_info(image_iter,gt_box_iter.copy(),iid,flip_box)
        
        box_iter = np.hstack((box_iter, np.array([[cf_label]])))
        gt_box_iter = np.hstack((gt_box_iter, np.array([[cf_label]])))

        ref_iter = torch.from_numpy(ref_iter).long()
        image_iter = self.transforms(image_iter)
        box_iter = torch.from_numpy(box_iter).float()
        gt_box_iter = torch.from_numpy(gt_box_iter).float()
        info_iter = np.array(info_iter)
        aw_iter = torch.from_numpy(aw_iter).long()
        negs_iter = torch.from_numpy(negs_iter).long()

        # r_iter = torch.from_numpy(r_iter).long()
        return ref_iter, image_iter, box_iter, gt_box_iter, info_iter, aw_iter, negs_iter

    def __len__(self):
        return self.data_size

    def shuffle_list(self, list):
        random.shuffle(list)
