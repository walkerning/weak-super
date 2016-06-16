# -*- coding: utf-8 -*-

from .trainer import Trainer
from .dataset import Dataset
from .proposal import Proposaler
from .feature import FeatureExtractor as _FE
from .detector import Detector

import random

class IterativeTrainer(Trainer):
    """
    迭代训练器
    """
    TYPE = "iterative"

    K = 10

    def __init__(self, cfg):
        self.cfg = cfg
        self.dataset = Dataset.get_registry(cfg["dataset"]["type"])(cfg)
        self.proposaler = Proposaler.get_registry(cfg["proposal"]["type"])(cfg)
        self.feat_ext = _FE.get_registry(cfg["feature"]["type"])(cfg)

        self.pos_feature_dict = {} # key: im_ind
        self.neg_feature_dict = {}
        self.rois_dict = {} # key: im_ind (Maybe we don't need, if we need, we should divide this dict into two parts —— positive one and negetive one)

    def _proposal_and_features(self, im_ind):
        # handle dataset
        im = self.dataset.get_image_at_index(im_ind) # get a image
        rois = self.proposaler.make_proposal(im) # get many proposals of one image
        # extract features
        return rois, self.feat_ext.extract_from_rois(im, rois) # every proposal has a feature

    def train(self):
        print "'{}' trainer start to train!".format(self.TYPE)

        for cls_ind in range(self.dataset.class_number):
            detector = Detector.get_registry(self.cfg["detector"]["type"])(self.cfg)
            
            print "start to train class ", self.dataset.class_names[cls_ind]
            pos_train_indexes = self.dataset.positive_train_indexes(cls_ind) # all positive image indexes in every class cls_ind
            neg_train_indexes = self.dataset.negative_train_indexes(cls_ind) # all negative image indexes in every class cls_ind
            hard_neg_feature = None 

            # get features of all rois of all positive images in every class 
            for im_ind in pos_train_indexes:
                if im_ind not in self.pos_feature_dict:
                    self.rois_dict[im_ind], self.pos_feature_dict[im_ind] = self._proposal_and_features(im_ind)
            # get features of all rois of all negative images in every class 
            for im_ind in neg_train_indexes:
                if im_ind not in self.neg_feature_dict:
                    self.rois_dict[im_ind], self.neg_feature_dict[im_ind] = self._proposal_and_features(im_ind)

            # initialization(choose one proposal from all proposals of every image)
            one_feat_dict = self.first_init(pos_train_indexes) # first initialization(use entire image up to a 4% border), one_feat_dict is a dict, key: im_ind
            

            # alternative deciding latent labels and training detectors
            results = None # Detect score列表
            while results is None or not self.judge_converge(results): # FIXME: 传参数
                # input: one proposal with feature every image
                # output: a detector
 
                # divide po-images into K folds
                lens = len(pos_train_indexes) # num of positive features for big detector(num of all positive images)
                s_lens = (lens / K - 1) * K # num of positive features for small detector 

                fold = np.zeros((K, lens/K)) # every row contains lens/K po_im_indexes
                random.shuffle(pos_train_indexes)
                for i in range(K):
                    fold[i, :] = pos_train_indexes[i:(i+1)*lens/K]
                    
                # For k = 1 to K
                for i in range(K):
                    # train using positive one_feature(one every image) in all folds but i
                    # set s_lens-s positive features and 2 * s_lens-s negative ones to de_features(np.array)
                    # first positive
                    for j in range(K):
                        if i != j:
                            # put all windows(features) of this fold to de_features
                            for ind in fold[j]:
                                de_features = np.vstack((de_features, np.array(one_feat_dict[ind])) # add a row to this array

                    # second negative (only select 2*s_lens negative ones randomly)
                    for j in range(2 * s_lens):
                        feat = random.sample(neg_feature_dict[random.sample(neg_feature_dict, 1)[0]])[0] # select a feature(from a box of a negative image)  
                        de_features = np.vstack((de_features, np.array(feat)))

                    # set their labels (1: positive, 0: negative)
                    de_labels = np.append(np.ones((1, s_lens)), np.zeros((1, 2 * s_lens)))
                    clf = detector.train(de_features, de_labels) # get a detector
                        
                    # relocalize positive rois(one every image) in fold i using this detector(choose the roi of which the propobility is the highest)
                    for im_ind in fold[i]:
                        # te_labels, te_proba are np.array
                        te_labels, te_proba = detector.test(pos_feature_dict[im_ind])
                        index = np.where(min(te_proba[np.where(te_labels == 1)]) == te_proba)[0][0] 
                        one_feat_dict[im_ind] = pos_feature_dict[im_ind][index]# labels == 1 && proba is minimum

                # train the final detector
                # positives(list)
                pos_feats = one_feat_dict.values()
                # negatives(np.array, if hard-negative features exist, use them)
                if hard_neg_feature not None:
                    neg_feats = hard_neg_feature[0: 2 * lens]
                    if len(neg_feats) < 2 * lens:
                        for j in range(2 * lens - len(neg_feats)):
                            feat = random.sample(neg_feature_dict[random.sample(neg_feature_dict, 1)[0]])[0] # select a feature(from a box of a negative image)  
                            neg_feats = np.vstack((neg_feats, np.array(feat))) 
                else:
                    for j in range(2 * lens):
                        feat = random.sample(neg_feature_dict[random.sample(neg_feature_dict, 1)[0]])[0] # select a feature(from a box of a negative image)  
                        neg_feats = np.vstack((neg_feats, np.array(feat)))
                    
                # set labels
                labels = np.append(np.ones((1, lens)), np.zeros((1, 2 * lens)))
                feats = np.vstack((pos_feats, neg_feats))
                detector.train(feats, labels) # 更新params

                # perform hard-negative mining using this detector(get some negative proposals for next iteration)
                results = detector.test(pos_feats) # 检测正例
                neg_results, proba = detector.test(neg_feats) # get hard-negative features
                hard_neg_feature = neg_feats[np.where((neg_results == 1))] 

            detector.save_param() # save all parameters of SVM after iteration 

    def judge_converge(self):
        # 判断收敛条件是否达到, FIXME: 没有指定参数
        pass

    def first_init(self, pos_train_indexes):
        # get a proposal up to 4% border
        return {im_ind:pos_feature_dict[im_ind][0] for im_ind in pos_train_indexes}

