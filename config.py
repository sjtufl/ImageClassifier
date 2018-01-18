#!/usr/bin/env python
#encoding:utf-8
"""
Set the config variable.
"""
import ConfigParser as cp
import json

config = cp.RawConfigParser()
config.read('./data/config/config.cfg')

orientations = json.loads(config.get("hog", "orientations"))
pixels_per_cell = json.loads(config.get("hog", "pixels_per_cell"))
cells_per_block = json.loads(config.get("hog", "cells_per_block"))
visualize = config.getboolean("hog", "visualize")
normalize = config.getboolean("hog", "normalize")
train_feat_path = config.get("path", "train_feat_path")
test_feat_path = config.get("path", "test_feat_path")
svm_model_path = config.get("path", "svm_model_path")
knn_model_path = config.get("path", "knn_model_path")
lda_model_path = config.get("path", "lda_model_path")
