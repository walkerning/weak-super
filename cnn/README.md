CNN
------------

```bash
cd py-faster-rcnn/caffe-fast-rcnn
cp Makefile.config.example Makefile.config # 并修改: BLAS=mkl, WITH_PYTHON_LAYER=1
make -j 20 && make pycaffe
cd ../lib
make
```

现在使用了在**ImageNet**的分类数据集上pre-trained的``bvlc_reference_caffenet``模型的卷积层提feature

