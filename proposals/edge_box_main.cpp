// edge box
// Some code are copied from https://github.com/samarth-robo/edges
#include <cstdio>
#include "opencv2/opencv.hpp"
#include "edge_boxes.h"
#include "edge_detect.h"

using namespace std;
using namespace cv;

void load_config() {
}

int main(int argc, char* argv[]) {
  if (argc != 3) {
    cout << "Usage: ./edge_box_main image_file cfg_file" << endl;
    return -1;
  }

  Mat im;
  im = imread(argv[1]);

  Mat ime, grad_ori, ime_t, grad_ori_t;
  EdgeBoxGenerator edgeBoxGen;
  Boxes boxes;
  // TODO: 统一的config object实现
  // 可能可以用 https://github.com/skystrife/cpptoml
  load_config();
  edgeBoxGen._alpha = 0.65; 
  edgeBoxGen._beta = 0.75;
  edgeBoxGen._eta = 1;
  edgeBoxGen._minScore = 0.1;
  edgeBoxGen._maxBoxes = 800;
  edgeBoxGen._edgeMinMag = 0.1;
  edgeBoxGen._edgeMergeThr = 0.5;
  edgeBoxGen._clusterMinMag = 0.5;
  edgeBoxGen._maxAspectRatio = 3;
  edgeBoxGen._minBoxArea = 1000;
  edgeBoxGen._gamma = 2;
  edgeBoxGen._kappa = 1.5;
  
  // FIXME: 作为配置传入sf的路径..
  edge_detect(im, ime, grad_ori, string("/home/foxfi/data/sf.dat"));

  transpose(ime, ime_t);
  transpose(grad_ori, grad_ori_t);
  if(!(ime_t.isContinuous() && grad_ori_t.isContinuous())) {
    cerr << "Matrices are not continuous, hence the Array struct will not work" << endl; 
  }
  arrayf E; E._x = ime_t.ptr<float>();
  arrayf O; O._x = grad_ori_t.ptr<float>();
  Size sz = ime.size();
  int h = sz.height; O._h=E._h=h;
  int w = sz.width; O._w=E._w=w;
    
  arrayf V;
  edgeBoxGen.generate(boxes, E, O, V);

  // output results
  int n = (int) boxes.size();
  cerr << "Found " << n << " boxes" << endl;
  for(int i = 0; i < n; i++) {
    printf("%d %d %d %d %f\n", boxes[i].c, boxes[i].r, 
	   boxes[i].c + boxes[i].w,
	   boxes[i].r + boxes[i].h,
	   boxes[i].s);
  }
  return 0;
}

