# how to compile and run paddle with openvino

## Clone the code
```
git clone https://github.com/luo-cheng2021/Paddle-Lite.git -b luocheng/int_ov
```

## Build paddle lite(default debug version)
```
source /openvino/path/bin/setupvars.sh
cd Paddle-Lite
./lite/tools/build.sh x86
```

## Build demo
```
cd build.lite.x86/inference_lite_lib/demo/cxx/mobilenetv1_full
chmod +x build.sh
./build.sh
```

## Run demo
Download resetnet50 to a dir, the model file should be 'model.pdmodel' and 'model.pdiparams'. Execute
```
export LD_LIBRARY_PATH=`pwd`/../../../third_party/mklml/lib:$LD_LIBRARY_PATH
export OV_FRONTEND_PATH=$LD_LIBRARY_PATH
./mobilenet_full_api path/to/resnet_dir 1,3,224,224 1 1 1
```
