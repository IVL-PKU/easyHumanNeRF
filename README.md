# easyHumanNeRF
End-to-end Implementation of HumanNeRF with custom dataset

[HumanNeRF](https://github.com/chungyiweng/humannerf)+[VIBE](https://github.com/mkocabas/VIBE)+[YOLOv7](https://github.com/WongKinYiu/yolov7)

<p float="center">
  <img src="./assets/easyHumanNeRFStructure.png" width="78%" />
</p>


### Fast Train

```sh
git clone https://github.com/IVL-PKU/easyHumanNeRF.git
cd easyHumanNeRF/
vim easy_train
```
set your path of images:
```
IMAGES="your/images/"  # please use absolute path
```

```sh
sh easy_train.sh
```
