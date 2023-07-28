# easyHumanNeRF
End-to-end Implementation of HumanNeRF with custom dataset

[HumanNeRF](https://github.com/chungyiweng/humannerf)+[VIBE](https://github.com/mkocabas/VIBE)+[YOLOv7](https://github.com/WongKinYiu/yolov7)

<p float="center">
  <img src="./assets/easyHumanNeRFStructure.png" width="78%" />
</p>


### Fast Train

```bash
git clone https://github.com/IVL-PKU/easyHumanNeRF.git
cd easyHumanNeRF/
vim easy_train
```
set your path of images:
```bash
IMAGES="your/images/"  # please use absolute path
```

```bash
sh easy_train.sh
```



## TODO LIST

- [ ] The schedule
  - [x] end-to-end training HumanNeRF
  - [ ] detailed README
  - [ ] acceleration
  - [ ] Multi-view HumanNeRF


## Acknowledge

easyHumanNeRF is an integration of [HumanNeRF](https://github.com/chungyiweng/humannerf), [VIBE](https://github.com/mkocabas/VIBE) and [YOLOv7](https://github.com/WongKinYiu/yolov7). If this is helpful to you, please give stars to the above works. Thanks!
