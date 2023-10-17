# NCHU_AI-Project_Sub_project-1
---

## (1) 專案描述
先前第一年的計畫中，我們使用了被廣為使用的EfficientNet來做遷移式學習，同時在資料處理上使用了資料擴增來增加訓練的資料，並針對真實圖片進行圖像分類這部分獲得了非常高的成績。雖然本次
實驗以少量的資料就獲得了不錯的結果但還是有許多可以改進的部分，若可獲得更多真實的圖片且保持各類別的平衡便可直接的使用不需強化資料，也不會有特定類別的精確度下降的問題，模型選擇上也可
以在不流失過多精確度的狀況下選擇較輕量化的模型，使得實際應用時減少延遲。

在第2年的計畫中，關於模型的選擇上我們主要區分為2種類別，分別是傳統的CNN的架構和CNN + Transformer這2種，而這2種的模型架構一樣都有使用到Transfer learning的部分，採用的預訓練模型都是ImageNet dataset，此外有使用到Transformer的模型
一般來說都較為複雜同時訓練時間會長很多，相較之下傳統CNN模型雖然不像Transformer那麼複雜，但在特定情況下可能會有不錯的性能表現，同時也能節省大量的訓練時間。另一方面，將CNN與Transformer結合的模型也是我們的研究重點。儘管這種組合可能
需要更長的訓練時間，但藉由複雜的模型更能夠捕捉到圖像中複雜的關係，以達到進一步提高分類的精確度。第2年的計劃將是建立在第一年成果的基礎上，通過對多種模型架構的訓練研究和在不同資料集
的選擇上，來考慮整體模型的精確度、速度和實際上應用需求的評估。

---
## (2)應用
圖像分類是計算機視覺中最基礎的一個任務，即在多種類別圖像中分類出正確的類別，本次任務目的即在正確分類牛隻身分上，並優化實驗結果使其能接軌後續計畫，資料選擇上使用完整的乳牛後背俯視圖
為主，模型架構方面除了先前的EfficientNet 外還加上了Resnet50和R50-ViT-B_16，並使用預訓練模型做遷移式學習以節省整體訓練時間。

---
## (3)資料集鏈結
University of Bristol Cows2021 Dataset (開放資料集)

https://data.bris.ac.uk/data/dataset/4vnrca7qw1642qlwxjadp87h7

---
## (4)程式碼

在文件夾`timm`下的`cowimg`裡放數據集，分成二個文件夾: `train/validate`，對應 訓練/驗證 數據文件夾,每個子文件夾下再依據分類的類別，每個類別建立一個對應的文件夾放置該類別的圖片。

另外test的部分在文件夾`timm`下的`inference_test`裡放要當測試集的數據圖片。

最終大概結構為：
```
- cowimg
  - train
    - class_0
      - 0.jpg
      - 1.jpg
      - ...
    - class_1
      - ...
    - ..
  - validate
    - class_0
      - 0.jpg
      - 1.jpg
      - ...
- ...
```

### 部分重要配置參數說明

針對`train.py`裡的部分重要參數說明如下：

- `--model`: 可選擇調整當前想要訓練的模型類別；
- `--image-size`: 輸入應該為兩個整數值，預訓練模型的輸入時正方形的，也就是[224,224]之類的；
   實際可以根據自己需要更改，數據預處理時，會將圖像等比例resize然後再padding（默認用0 padding）到指定的輸入尺寸。
- `--num-classes`: 分類模型的預測類別數；
- `-b`: 設置batch size大小，可根據GPU顯存規格設置；
- `--data_dir`: 設置訓練的資料集資料夾路徑；
- `--dataset`: 設置訓練的資料集資料夾名稱；
- `--initial-checkpoint`: 如果訓練有中斷或是想用其他的權重接著訓練可以參考設置；
- `--epochs`: 訓練的總迭代次數；

針對`validate.py`裡的部分重要參數說明如下：

- `--model`: 需要調整和train的模型相同名稱；
- `--data_dir`: 設置訓練的資料集資料夾路徑；
- `--dataset`: 設置訓練的資料集資料夾名稱；
- `--checkpointt`: 選擇訓練完想要拿來測試的權重檔案的路徑位置；

---
### Train
1. 資料夾 `pytorch-image-models-main` 下的 `train.py`檔案。
2. `train.py` 由上述引導設定。(重點data、image-size、num_classes、train、epochs、batch_size...)
3. 運行`train.py` 開始訓練。
4. 這邊會將運行結果的權重檔案存放在文件夾`output`下的`train`。

### Test  

1. 運行`validate.py`，這裡的validate程式碼就是我們test的部分。
2. 這邊會將運行結果存放在文件夾`output`下的`inference`。

---
### Result 
- 實驗1 (Resnet50) 資料集訓練/驗證/測試分割為(7:1:2)
![image](https://github.com/dannyFan-0201/NCHU_AI-Project_Sub_project-1/assets/47968782/6fc72645-3dc5-462d-8bf5-8bd43d226d06)

- 實驗2 (R50-ViT-B_16) 資料集訓練/驗證/測試分割為(7:1:2)
![image](https://github.com/dannyFan-0201/NCHU_AI-Project_Sub_project-1/assets/47968782/cfa8da18-0c13-412e-a435-25c1c9b7f55b)

- 實驗3 (Efficientnet_b4) 資料集訓練/驗證/測試分割為(7:1:2)
![image](https://github.com/dannyFan-0201/NCHU_AI-Project_Sub_project-1/assets/47968782/36249a5d-9895-447d-8d04-d3688ee787d0)

-各模型綜合參數比較表

![image](https://github.com/dannyFan-0201/NCHU_AI-Project_Sub_project-1/assets/47968782/3c5241ea-cb47-48ad-8fb5-e3d51b788c9e)


---
## (5) 感謝下列計畫的支持
-(1) 計畫編號： MOST 110-2634-F-005-006

-(2)計畫編號： NSTC 111-2634-F-005-001

---
## (6)Reference

 [Oquab, M., Bottou, L., Laptev, I., & Sivic, J. (2014). Learning and transferring mid-level image representations using convolutional neural networks. In Proceedings of the IEEE conference on computer vision 
 and pattern recognition (pp. 1717-1724). ](https://ieeexplore.ieee.org/document/6909618)

[Tan, M., and Le, Q. (2019 May). Efficientnet: Rethinking model scaling for convolutional neural networks. In International conference on machine learning, PMLR (pp. 6105-6114).](https://proceedings.mlr.press/v97/tan19a/tan19a.pdf)

[Deng, J., Dong, W., Socher, R., Li, L. J., Li, K., & Fei-Fei, L. (2009, June). Imagenet: A large-scale hierarchical image database. In IEEE conference on computer vision and pattern recognition (pp. 248-255).](https://ieeexplore.ieee.org/document/5206848)

( https://data.bris.ac.uk/data/dataset/4vnrca7qw1642qlwxjadp87h7)

(https://ithelp.ithome.com.tw/m/articles/10264843)

(https://zhuanlan.zhihu.com/p/464920124)

(https://github.com/rwightman/pytorch-image-models)

---
## Contact
If you have any question, feel free to contact danny80351@gmail.com
