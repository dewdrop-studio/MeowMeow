---
marp: true
# backgroundColor: '#222'
theme: uncover
_class: invert

---

![bg right:40% 80%](https://em-content.zobj.net/source/microsoft-teams/363/cat-face_1f431.png)

# MeowMeow
image classification project using CNN  

>by Radhey & Aabish
---

## Project Overview 

![bg 40% opacity:0.4](https://em-content.zobj.net/source/microsoft-teams/363/camera_1f4f7.png)

- 🐱 dataset of labeled images marked silly or not-silly
- 🤖 Uses Keras for model training
- 📊 Streamlit dashboard for a clean interface

---

## Dataset & Model 

- Images loaded from local folders
- `isSilly` label: 1 (Target), 0 (Salt)
- Model: Keras-based neural network
- Visualizes dataset stats and image samples

<div style="position: absolute; top: 0; left: 0; width: 100vw; height: 100vh; margin: 0; padding: 0; overflow: hidden; z-index: -1; opacity: 0.5;">
    <iframe src="https://dewdrop-studio.github.io/MeowMeow/presentation/visu.html" style="position: absolute; top: 0; left: 0; border: none; width: 100vw; height: 100vh;"></iframe>
</div>

---



## Results & Visualization
<div style="font-size: 0.5em;">

| Layer (type)                | Output Shape         | Param #      |
|-----------------------------|---------------------|--------------|
| random_flip (RandomFlip)    | (None, 216, 235, 3) | 0            |
| random_rotation (RandomRotation) | (None, 216, 235, 3) | 0        |
| conv2d (Conv2D)             | (None, 214, 233, 32)| 896          |
| max_pooling2d (MaxPooling2D)| (None, 107, 116, 32)| 0            |
| conv2d_1 (Conv2D)           | (None, 105, 114, 32)| 9,248        |
| max_pooling2d_1 (MaxPooling2D)| (None, 52, 57, 32)| 0            |
| flatten (Flatten)           | (None, 94848)       | 0            |
| dense (Dense)               | (None, 64)          | 6,070,336    |
| dense_1 (Dense)             | (None, 64)          | 4,160        |
| dense_2 (Dense)             | (None, 1)           | 65           |

**Total params:** 18,254,117 (69.63 MB)  
**Trainable params:** 6,084,705 (23.21 MB)  
**Non-trainable params:** 0 (0.00 B)  
**Optimizer params:** 12,169,412 (46.42 MB)

</div>

---
### `Lets check whether your billi is silly`
![bg](https://media.giphy.com/media/13borq7Zo2kulO/giphy.gif)


---
## Scan me
(or just go to [sillibilli](https://sillibilli.streamlit.app))
![bg left:40% 80%](https://github.com/dewdrop-studio/MeowMeow/blob/master/presentation/qr-code.png?raw=true)

---