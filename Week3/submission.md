# Stable-Diffusion-webui部署及Lora微调



## sd部署

​	在autodl云服务器上成功部署文档中推荐的[stable-diffusion-webui基础版本]([AUTOMATIC1111/stable-diffusion-webui: Stable Diffusion web UI (github.com)](https://github.com/AUTOMATIC1111/stable-diffusion-webui))，并成功启动webui界面进行ai创作，在部署过程中遇到过环境配置失败，网速连接太慢，项目文件缺少，git无法拉取等问题，也都一一能够解决。

![stable-diffusion部署](https://github.com/puraGH/UniqueAI2024_SummerCamp/blob/main/Week3/pictures/sd%E9%83%A8%E7%BD%B2.png?raw=true)



## Lora训练器部署

在同一台云服务器上成功部署[kohya_ss官方项目]([bmaltais/kohya_ss (github.com)](https://github.com/bmaltais/kohya_ss))，其能够在GUI里进行Lora微调训练，在部署过程中也是遇到与sd部署类似的问题。

![Lora服务器部署](https://github.com/puraGH/UniqueAI2024_SummerCamp/blob/main/Week3/pictures/Lora%E8%AE%AD%E7%BB%83%E9%83%A8%E7%BD%B2.png?raw=true)



## Lora微调结果

选择的基础模型是官方模型sd-v1-4.skpt, 前后进行了两次微调

第一次微调图片集是原神角色凝光，使用19张图片。这个角色形象在sd-v1-4.skpt模型里是不存在的

原神凝光-基模：

![原神凝光-基模](https://github.com/puraGH/UniqueAI2024_SummerCamp/blob/main/Week3/pictures/%E5%8E%9F%E7%A5%9E%E5%87%9D%E5%85%89-%E5%9F%BA%E6%A8%A1.png?raw=true)



原神凝光-Lora

![原神凝光-Lora](https://github.com/puraGH/UniqueAI2024_SummerCamp/blob/main/Week3/pictures/%E5%8E%9F%E7%A5%9E%E5%87%9D%E5%85%89-Lora.png?raw=true)



第二次微调图片集是海贼王路飞尼卡形态，图片全从网上选取，官方模型sd-v1-4.skpt只认识路飞这个角色，并不认识它的尼卡形态。第一次微调使用28张图片，微调出来的结果很不理想，甚至有概率出现脸上全是皱纹的老人路飞，第二次微调对数据集进行了挑选，只使用20张图片并修改了一些标签，微调出来的结果有明显提升。

路飞尼卡-基模：
![路飞尼卡-基模](https://github.com/puraGH/UniqueAI2024_SummerCamp/blob/main/Week3/pictures/%E8%B7%AF%E9%A3%9E%E5%B0%BC%E5%8D%A1-%E5%9F%BA%E6%A8%A1.png?raw=true)

路飞尼卡-Lora：

![路飞尼卡-Lora](https://github.com/puraGH/UniqueAI2024_SummerCamp/blob/main/Week3/pictures/%E8%B7%AF%E9%A3%9E%E5%B0%BC%E5%8D%A1-Lora.png?raw=true)
