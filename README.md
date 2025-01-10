## Language

- [English](#english)
- [中文](#中文)

### English
Requirements: 
Licensed installation of TopazVideoAI

Installation:
git clone this project to custom_nodes folder
```
git clone https://github.com/sh570655308/ComfyUI-TopazVideoAI.git
```

First, set up environment variables
Open Topaz Video AI UI, log in to your account, press Ctrl+Shift+T to open the command window, and it will automatically set the model directory in environment variables
Then add Topaz's installation directory to the PATH. This step is for setting up FFmpeg. If you have previously set other paths, please remove them
Close the GUI, open shell terminal

```cd "C:\Program Files\Topaz Labs LLC\Topaz Video AI"```

```.\login```

The path may vary when you have custom installation path
Then close the shell terminal, now you can use this node normally

Usage:
Simply connect this nodes between video output and video save
![image](https://github.com/user-attachments/assets/18e10017-ebb3-4e9f-a4ae-45e33c641ff0)

Notification:
Use 2 or 4 for upscale factor, others may cause error now
~~Model selection is not ready yet, now only works with default~~
This node is designed for short AI generated videos. I didn't test it with long video, because comfyui transfer video as image batch, the node will encode and decode which cost longer time than TopazVideoAI GUI. 

Common errors:
No such filter: 'tvai_up'
Make sure the ffmpeg path is correct and unique - you must use the ffmpeg that comes with TopazVideoAI. Remember to remove ffmpeg from other paths and from the ComfyUI environment.

### 中文
要求：
已安装的 TopazVideoAI，要登录账户

安装：
将此项目克隆到 custom_nodes 文件夹

```
git clone https://github.com/sh570655308/ComfyUI-TopazVideoAI.git
```

使用：
首先要设置环境变量

打开topaz video ai的应用程序，登录账号，快捷键ctrl+shift+T打开命令窗口，会自动设置模型目录到环境变量。

如果自动设置模型目录失败，则需要手动设置，在用户环境配置中添加TVAI_MODEL_DATA_DIR 和TVAI_MODEL_DIR，数值为模型对应的目录

然后在path中添加topaz的安装目录，这一步是设置ffmpeg，如果之前有设置过其他路径的ffmpeg请删除

设置完成后关闭gui，打开shell终端输入：

```cd "C:\Program Files\Topaz Labs LLC\Topaz Video AI"```

```.\login```

之后就可以正常使用了

注意事项：
放大倍数请使用2或4，其他数值目前可能导致错误
~~模型选择功能尚未就绪，目前仅支持默认设置~~
此节点专为AI生成的短视频设计。由于ComfyUI以图像批次方式传输视频，节点需要进行编码和解码，因此相比TopazVideoAI图形界面处理时间更长，故未对长视频进行测试。

常见错误：
No such filter: 'tvai_up’
确保ffmpeg的path路径正确，必须用TopazVideoAI自带的ffmpeg。记得删除其他路径的ffmpeg以及comfyui环境中的ffmpeg。
