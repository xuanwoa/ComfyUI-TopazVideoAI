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
```
cd "C:\Program Files\Topaz Labs LLC\Topaz Video AI"
.\login
```
The path may vary when you have custom installation path
Then close the shell terminal, now you can use this node normally

Usage:
Simply connect this nodes between video output and video save
![image](https://github.com/user-attachments/assets/18e10017-ebb3-4e9f-a4ae-45e33c641ff0)

Notification:
Use 2 or 4 for upscale factor, others may cause error now
Model selection is not ready yet, now only works with default
This node is designed for short AI generated videos. I didn't test it with long video, because comfyui transfer video as image batch, the node will encode and decode which cost longer time than TopazVideoAI GUI. 
