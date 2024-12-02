配置环境python version == 3.11

安装依赖
pip install -r requirements.txt
建议先单独cupy安装pip install cupy-cuda12x
再运行requirements

文件包含：
main：主程序，运行即可展示掩码图像，可调参数方差计算窗口以及阈值大小，可作剪切操作（效果不明显），对HEplan.enhance_contrast可选择CLAHE模式和LRSHE模		    式，默认LRSHE
DVAbase：基础调用，包含方差计算以及循环展示部分函数
HEplan：局部均衡化算法
Viewdicom：读取.dcm文件并逐帧展示
Flow：经典光流，已废弃（暂时）
TestDVA：测试用


文件夹：
demo1：当前效果存储
Input：输入文件存储