import torch
import numpy as np
import tkinter as tk
from tkinter import font
from PIL import Image, ImageDraw
from CNN import CNN

class HandwritingRecognizer:
    def __init__(self, root, model):
        chinese_font = font.Font(family="WenQuanYi Zen Hei 12", size=12) # 创建一个支持中文的字体对象
        root.option_add("*Font", chinese_font) # 将Tkinter的默认字体替换为此字体(影响标题栏、按钮、标签等)
        root.geometry("400x380")  # 设置窗口宽400高300
        root.title("数字手写板") # 设置窗口主题

        self.last_x = None
        self.last_y = None

        self.model = model

        self.canvas = tk.Canvas(root, width=280, height=280, bg='white') # 创建画布：宽高 280 像素，背景黑色
        self.canvas.pack(pady=10) # 将画布放在窗口内，靠上边缘距离10个像素
        # 绑定事件:按住鼠标左键并移动时触发 paint
        # B1:鼠标左键 Motion:移动鼠标
        self.canvas.bind('<B1-Motion>', self.paint)
        # 鼠标松开时重置，避免两笔及以上的写法都连接上
        self.canvas.bind('<ButtonRelease-1>', self.reset_last)

        # 初始化内存图像（灰度模式 'L'，背景白色）
        self.image = Image.new('L', (280, 280), 'white')
        self.draw = ImageDraw.Draw(self.image)

        btn_frame = tk.Frame(root) # 按钮框架，用于整齐排列按钮
        btn_frame.pack() # 该按钮框架默认放在中间、顶部，由于有画布，所以在画布下方

        # 按钮放在按钮框架内，顶框架左侧，距离5个像素排列
        tk.Button(btn_frame, text='recognize', command=self.recognize).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text='clear', command=self.clear_canvas).pack(side=tk.LEFT, padx=5)

        self.result_label = tk.Label(root, text='result', font=("Arial", 18))
        self.result_label.pack(pady=10)

    def reset_last(self, event):
        self.last_x = None
        self.last_y = None

    # 鼠标拖拽时调用,在画布上绘制黑色圆点
    def paint(self, event):
        x, y = event.x, event.y
        r = 6
        if self.last_x is not None and self.last_y is not None:
            # 不是第一个点,那就连接当前点和上一个点,画线
            self.canvas.create_line(self.last_x, self.last_y, x, y, fill='black', width=r*2, capstyle=tk.ROUND, smooth=True)
            self.draw.line([self.last_x, self.last_y, x, y], fill='black', width=r*2)
        else:
            # 第一个点,在画布上画圆,填充和轮廓均为黑色
            self.canvas.create_oval(x-r, y-r, x+r, y+r, fill='black', outline='black')
            self.draw.ellipse([x-r, y-r, x+r, y+r], fill='black') # 在内存图像上绘制(PIL 图像)
        # # 每次都在当前点画一个圆点作为补充（加强覆盖）
        # self.canvas.create_oval(x-r, y-r, x+r, y+r, fill='black', outline='black')
        # self.draw.ellipse([x-r, y-r, x+r, y+r], fill='black')
        self.last_x = x
        self.last_y = y

    # 将内存图像保存到文件
    def save_image(self):
        self.image.save('digit.png')
        print('图像已保存为 digit.png')

    # 清空画布上的所有图形
    def clear_canvas(self):
        self.canvas.delete('all')
        # 同时清空内存图像:重新创建一张白色背景图
        self.image = Image.new('L', (280, 280), 'white')
        self.draw = ImageDraw.Draw(self.image)
        self.result_label.config(text="result")
        self.last_x = None
        self.last_y = None

    # 内存图像预处理，缩放、归一化、转张量
    def preprocess(self):
        img = self.image.copy().resize((28, 28), Image.Resampling.LANCZOS)
        img_arr = np.array(img, dtype=np.float32) / 255.0
        img_arr = 1.0 - img_arr
        tensor = torch.tensor(img_arr).unsqueeze(0).unsqueeze(0)
        return tensor
    
    def recognize(self):
        self.save_image()
        input_tensor = self.preprocess().cuda()
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            pred = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0, pred].item() * 100

        self.result_label.config(text=f'result:{pred} (confidence:{confidence:.1f}%)')

def load_model(model_path):
    cnn = CNN()
    cnn.load_state_dict(torch.load(model_path, map_location='cuda'))
    cnn.cuda()
    cnn.eval()
    return cnn

if __name__ == '__main__':
    model = load_model('model/mnist_model.pth')
    root = tk.Tk() # 创建主窗口
    app = HandwritingRecognizer(root, model)
    root.mainloop() # 显示主窗口