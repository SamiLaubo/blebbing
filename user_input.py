from PIL import ImageTk, Image, ImageDraw
import PIL
import colorsys
import numpy as np
from tkinter import *
import cv2
import os

class Interface(Tk):
    def __init__(self, image_sequence, snakes):
        super(Interface, self).__init__()

        # Storage for drawings
        self.snakes = snakes
        self.buffer = []
        self.snake = []

        # Image
        self.image_sequence = image_sequence
        self.frame_idx = 0
        self.update_image() # Creates self.image
        self.w, self.h = self.image.width(), self.image.height()

        # Don't make window larger than computer screen
        self.canvas = Canvas(self, width=self.w, height=min(self.h, self.winfo_screenheight() - 200))
        self.canvas.pack()
        self.create_new()
        self.canvas.bind("<B1-Motion>", self.paint)

        # Cycle frame buttons
        buttons_cycle = Frame()
        buttons_cycle.pack()

        self.button_prev = Button(buttons_cycle, text="<-- Previous frame", command=self.prev_frame)
        self.button_prev.pack(side=LEFT)

        self.text_frame = Text(buttons_cycle, height=1, width=10, background=self.canvas["background"])
        self.text_frame.tag_configure("center", justify='center')
        self.text_frame.insert(INSERT, f"[1/{self.image_sequence.shape[0]}]")
        self.text_frame.tag_add("center", "1.0", "end")
        self.text_frame.pack(side=LEFT)

        self.button_next = Button(buttons_cycle, text="Next frame -->", command=self.next_frame)
        self.button_next.pack()

        # Saving buttons
        self.button1 = Button(self, text="save current selection and start new", command=self.save_and_add_new)
        self.button1.pack()

        self.button2 = Button(self, text="clear current selection and start new", command=self.clear)
        self.button2.pack()

        self.button3 = Button(self, text="save current selection and finish", command=self.save_and_quit)
        self.button3.pack()

        self.button4 = Button(self, text="cancel current selection and finish", command=self.quit)
        self.button4.pack()


    def create_new(self):
        self.image_canvas = self.canvas.create_image(self.w // 2, self.h // 2, image=self.image)
        self.output_image = PIL.Image.new("L", (self.w, self.h), 0)
        self.draw = ImageDraw.Draw(self.output_image)
        self.x = -1
        self.y = -1
        self.start_x = -1
        self.start_y = -1

    def clear(self):
        self.canvas.delete("all")
        self.create_new()
        self.snake = []

    def save_and_add_new(self):
        self.save_shape()
        self.update_background()
        self.clear()

    def save_and_quit(self):
        self.save_shape()
        self.buffer = np.array(self.buffer)

        # Delete the temporary image we used for annotation
        os.remove("data/tmp.png")

        self.destroy()

    def quit(self):
        self.buffer = np.array(self.buffer)
        self.destroy()

    def save_shape(self):
        # save user input in the output
        self.draw.line([self.x, self.y, self.start_x, self.start_y], fill="white", width=1)
        self.canvas.create_line(self.x, self.y, self.start_x, self.start_y, fill="white", width=1)
        self.buffer.append(np.array(self.output_image))

        # Save snake coordinate list
        self.snakes.append({
            'start_frame': self.frame_idx,
            'data': np.array(self.snake).copy(),
            'color': self.get_color(len(self.buffer)),
            'name': f'Bleb {len(self.buffer)}'
        })
        self.snake = []

    def paint(self, event):
        x, y = event.x, event.y
        if self.x >= 0 and self.y >= 0:
            self.draw.line([self.x, self.y, x, y], fill="white", width=1)
            self.canvas.create_line(self.x, self.y, x, y, fill="white", width=1)
            if self.start_x < 0 or self.start_y <0:
                self.start_x, self.start_y = self.x, self.y
        self.x, self.y = x, y

        self.snake.append((y,x))



    def update_background(self):
        r, g, b = self.get_color(len(self.buffer) - 1)
        for y in range(self.image.height()):
            for x in range(self.image.width()):
                if self.buffer[-1][y, x] > 0:
                    self.image.put("#%02x%02x%02x" % (r, g, b), (x, y))

    def get_color(self, index):
        hue = (index * 0.382) % 1 # a 0.382th of a circle is the golden angle which yields hues that have great differences
        r, g, b = colorsys.hsv_to_rgb(hue,1,1)
        return np.array([(int)(255*r), (int)(255*g), (int)(255*b)])
    
    def next_frame(self):
        self.frame_idx = min(self.frame_idx + 1, self.image_sequence.shape[0] - 1)
        self.update_image()
        self.change_frame()

    def prev_frame(self):
        self.frame_idx = max(self.frame_idx - 1, 0)
        self.update_image()
        self.change_frame()

    def change_frame(self):
        # Update frame number text
        self.text_frame.delete(1.0, END)
        text = f"[{self.frame_idx+1}/{self.image_sequence.shape[0]}]"
        self.text_frame.insert(END, text, "center")
        self.text_frame.tag_add("center", "1.0", "end")

        # Update image
        self.canvas.itemconfig(self.image_canvas, image = self.image)

    def update_image(self):
        # Load the image we want to annotate
        img = cv2.cvtColor((self.image_sequence[self.frame_idx]*255).astype('uint8'), cv2.COLOR_RGB2BGR)
        
        # Create directory
        if not os.path.exists("data"): os.makedirs("data")
        # Save it in a format that photo image can handle (png)
        cv2.imwrite("data/tmp.png", img)

        # Create PhotoImage
        self.image = PhotoImage(file="data/tmp.png", master=self)
        
        