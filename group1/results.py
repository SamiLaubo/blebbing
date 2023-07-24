from PIL import ImageTk, Image, ImageDraw
import PIL
import colorsys
import numpy as np
from tkinter import *
import cv2
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from os import path as os_path
import pandas as pd
import imageio

class Interface(Tk):
    def __init__(self, image_sequence, snakes):
        super(Interface, self).__init__()

        # Image
        self.image_sequence = image_sequence
        self.snakes = snakes
        self.frame_idx = 0
        self.update_image() # Creates self.image
        self.w, self.h = self.image.width(), self.image.height()

        # Don't make window larger than computer screen
        canvas_frame = Frame(self)
        canvas_frame.pack()
        self.canvas = Canvas(canvas_frame, width=self.w, height=min(self.h, self.winfo_screenheight() - 200))
        self.canvas.pack(side=LEFT)
        self.create_new()

        self.plot_area(canvas_frame)

        # Cycle frame buttons
        buttons_cycle = Frame(self)
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
        self.button1 = Button(self, text="Save as .gif", command=self.save_gif)
        self.button1.pack()

        # self.button2 = Button(self, text="Save as .tif", command=self.save_tif)
        # self.button2.pack()

        self.button3 = Button(self, text="Save area", command=self.save_area)
        self.button3.pack()

        self.button3 = Button(self, text="Quit", command=self.quit)
        self.button3.pack()


    def create_new(self):
        self.image_canvas = self.canvas.create_image(self.w // 2, self.h // 2, image=self.image)

    def quit(self):
        plt.close()
        self.destroy()
    
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
        
        # Draw snakes
        for snake in self.snakes:
            if snake.get('start_frame') <= self.frame_idx:
                snake_idx = self.frame_idx - snake.get('start_frame')
                contour = np.flip(snake.get('data')[snake_idx], axis=1).astype(np.int32)
                c = tuple(snake.get('color').tolist())
                cv2.drawContours(img, [contour], 0, color=c)

                text_coord = (np.min(contour[:,0], axis=0)-20, np.max(contour[:,1], axis=0)+20)
                cv2.putText(img, snake.get('name'), text_coord, 
                            cv2.FONT_HERSHEY_SIMPLEX, .5, c, 2)

        # save it in a format that photo image can handle (png)
        cv2.imwrite("data/tmp.png", img)

        # Create PhotoImage
        self.image = PhotoImage(file="data/tmp.png", master=self)

    def plot_area(self, master):
        self.area_fig = plt.figure(figsize=(6, 5), dpi=100)
        plt.title('Area of blebs')
        plt.xlabel('Frame')
        plt.ylabel('Area in pixels')

        for snake in self.snakes:
            x = np.arange(len(snake.get('area'))) + snake.get('start_frame')
            plt.plot(x, snake.get('area'), color=tuple(snake.get('color')/255), label=snake.get('name'))

        plt.legend(loc='best')

        graph = FigureCanvasTkAgg(self.area_fig, master)
        graph.get_tk_widget().pack(side=LEFT, fill=BOTH)

    def save_gif(self):
        # Create directory
        path = "output"
        if not os_path.exists(path):
            os.makedirs(path)

        new_name = "blebs"
        if os_path.exists(f"{path}\{new_name}.gif"):
            i = 1
            new_name = "blebs_1"
            while os_path.exists(f"{path}\{new_name}.gif") and i < 1000:
                i += 1
                new_name = "blebs_" + str(i)

        with imageio.get_writer(f"{path}\{new_name}.gif", mode='I', duration='0.25') as writer:
            for idx, frame in enumerate(self.image_sequence):
                # Draw snakes
                img = (frame*255).astype('uint8')

                for snake in self.snakes:
                    if snake.get('start_frame') <= idx:
                        snake_idx = idx - snake.get('start_frame')
                        contour = np.flip(snake.get('data')[snake_idx], axis=1).astype(np.int32)
                        c = tuple(snake.get('color').tolist())
                        cv2.drawContours(img, [contour], 0, color=c)

                        text_coord = (np.min(contour[:,0], axis=0)-20, np.max(contour[:,1], axis=0)+20)
                        cv2.putText(img, snake.get('name'), text_coord, 
                                    cv2.FONT_HERSHEY_SIMPLEX, .5, c, 2)
                        
                writer.append_data(img)
                
        print(f'Saved bleb gif in: {path}\{new_name}.gif')


    def save_area(self):
        # Create directory
        path = "output"
        if not os_path.exists(path):
            os.makedirs(path)

        # Figure
        new_name = "bleb_area"

        i = 0
        if os_path.exists(f"{path}\{new_name}.pdf"):
            i = 1
            new_name = "bleb_area_1"
            while os_path.exists(f"{path}\{new_name}.pdf") and i < 1000:
                i += 1
                new_name = "bleb_area_" + str(i)

        self.area_fig.savefig(f"{path}\{new_name}.pdf")
        print(f'Saved area figure in: {path}\{new_name}.pdf')

        # csv data
        frame_count = self.image_sequence.shape[0]
        d = {'Frame': list(range(1, frame_count+1))}

        for snake in self.snakes:
            snake_frames = list(range(snake.get('start_frame')+1, frame_count+1))
            d[snake.get('name')] = pd.Series(snake.get('area'), index=snake_frames)

        df = pd.DataFrame(data=d, index=list(range(1, frame_count+1)))

        if i == 0:
            df.to_excel(f'{path}\\area_data.xlsx', index = False)
            print(f'Saved area data in: {path}\\area_data.xlsx')
        else:
            df.to_excel(f'{path}\\area_data_{i}.xlsx',index = False)
            print(f'Saved area data in: {path}\\area_data_{i}.xlsx')
        