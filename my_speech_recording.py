import tkinter as tk
from tkinter import messagebox as msb

import sounddevice as sd
import queue
import soundfile as sf
import threading

class App(tk.Tk):
    def	__init__(self):
        super().__init__()
        #Create a queue to contain the audio data
        self.q = queue.Queue()
        #Declare variables and initialise them
        self.recording = False
        self.file_exists = False  
        self.data = None
        self.index = 0
        
        
        
        
        self.title('Speech Signal Processing')
        # Tạo widget
        self.cvs_figure = tk.Canvas(self, width= 600, height= 300)
        
        lblf_upper = tk.LabelFrame(self)
        btn_open = tk.Button(lblf_upper, text = 'Open', width= 8)
        btn_record = tk.Button(lblf_upper, text = 'Record', width= 8, command=lambda m=1:self.threading_rec(m))
        btn_stop = tk.Button(lblf_upper, text= 'Stop', width= 8, command=lambda m=2:self.threading_rec(m))
        btn_play = tk.Button(lblf_upper, text= 'Play', width= 8, command=lambda m=3:self.threading_rec(m))
        
        
        
        btn_open.grid(row= 0, padx= 5, pady= 5)
        btn_record.grid(row= 1, padx= 5, pady= 5)
        btn_stop.grid(row= 2, padx= 5, pady= 5)
        btn_play.grid(row= 3, padx= 5, pady= 5)
        
        
        lblf_lower = tk.LabelFrame(self)
        btn_zoom = tk.Button(lblf_lower, text = 'Zoom', width= 8, command = self.btn_zoom_click)
        btn_next = tk.Button(lblf_lower, text = 'Next', width= 8, command = self.btn_next_click)
        btn_prev = tk.Button(lblf_lower, text= 'Prev', width= 8, command = self.btn_prev_click)
        
        
        btn_zoom.grid(row= 0, padx= 5, pady= 5)
        btn_next.grid(row= 1, padx= 5, pady=5)
        btn_prev.grid(row= 2, padx= 5, pady= 5)
        
        
        # Đưa widget lên lưới
        self.cvs_figure.grid(row = 0, column = 0, rowspan=2, padx = 5, pady = 5)
        lblf_upper.grid(row = 0, column= 1, padx = 5, pady =6, sticky= tk.N)
        lblf_lower.grid(row = 1, column= 1, padx = 5, pady = 6, sticky= tk.S)
        
        
        
       
        self.cvs_figure.bind("<Button-1>", self.xu_ly_mouse)
        
    def xu_ly_mouse(self, event):
        x = event.x
        y = event.y
        print("Toạ độ x= ", x, " y= ", y)
        
    #Fit data into queue
    def callback(self, indata, frames, time, status):
        self.q.put(indata.copy())
        
    #Functions to play, stop and record audio
    #The recording is done as a thread to prevent it being the main process
    def threading_rec(self, x):
        if x == 1:
            #If recording is selected, then the thread is activated
            t1=threading.Thread(target= self.record_audio)
            t1.start()
        elif x == 2:
            #To stop, set the flag to false
            self.recording = False
            msb.showinfo(title="Stop Recording",message="Recording finished")
            self.data, fs = sf.read("trial.wav", dtype='int16')
            L = len(self.data)
            N = L // 600
            yc = 150
            #Xoá canvas cũ
            self.cvs_figure.delete('all')
            for x in range (0,600-1):
                a = int(self.data[x*N])
                b = int(self.data[(x+1)*N])
                y1 = (a+32767)*300//65535 - 150
                y2 = (b+32767)*300//65535 - 150
                self.cvs_figure.create_line(x,yc-y1,x+1,yc-y2, fill='green')
        elif x == 3:
            #To play a recording, it must exist.
            if self.file_exists:
                #Read the recording if it exists and play it
                data, fs = sf.read("trial.wav", dtype='float32') 
                sd.play(data,fs)
                sd.wait()
            else:
                #Display and error if none is found
                msb.showerror(title="Error", message="Record something to play")
                
                
    #Recording function
    def record_audio(self):
        #Set to True to record
        self.recording= True   
        #Create a file to save the audio
        msb.showinfo(title= "Recording Speech", message="Speak into the mic")
        with sf.SoundFile("trial.wav", mode='w', samplerate= 16000, #44100,
                            channels=1) as file:
        #Create an input stream to record audio without a preset time
                with sd.InputStream(samplerate=16000, channels=1, callback=self.callback):
                    while self.recording == True:
                        #Set the variable to True to allow playing the audio later
                        self.file_exists =True
                        #write into file
                        file.write(self.q.get())

    def btn_zoom_click(self):
        self.cvs_figure.delete('all')
        yc = 150
        i = self.index
        for x in range (0,600-1):
                a = int(self.data[x+i])
                b = int(self.data[(x+i+1)])
                y1 = (a+32767)*300//65535 - 150
                y2 = (b+32767)*300//65535 - 150
                self.cvs_figure.create_line(x,yc-y1,x+1,yc-y2, fill='green')
     
    def btn_next_click(self):
        self.cvs_figure.delete('all')   
        yc = 150
        self.index += 1
        i = self.index 
        for x in range (0,600-1):
                a = int(self.data[i*600+x])
                b = int(self.data[(i*600+x+1)])
                y1 = (a+32767)*300//65535 - 150
                y2 = (b+32767)*300//65535 - 150
                self.cvs_figure.create_line(x,yc-y1,x+1,yc-y2, fill='green')
        
    def btn_prev_click(self):
        self.cvs_figure.delete('all')   
        yc = 150
        self.index -= 1
        i = self.index
        for x in range (0,600-1):
                a = int(self.data[i*600+x])
                b = int(self.data[(i*600+x+1)])
                y1 = (a+32767)*300//65535 - 150
                y2 = (b+32767)*300//65535 - 150
                self.cvs_figure.create_line(x,yc-y1,x+1,yc-y2, fill='green')
if	__name__ ==	"__main__":
    app	= App()
    app.mainloop()