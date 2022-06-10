from mimetypes import init
from tkinter import *
from PIL import ImageTk, Image
from tkinter import filedialog
from tkinter.messagebox import *
import numpy as np

#window 기본 설정
window = Tk()
window.title("이모지 만들어주는 프로그램")
window.geometry("640x640+650+200")
window.resizable(False,False)
init_image = "./"

panedwindow1= PanedWindow(width = "300", height = "300" ,relief="sunken", bd=5)
panedwindow1.pack(expand=True)

init_image = Image.open('./camera.png')
imageForInit = ImageTk.PhotoImage(init_image.resize((320,320)))
imageLabel = Label(panedwindow1, image=imageForInit)
imageLabel.pack()

#이미지 선택을 했는지 체크
IsImageSelected = False

def btn_click():
    if IsImageSelected == False:
        showerror("오류", "이미지를 선택해야합니다!")
    else:
        #수정 필요@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@        
        print("이모지가 생성되었습니다")
    
def open():
    global IsImageSelected
    global my_image # 함수에서 이미지를 기억하도록 전역변수 선언 (안하면 사진이 안보임)
    panedwindow1.filename = filedialog.askopenfilename(initialdir='', title='파일선택', filetypes=(
    ('png files', '*.png'), ('jpg files', '*.jpg'), ('all files', '*.*')))
    
    #선택을 했을때만 실행
    if panedwindow1.filename != "":
        IsImageSelected = True
        Label(panedwindow1, text=panedwindow1.filename).pack() # 파일경로 view
        my_image = ImageTk.PhotoImage(Image.open(panedwindow1.filename))
        Label(panedwindow1, image=my_image).pack() #사진 view
        imageLabel.pack_forget()
        

btn_create = Button(window, text='이모지 만들기', command = btn_click)
btn_create.pack(side = "bottom", padx = "20" ,pady="20", fill= "x")

btn_load = Button(window, text='증명사진 불러오기', command = open)
btn_load.pack(side="bottom", padx = "20" ,pady="20", fill= "x")

window.mainloop()
