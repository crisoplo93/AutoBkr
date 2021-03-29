import cv2
import numpy as np
import pytesseract
import time
from pynput.keyboard import Key, Controller
import win32gui, win32ui, win32con, win32api
import math
from random import seed
from random import randint
from alexnet import alexnet
import os
from tkinter import *

root = Tk()
root.geometry("350x470+0+0")
fontSize = 14

MyLabel = Label(root, text = "Balance")
MyLabel.config(font=("Courier", fontSize))
MyLabel2 = Label(root, text = "Total", justify='left')
MyLabel2.config(font=("Courier", fontSize))
MyLabel3 = Label(root, text = "Successful", justify='left')
MyLabel3.config(font=("Courier", fontSize))
MyLabel4 = Label(root, text = "Success Rate", justify='left')
MyLabel4.config(font=("Courier", fontSize))
MyLabel5 = Label(root, text = "Profit", justify='left')
MyLabel5.config(font=("Courier", fontSize))
MyLabel.pack()
MyLabel2.pack()
MyLabel3.pack()
MyLabel4.pack()
MyLabel5.pack()

WIDTH = 110
HEIGHT = 58
LR = 1e-3
EPOCHS = 10
MODEL_NAME = 'autobrk-{}-{}-{}-epochs-3K-data.model'.format(LR, 'alexnetv2', EPOCHS)

model = alexnet(WIDTH, HEIGHT, LR)
model.load(MODEL_NAME)

np_load_old = np.load

trade_count = 0
succesful = 0

up = [1, 0]

down = [0, 1]

np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)

seed(1)

contador = 0


starting_balance = 0

profit = 0

wait_time = 1*60+3

n_igual = 1

high = 0.995

low = 0.01

inspecting_time = 15

def grab_screen(region=None):
    hwin = win32gui.GetDesktopWindow()

    if region:
        left, top, x2, y2 = region
        width = x2 - left + 1
        height = y2 - top + 1
    else:
        width = win32api.GetSystemMetrics(win32con.SM_CXVIRTUALSCREEN)
        height = win32api.GetSystemMetrics(win32con.SM_CYVIRTUALSCREEN)
        left = win32api.GetSystemMetrics(win32con.SM_XVIRTUALSCREEN)
        top = win32api.GetSystemMetrics(win32con.SM_YVIRTUALSCREEN)

    hwindc = win32gui.GetWindowDC(hwin)
    srcdc = win32ui.CreateDCFromHandle(hwindc)
    memdc = srcdc.CreateCompatibleDC()
    bmp = win32ui.CreateBitmap()
    bmp.CreateCompatibleBitmap(srcdc, width, height)
    memdc.SelectObject(bmp)
    memdc.BitBlt((0, 0), (width, height), srcdc, (left, top), win32con.SRCCOPY)

    signedIntsArray = bmp.GetBitmapBits(True)
    img = np.frombuffer(signedIntsArray, dtype='uint8')
    img.shape = (height, width, 4)

    srcdc.DeleteDC()
    memdc.DeleteDC()
    win32gui.ReleaseDC(hwin, hwindc)
    win32gui.DeleteObject(bmp.GetHandle())

    return img

def balanceToText(img_balance: None):

        text = pytesseract.image_to_string(img_balance)
        text = text.strip()
        text = text[1:]
        text = text.replace(' ', '')
        text = text.replace(',','')
        text = text.replace('.', '')
        # belements = text.split(',')
        # if len(belements) > 1:
        #     text = belements[0].strip() + belements[1].strip()
        #     text = float(text)
        # else:
        text = float(text)
        return text

def pressUD(number: None):
    if number == 0:
        keyboard.press('w')
        keyboard.release('w')
        output = [1, 0]
        return output
    else:
        keyboard.press('s')
        keyboard.release('s')
        output = [0, 1]
        return output

def refresh():
    keyboard.press(Key.f5)
    keyboard.press(Key.f5)
    time.sleep(5);


pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
keyboard = Controller()


print("Start")
while(True):
    c_up = 0

    c_down = 0

    up_average = 0

    down_average = 0

    img2 = grab_screen(region=(700, 280, 855, 295))
    # cv2.namedWindow("window", cv2.WINDOW_NORMAL)  # Create window with freedom of dimensions
    # imS = cv2.resize(img, (800, 600))  # Resize image
    # cv2.imshow("window", imS)

    #cv2.imshow('window1', cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    #img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)

    #cv2.imshow('window', img)
    #cv2.imshow('window2', img2)
    text = pytesseract.image_to_string(img2)

    #print(balance_text)
    #print(text)
    balance = grab_screen(region=(1500, 167, 1620, 195))

    #cv2.imshow('window1', cv2.cvtColor(balance, cv2.COLOR_BGR2GRAY))
    #cv2.imshow('window2', cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    balance_text = balanceToText(balance)
    MyLabel.config(text = str(balance_text))
    MyLabel.pack()
    root.update_idletasks()
    root.update()
    if starting_balance == 0:
        starting_balance = balance_text
    img = grab_screen(region=(500, 320, 1605, 900))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img_save = img
    img = cv2.resize(img, (110, 58))

    prediction = model.predict([img.reshape(WIDTH,HEIGHT,1)])[0]

    t_end = time.time() + inspecting_time
    print(c_up, c_down)
    # while time.time() < t_end:
    #     img = grab_screen(region=(500, 320, 1605, 900))
    #     img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    #     img_save = img
    #     img = cv2.resize(img, (110, 58))
    #
    #     prediction = model.predict([img.reshape(WIDTH,HEIGHT,1)])[0]
    #     #print('precdiction :', prediction)
    #     if prediction[0] > prediction[1] and prediction[0] > high:
    #         c_up += 1
    #         up_average = up_average + prediction[0]
    #         #c_down = 0
    #     elif prediction[1] > prediction[0] and prediction[1] > high:
    #         c_down += 1
    #         down_average = down_average + prediction[1]
    #         #c_up = 0
    #
    # print(c_up, c_down)
    #
    # if c_up > 0:
    #     up_average = up_average/c_up
    # else:
    #     up_average = 0
    #
    # if c_down>0:
    #     down_average = down_average/c_down
    # else:
    #     down_average = 0
    # if c_up > c_down and up_average >= high and c_up > (c_up + c_down)/2:
    #     moves = [1,0]
    # elif c_down > c_up and down_average >= high and c_down > (c_up + c_down)/2:
    #     moves = [0,1]
    # else:
    #     moves = [0,0]

    if prediction[0] >= high and prediction[1] <= low:
        moves = [1,0]
    elif prediction[1] >= high and prediction[1] <= low:
        moves = [0,1]
    else:
        moves = [0,0]

    if moves == [1, 0]:
        pressUD(0)
        c_up = 0
        contador += 1
        trade_count += 1
        time.sleep(wait_time)
        saldo = grab_screen(region=(1500, 167, 1620, 195))
        balance_after = balanceToText(saldo)
        if balance_after > balance_text:
            succesful += 1
            succes_rate = (succesful*100)/trade_count
            #print('up', prediction, succes_rate)
            cv2.imwrite('scfl_trades/up'+str(succesful)+'.jpg', img_save)
            profit = balance_after - starting_balance
            profit = profit/100
            info = 'Total : ' + str(trade_count)
            info1 = ' Successful : ' + str(succesful)
            info2 = ' Rate: ' + str(math.trunc(succes_rate)) + '%'
            info3 = ' Profit: ' + str(round(profit,2))
            MyLabel2.config(text=str(info))
            MyLabel2.pack()
            MyLabel3.config(text=str(info1))
            MyLabel3.pack()
            MyLabel4.config(text=str(info2))
            MyLabel4.pack()
            MyLabel5.config(text=str(info3))
            MyLabel5.pack()
        else:
            profit = balance_after - starting_balance
            profit = profit / 100
            succes_rate = (succesful * 100) / trade_count
            info = 'Total : ' + str(trade_count)
            info1 = ' Successful : ' + str(succesful)
            info2 = ' Rate: ' + str(math.trunc(succes_rate)) + '%'
            info3 = ' Profit: ' + str(round(profit,2))
            MyLabel2.config(text=str(info))
            MyLabel2.pack()
            MyLabel3.config(text=str(info1))
            MyLabel3.pack()
            MyLabel4.config(text=str(info2))
            MyLabel4.pack()
            MyLabel5.config(text=str(info3))
            MyLabel5.pack()
    elif moves == [0,1]:
        pressUD(1)
        c_down = 0
        contador += 1
        trade_count += 1
        time.sleep(wait_time)
        saldo = grab_screen(region=(1500, 167, 1620, 195))
        balance_after = balanceToText(saldo)
        if balance_after > balance_text:
            succesful += 1
            succes_rate = (succesful * 100) / trade_count
            #print('down', prediction, succes_rate)
            cv2.imwrite('scfl_trades/down'+str(succesful)+'.jpg', img_save)
            profit = balance_after - starting_balance
            profit = profit / 100
            info = 'Total : ' + str(trade_count)
            info1 = ' Successful : ' + str(succesful)
            info2 = ' Rate: ' + str(math.trunc(succes_rate)) + '%'
            info3 = ' Profit: ' + str(round(profit,2))
            MyLabel2.config(text=str(info))
            MyLabel2.pack()
            MyLabel3.config(text=str(info1))
            MyLabel3.pack()
            MyLabel4.config(text=str(info2))
            MyLabel4.pack()
            MyLabel5.config(text=str(info3))
            MyLabel5.pack()
        else:
            profit = balance_after - starting_balance
            profit = profit / 100
            succes_rate = (succesful * 100) / trade_count
            info = 'Total : ' + str(trade_count)
            info1 = ' Successful : ' + str(succesful)
            info2 = ' Rate: ' + str(math.trunc(succes_rate)) + '%'
            info3 = ' Profit: ' + str(round(profit,2))
            MyLabel2.config(text=str(info))
            MyLabel2.pack()
            MyLabel3.config(text=str(info1))
            MyLabel3.pack()
            MyLabel4.config(text=str(info2))
            MyLabel4.pack()
            MyLabel5.config(text=str(info3))
            MyLabel5.pack()
    else:
        print('unable to decide', prediction)
        time.sleep(5)

    if contador == 10:
        contador =0
        refresh()

    root.update_idletasks()
    root.update()

    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        np.load = np_load_old
        break

    # if profit > 1.0:
    #     break