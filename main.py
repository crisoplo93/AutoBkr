import cv2
import numpy as np
import pytesseract
import time
from pynput.keyboard import Key, Controller
import win32gui, win32ui, win32con, win32api
from random import seed
from random import randint
import os

np_load_old = np.load

trade_count = 0
succesful = 0

up = [1, 0]

down = [0, 1]

np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)

seed(1)

file_name = 'training_data_OTAI.npy'

contador = 0

if os.path.isfile(file_name):
    print('File exists, loading previous data!')
    training_data = list(np.load(file_name))
    np.load = np_load_old
else:
    print('File does not exist, starting fresh!')
    training_data = []


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
        belements = text.split(',')
        if len(belements) > 1:
            text = belements[0] + belements[1]
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
    time.sleep(10);


pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
keyboard = Controller()

time.sleep(20)
while(True):

    #img2 = grab_screen(region=(700, 280, 855, 295))
    # cv2.namedWindow("window", cv2.WINDOW_NORMAL)  # Create window with freedom of dimensions
    # imS = cv2.resize(img, (800, 600))  # Resize image
    # cv2.imshow("window", imS)

    #cv2.imshow('window1', cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    #img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)

    #cv2.imshow('window', img)
    #scv2.imshow('window2', img2)
    #text = pytesseract.image_to_string(img2)

    #print(balance_text)
    #print(text)

    print("Start")
    contador += 1
    trade_count += 1
    img = grab_screen(region=(500, 320, 1605, 900))
    balance = grab_screen(region=(1500, 160, 1620, 188))
    #balance = grab_screen(region=(1370, 160, 1470, 188))
    #cv2.imshow('window', balance)
    balance_text = balanceToText(balance)
    print(balance_text)
    movimiento = pressUD(randint(0, 1))
    time.sleep(63)
    saldo = grab_screen(region=(1500, 161, 1620, 188))
    balance_after = balanceToText(saldo)
    #print(balance_after)
    if balance_after > balance_text:

        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = cv2.resize(img, (110, 58))
        training_data.append([img, movimiento])
        print(len(training_data))
        np.save(file_name, training_data)
        print("Profit " + str(balance_after) + "|" + str(balance_text))
        succesful += 1
        succes_rate = (succesful*100)/trade_count
        print(str(succes_rate))
    elif balance_after < balance_text:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = cv2.resize(img, (110, 58))
        if movimiento == up:
            training_data.append([img, down])
            print(len(training_data))
            np.save(file_name, training_data)
        elif movimiento == down:
            training_data.append([img, up])
            print(len(training_data))
            np.save(file_name, training_data)

    time.sleep(3)
    if contador == 10:
        refresh()
        contador = 0;




    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        np.load = np_load_old
        break
