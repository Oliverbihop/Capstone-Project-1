import serial
import time
num_array=[]
arduino = serial.Serial(port='COM11', baudrate=115200, timeout=.1)
ROS_DATA = [0,0]
def write_read(x):
    arduino.write(bytes(x, 'utf-8'))
    time.sleep(0.05)
    data = arduino.readline()
    return data

# def Tach_du_lieu(goc_chay):
#     gia_tri_dau = 0
#     gia_tri_goc = 0
#     if(goc_chay > 0):
#          gia_tri_dau = 1
#          gia_tri_goc = abs(goc_chay)
#     else:
#          gia_tri_dau = 0
#          gia_tri_goc = abs(goc_chay)
#     return gia_tri_dau, gia_tri_goc
while 1:
    num = input("Enter a number: ") # Taking input from user
    #bit_1, bit_2 = Tach_du_lieu(int(num))

    value = write_read(num)
    print(value) # printing the value