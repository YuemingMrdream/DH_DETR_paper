import matplotlib.pyplot as plt
import numpy as np

def read_log(filename):
    fp=open(filename)
    AP = []
    for line in fp.readlines():
        train_AP= float(line[41 :47 ])   # 29: 35    41 :47    53 : 59     70 :76
        AP.append(train_AP)
        # train_acc=line[-8:]
    draw_curve(AP[:20],AP[20:40],AP[50:60],AP[-19:-10],AP[-10:])
    fp.close()



def draw_curve(data0,data1,data2,data3,data4):
    # y1=data0
    # x1=range(len(data0))
    # plt.plot(x1,y1,label='Fist line',linewidth=1,color='r',marker='o',
    #          markerfacecolor='red',markersize=3)
    # y2 = data1
    # x2 = range(len(data1))
    # plt.plot(x2,y2,label='Fist line',linewidth=1,color='b',marker='o',
    #          markerfacecolor='blue',markersize=3)
    y3 = data2
    x3 = range(len(data2))
    plt.plot(x3, y3, label='Fist line', linewidth=1, color='y', marker='o',
             markerfacecolor='yellow', markersize=3)
    y4 = data3
    x4 = range(len(data3))
    plt.plot(x4, y4, label='Fist line', linewidth=1, color='g', marker='o',
             markerfacecolor='green', markersize=3)
    y5 = data4
    x5 = range(len(data4))
    plt.plot(x5, y5, label='Fist line', linewidth=1, color='black', marker='o',
             markerfacecolor='black', markersize=3)
    plt.legend(['iter-DET','4-1','D-DETR'])
    # plt.legend(['2,2', '2.1','baseline','4.1','D-DETR'])
    plt.show()

def main():
    filename='record-final.txt'
    read_log(filename)

    print('Finish')


if __name__=='__main__':
    main()
