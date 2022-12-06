import numpy as np
# from numpy.fft import fft2, ifft2, fftshift
from scipy.signal import convolve2d, correlate2d
import matplotlib.pyplot as plt

from Functions import *
from gaussfft import gaussfft


# Either write your code in a file like this or use a Jupyter notebook.
#
# A good idea is to use switches, so that you can turn things on and off
# depending on what you are working on. It should be fairly easy for a TA
# to go through all parts of your code though.

def deltax(operator):
    if operator == 'sdo':
        dxmask = np.mat([-1, 0, 1])
    if operator == 'cdo':
        dxmask = np.mat([-0.5, 0, 0.5])
    if operator == 'robert':
        dxmask = np.mat([[1, 0], [0, -1]])
    if operator == 'sobel':
        dxmask = np.mat([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    return dxmask


def deltay(operator):
    if operator == 'sdo':
        dymask = np.transpose(np.mat([-1, 0, 1]))
    if operator == 'cdo':
        dymask = np.transpose(np.mat([-0.5, 0, 0.5]))
    if operator == 'robert':
        dymask = np.mat([[0, 1], [-1, 0]])
    if operator == 'sobel':
        dymask = np.mat([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    return dymask


def Lv(inpic, dxmask, dymask, shape='same'):
    Lx = convolve2d(inpic, dxmask, shape)
    Ly = convolve2d(inpic, dymask, shape)
    return np.sqrt(Lx ** 2 + Ly ** 2)


def Lvvtilde(inpic, shape='same'):
    dxmask = np.mat([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0.5, 0, -0.5, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]])
    dxxmask = np.mat([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 1, -2, 1, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]])
    dymask, dyymask = dxmask.T, dxxmask.T
    dxymask = convolve2d(dxmask, dymask, shape)
    Lx = convolve2d(inpic, dxmask, shape)
    Ly = convolve2d(inpic, dymask, shape)
    Lxx = convolve2d(inpic, dxxmask, shape)
    Lyy = convolve2d(inpic, dyymask, shape)
    Lxy = convolve2d(inpic, dxymask, shape)
    return np.power(Lx, 2) * Lxx + 2 * Lx * Ly * Lxy + np.power(Ly, 2) * Lyy


def Lvvvtilde(inpic, shape='same'):
    dxmask = np.mat([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0.5, 0, -0.5, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]])
    dxxmask = np.mat([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 1, -2, 1, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]])
    dymask, dyymask = dxmask.T, dxxmask.T

    dxxxmask = convolve2d(dxmask, dxxmask, shape)
    dyyymask = convolve2d(dymask, dyymask, shape)

    dxxymask = convolve2d(dxxmask, dymask, shape)
    dxyymask = convolve2d(dyymask, dxmask, shape)

    Lx = convolve2d(inpic, dxmask, shape)
    Ly = convolve2d(inpic, dymask, shape)
    Lxxx = convolve2d(inpic, dxxxmask, shape)
    Lyyy = convolve2d(inpic, dyyymask, shape)
    Lxxy = convolve2d(inpic, dxxymask, shape)
    Lxyy = convolve2d(inpic, dxyymask, shape)
    return np.power(Lx, 3) * Lxxx + 3 * np.power(Lx, 2) * Ly * Lxxy + 3 * Lx * np.power(Ly, 2) * Lxyy + np.power(Ly,
                                                                                                                 3) * Lyyy


def extractedge(inpic, scale, threshold, shape='same'):
    smoothpic, _, _, _ = discgaussfft(inpic, scale)
    lv = Lv(smoothpic, deltax('robert'), deltay('robert'), shape)
    zeropic = Lvvtilde(smoothpic, shape)
    extpic = Lvvvtilde(smoothpic, shape)
    maskpic1 = extpic < 0
    maskpic2 = lv > threshold

    curves = zerocrosscurves(zeropic, maskpic1)
    edgecurves = thresholdcurves(curves, maskpic2)
    return edgecurves


def houghline(pic, curves, magnitude, nrho, ntheta, threshold, nlines=20, verbose=False):
    acc = np.zeros((nrho, ntheta))
    thetas = np.linspace(-np.pi / 2, np.pi / 2, ntheta)
    d = np.sqrt(magnitude.shape[0] ** 2 + magnitude.shape[1] ** 2)
    rhos = np.linspace(-d, d, nrho)

    for i in range(len(curves[0])):
        x, y = curves[0][i], curves[1][i]
        if magnitude[x, y] < threshold:
            continue
        for j, theta in enumerate(thetas):
            rho = x * np.cos(theta) + y * np.sin(theta)
            rho_idx = np.argmin(abs(rhos - rho))
            acc[rho_idx][j] += 1
            # -----Question 10-----
            # acc[rho_idx][j] += magnitude[x, y]
            # acc[rho_idx][j] += np.log(magnitude[x, y])

    pos, value, _ = locmax8(acc)
    indexvector = np.argsort(value)[-nlines:]
    pos = pos[indexvector]
    linepar = []

    f = plt.figure(figsize=(8, 8), dpi=200)
    f.subplots_adjust(wspace=0.2, hspace=0.4)
    plt.rc('axes', titlesize=10)
    a = f.add_subplot(1, 2, 1)
    showgrey(pic, False)
    a.title.set_text("original")
    a = f.add_subplot(1, 2, 2)
    showgrey(pic, False)

    for idx in range(nlines):
        thetaidxacc = pos[idx, 0]
        rhoidxacc = pos[idx, 1]
        rhoLM = rhos[rhoidxacc]
        thetaLM = thetas[thetaidxacc]
        linepar.append([rhoLM, thetaLM])

        x0 = rhoLM * np.cos(thetaLM)
        y0 = rhoLM * np.sin(thetaLM)
        dx = d * (-np.sin(thetaLM))
        dy = d * np.cos(thetaLM)
        plt.plot([y0 - dy, y0, y0 + dy], [x0 - dx, x0, x0 + dx], 'r-')
    a.title.set_text("curves")

    g = plt.figure(figsize=(8, 8), dpi=200)
    g.subplots_adjust(wspace=0.2, hspace=0.4)
    plt.rc('axes', titlesize=10)
    b = g.add_subplot(1, 2, 1)
    showgrey(acc, False)
    b.title.set_text("acc")

    b = g.add_subplot(1, 2, 2)
    overlaycurves(pic, curves)
    b.title.set_text(f'threshold={threshold} nrho={nrho} ntheta={ntheta} nlines={nlines}')
    plt.show()

    return linepar, acc


def houghedgeline(pic, scale, gradmagnthreshold, nrho, ntheta, nlines=20, verbose=False):
    edgecurves = extractedge(pic, scale, gradmagnthreshold)
    magnitude = Lv(pic, deltax('robert'), deltay('robert'))
    linepar, acc = houghline(pic, edgecurves, magnitude, nrho, ntheta, gradmagnthreshold, nlines)
    return linepar, acc


def different_operator():
    tools = np.load("Images-npy/few256.npy")
    operator = ['sdo', 'cdo', 'robert', 'sobel']

    f = plt.figure(1)
    f.subplots_adjust(wspace=0.2, hspace=0.2)
    f.suptitle('Different Operator')

    for i, ope in enumerate(operator):
        dxtools = convolve2d(tools, deltax(operator=ope), 'valid')
        dytools = convolve2d(tools, deltay(operator=ope), 'valid')
        print(f'tools shape of {ope}: {tools.shape}')
        print(f'dxtools shape of {ope}: {dxtools.shape}')

        a2 = f.add_subplot(4, 2, 2 * i + 1)
        showgrey(dxtools, False)
        a2.set_title(f'dx of {ope}')

        a3 = f.add_subplot(4, 2, 2 * i + 2)
        showgrey(dytools, False)
        a3.set_title(f'dy of {ope}')

    plt.show()


def thresholding():
    tools = np.load("Images-npy/godthem256.npy")
    gradmagntools = Lv(tools, deltax('robert'), deltay('robert'))
    counts, bins = np.histogram(gradmagntools, 50)
    plt.figure(1)
    plt.hist(bins[:-1], len(bins), weights=counts, edgecolor='black')
    plt.title('Histogram of gradient magnitude tools')

    threshold = [10, 20, 30, 40, 50, 60, 70, 80]
    f = plt.figure(2)
    for i, th in enumerate(threshold):
        a = f.add_subplot(1, len(threshold), i + 1)
        showgrey((gradmagntools > th).astype(int), False)
        a.set_title(f't = {th}')

    sigma = [0, 0.1, 0.3, 0.5, 1, 2]
    g = plt.figure(3)
    for i, s in enumerate(sigma):
        a = g.add_subplot(1, len(sigma), i + 1)
        smoothimg, _, _, _ = discgaussfft(tools, s)
        gradmagntools2 = Lv(smoothimg, deltax('robert'), deltay('robert'))
        showgrey((gradmagntools2 > 30).astype(int), False)
        a.set_title(f'sigma = {s}')
    plt.show()


def geometry():
    house = np.load("Images-npy/godthem256.npy")
    scale = [0.0001, 1.0, 4.0, 16.0, 64.0]
    f = plt.figure(1)
    f.suptitle('Lvvtilde')

    for i, s in enumerate(scale):
        a = f.add_subplot(1, len(scale), i + 1)
        smoothimg, _, _, _ = discgaussfft(house, s)
        showgrey(contour(Lvvtilde(smoothimg, 'same')), False)
        a.set_title(f'scale = {s}')

    tools = np.load("Images-npy/few256.npy")
    g = plt.figure(2)
    g.suptitle('Lvvvtilde')
    for i, s in enumerate(scale):
        a = g.add_subplot(1, len(scale), i + 1)
        smoothimg, _, _, _ = discgaussfft(tools, s)
        showgrey((Lvvvtilde(smoothimg, 'same') < 0).astype(int), False)
        a.set_title(f'scale = {s}')
    plt.show()


def extraction():
    house = np.load("Images-npy/godthem256.npy")
    tools = np.load("Images-npy/few256.npy")
    scale = [1.0, 4.0, 8.0, 16.0]
    threshold = [4, 6, 8, 10]

    for s in scale:
        for t in threshold:
            edgecurves = extractedge(house, s, t)
            overlaycurves(house, edgecurves)
            plt.title(f'scale = {s}, threshold = {t}')
            plt.show()

    for s in scale:
        for t in threshold:
            edgecurves = extractedge(tools, s, t)
            overlaycurves(tools, edgecurves)
            plt.title(f'scale = {s}, threshold = {t}')
            plt.show()


def main():
    # different_operator()
    # thresholding()
    # geometry()
    # extraction()
    testimage1 = np.load("Images-npy/triangle128.npy")
    # smalltest1 = binsubsample(testimage1)
    testimage2 = np.load("Images-npy/houghtest256.npy")
    # smalltest2 = binsubsample(binsubsample(testimage2))

    # linepar, acc = houghedgeline(testimage1, 4, 10, 400, 300, 3)
    # linepar, acc = houghedgeline(testimage2, 4, 25, 175, 120, 10)

    tools = np.load("Images-npy/few256.npy")
    phone = np.load("Images-npy/phonecalc256.npy")
    house = np.load("Images-npy/godthem256.npy")
    # linepar, acc = houghedgeline(tools, 4, 10, 512, 64, 15)
    # linepar, acc = houghedgeline(phone, 3, 30, 1200, 40, 15)
    linepar, acc = houghedgeline(house, 3, 5, 800, 100, 15)


if __name__ == '__main__':
    main()
