import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft2, ifft2, fftshift
from scipy.signal import convolve2d

from Functions import *
from gaussfft import gaussfft
from fftwave import fftwave


# Either write your code in a file like this or use a Jupyter notebook.
#
# A good idea is to use switches, so that you can turn things on and off
# depending on what you are working on. It should be fairly easy for a TA
# to go through all parts of your code though.

# Exercise 1.3
def basic_functions():
    '''
    p, q = 5, 9
    Fhat = np.zeros((128, 128))
    Fhat[p, q] = 1
    showgrey(Fhat)

    F = ifft2(Fhat)
    Fabsmax = np.max(np.abs(F))
    showgrey(np.real(F), True, 64, -Fabsmax, Fabsmax)
    showgrey(np.imag(F), True, 64, -Fabsmax, Fabsmax)
    showgrey(np.abs(F), True, 64, -Fabsmax, Fabsmax)
    showgrey(np.angle(F), True, 64, -np.pi, np.pi)
    '''
    coordinates = [(5, 9), (9, 5), (17, 9), (17, 121), (5, 1), (125, 1)]
    for idx, c in enumerate(coordinates):
        p, q = c
        fftwave(p, q, idx + 1, is_pltshow=False)
    plt.show()


# Exercise 1.4
def linearity():
    F = np.concatenate([np.zeros((56, 128)), np.ones((16, 128)), np.zeros((56, 128))])
    G = F.T
    H = F + 2 * G

    Fhat = fft2(F)
    Ghat = fft2(G)
    Hhat = fft2(H)

    f = plt.figure(1)
    f.subplots_adjust(wspace=0.2, hspace=0.4)
    plt.rc('axes', titlesize=10)
    a1 = f.add_subplot(3, 3, 1)
    showgrey(F, False)
    a1.title.set_text("F")

    b1 = f.add_subplot(3, 3, 2)
    showgrey(np.log(1 + np.abs(Fhat)), False)
    b1.title.set_text("log(1+abs(Fhat))")

    c1 = f.add_subplot(3, 3, 3)
    showgrey(np.log(1 + np.abs(fftshift(Fhat))), False)
    c1.title.set_text("log(1+abs(fftshift(Fhat)))")

    a2 = f.add_subplot(3, 3, 4)
    showgrey(G, False)
    a2.title.set_text("G=F.T")

    b2 = f.add_subplot(3, 3, 5)
    showgrey(np.log(1 + np.abs(Ghat)), False)
    b2.title.set_text("log(1+abs(Ghat))")

    c2 = f.add_subplot(3, 3, 6)
    showgrey(np.log(1 + np.abs(fftshift(Ghat))), False)
    c2.title.set_text("log(1+abs(fftshift(Ghat)))")

    a3 = f.add_subplot(3, 3, 7)
    showgrey(H, False)
    a3.title.set_text("H=F+2G")

    b3 = f.add_subplot(3, 3, 8)
    showgrey(np.log(1 + np.abs(Hhat)), False)
    b3.title.set_text("log(1+abs(Hhat))")

    c3 = f.add_subplot(3, 3, 9)
    showgrey(np.log(1 + np.abs(fftshift(Hhat))), False)
    c3.title.set_text("log(1+abs(fftshift(Hhat)))")

    plt.show()


# Exercise 1.5
def multiplication():
    F = np.concatenate([np.zeros((56, 128)), np.ones((16, 128)), np.zeros((56, 128))])
    G = F.T

    Fhat = fft2(F)
    Ghat = fft2(G)

    f = plt.figure(1)
    f.subplots_adjust(wspace=0.2, hspace=0.4)
    plt.rc('axes', titlesize=10)

    a1 = f.add_subplot(1, 3, 1)
    showgrey(F * G, False)
    a1.title.set_text("F*G")

    a2 = f.add_subplot(1, 3, 2)
    showfs(fft2(F * G), False)
    a2.title.set_text("fft(F*G)")

    a3 = f.add_subplot(1, 3, 3)
    showgrey(np.log(1 + np.abs(convolve2d(Fhat, Ghat, mode='same', boundary='wrap') / (128 ** 2))), False)
    a3.title.set_text("conv(Fhat, Ghat)")

    plt.show()


# Exercise 1.6
def scaling():
    F = np.concatenate([np.zeros((60, 128)), np.ones((8, 128)), np.zeros((60, 128))]) * \
        np.concatenate([np.zeros((128, 48)), np.ones((128, 32)), np.zeros((128, 48))], axis=1)
    Fhat = fft2(F)

    f = plt.figure(1)
    f.subplots_adjust(wspace=0.2, hspace=0.4)
    plt.rc('axes', titlesize=10)

    a1 = f.add_subplot(1, 2, 1)
    showgrey(F, False)
    a1.title.set_text("F")

    a2 = f.add_subplot(1, 2, 2)
    showfs(Fhat, False)
    a2.title.set_text("Fhat")

    plt.show()


# Exercise 1.7
def rotation():
    alphas = [0, 30, 45, 60, 90]

    F = np.concatenate([np.zeros((60, 128)), np.ones((8, 128)), np.zeros((60, 128))]) * \
        np.concatenate([np.zeros((128, 48)), np.ones((128, 32)), np.zeros((128, 48))], axis=1)

    f = plt.figure(1)
    f.subplots_adjust(wspace=0.2, hspace=0.4)
    plt.rc('axes', titlesize=10)

    for i, alpha in enumerate(alphas):
        G = rot(F, alpha)
        Ghat = fft2(G)
        Hhat = rot(fftshift(Ghat), -alpha)

        a1 = f.add_subplot(5, 3, 1 + 3 * i)
        showgrey(G, False)
        a1.title.set_text("G for alpha={}".format(alpha))

        a2 = f.add_subplot(5, 3, 2 + 3 * i)
        showfs(Ghat, False)
        a2.title.set_text("Ghat for alpha={}".format(alpha))

        a3 = f.add_subplot(5, 3, 3 + 3 * i)
        showgrey(np.log(1 + np.abs(Hhat)), False)
        a3.title.set_text("Hhat for alpha={}".format(alpha))

    plt.show()


# Exercise 1.8
def phase_magnitude():
    img1 = np.load("Images-npy/phonecalc128.npy")
    img2 = np.load("Images-npy/few128.npy")
    img3 = np.load("Images-npy/nallo128.npy")

    f = plt.figure(1)
    f.subplots_adjust(wspace=0.2, hspace=0.4)
    plt.rc('axes', titlesize=10)

    images = [img1, img2, img3]
    for i, img in enumerate(images):
        pow = pow2image(img)
        randphase = randphaseimage(img)

        a1 = f.add_subplot(3, 3, 1 + 3 * i)
        showgrey(img, display=False)
        a1.title.set_text('Original Image')

        a2 = f.add_subplot(3, 3, 2 + 3 * i)
        showgrey(pow, display=False)
        a2.title.set_text('Pow2Image')

        a3 = f.add_subplot(3, 3, 3 + 3 * i)
        showgrey(randphase, display=False)
        a3.title.set_text('RandPhase Image')

    plt.show()


def gauss_test():
    ts = [0.1, 0.3, 1.0, 10.0, 100.0]
    tss = [1.0, 4.0, 16.0, 64.0, 256.0]
    img = np.load("Images-npy/phonecalc128.npy")

    f = plt.figure(1)
    f.subplots_adjust(wspace=0.2, hspace=0.4)
    plt.rc('axes', titlesize=10)

    for i, t in enumerate(ts):
        psf, X, Y, gauss = gaussfft(deltafcn(128, 128), t)
        # psf, X, Y, gauss = discgaussfft(deltafcn(128, 128), t)
        var = variance(psf)
        var = [[round(j, 3) for j in var[i]] for i in range(len(var))]

        a = f.add_subplot(2, 5, 1 + i)
        showfs(fftshift(psf), False)
        a.title.set_text('t={}\nvar={}'.format(t, var))
        a.set_axis_off()

        b = f.add_subplot(2, 5, 6 + i, projection='3d')
        b.plot_surface(X, Y, gauss, rstride=1, cstride=1)
        b.set_axis_off()

    g = plt.figure(2)
    g.subplots_adjust(wspace=0.2, hspace=0.4)
    plt.rc('axes', titlesize=10)

    for i, t in enumerate(tss):
        psf, _, _, _ = gaussfft(img, t)

        a = g.add_subplot(1, 5, i + 1)
        showfs(fftshift(psf), False)
        a.title.set_text('t={}'.format(t))
        a.set_axis_off()

    plt.show()


def smoothing():
    office = np.load("Images-npy/office256.npy")

    # image
    f1 = plt.figure(1)
    f1.subplots_adjust(wspace=0.2, hspace=0.4)
    plt.rc('axes', titlesize=10)

    a1 = f1.add_subplot(1, 3, 1)
    showgrey(office, False)
    a1.title.set_text('Original')

    add = gaussnoise(office, 16)
    sap = sapnoise(office, 0.1, 255)

    a2 = f1.add_subplot(1, 3, 2)
    showgrey(add, False)
    a2.title.set_text('AWGN')

    a3 = f1.add_subplot(1, 3, 3)
    showgrey(sap, False)
    a3.title.set_text('SAPN')

    # gaussfft
    f2 = plt.figure(2)
    f2.subplots_adjust(wspace=0.2, hspace=0.4)
    plt.rc('axes', titlesize=10)
    f2.suptitle('Gaussfft-AWGN')
    ts = [0.1, 0.3, 0.5, 1.0, 2.0, 10.0]
    for i, t in enumerate(ts):
        psf, _, _, _ = gaussfft(add, t)
        a = f2.add_subplot(2, 3, i + 1)
        showfs(fftshift(psf), False)
        a.set_title('t={}'.format(t))

    f3 = plt.figure(3)
    f3.subplots_adjust(wspace=0.2, hspace=0.4)
    plt.rc('axes', titlesize=10)
    f3.suptitle('Gaussfft-SAPN')
    for i, t in enumerate(ts):
        psf, _, _, _ = gaussfft(sap, t)
        a = f3.add_subplot(2, 3, i + 1)
        showfs(fftshift(psf), False)
        a.set_title('t={}'.format(t))

    # median filter
    f4 = plt.figure(4)
    f4.subplots_adjust(wspace=0.2, hspace=0.4)
    plt.rc('axes', titlesize=10)
    f4.suptitle('median-AWGN')
    ws = [1, 3, 5, 7, 9, 11]
    for i, w in enumerate(ws):
        psf = medfilt(add, w)
        a = f4.add_subplot(2, 3, i + 1)
        showfs(fftshift(psf), False)
        a.set_title('window size={}'.format(w))

    f5 = plt.figure(5)
    f5.subplots_adjust(wspace=0.2, hspace=0.4)
    plt.rc('axes', titlesize=10)
    f5.suptitle('median-SAPN')
    for i, w in enumerate(ws):
        psf = medfilt(sap, w)
        a = f5.add_subplot(2, 3, i + 1)
        showfs(fftshift(psf), False)
        a.set_title('window size={}'.format(w))

    # ideal low-pass
    f6 = plt.figure(6)
    f6.subplots_adjust(wspace=0.2, hspace=0.4)
    plt.rc('axes', titlesize=10)
    f6.suptitle('lowpass-AWGN')
    cs = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
    for i, c in enumerate(cs):
        psf = ideal(add, c)
        a = f6.add_subplot(2, 3, i + 1)
        showfs(fftshift(psf), False)
        a.set_title('cut-off={}'.format(c))

    f7 = plt.figure(7)
    f7.subplots_adjust(wspace=0.2, hspace=0.4)
    plt.rc('axes', titlesize=10)
    f7.suptitle('lowpass-SAPN')
    for i, c in enumerate(cs):
        psf = ideal(sap, c)
        a = f7.add_subplot(2, 3, i + 1)
        showfs(fftshift(psf), False)
        a.set_title('cut-off={}'.format(c))

    plt.show()


def smoothing_subsampling():
    img = np.load("Images-npy/phonecalc256.npy")
    smoothimg1 = img
    smoothimg2 = img
    N = 5
    f = plt.figure()
    f.subplots_adjust(wspace=0, hspace=0)
    f.suptitle('t=1, cut-off=0.2')
    for i in range(N):
        if i > 0:  # generate subsampled versions
            img = rawsubsample(img)
            smoothimg1, _, _, _ = gaussfft(smoothimg1, 1.0)  # <call_your_filter_here>(smoothimg, <params>)
            smoothimg1 = rawsubsample(smoothimg1)
            smoothimg2 = ideal(smoothimg2, 0.2)
            smoothimg2 = rawsubsample(smoothimg2)
        a = f.add_subplot(3, N, i + 1)
        showgrey(img, False)
        x, y = np.shape(img)
        a.set_title('{}×{}'.format(x, y))
        f.add_subplot(3, N, i + N + 1)
        showgrey(smoothimg1, False)
        f.add_subplot(3, N, i + 2*N + 1)
        showgrey(smoothimg2, False)

    plt.show()


def main():
    # part 1
    # basic_functions()
    # linearity()
    # multiplication()
    # scaling()
    # rotation()
    # phase_magnitude()

    # part 2
    # gauss_test()
    # smoothing()
    smoothing_subsampling()


if __name__ == '__main__':
    main()
