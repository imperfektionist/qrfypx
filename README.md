# QRfypx - Generate QR Codes from Images - README
February 3rd, 2025 - Anton Hoyer

## Summary

QRfypx, pronounced “qurify pix,” is a Python tool freely available to everyone on GitHub.
If you want to run it on Windows, you can simply download the release version.
In case of Mac or Linux, or if you want to take a closer look at the code, you can download
the Python script but may need to install additional modules:
`os, numpy, random, itertools, tkinter, PIL, qrcode, concurrent.futures, multiprocessing`

The tool accepts an input image and a URL, combines them, and lets you mess around with
different patterns and brightness settings to find the ideal balance between aesthetics
and machine interpretability. Output images are regenerated whenever the UI detects a value
change, running reasonably fast on a high-end CPU despite using Python. However, machine
interpretability of the output images depends very much on the camera device, and sometimes
stepping away from the code makes it easier to be read. If maximum compatibility and clear
communication are key, do not generate your QR codes with QRfypx. But if you want to append
them to your Instagram posts and take part in a bit of steganography, go ahead.

## Installation

You do not need to install the program. Simply run `QRfypx.exe` or the `QRfypx.py` script.

## Further Information

For more information and to understand the reasoning behind QRfypx, refer to the website:
[https://antonhoyer.com/qrfypx/].
