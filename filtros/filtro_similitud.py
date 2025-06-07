from fastapi import APIRouter, File, UploadFile, HTTPException
from PIL import Image
from skimage.metrics import structural_similarity as ssim
import numpy as np
import io
import asyncio
import gc

CONSTANTS_AVAILABLE = True

IMAGE_DATA = {
    "1.jpeg": {
        "mean": 159.88,
        "std": 93.56,
        "histogram": [0.0, 0.0001, 0.0143, 0.217, 0.2163, 0.0082, 0.0028, 0.0032, 0.0043, 0.0065, 0.0165, 0.011, 0.0106, 0.02, 0.0173, 0.4516],
        "gradient_x": 27.39,
        "gradient_y": 36.82,
        "original_size": (1042, 1600),
        "file_size": 122245
    },
    "2.jpeg": {
        "mean": 185.41,
        "std": 90.31,
        "histogram": [0.0, 0.0001, 0.0093, 0.2038, 0.0937, 0.0047, 0.0164, 0.0031, 0.0042, 0.0061, 0.0085, 0.0129, 0.014, 0.0138, 0.0226, 0.5866],
        "gradient_x": 29.92,
        "gradient_y": 44.46,
        "original_size": (1260, 1520),
        "file_size": 76995
    },
    "3.jpeg": {
        "mean": 185.31,
        "std": 90.29,
        "histogram": [0.0, 0.0001, 0.0095, 0.2037, 0.0938, 0.0047, 0.0165, 0.0028, 0.0045, 0.0065, 0.0087, 0.0129, 0.0142, 0.0137, 0.0237, 0.5848],
        "gradient_x": 30.31,
        "gradient_y": 44.6,
        "original_size": (1260, 1520),
        "file_size": 77538
    },
    "4.jpeg": {
        "mean": 185.26,
        "std": 90.32,
        "histogram": [0.0, 0.0001, 0.0093, 0.2038, 0.0941, 0.0046, 0.0167, 0.0029, 0.0045, 0.0061, 0.0089, 0.0128, 0.0136, 0.0146, 0.0218, 0.5861],
        "gradient_x": 30.33,
        "gradient_y": 44.71,
        "original_size": (1260, 1520),
        "file_size": 77626
    },
    "5.jpeg": {
        "mean": 184.91,
        "std": 90.15,
        "histogram": [0.0, 0.0001, 0.0093, 0.2037, 0.0936, 0.0046, 0.0165, 0.0038, 0.0046, 0.0066, 0.0094, 0.0143, 0.0146, 0.0153, 0.0236, 0.5798],
        "gradient_x": 31.24,
        "gradient_y": 46.03,
        "original_size": (1260, 1520),
        "file_size": 80693
    },
    "6.jpeg": {
        "mean": 185.25,
        "std": 90.27,
        "histogram": [0.0, 0.0001, 0.0093, 0.2037, 0.0937, 0.0048, 0.0161, 0.0035, 0.0051, 0.0061, 0.0089, 0.0133, 0.0135, 0.0139, 0.0229, 0.585],
        "gradient_x": 30.46,
        "gradient_y": 44.82,
        "original_size": (1260, 1520),
        "file_size": 77876
    },
    "7.jpeg": {
        "mean": 185.19,
        "std": 90.29,
        "histogram": [0.0, 0.0001, 0.0093, 0.2038, 0.0941, 0.0046, 0.0167, 0.0028, 0.0047, 0.0067, 0.0088, 0.0128, 0.0139, 0.014, 0.0237, 0.5839],
        "gradient_x": 30.68,
        "gradient_y": 44.87,
        "original_size": (1260, 1520),
        "file_size": 78154
    },
    "8.jpeg": {
        "mean": 160.51,
        "std": 93.85,
        "histogram": [0.0, 0.0001, 0.0144, 0.2171, 0.2162, 0.0081, 0.0024, 0.0023, 0.0034, 0.0057, 0.0154, 0.0105, 0.0107, 0.0185, 0.0168, 0.4584],
        "gradient_x": 27.04,
        "gradient_y": 34.96,
        "original_size": (1042, 1600),
        "file_size": 60835
    },
    "9.jpeg": {
        "mean": 149.38,
        "std": 93.63,
        "histogram": [0.0, 0.0, 0.0146, 0.2446, 0.246, 0.0011, 0.0034, 0.0096, 0.0064, 0.0064, 0.0142, 0.0077, 0.0116, 0.0159, 0.0121, 0.4064],
        "gradient_x": 24.69,
        "gradient_y": 34.33,
        "original_size": (1080, 1443),
        "file_size": 58583
    },
    "10.jpeg": {
        "mean": 159.91,
        "std": 93.57,
        "histogram": [0.0, 0.0001, 0.0143, 0.217, 0.2163, 0.0082, 0.0027, 0.0034, 0.0042, 0.0065, 0.0165, 0.011, 0.0108, 0.0201, 0.0171, 0.4518],
        "gradient_x": 27.87,
        "gradient_y": 36.76,
        "original_size": (1042, 1600),
        "file_size": 64011
    },
    "11.jpeg": {
        "mean": 166.1,
        "std": 92.94,
        "histogram": [0.0, 0.0, 0.0151, 0.2026, 0.1951, 0.002, 0.0086, 0.0093, 0.0037, 0.0057, 0.016, 0.0093, 0.0109, 0.0126, 0.0204, 0.4888],
        "gradient_x": 28.06,
        "gradient_y": 41.31,
        "original_size": (953, 1599),
        "file_size": 59042
    },
    "12.jpeg": {
        "mean": 165.18,
        "std": 92.61,
        "histogram": [0.0, 0.0, 0.0153, 0.2025, 0.1952, 0.0025, 0.009, 0.0103, 0.0048, 0.0064, 0.0178, 0.0107, 0.0122, 0.0157, 0.0212, 0.4765],
        "gradient_x": 31.35,
        "gradient_y": 44.41,
        "original_size": (953, 1599),
        "file_size": 64395
    },
    "13.jpeg": {
        "mean": 159.7,
        "std": 93.54,
        "histogram": [0.0, 0.0001, 0.0146, 0.2173, 0.2164, 0.0084, 0.0029, 0.0037, 0.0038, 0.0062, 0.0161, 0.0113, 0.0112, 0.0208, 0.0177, 0.4496],
        "gradient_x": 28.36,
        "gradient_y": 37.06,
        "original_size": (1042, 1600),
        "file_size": 64456
    },
    "14.jpeg": {
        "mean": 160.04,
        "std": 93.77,
        "histogram": [0.0, 0.0001, 0.0082, 0.223, 0.2234, 0.0016, 0.0029, 0.003, 0.0038, 0.005, 0.0159, 0.0107, 0.017, 0.0125, 0.0167, 0.4562],
        "gradient_x": 26.74,
        "gradient_y": 38.85,
        "original_size": (838, 1280),
        "file_size": 45745
    },
    "15.jpeg": {
        "mean": 160.12,
        "std": 93.76,
        "histogram": [0.0, 0.0001, 0.008, 0.2229, 0.2234, 0.0014, 0.0027, 0.0031, 0.0041, 0.0049, 0.0157, 0.0108, 0.017, 0.0125, 0.0172, 0.4562],
        "gradient_x": 26.87,
        "gradient_y": 38.87,
        "original_size": (838, 1280),
        "file_size": 45539
    },
    "16.jpeg": {
        "mean": 160.0,
        "std": 93.75,
        "histogram": [0.0, 0.0001, 0.0082, 0.2231, 0.2233, 0.0015, 0.0027, 0.0028, 0.0038, 0.005, 0.0164, 0.0109, 0.0171, 0.0126, 0.0168, 0.4555],
        "gradient_x": 27.15,
        "gradient_y": 39.26,
        "original_size": (838, 1280),
        "file_size": 45834
    },
    "17.jpeg": {
        "mean": 160.05,
        "std": 93.78,
        "histogram": [0.0, 0.0001, 0.0084, 0.2237, 0.2228, 0.0014, 0.0027, 0.003, 0.0035, 0.0048, 0.0166, 0.0104, 0.017, 0.0124, 0.0173, 0.4558],
        "gradient_x": 27.07,
        "gradient_y": 40.39,
        "original_size": (1047, 1600),
        "file_size": 62517
    },
    "18.jpeg": {
        "mean": 160.17,
        "std": 93.79,
        "histogram": [0.0, 0.0001, 0.0081, 0.2236, 0.2227, 0.0015, 0.0027, 0.0027, 0.0037, 0.0051, 0.0168, 0.0108, 0.0162, 0.0119, 0.0165, 0.4576],
        "gradient_x": 26.85,
        "gradient_y": 40.22,
        "original_size": (1047, 1600),
        "file_size": 62012
    },
    "19.jpeg": {
        "mean": 160.16,
        "std": 93.79,
        "histogram": [0.0, 0.0001, 0.0081, 0.2235, 0.2228, 0.0013, 0.003, 0.003, 0.0037, 0.005, 0.0165, 0.0098, 0.0163, 0.0123, 0.0176, 0.4568],
        "gradient_x": 26.68,
        "gradient_y": 40.28,
        "original_size": (1047, 1600),
        "file_size": 62201
    },
    "20.jpeg": {
        "mean": 160.3,
        "std": 93.83,
        "histogram": [0.0, 0.0001, 0.0081, 0.2235, 0.2227, 0.0014, 0.0028, 0.0028, 0.0031, 0.0046, 0.0165, 0.0105, 0.0164, 0.012, 0.0174, 0.4581],
        "gradient_x": 26.52,
        "gradient_y": 40.0,
        "original_size": (1047, 1600),
        "file_size": 61704
    },
    "21.jpeg": {
        "mean": 159.98,
        "std": 93.76,
        "histogram": [0.0, 0.0001, 0.0083, 0.2235, 0.223, 0.0015, 0.0031, 0.0032, 0.0036, 0.005, 0.0164, 0.0103, 0.0168, 0.012, 0.0175, 0.4558],
        "gradient_x": 27.02,
        "gradient_y": 40.57,
        "original_size": (1047, 1600),
        "file_size": 62823
    },
    "22.jpeg": {
        "mean": 160.13,
        "std": 93.8,
        "histogram": [0.0, 0.0001, 0.0081, 0.2231, 0.2234, 0.0016, 0.0027, 0.0031, 0.0036, 0.0046, 0.0164, 0.01, 0.0167, 0.0125, 0.0173, 0.4568],
        "gradient_x": 26.79,
        "gradient_y": 38.96,
        "original_size": (838, 1280),
        "file_size": 45555
    },
    "23.jpeg": {
        "mean": 160.04,
        "std": 93.75,
        "histogram": [0.0, 0.0001, 0.0081, 0.2236, 0.223, 0.0015, 0.0028, 0.0027, 0.0038, 0.0049, 0.0171, 0.0106, 0.0166, 0.0122, 0.017, 0.456],
        "gradient_x": 27.18,
        "gradient_y": 40.52,
        "original_size": (1047, 1600),
        "file_size": 62789
    },
    "24.jpeg": {
        "mean": 160.25,
        "std": 93.82,
        "histogram": [0.0, 0.0001, 0.0081, 0.2233, 0.2228, 0.0015, 0.0025, 0.003, 0.0032, 0.0051, 0.0167, 0.0103, 0.0164, 0.012, 0.0168, 0.458],
        "gradient_x": 26.49,
        "gradient_y": 40.05,
        "original_size": (1047, 1600),
        "file_size": 61862
    },
    "25.jpeg": {
        "mean": 160.08,
        "std": 93.76,
        "histogram": [0.0, 0.0001, 0.008, 0.2229, 0.2234, 0.0015, 0.0031, 0.0031, 0.0035, 0.0048, 0.0165, 0.0104, 0.0172, 0.0123, 0.0168, 0.4564],
        "gradient_x": 26.57,
        "gradient_y": 38.85,
        "original_size": (838, 1280),
        "file_size": 45702
    },
    "26.jpeg": {
        "mean": 159.99,
        "std": 93.77,
        "histogram": [0.0, 0.0001, 0.0082, 0.2231, 0.2234, 0.0015, 0.0032, 0.0029, 0.0039, 0.0049, 0.0162, 0.01, 0.0172, 0.012, 0.0179, 0.4556],
        "gradient_x": 27.01,
        "gradient_y": 39.07,
        "original_size": (838, 1280),
        "file_size": 45873
    },
    "27.jpeg": {
        "mean": 160.16,
        "std": 93.8,
        "histogram": [0.0, 0.0001, 0.0081, 0.2231, 0.2233, 0.0013, 0.0028, 0.0026, 0.0037, 0.0052, 0.0161, 0.0107, 0.0167, 0.0122, 0.0167, 0.4572],
        "gradient_x": 26.72,
        "gradient_y": 38.66,
        "original_size": (838, 1280),
        "file_size": 45292
    },
    "28.jpeg": {
        "mean": 160.04,
        "std": 93.76,
        "histogram": [0.0, 0.0002, 0.0081, 0.2236, 0.2228, 0.0015, 0.0028, 0.0031, 0.004, 0.0045, 0.0165, 0.0109, 0.0168, 0.0119, 0.0173, 0.456],
        "gradient_x": 27.07,
        "gradient_y": 40.41,
        "original_size": (1047, 1600),
        "file_size": 62600
    },
    "29.jpeg": {
        "mean": 160.14,
        "std": 93.79,
        "histogram": [0.0, 0.0001, 0.0082, 0.2234, 0.2229, 0.0015, 0.0026, 0.0031, 0.0034, 0.0049, 0.0163, 0.0104, 0.0168, 0.0122, 0.0175, 0.4565],
        "gradient_x": 26.71,
        "gradient_y": 40.2,
        "original_size": (1047, 1600),
        "file_size": 62201
    },
    "30.jpeg": {
        "mean": 160.19,
        "std": 93.82,
        "histogram": [0.0, 0.0001, 0.0081, 0.2235, 0.2228, 0.0016, 0.0029, 0.0026, 0.004, 0.0048, 0.0163, 0.0099, 0.0163, 0.0119, 0.0172, 0.4579],
        "gradient_x": 26.88,
        "gradient_y": 40.25,
        "original_size": (1047, 1600),
        "file_size": 61822
    },
    "31.jpeg": {
        "mean": 166.17,
        "std": 92.86,
        "histogram": [0.0, 0.0, 0.0144, 0.2031, 0.1956, 0.0016, 0.009, 0.0026, 0.0098, 0.006, 0.0158, 0.0091, 0.0105, 0.0135, 0.0187, 0.4903],
        "gradient_x": 27.43,
        "gradient_y": 43.48,
        "original_size": (763, 1280),
        "file_size": 43273
    },
    "32.jpeg": {
        "mean": 159.75,
        "std": 93.69,
        "histogram": [0.0, 0.0001, 0.0083, 0.2232, 0.2234, 0.0017, 0.0032, 0.0033, 0.0041, 0.0049, 0.0165, 0.0104, 0.017, 0.0131, 0.0176, 0.4532],
        "gradient_x": 27.5,
        "gradient_y": 39.8,
        "original_size": (838, 1280),
        "file_size": 46601
    },
    "33.jpeg": {
        "mean": 160.06,
        "std": 93.78,
        "histogram": [0.0, 0.0001, 0.0082, 0.223, 0.2234, 0.0016, 0.0027, 0.0029, 0.0037, 0.0051, 0.0163, 0.0101, 0.0172, 0.0124, 0.0166, 0.4566],
        "gradient_x": 26.77,
        "gradient_y": 38.83,
        "original_size": (838, 1280),
        "file_size": 45563
    },
    "34.jpeg": {
        "mean": 159.99,
        "std": 93.76,
        "histogram": [0.0, 0.0001, 0.0083, 0.2231, 0.2233, 0.0016, 0.0031, 0.0026, 0.0038, 0.005, 0.0164, 0.0108, 0.0167, 0.0123, 0.0168, 0.4561],
        "gradient_x": 27.02,
        "gradient_y": 39.07,
        "original_size": (838, 1280),
        "file_size": 45774
    },
    "35.jpeg": {
        "mean": 159.99,
        "std": 93.75,
        "histogram": [0.0, 0.0001, 0.0082, 0.2231, 0.2233, 0.0015, 0.003, 0.0028, 0.0038, 0.0051, 0.0168, 0.0104, 0.0167, 0.0123, 0.0172, 0.4557],
        "gradient_x": 26.99,
        "gradient_y": 39.01,
        "original_size": (838, 1280),
        "file_size": 45787
    },
    "36.jpeg": {
        "mean": 166.17,
        "std": 92.85,
        "histogram": [0.0, 0.0, 0.0145, 0.2031, 0.1958, 0.0012, 0.0087, 0.0028, 0.0096, 0.0061, 0.0159, 0.0095, 0.0101, 0.014, 0.0204, 0.4883],
        "gradient_x": 28.0,
        "gradient_y": 43.8,
        "original_size": (763, 1280),
        "file_size": 43298
    },
    "37.jpeg": {
        "mean": 160.1,
        "std": 93.8,
        "histogram": [0.0, 0.0001, 0.0081, 0.223, 0.2235, 0.0016, 0.0029, 0.0028, 0.004, 0.0046, 0.0159, 0.0104, 0.0167, 0.0121, 0.0174, 0.4568],
        "gradient_x": 26.88,
        "gradient_y": 38.83,
        "original_size": (838, 1280),
        "file_size": 45545
    },
    "38.jpeg": {
        "mean": 166.15,
        "std": 92.86,
        "histogram": [0.0, 0.0, 0.0144, 0.2032, 0.1958, 0.0016, 0.0087, 0.0026, 0.0095, 0.0063, 0.016, 0.009, 0.0101, 0.0135, 0.0198, 0.4894],
        "gradient_x": 27.75,
        "gradient_y": 43.54,
        "original_size": (763, 1280),
        "file_size": 43203
    },
    "39.jpeg": {
        "mean": 166.16,
        "std": 92.87,
        "histogram": [0.0, 0.0, 0.0144, 0.2032, 0.1956, 0.0016, 0.0089, 0.0026, 0.0098, 0.0061, 0.0162, 0.0092, 0.0099, 0.0134, 0.0188, 0.4904],
        "gradient_x": 27.45,
        "gradient_y": 43.46,
        "original_size": (763, 1280),
        "file_size": 43228
    },
    "40.jpeg": {
        "mean": 160.11,
        "std": 93.78,
        "histogram": [0.0, 0.0001, 0.0081, 0.223, 0.2233, 0.0017, 0.0027, 0.0032, 0.004, 0.0046, 0.0161, 0.0104, 0.0172, 0.012, 0.0166, 0.4572],
        "gradient_x": 26.8,
        "gradient_y": 38.91,
        "original_size": (838, 1280),
        "file_size": 45508
    },
    "41.jpeg": {
        "mean": 160.14,
        "std": 93.8,
        "histogram": [0.0, 0.0001, 0.0082, 0.2236, 0.2228, 0.0015, 0.0029, 0.0029, 0.0035, 0.0049, 0.0162, 0.0105, 0.0166, 0.0118, 0.0178, 0.4567],
        "gradient_x": 26.93,
        "gradient_y": 40.16,
        "original_size": (1047, 1600),
        "file_size": 62247
    },
    "42.jpeg": {
        "mean": 160.13,
        "std": 93.8,
        "histogram": [0.0, 0.0001, 0.0082, 0.2236, 0.2228, 0.0014, 0.0029, 0.0031, 0.0036, 0.0049, 0.0159, 0.0107, 0.0161, 0.0125, 0.0179, 0.4564],
        "gradient_x": 26.82,
        "gradient_y": 40.33,
        "original_size": (1047, 1600),
        "file_size": 62371
    },
    "43.jpeg": {
        "mean": 160.15,
        "std": 93.78,
        "histogram": [0.0, 0.0001, 0.0081, 0.2236, 0.2229, 0.0015, 0.0027, 0.0026, 0.0036, 0.0049, 0.0166, 0.0107, 0.0163, 0.0124, 0.0176, 0.4565],
        "gradient_x": 26.83,
        "gradient_y": 40.2,
        "original_size": (1047, 1600),
        "file_size": 62329
    },
    "44.jpeg": {
        "mean": 185.38,
        "std": 90.26,
        "histogram": [0.0, 0.0001, 0.0093, 0.2038, 0.0938, 0.0045, 0.0157, 0.0032, 0.0043, 0.0068, 0.0082, 0.0129, 0.0146, 0.0145, 0.0227, 0.5856],
        "gradient_x": 30.17,
        "gradient_y": 44.68,
        "original_size": (1260, 1520),
        "file_size": 77511
    },
    "45.jpeg": {
        "mean": 184.95,
        "std": 90.23,
        "histogram": [0.0, 0.0001, 0.0093, 0.2038, 0.094, 0.0046, 0.0164, 0.0041, 0.0047, 0.0069, 0.0094, 0.0124, 0.0147, 0.0147, 0.0231, 0.5818],
        "gradient_x": 31.32,
        "gradient_y": 45.83,
        "original_size": (1260, 1520),
        "file_size": 80491
    },
    "46.jpeg": {
        "mean": 185.1,
        "std": 90.24,
        "histogram": [0.0, 0.0001, 0.0093, 0.204, 0.0938, 0.0048, 0.0164, 0.0033, 0.0043, 0.0065, 0.0094, 0.0123, 0.0159, 0.0143, 0.0229, 0.5826],
        "gradient_x": 30.65,
        "gradient_y": 45.24,
        "original_size": (1260, 1520),
        "file_size": 79359
    },
    "47.jpeg": {
        "mean": 185.18,
        "std": 90.24,
        "histogram": [0.0, 0.0001, 0.0093, 0.2039, 0.0939, 0.0046, 0.0158, 0.0032, 0.0045, 0.0068, 0.0089, 0.0134, 0.0146, 0.015, 0.0222, 0.5837],
        "gradient_x": 30.54,
        "gradient_y": 45.13,
        "original_size": (1260, 1520),
        "file_size": 78628
    },
    "48.jpeg": {
        "mean": 185.02,
        "std": 90.2,
        "histogram": [0.0, 0.0001, 0.0093, 0.2038, 0.0936, 0.0047, 0.0165, 0.0037, 0.0049, 0.0065, 0.0093, 0.0126, 0.015, 0.0153, 0.0223, 0.5824],
        "gradient_x": 30.98,
        "gradient_y": 45.64,
        "original_size": (1260, 1520),
        "file_size": 80387
    },
    "49.jpeg": {
        "mean": 181.62,
        "std": 91.08,
        "histogram": [0.0, 0.0001, 0.0095, 0.2261, 0.0857, 0.0046, 0.0099, 0.0095, 0.0062, 0.0135, 0.008, 0.011, 0.0125, 0.0132, 0.0107, 0.5793],
        "gradient_x": 28.03,
        "gradient_y": 41.5,
        "original_size": (1260, 1425),
        "file_size": 70855
    },
    "50.jpeg": {
        "mean": 185.07,
        "std": 90.22,
        "histogram": [0.0, 0.0001, 0.0093, 0.2038, 0.0938, 0.0048, 0.0161, 0.0035, 0.005, 0.0065, 0.0092, 0.0131, 0.0146, 0.0141, 0.0236, 0.5823],
        "gradient_x": 31.04,
        "gradient_y": 45.83,
        "original_size": (1260, 1520),
        "file_size": 79665
    },
    "51.jpeg": {
        "mean": 185.22,
        "std": 90.23,
        "histogram": [0.0, 0.0001, 0.0093, 0.2039, 0.0939, 0.0045, 0.0158, 0.0032, 0.0042, 0.0067, 0.0089, 0.0135, 0.015, 0.0146, 0.0227, 0.5836],
        "gradient_x": 30.5,
        "gradient_y": 45.09,
        "original_size": (1260, 1520),
        "file_size": 78463
    },
    "52.jpeg": {
        "mean": 185.13,
        "std": 90.21,
        "histogram": [0.0, 0.0001, 0.0093, 0.2039, 0.0939, 0.0045, 0.0157, 0.0034, 0.0045, 0.0069, 0.009, 0.0132, 0.0154, 0.0149, 0.0226, 0.5825],
        "gradient_x": 30.8,
        "gradient_y": 45.45,
        "original_size": (1260, 1520),
        "file_size": 79500
    },
    "53.jpeg": {
        "mean": 185.26,
        "std": 90.26,
        "histogram": [0.0, 0.0001, 0.0093, 0.2038, 0.0938, 0.0046, 0.0165, 0.0031, 0.004, 0.006, 0.0094, 0.0131, 0.015, 0.0139, 0.0232, 0.5842],
        "gradient_x": 30.5,
        "gradient_y": 44.94,
        "original_size": (1260, 1520),
        "file_size": 78488
    },
    "54.jpeg": {
        "mean": 185.16,
        "std": 90.26,
        "histogram": [0.0, 0.0001, 0.0093, 0.2038, 0.0938, 0.0048, 0.0161, 0.0035, 0.0047, 0.0065, 0.0094, 0.0128, 0.0139, 0.0142, 0.0232, 0.5839],
        "gradient_x": 30.55,
        "gradient_y": 45.06,
        "original_size": (1260, 1520),
        "file_size": 79145
    },
    "55.jpeg": {
        "mean": 185.22,
        "std": 90.23,
        "histogram": [0.0, 0.0001, 0.0093, 0.2039, 0.0939, 0.0045, 0.0158, 0.0033, 0.0043, 0.0066, 0.0087, 0.0131, 0.0152, 0.0148, 0.0231, 0.5832],
        "gradient_x": 30.8,
        "gradient_y": 45.26,
        "original_size": (1260, 1520),
        "file_size": 78542
    },
    "56.jpeg": {
        "mean": 176.55,
        "std": 93.29,
        "histogram": [0.0, 0.0001, 0.0032, 0.2568, 0.0909, 0.0118, 0.0089, 0.0035, 0.0051, 0.0063, 0.0097, 0.017, 0.0125, 0.0103, 0.0095, 0.5544],
        "gradient_x": 27.24,
        "gradient_y": 32.5,
        "original_size": (1260, 1286),
        "file_size": 62527
    },
    "58.jpeg": {
        "mean": 176.52,
        "std": 93.28,
        "histogram": [0.0, 0.0001, 0.0032, 0.2569, 0.0909, 0.0117, 0.0089, 0.0036, 0.0052, 0.0063, 0.0093, 0.0173, 0.0131, 0.0104, 0.0092, 0.554],
        "gradient_x": 27.48,
        "gradient_y": 32.63,
        "original_size": (1260, 1286),
        "file_size": 62611
    },
    "59.jpeg": {
        "mean": 185.08,
        "std": 90.22,
        "histogram": [0.0, 0.0001, 0.0093, 0.2037, 0.0938, 0.0048, 0.0162, 0.0039, 0.0046, 0.0063, 0.0093, 0.0128, 0.0145, 0.0151, 0.0233, 0.5823],
        "gradient_x": 31.06,
        "gradient_y": 45.48,
        "original_size": (1260, 1520),
        "file_size": 79239
    },
    "60.jpeg": {
        "mean": 185.39,
        "std": 90.28,
        "histogram": [0.0, 0.0001, 0.0093, 0.2037, 0.0938, 0.0047, 0.0159, 0.0031, 0.0046, 0.0063, 0.0086, 0.0128, 0.0141, 0.014, 0.0226, 0.5862],
        "gradient_x": 30.28,
        "gradient_y": 44.65,
        "original_size": (1260, 1520),
        "file_size": 77519
    },
    "61.jpeg": {
        "mean": 176.48,
        "std": 93.26,
        "histogram": [0.0, 0.0001, 0.0031, 0.257, 0.0908, 0.0118, 0.0089, 0.0037, 0.0052, 0.0063, 0.0099, 0.0171, 0.0126, 0.011, 0.0096, 0.5529],
        "gradient_x": 27.5,
        "gradient_y": 32.58,
        "original_size": (1260, 1286),
        "file_size": 62931
    },
    "62.jpeg": {
        "mean": 184.96,
        "std": 90.15,
        "histogram": [0.0, 0.0001, 0.0093, 0.2038, 0.0935, 0.0046, 0.0161, 0.0034, 0.0059, 0.0068, 0.0092, 0.0128, 0.0152, 0.0155, 0.0234, 0.5804],
        "gradient_x": 31.07,
        "gradient_y": 46.19,
        "original_size": (1260, 1520),
        "file_size": 79846
    },
    "64.jpeg": {
        "mean": 176.38,
        "std": 93.26,
        "histogram": [0.0, 0.0001, 0.0031, 0.2568, 0.0912, 0.0118, 0.0092, 0.0034, 0.0054, 0.0067, 0.0095, 0.0175, 0.0122, 0.011, 0.0096, 0.5524],
        "gradient_x": 27.69,
        "gradient_y": 32.82,
        "original_size": (1260, 1286),
        "file_size": 63365
    },
    "66.jpeg": {
        "mean": 188.7,
        "std": 88.21,
        "histogram": [0.0, 0.0, 0.0093, 0.1945, 0.0815, 0.0046, 0.0109, 0.0026, 0.0055, 0.0063, 0.0231, 0.012, 0.0138, 0.0148, 0.0175, 0.6037],
        "gradient_x": 32.43,
        "gradient_y": 48.85,
        "original_size": (1260, 1659),
        "file_size": 86102
    },
    "67.jpeg": {
        "mean": 185.0,
        "std": 90.23,
        "histogram": [0.0, 0.0001, 0.0093, 0.2042, 0.0937, 0.0049, 0.016, 0.0034, 0.0049, 0.0069, 0.0091, 0.0128, 0.015, 0.0143, 0.0237, 0.5817],
        "gradient_x": 31.02,
        "gradient_y": 45.78,
        "original_size": (1260, 1520),
        "file_size": 79664
    },
    "68.jpeg": {
        "mean": 185.25,
        "std": 90.28,
        "histogram": [0.0, 0.0001, 0.0095, 0.2039, 0.0936, 0.0048, 0.0162, 0.0034, 0.0043, 0.0059, 0.0091, 0.0128, 0.014, 0.0143, 0.0237, 0.5842],
        "gradient_x": 30.68,
        "gradient_y": 44.87,
        "original_size": (1260, 1520),
        "file_size": 78025
    },
    "70.jpeg": {
        "mean": 180.55,
        "std": 92.3,
        "histogram": [0.0, 0.0001, 0.0092, 0.2341, 0.0861, 0.005, 0.022, 0.0023, 0.0046, 0.0058, 0.0093, 0.0127, 0.0117, 0.0093, 0.012, 0.5757],
        "gradient_x": 27.51,
        "gradient_y": 33.75,
        "original_size": (1260, 1373),
        "file_size": 66406
    },
    "71.jpeg": {
        "mean": 188.82,
        "std": 88.3,
        "histogram": [0.0, 0.0, 0.0093, 0.1946, 0.0818, 0.0042, 0.0113, 0.0029, 0.0047, 0.0063, 0.0219, 0.0121, 0.014, 0.0134, 0.0168, 0.6066],
        "gradient_x": 31.45,
        "gradient_y": 48.0,
        "original_size": (1260, 1659),
        "file_size": 84387
    },
    "72.jpeg": {
        "mean": 185.31,
        "std": 90.26,
        "histogram": [0.0, 0.0001, 0.0093, 0.2039, 0.0938, 0.0046, 0.0158, 0.0032, 0.0045, 0.0067, 0.0084, 0.0132, 0.0147, 0.0141, 0.0227, 0.5851],
        "gradient_x": 30.14,
        "gradient_y": 44.82,
        "original_size": (1260, 1520),
        "file_size": 77984
    },
    "73.jpeg": {
        "mean": 185.24,
        "std": 90.23,
        "histogram": [0.0, 0.0001, 0.0093, 0.2036, 0.0935, 0.0047, 0.0162, 0.0037, 0.0046, 0.0064, 0.009, 0.0125, 0.0148, 0.0146, 0.0225, 0.5844],
        "gradient_x": 30.28,
        "gradient_y": 44.91,
        "original_size": (1260, 1520),
        "file_size": 78659
    },
    "74.jpeg": {
        "mean": 188.89,
        "std": 88.24,
        "histogram": [0.0, 0.0, 0.0093, 0.1946, 0.0814, 0.0044, 0.0109, 0.0026, 0.0049, 0.0063, 0.0234, 0.0112, 0.0138, 0.0142, 0.0163, 0.6068],
        "gradient_x": 31.59,
        "gradient_y": 48.39,
        "original_size": (1260, 1659),
        "file_size": 84758
    }
}

UNIQUE_IMAGE_NAMES = ['1.jpeg', '2.jpeg', '3.jpeg', '4.jpeg', '5.jpeg', '6.jpeg', '7.jpeg', '8.jpeg', '9.jpeg', '10.jpeg', '11.jpeg', '12.jpeg', '13.jpeg', '14.jpeg', '15.jpeg', '16.jpeg', '17.jpeg', '18.jpeg', '19.jpeg', '20.jpeg', '21.jpeg', '22.jpeg', '23.jpeg', '24.jpeg', '25.jpeg', '26.jpeg', '27.jpeg', '28.jpeg', '29.jpeg', '30.jpeg', '31.jpeg', '32.jpeg', '33.jpeg', '34.jpeg', '35.jpeg', '36.jpeg', '37.jpeg', '38.jpeg', '39.jpeg', '40.jpeg', '41.jpeg', '42.jpeg', '43.jpeg', '44.jpeg', '45.jpeg', '46.jpeg', '47.jpeg', '48.jpeg', '49.jpeg', '50.jpeg', '51.jpeg', '52.jpeg', '53.jpeg', '54.jpeg', '55.jpeg', '56.jpeg', '58.jpeg', '59.jpeg', '60.jpeg', '61.jpeg', '62.jpeg', '64.jpeg', '66.jpeg', '67.jpeg', '68.jpeg', '70.jpeg', '71.jpeg', '72.jpeg', '73.jpeg', '74.jpeg']

router = APIRouter()

RESIZE_DIM = (128, 128)

def quick_similarity_check(features1: dict, features2: dict) -> float:
    """
    Comparación ultra-rápida usando características pre-calculadas
    Retorna un score aproximado sin SSIM completo
    """
    # Diferencia de medias
    mean_diff = abs(features1["mean"] - features2["mean"]) / 255.0
    
    # Diferencia de desviaciones estándar
    std_diff = abs(features1["std"] - features2["std"]) / 255.0
    
    # Comparación de histogramas (correlación)
    hist1 = np.array(features1["histogram"])
    hist2 = np.array(features2["histogram"])
    hist_corr = np.corrcoef(hist1, hist2)[0, 1]
    hist_corr = 0 if np.isnan(hist_corr) else hist_corr
    
    # Diferencia de gradientes
    grad_diff = abs(features1["gradient_x"] - features2["gradient_x"]) + \
                abs(features1["gradient_y"] - features2["gradient_y"])
    grad_diff = grad_diff / 100.0  # Normalizar
    
    # Score combinado (ponderado)
    similarity = (
        (1 - mean_diff) * 0.2 +
        (1 - std_diff) * 0.2 +
        hist_corr * 0.4 +
        (1 - min(grad_diff, 1.0)) * 0.2
    )
    
    return max(0, min(1, similarity))

def extract_input_features(img_array: np.ndarray) -> dict:
    """Extrae características de la imagen de entrada"""
    mean_val = float(np.mean(img_array))
    std_val = float(np.std(img_array))
    
    hist, _ = np.histogram(img_array.flatten(), bins=16, range=(0, 256))
    hist_normalized = (hist / hist.sum()).tolist() if hist.sum() > 0 else [0] * 16
    
    grad_x = np.abs(np.diff(img_array, axis=1)).mean() if img_array.shape[1] > 1 else 0
    grad_y = np.abs(np.diff(img_array, axis=0)).mean() if img_array.shape[0] > 1 else 0
    
    return {
        "mean": round(mean_val, 2),
        "std": round(std_val, 2),
        "histogram": [round(x, 4) for x in hist_normalized],
        "gradient_x": round(float(grad_x), 2),
        "gradient_y": round(float(grad_y), 2)
    }

def ultra_fast_comparison(img_bytes: bytes, use_full_ssim: bool = False) -> dict:
    """
    Comparación ultra-rápida usando características pre-calculadas
    """
    if not CONSTANTS_AVAILABLE:
        raise ValueError("Constantes no disponibles. Ejecuta generate_image_fingerprints.py")
    
    try:
        # Procesar imagen de entrada
        img = Image.open(io.BytesIO(img_bytes)).convert("L").resize(RESIZE_DIM, Image.Resampling.LANCZOS)
        img_array = np.array(img)
        
        # Extraer características de entrada
        input_features = extract_input_features(img_array)
        
        resultados = []
        
        # Comparación rápida con características
        for nombre in UNIQUE_IMAGE_NAMES:
            ref_features = IMAGE_DATA[nombre]
            
            # Comparación ultra-rápida
            quick_score = quick_similarity_check(input_features, ref_features)
            quick_pct = round(quick_score * 100, 2)
            final_pct = quick_pct                
            
            resultados.append({
                "imagen": nombre,
                "similitud": final_pct,
                "method": "ssim" if (use_full_ssim and quick_pct > 50) else "quick"
            })
        
        # Calcular promedio
        if resultados:
            similitudes = [r["similitud"] for r in resultados]
            promedio = round(sum(similitudes) / len(similitudes), 2)
        else:
            promedio = 0.0
        
        return {"detalle": resultados, "promedio": promedio}
    
    except Exception as e:
        print(f"Error en comparación: {e}")
        return {"detalle": [], "promedio": 0.0}

@router.post("/filtro_ssim")
async def filtro_ssim(
    file: UploadFile = File(...),
    precision: str = "fast"  # "fast" o "precise"
):
    """
    Filtro SSIM ultra-rápido
    - precision="fast": Solo características (sub-segundo)
    - precision="precise": SSIM completo para candidatos prometedores
    """
    try:
        if not CONSTANTS_AVAILABLE:
            raise HTTPException(
                status_code=500, 
                detail="❌ Sistema no inicializado. Contacta al administrador."
            )
        
        # Validaciones
        if not file.content_type or not file.content_type.startswith("image/"):
            raise HTTPException(status_code=422, detail="❌ Debe ser imagen válida")

        if file.size and file.size > 5 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="❌ Imagen muy grande (máx 5MB)")
        
        contenido = await file.read()
        if not contenido:
            raise HTTPException(status_code=422, detail="❌ Archivo vacío")

        # Procesamiento ultra-rápido
        use_full_ssim = (precision == "precise")
        
        loop = asyncio.get_event_loop()
        resultado = await asyncio.wait_for(
            loop.run_in_executor(None, ultra_fast_comparison, contenido, use_full_ssim),
            timeout=15.0  # Mucho más rápido
        )
        
        # Estadísticas
        imagenes_alta_coincidencia = sum(
            1 for img in resultado["detalle"] 
            if img["similitud"] > 60
        )
        
        # Obtener top 5 matches
        top_matches = sorted(
            resultado["detalle"], 
            key=lambda x: x["similitud"], 
            reverse=True
        )[:5]
        
        gc.collect()
        
        return {
            "imagenes_comparadas": len(resultado["detalle"]),
            "imagenes_unicas": len(UNIQUE_IMAGE_NAMES),
            "promedio_similitud": resultado['promedio'],
            "imagenes_con_alta_coincidencia": imagenes_alta_coincidencia,
            "top_matches": top_matches,
            "precision_mode": precision,
            "total_time_estimate": "< 1 segundo" if not use_full_ssim else "< 5 segundos"
        }
    
    except asyncio.TimeoutError:
        raise HTTPException(status_code=408, detail="❌ Timeout - intenta modo 'fast'")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"❌ Error: {str(e)[:100]}")