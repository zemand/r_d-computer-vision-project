"""Декодери: OpenCV, WeChat QR, zxing-cpp. Усі повертають однаковий формат.

Результат кожного декодера: список словників {"text": ..., "format": ...}.
Порожній список = нічого не прочитано.
"""

import cv2
import zxingcpp

# Детектори створюємо один раз: створення займає час
_opencv_detector = cv2.barcode.BarcodeDetector()
_wechat_detector = cv2.wechat_qrcode_WeChatQRCode()

# Які формати вміє кожен декодер (OpenCV barcode: лише EAN/UPC)
SUPPORTED = {
    "opencv": ["EAN13"],
    "wechat": ["QR"],
    "zxing": ["EAN13", "CODE128", "QR"],
}


def normalize_format(name):
    """'EAN_13' / 'EAN-13' / 'Code 128' / 'QRCode' -> 'EAN13' / 'CODE128' / 'QR'."""
    clean = str(name).upper().replace("-", "").replace("_", "").replace(" ", "")
    if clean.startswith("QR"):
        return "QR"
    return clean


def decode_opencv(image):
    ok, texts, types, points = _opencv_detector.detectAndDecodeWithType(image)
    results = []
    if not ok:
        return results
    for index in range(len(texts)):
        # OpenCV може знайти баркод, але не прочитати: тоді текст порожній
        if texts[index] != "":
            results.append({"text": texts[index],
                            "format": normalize_format(types[index])})
    return results


def decode_wechat(image):
    texts, points = _wechat_detector.detectAndDecode(image)
    results = []
    for text in texts:
        if text != "":
            results.append({"text": text, "format": "QR"})
    return results


def decode_zxing(image):
    results = []
    for barcode in zxingcpp.read_barcodes(image):
        results.append({"text": barcode.text,
                        "format": normalize_format(barcode.format)})
    return results


DECODERS = {
    "opencv": decode_opencv,
    "wechat": decode_wechat,
    "zxing": decode_zxing,
}


def decoders_for(code_format):
    """Назви декодерів, які підтримують цей формат."""
    names = []
    for name in DECODERS:
        if code_format in SUPPORTED[name]:
            names.append(name)
    return names
