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
    "opencv": ["EAN13", "EAN8"],
    "wechat": ["QR"],
    "zxing": ["EAN13", "EAN8", "CODE128", "CODE39", "QR"],
}


def normalize_format(name):
    """'EAN_13' / 'EAN-13' / 'Code 128' / 'QRCode' -> 'EAN13' / 'CODE128' / 'QR'."""
    clean = str(name).upper().replace("-", "").replace("_", "").replace(" ", "")
    if clean.startswith("QR"):
        return "QR"
    return clean


def decode_opencv(image):
    ok, texts, types, points = _opencv_detector.detectAndDecodeWithType(image)
    if not ok:
        return []
    # OpenCV може знайти баркод, але не прочитати: тоді текст порожній
    return [{"text": text, "format": normalize_format(kind)}
            for text, kind in zip(texts, types) if text != ""]


def decode_wechat(image):
    texts, points = _wechat_detector.detectAndDecode(image)
    return [{"text": text, "format": "QR"} for text in texts if text != ""]


def decode_zxing(image):
    return [{"text": barcode.text, "format": normalize_format(barcode.format)}
            for barcode in zxingcpp.read_barcodes(image)]


DECODERS = {
    "opencv": decode_opencv,
    "wechat": decode_wechat,
    "zxing": decode_zxing,
}


def decoders_for(code_format):
    """Назви декодерів, які підтримують цей формат."""
    return [name for name in DECODERS if code_format in SUPPORTED[name]]


def check(results, true_text):
    """'correct' / 'wrong' (прочитав, але не те) / 'none' (нічого не прочитав)."""
    if len(results) == 0:
        return "none"
    if any(result["text"] == true_text for result in results):
        return "correct"
    return "wrong"
