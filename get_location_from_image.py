import cv2
import os
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
import requests
from geopy.geocoders import Nominatim
from tkinter import filedialog, Tk

# --------- Function 1: Open File Dialog for Upload ---------
def select_image():
    root = Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
    return file_path

# --------- Function 2: Take Photo from Webcam ----------
def capture_image():
    cam = cv2.VideoCapture(0)
    cv2.namedWindow("Press SPACE to capture")

    while True:
        ret, frame = cam.read()
        cv2.imshow("Press SPACE to capture", frame)
        key = cv2.waitKey(1)
        if key % 256 == 32:
            filename = "captured_image.jpg"
            cv2.imwrite(filename, frame)
            break
        elif key % 256 == 27:
            filename = None
            break

    cam.release()
    cv2.destroyAllWindows()
    return filename

# --------- Function 3: Get GPS from Image (Mobile Photo) ---------
def extract_gps(img_path):
    def to_degrees(value):
        d, m, s = value
        return d[0]/d[1] + m[0]/m[1]/60 + s[0]/s[1]/3600

    img = Image.open(img_path)
    exif_data = img._getexif()
    gps = {}

    if exif_data is None:
        return None

    for tag, val in exif_data.items():
        if TAGS.get(tag) == "GPSInfo":
            for key in val:
                gps[GPSTAGS.get(key)] = val[key]

    if "GPSLatitude" in gps and "GPSLongitude" in gps:
        lat = to_degrees(gps["GPSLatitude"])
        lon = to_degrees(gps["GPSLongitude"])
        if gps["GPSLatitudeRef"] != "N":
            lat = -lat
        if gps["GPSLongitudeRef"] != "E":
            lon = -lon
        return (lat, lon)
    return None

# --------- Function 4: Fallback to IP-based Location ---------
def get_ip_location():
    try:
        res = requests.get("https://ipinfo.io/json")
        loc = res.json()['loc'].split(',')
        return float(loc[0]), float(loc[1])
    except:
        return None

# --------- Function 5: Convert Coordinates to Address ---------
def get_address(coords):
    geo = Nominatim(user_agent="geoapi")
    return geo.reverse(coords, language='en').address

# --------- MAIN FUNCTION ---------
def main():
    choice = input("Do you want to (1) Upload an image or (2) Capture from webcam? Enter 1 or 2: ")
    
    if choice == "1":
        path = select_image()
        if not path:
            print("❌ No image selected.")
            return
    elif choice == "2":
        path = capture_image()
        if not path:
            print("❌ No image captured.")
            return
    else:
        print("❌ Invalid choice.")
        return

    # Try extracting GPS from image
    gps = extract_gps(path)
    if gps:
        print(f"✅ GPS found: {gps}")
        print("📌 Location:", get_address(gps))
    else:
        print("⚠️ No GPS found in image, trying IP-based location...")
        coords = get_ip_location()
        if coords:
            print(f"📍 Approximate IP Location: {coords}")
            print("📌 Location:", get_address(coords))
        else:
            print("❌ Could not determine location.")

if __name__ == "__main__":
    main()
