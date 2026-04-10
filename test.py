import requests
import base64

# 1. Configuration
URL = "http://127.0.0.1:5001/segment"
IMAGE_PATH = "dataset/train/images/2.png"

# 2. Encode image to Base64
with open(IMAGE_PATH, "rb") as image_file:
    encoded_string = base64.b64encode(image_file.read()).decode('utf-8')

# 3. Send POST request
payload = {"image": encoded_string}
response = requests.post(URL, json=payload)

# 4. Handle response
if response.status_code == 200:
    mask_data = response.json()["mask"]
    # Decode and save the mask
    with open("result_mask.png", "wb") as f:
        f.write(base64.b64decode(mask_data))
    print("Success! Mask saved as result_mask.png")
else:
    print(f"Error: {response.json()}")