from PIL import Image

def make_white_transparent(input_path, output_path):
    # 1. Open the image and convert it to RGBA (so it has an alpha channel)
    img = Image.open(input_path).convert("RGBA")
    datas = img.getdata()

    newData = []
    # Define what you consider "white".
    # (255, 255, 255) is pure white. We use a small threshold (e.g., > 240)
    # to catch pixels that are almost white.
    threshold = 240

    # 2. Loop through every pixel
    for item in datas:
        # item is a tuple of (R, G, B, A)
        # Check if Red, Green, and Blue are all brighter than the threshold
        if item[0] > threshold and item[1] > threshold and item[2] > threshold:
            # If it's white, make it fully transparent (Alpha = 0)
            newData.append((255, 255, 255, 0))
        else:
            # Otherwise, keep the original pixel color
            newData.append(item)

    # 3. Save the new data into the image
    img.putdata(newData)
    img.save(output_path, "PNG")
    print(f"Successfully saved transparent image to: {output_path}")

# --- Use the function ---
# Replace with your actual file paths
input_file = "./Figure_1.png"
output_file = "./Figure_1b.png"

make_white_transparent(input_file, output_file)