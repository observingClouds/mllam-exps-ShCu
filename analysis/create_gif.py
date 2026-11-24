#!/leonardo_work/DestE_340_26/users/hschulz0/repos/mllam-exps-ShCu/.venv/bin/python
"""Create an animated GIF from the timestep comparison PNG files."""

from PIL import Image, ImageDraw, ImageFont
import glob
from pathlib import Path

# Get all timestep PNG files in order
png_files = sorted(glob.glob("single_timestep_comparison_t*.png"))

if not png_files:
    print("No PNG files found matching pattern 'single_timestep_comparison_t*.png'")
    exit(1)

print(f"Found {len(png_files)} PNG files")
print(f"Creating GIF from: {png_files[0]} to {png_files[-1]}")

# Define target screen resolution (width)
target_width = 1920  # Standard HD screen width
# Or use 1280 for smaller screens, 2560 for 2K, 3840 for 4K

# Footer settings
footer_text = "Hauke Schulz, 2026-05-03"
footer_height = 40  # pixels for footer area
footer_bg_color = (255, 255, 255)  # white background
text_color = (0, 0, 0)  # black text

# Load and resize all images
images = []
for png_file in png_files:
    img = Image.open(png_file)
    
    # Calculate new size maintaining aspect ratio
    aspect_ratio = img.height / img.width
    new_width = target_width
    new_height = int(new_width * aspect_ratio)
    
    # Resize image with high-quality resampling
    img_resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    # Create new image with footer space
    img_with_footer = Image.new('RGB', (new_width, new_height + footer_height), footer_bg_color)
    img_with_footer.paste(img_resized, (0, 0))
    
    # Add footer text
    draw = ImageDraw.Draw(img_with_footer)
    try:
        # Try to use a nice font
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
    except:
        # Fall back to default font
        font = ImageFont.load_default()
    
    # Calculate text position (centered at bottom)
    bbox = draw.textbbox((0, 0), footer_text, font=font)
    text_width = bbox[2] - bbox[0]
    text_x = (new_width - text_width) // 2
    text_y = new_height + (footer_height - (bbox[3] - bbox[1])) // 2
    
    draw.text((text_x, text_y), footer_text, fill=text_color, font=font)
    
    images.append(img_with_footer)
    print(f"Loaded and resized {png_file} from {img.size} to {img_with_footer.size}")

# Save as animated GIF
output_file = "timestep_comparison_animation.gif"
images[0].save(
    output_file,
    save_all=True,
    append_images=images[1:],
    duration=500,  # milliseconds per frame (500ms = 0.5s per frame)
    loop=0  # 0 means loop forever
)

print(f"\nSuccessfully created {output_file}")
print(f"Animation duration: {len(images) * 0.5:.1f} seconds (looping)")
print(f"Frame rate: 2 frames per second")
