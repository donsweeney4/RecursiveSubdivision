import plotly.graph_objects as go
import matplotlib.pyplot as plt
import numpy as np
import os
import uuid # To generate unique filenames

# --- Configuration ---
# Directory where temporary images will be saved
# This will be a subdirectory within the directory where this script is run.
TEMP_IMAGE_DIR = 'temp_hover_plots'

# Port for the local HTTP server.
# This should match the port you use when running `python -m http.server`.
LOCAL_SERVER_PORT = 8000

# Base URL for accessing images via the local server.
# Assumes the server is run from the same directory as this script,
# and the images are in the 'temp_hover_plots' subdirectory.
LOCAL_SERVER_URL_BASE = f"http://localhost:{LOCAL_SERVER_PORT}/{TEMP_IMAGE_DIR}/"

# Name of the HTML file that will be generated for the Plotly chart
OUTPUT_HTML_FILE = 'plotly_hover_chart.html'

# -------------------

# Ensure the temporary image directory exists
# This will create 'temp_hover_plots' in the same directory as the script.
os.makedirs(TEMP_IMAGE_DIR, exist_ok=True)

# Sample data for the main plot (products)
products = ['Product A', 'Product B', 'Product C', 'Product D']
main_x = [1, 2, 3, 4]
main_y = [5, 3, 6, 2]

# Function to generate a miniature sales history plot and save it to a file
def create_sales_plot_file(product_name, output_dir=TEMP_IMAGE_DIR):
    # Simulate sales history data
    time = np.arange(1, 6)
    sales = np.random.randint(1, 10, size=5)
    plt.figure(figsize=(2, 1)) # Adjust size as needed (e.g., 2 inches wide, 1 inch tall)
    plt.plot(time, sales, marker='o', markersize=3) # Added markers for clarity
    plt.title(f'{product_name} Sales', fontsize=8)
    plt.xlabel('Time', fontsize=6)
    plt.ylabel('Sales', fontsize=6)
    plt.xticks(fontsize=5)
    plt.yticks(fontsize=5)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout() # Adjust layout to prevent labels from overlapping

    # Generate a unique filename to avoid conflicts
    filename = f"{product_name.replace(' ', '_').lower()}_{uuid.uuid4().hex}.png"
    filepath = os.path.join(output_dir, filename)

    plt.savefig(filepath, format='png', dpi=100) # Save the plot to a file
    plt.close() # Close the matplotlib figure to free memory
    return filename # Return just the filename, which will be appended to the base URL

# --- Main Script Logic ---
image_filenames = []
for product in products:
    filename = create_sales_plot_file(product)
    image_filenames.append(filename)

# Construct the full URLs that the browser will try to access
custom_image_urls = [f"{LOCAL_SERVER_URL_BASE}{filename}" for filename in image_filenames]

# Create the Plotly scatter plot
fig = go.Figure(data=[go.Scatter(x=main_x, y=main_y,
                                 mode='markers',
                                 text=products,
                                 # Store the full image URLs in customdata as a list of lists
                                 customdata=[[url] for url in custom_image_urls], # Changed from np.array().reshape()
                                 hoverinfo='text',
                                 # Reference customdata in the hovertemplate
                                 hovertemplate="<b>%{text}</b><br>" +
                                               "<img src='%{customdata[0]}' width='200' height='100'><br>" + # Adjusted size
                                               "Value X: %{x}<br>" +
                                               "Value Y: %{y}<extra></extra>"
                                )])

fig.update_layout(
    title='Main Product Plot with Sales History Popups (Images from Local Server)',
    hovermode='closest' # Ensures hover shows info for the closest point
)

# Save the Plotly figure to an HTML file
fig.write_html(OUTPUT_HTML_FILE)

print(f"\nImages saved to: '{os.path.abspath(TEMP_IMAGE_DIR)}'")
print(f"Plotly chart saved to: '{os.path.abspath(OUTPUT_HTML_FILE)}'")
print(f"To view images in the hover popup, you must run a simple HTTP server.")
print(f"Navigate to the directory *containing this Python script*.")
print(f"Then run this command in your terminal:")
print(f"   python -m http.server {LOCAL_SERVER_PORT}")
print(f"Ensure the server is running *before* you open or refresh the Plotly chart in your browser.")
print(f"Finally, open the '{OUTPUT_HTML_FILE}' file in your web browser.")
