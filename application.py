import os
import json
import tkinter as tk
from tkinter import ttk
from PIL import ImageTk, Image
import csv
import subprocess
import math

# Get the current directory path
current_dir = os.path.dirname(os.path.abspath(__file__))

# Load the prompts from the JSON file
prompts_file = os.path.join(current_dir, "prompts.json")
with open(prompts_file, "r") as file:
    prompts_data = json.load(file)

# Extract the prompts and directory paths from the prompts data
prompts = []
directory_paths = []
for entry in prompts_data:
    prompts.append(entry["prompt"])
    directory_path = os.path.join(current_dir, os.path.basename(entry["directory_path"]))
    directory_paths.append(directory_path)

# Remove duplicate prompts and directory paths
prompts = list(dict.fromkeys(prompts))
directory_paths = list(dict.fromkeys(directory_paths))

# Initialize an empty dictionary to store the ratings
ratings = {}

def show_start_page():
    start_frame.pack(pady=20)

def show_prompt_page(prompt_index):
    start_frame.pack_forget()
    prompt_frame.pack_forget()
    
    prompt = prompts[prompt_index]
    prompt_label.config(text=f"Description: {prompt}")
    
    prompt_dir = directory_paths[prompt_index]
    image_files = [f for f in os.listdir(prompt_dir) if f.endswith(".png") or f.endswith(".jpg")]
    
    # for i, image_file in enumerate(image_files):
    #     image_path = os.path.join(prompt_dir, image_file)
    #     image = Image.open(image_path)
    #     image.thumbnail((100, 100))  # Adjust the image size to fit the canvas
    #     photo = ImageTk.PhotoImage(image)
    #     canvas = tk.Canvas(image_frame, width=100, height=100)  # Set the canvas size
    #     canvas.grid(row=i, column=0, padx=10, pady=10)
    #     canvas.create_image(0, 0, anchor=tk.NW, image=photo)
    #     canvas.image = photo
    #     rating_entries[i].delete(0, tk.END)
    
    # for i in range(len(image_files), len(rating_entries)):
    #     rating_entries[i].delete(0, tk.END)

    num_cols = 3
    num_rows = 2

    for i, image_file in enumerate(image_files):
        if i >= num_cols * num_rows:
            break

        image_path = os.path.join(prompt_dir, image_file)
        image = Image.open(image_path)
        image.thumbnail((300, 300))  # Adjust the image size to fit the canvas
        photo = ImageTk.PhotoImage(image)

        col = i % num_cols
        row = i // num_cols

        canvas = tk.Canvas(image_frame, width=300, height=300)  # Set the canvas size
        canvas.grid(row=row, column=col, padx=10, pady=10)
        canvas.create_image(0, 0, anchor=tk.NW, image=photo)
        canvas.image = photo

        rating_entry = ttk.Entry(image_frame, width=5)
        rating_entry.grid(row=row, column=col, padx=10, pady=(310, 10))  # Position the input box below the canvas
        rating_entries.append(rating_entry)

    for i in range(len(image_files), len(rating_entries)):
        rating_entries[i].delete(0, tk.END)
    
    prompt_progress_label.config(text=f"Please rate on a scale from 1 to 10, based on how well it matches the given description.\n A rating of 1 means the image does not match the description at all, while a rating of 10 means the image perfectly matches the description.\n Prompt {prompt_index + 1}/{len(prompts)}, {len(prompts) - prompt_index - 1} left to be rated")
    
    prompt_frame.pack()
    
    if prompt_index == 0:
        prev_button.config(state=tk.DISABLED)
    else:
        prev_button.config(state=tk.NORMAL)
    
    if prompt_index == len(prompts) - 1:
        next_button.config(text="Finish", command=show_finish_page)
    else:
        next_button.config(text="Next", command=lambda: show_prompt_page(prompt_index + 1))

def show_finish_page():
    prompt_frame.pack_forget()
    finish_frame.pack(pady=20)
    save_ratings()
    report_button.config(state=tk.NORMAL)  # Enable the "Report" button after saving the ratings

def save_ratings():
    for i, prompt_dir in enumerate(directory_paths):
        image_files = [f for f in os.listdir(prompt_dir) if f.endswith(".png") or f.endswith(".jpg")]
        for j, image_file in enumerate(image_files):
            rating = rating_entries[j].get()
            ratings[f"{prompt_dir}_{image_file}"] = rating
    
    # Save the ratings to a CSV file
    ratings_file = os.path.join(current_dir, "ratings.csv")
    with open(ratings_file, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Image", "Rating"])
        for image, rating in ratings.items():
            writer.writerow([image, rating])

def open_report():
    ratings_file = os.path.join(current_dir, "ratings.csv")
    if os.path.exists(ratings_file):
        if os.name == "nt":  # Windows
            os.startfile(ratings_file)
        elif os.name == "posix":  # macOS and Linux
            subprocess.call(["open", ratings_file])
    else:
        print("Ratings file does not exist.")

# Create the main window
window = tk.Tk()
window.title("MJ-Bench Rating App")
window.geometry("1200x900")  # Set the window size

# Create the start frame
start_frame = ttk.Frame(window, width=1200, height=900)
start_frame.pack_propagate(False)
start_frame.pack(fill='both', expand=True) 

# Load the logo image
logo_image = Image.open("logo.png")
logo_image = logo_image.resize((250, 250))  
logo_photo = ImageTk.PhotoImage(logo_image)

# Create a label to display the logo
logo_label = ttk.Label(start_frame, image=logo_photo)
logo_label.pack(pady=(20, 0))  # Add some padding above the logo

welcome_label = ttk.Label(start_frame, text="Welcome to the MJ-Bench Rating App", font=("Arial", 16))
welcome_label.place(relx=0.5, rely=0.4, anchor='center')


description_text = (
    "We appreciate your participation in helping us evaluate AI-generated images. "
    "Your task is to rate each picture on a scale from 1 to 10, based on how well it matches the given description. "
    "A rating of 1 means the image does not match the description at all, while a rating of 10 means the image perfectly matches the description.\n"
    "Thank you for your valuable contribution to developing safer and more accurate AI technology!"
)
description_label = ttk.Label(start_frame, text=description_text, wraplength=1000, justify="center")
description_label.place(relx=0.5, rely=0.5, anchor='center')

start_button = ttk.Button(start_frame, text="Start", command=lambda: show_prompt_page(0))
start_button.place(relx=0.5, rely=0.6, anchor='center')

# Create the prompt frame
prompt_frame = ttk.Frame(window, width=1200, height=800)

prompt_label = ttk.Label(prompt_frame, font=("Arial", 16))
prompt_label.pack(pady=10)

prompt_progress_label = ttk.Label(prompt_frame, font=("Arial", 12), justify="center", wraplength=1500)
prompt_progress_label.pack()

image_frame = ttk.Frame(prompt_frame)
image_frame.pack()

# Determine the maximum number of images per prompt
max_images_per_prompt = max(len(os.listdir(path)) for path in directory_paths)

rating_entries = []
# for i in range(max_images_per_prompt):
#     rating_entry = ttk.Entry(image_frame, width=5)
#     rating_entry.grid(row=i, column=1, padx=10, pady=10)
#     rating_entries.append(rating_entry)

button_frame = ttk.Frame(prompt_frame)
button_frame.pack(pady=10)

prev_button = ttk.Button(button_frame, text="Previous", command=lambda: show_prompt_page(max(0, prompt_index - 1)))
prev_button.pack(side=tk.LEFT, padx=10)

next_button = ttk.Button(button_frame, text="Next", command=lambda: show_prompt_page(prompt_index + 1))
next_button.pack(side=tk.LEFT)

# Create the finish frame
finish_frame = ttk.Frame(window)

finish_label = ttk.Label(finish_frame, text="Thank you for your participation!", font=("Arial", 16))
finish_label.pack(pady=10)

report_button = ttk.Button(finish_frame, text="Report", command=open_report, state=tk.DISABLED)  # Disable the "Report" button initially
report_button.pack(pady=10)

# Show the start page initially
show_start_page()

# Start the GUI
window.mainloop()