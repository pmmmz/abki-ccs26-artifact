import logging
from pynput import keyboard
from datetime import datetime

# Set log file path
log_file = "./keystroke_time/key_log_both.txt"

# Create the log file and write header columns
with open(log_file, "w") as file:
    file.write("Key arrived\tKey released\tKey name\n")

# Configure logging format (append mode)
logging.basicConfig(
    filename=log_file,
    level=logging.DEBUG,
    format='%(message)s',
    filemode='a'
)

# Dictionary to store key press start times
key_press_times = {}

# Callback function for key press events
def on_press(key):
    try:
        key_char = key.char
    except AttributeError:
        # Special keys (e.g., Key.space, Key.enter)
        key_char = str(key)
    
    # Record key press timestamp
    key_press_times[key_char] = datetime.now()

# Callback function for key release events
def on_release(key):
    try:
        key_char = key.char
    except AttributeError:
        # Special keys (e.g., Key.space, Key.enter)
        key_char = str(key)
    
    # Retrieve press time and record release time
    press_time = key_press_times.get(key_char, "N/A")
    release_time = datetime.now()
    logging.info(f"{press_time}\t{release_time}\t{key_char}")

    # Stop listener if ESC key is pressed
    if key == keyboard.Key.esc:
        return False

# Start keyboard listener
def start_listener():
    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        listener.join()

# Entry point
if __name__ == "__main__":
    start_listener()
