import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageStat, ImageOps, ImageEnhance, ImageTk
import qrcode
import numpy as np
import os
from random import shuffle, choice
import concurrent.futures
from itertools import repeat

def process_dark_cell(cell, gray_thresh, brightness_dark_max, contrast_factor):
    """
    Adjust the brightness of a dark cell based on the target gray threshold.
    """
    cell_rgb, pos, mean_val = cell
    img_array = np.array(cell_rgb)
    # Map brightness from [0, gray_thresh] to [0, brightness_dark_max]
    mapped_brightness = np.interp(mean_val, [0, gray_thresh], [0, brightness_dark_max])
    factor = mapped_brightness / mean_val if mean_val != 0 else 1
    img_array = np.clip(img_array * factor, 0, 255).astype(np.uint8)
    adjusted_img = Image.fromarray(img_array)
    enhancer = ImageEnhance.Contrast(adjusted_img)
    return (enhancer.enhance(contrast_factor), pos, mean_val)

def process_bright_cell(cell, gray_thresh, brightness_bright_min, contrast_factor):
    """
    Adjust the brightness of a bright cell based on the target gray threshold.
    """
    cell_rgb, pos, mean_val = cell
    img_array = np.array(cell_rgb)
    # Map brightness from [gray_thresh, 255] to [brightness_bright_min, 255]
    mapped_brightness = np.interp(mean_val, [gray_thresh, 255], [brightness_bright_min, 255])
    factor = mapped_brightness / mean_val if mean_val != 0 else 1
    img_array = np.clip(img_array * factor, 0, 255).astype(np.uint8)
    adjusted_img = Image.fromarray(img_array)
    enhancer = ImageEnhance.Contrast(adjusted_img)
    return (enhancer.enhance(contrast_factor), pos, mean_val)

class App:
    def __init__(self, master):
        # Initialize main window and set its title.
        self.master = master
        master.title("QRfypx")
        
        # Load the original image from a default path.
        self.original_img = None
        self.target_img = None
        self.input_path = "inages/baboon.png"
        try:
            self.original_img = Image.open(self.input_path)
        except:
            self.original_img = None

        # Define UI variables.
        self.qr_url = tk.StringVar(value="https://antonhoyer.com/?random_post=4")
        self.output_width = tk.IntVar(value=1024)  # User-defined target width.
        self.contrast_adjustment_factor = tk.DoubleVar(value=1.0)
        self.gray_thresh = tk.IntVar(value=128)
        self.gray_margin = tk.IntVar(value=56)

        # Define threshold mode options and mapping.
        self.threshold_mode_options = [("median", 2), ("mean", 1), ("custom", 0)]
        self.threshold_mode_map = dict(self.threshold_mode_options)
        self.threshold_mode = tk.StringVar(value="mean")

        # Define position mode options and mapping.
        # "vertical" (6): vertical bleed mode (search horizontally).
        # "horizontal" (7): horizontal blinds mode (search vertically).
        self.position_mode_options = [
            ("nearest", 5),
            ("maintain", 3),
            ("vortex", 4),
            ("index", 2),
            ("reverse", 1),            
            ("vertical", 6),
            ("horizontal", 7),
            ("random", 0)
        ]
        self.position_mode_map = dict(self.position_mode_options)
        self.position_mode = tk.StringVar(value="maintain")

        # Define additional processing options.
        self.equalize = tk.BooleanVar(value=False)
        self.grayscale = tk.BooleanVar(value=False)

        # Create frames for UI elements and image display.
        self.left_frame = tk.Frame(master)
        self.left_frame.pack(side=tk.LEFT, padx=10, pady=10)
        self.right_frame = tk.Frame(master)
        self.right_frame.pack(side=tk.RIGHT, padx=10, pady=10)

        # Create UI elements in the left frame.
        tk.Button(self.left_frame, text="Load Image", command=self.load_file).pack(fill=tk.X)
        tk.Label(self.left_frame, text="Your URL").pack(anchor="w")
        tk.Entry(self.left_frame, textvariable=self.qr_url).pack(fill=tk.X)
        tk.Label(self.left_frame, text="Image Width (100-3000)").pack(anchor="w")
        # Spinbox enforces width limits; additional clamping is performed during processing.
        tk.Spinbox(self.left_frame, from_=100, to=3000, textvariable=self.output_width).pack(fill=tk.X)
        
        # Create a dedicated frame for the position mode selection.
        pos_frame = tk.Frame(self.left_frame)
        pos_frame.pack(fill=tk.X, pady=(5, 5))
        # Row 0: "Pattern" label and arrow buttons placed side by side.
        tk.Label(pos_frame, text="Pattern").grid(row=0, column=0, sticky="w")
        self.left_arrow = tk.Button(pos_frame, text="←", command=lambda: self.switch_position_mode(-1), width=2)
        self.left_arrow.grid(row=0, column=1, padx=(5, 0))
        self.right_arrow = tk.Button(pos_frame, text="→", command=lambda: self.switch_position_mode(1), width=2)
        self.right_arrow.grid(row=0, column=2, padx=(5, 0))
        # Row 1: Dropdown menu for position mode.
        self.position_menu = tk.OptionMenu(pos_frame, self.position_mode, *[opt for opt, _ in self.position_mode_options])
        self.position_menu.grid(row=1, column=0, columnspan=3, sticky="ew")
        pos_frame.columnconfigure(0, weight=1)


        # Threshold selection.
        tk.Label(self.left_frame, text="Threshold").pack(anchor="w")
        tk.OptionMenu(self.left_frame, self.threshold_mode, *[opt for opt, _ in self.threshold_mode_options]).pack(fill=tk.X)
        # Checkbuttons for Equalize and Grayscale options.
        tk.Checkbutton(self.left_frame, text="Equalize", variable=self.equalize,
                       command=self.schedule_process_image, anchor="w", justify="left").pack(fill=tk.X, anchor="w")
        tk.Checkbutton(self.left_frame, text="Grayscale", variable=self.grayscale,
                       command=self.schedule_process_image, anchor="w", justify="left").pack(fill=tk.X, anchor="w")
        # Contrast and gray settings.
        tk.Label(self.left_frame, text="Contrast").pack(anchor="w")
        tk.Scale(self.left_frame, variable=self.contrast_adjustment_factor, from_=0.5, to=2,
                 resolution=0.01, orient=tk.HORIZONTAL).pack(fill=tk.X)
        tk.Label(self.left_frame, text="Gray Threshold").pack(anchor="w")
        self.gray_thresh_scale = tk.Scale(self.left_frame, variable=self.gray_thresh, from_=0, to=255, orient=tk.HORIZONTAL)
        self.gray_thresh_scale.pack(fill=tk.X)
        tk.Label(self.left_frame, text="Gray Margin").pack(anchor="w")
        tk.Scale(self.left_frame, variable=self.gray_margin, from_=0, to=128, orient=tk.HORIZONTAL).pack(fill=tk.X)
        tk.Button(self.left_frame, text="Save Image", command=self.save_image).pack(fill=tk.X, pady=(10, 0))

        # Create a label in the right frame to display the output image.
        self.image_label = tk.Label(self.right_frame)
        self.image_label.pack()

        # Set up variable traces to trigger reprocessing when values change.
        self.qr_url.trace_add("write", lambda *args: self.schedule_process_image())
        self.output_width.trace_add("write", lambda *args: self.schedule_process_image())
        self.contrast_adjustment_factor.trace_add("write", lambda *args: self.schedule_process_image())
        self.gray_thresh.trace_add("write", lambda *args: self.schedule_process_image())
        self.gray_margin.trace_add("write", lambda *args: self.schedule_process_image())
        self.threshold_mode.trace_add("write", lambda *args: (self.update_gray_threshold_visibility(), self.schedule_process_image()))
        self.position_mode.trace_add("write", lambda *args: self.schedule_process_image())

        self.after_id = None
        self.nearest_rotation = 0
        self.update_gray_threshold_visibility()
        self.schedule_process_image()

    def switch_position_mode(self, direction):
        """
        Cycle through the position mode options.
        direction: -1 for previous, +1 for next.
        """
        options = [opt for opt, _ in self.position_mode_options]
        current = self.position_mode.get()
        try:
            index = options.index(current)
        except ValueError:
            index = 0
        new_index = (index + direction) % len(options)
        self.position_mode.set(options[new_index])
        self.schedule_process_image()

    def update_gray_threshold_visibility(self):
        """
        Enable or disable the gray threshold slider based on the threshold mode.
        """
        if self.threshold_mode.get() == "custom":
            self.gray_thresh_scale.config(state="normal", bg="white")
            self.gray_thresh_scale.configure(fg="black")
        else:
            self.gray_thresh_scale.config(state="disabled", bg="light gray")
            self.gray_thresh_scale.configure(fg="gray")

    def schedule_process_image(self):
        """
        Schedule the image processing to occur after 50ms.
        Cancel any previously scheduled processing.
        """
        if self.after_id is not None:
            self.master.after_cancel(self.after_id)
        self.after_id = self.master.after(50, self.process_image)

    def load_file(self):
        """
        Open a file dialog to load an image.
        """
        file_path = filedialog.askopenfilename(
            filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp;*.gif"), ("All Files", "*.*")]
        )
        if file_path:
            self.input_path = file_path
            try:
                self.original_img = Image.open(file_path)
            except:
                self.original_img = None
            self.schedule_process_image()

    def generate_vortex_order(self, n):
        """
        Generate a vortex ordering of coordinates for an n x n grid.
        """
        r = (n - 1) // 2
        c = (n - 1) // 2
        result = [(r, c)]
        step = 1
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        while len(result) < n * n:
            for d in directions[0:2]:
                for _ in range(step):
                    r += d[0]
                    c += d[1]
                    if 0 <= r < n and 0 <= c < n:
                        result.append((r, c))
            step += 1
            for d in directions[2:4]:
                for _ in range(step):
                    r += d[0]
                    c += d[1]
                    if 0 <= r < n and 0 <= c < n:
                        result.append((r, c))
            step += 1
        return result[:n * n]

    def find_nearest_candidate(self, target, candidates, grid_size):
        """
        Find the candidate with the minimum toroidal distance to the target.
        Remove the chosen candidate from the list.
        """
        if not candidates:
            return None
        best_distance = None
        best_candidates = []
        for cand in candidates:
            dr = cand[1][0] - target[0]
            dc = cand[1][1] - target[1]
            dist = dr * dr + dc * dc
            if best_distance is None or dist < best_distance:
                best_distance = dist
                best_candidates = [cand]
            elif dist == best_distance:
                best_candidates.append(cand)
        index = self.nearest_rotation % len(best_candidates)
        chosen = best_candidates[index]
        self.nearest_rotation += 1
        candidates.remove(chosen)
        return chosen

    def process_image(self):
        """
        Process the original image to produce the target image based on the QR code and user settings.
        Enforce that the output width is clamped between 100 and 3000 pixels.
        """
        try:
            thresh_mode = self.threshold_mode_map[self.threshold_mode.get()]
            pos_mode = self.position_mode_map[self.position_mode.get()]

            # Generate the QR code image from the provided URL.
            qr = qrcode.QRCode(version=3, box_size=1, border=0)
            qr.add_data(self.qr_url.get())
            qr.make(fit=True)
            qr_img = qr.make_image(fill='black', back_color='white').convert('L')
            qr_array = np.array(qr_img, dtype=bool)
            qr_size = qr_array.shape[0]

            # Clamp the output width to the range [100, 3000].
            width_value = self.output_width.get()
            out_width = max(100, min(width_value, 3000))
            cell_size = out_width // qr_size

            if self.original_img is None:
                raise Exception("No image loaded")
            # Prepare the original image.
            img = self.original_img.copy().convert('RGB')
            width, height = img.size
            if width != height:
                min_side = min(width, height)
                left = (width - min_side) // 2
                top = (height - min_side) // 2
                img = img.crop((left, top, left + min_side, top + min_side))
            # Resize the image to match the QR code grid.
            img = img.resize((qr_size * cell_size, qr_size * cell_size))
            if self.equalize.get():
                img = ImageOps.equalize(img)
            img_gray = img.convert('L')

            # Determine the gray threshold using the selected threshold mode.
            gray_thresh = self.gray_thresh.get()
            if thresh_mode == 1:
                stat = ImageStat.Stat(img_gray)
                gray_thresh = max(stat.mean[0], 1e-6)
            elif thresh_mode == 2:
                arr = np.array(img_gray)
                gray_thresh = max(float(np.median(arr)), 1e-6)
            gray_margin = self.gray_margin.get()
            brightness_dark_max = max(0, min(255, gray_thresh - gray_margin))
            brightness_bright_min = max(0, min(255, gray_thresh + gray_margin))

            # Split the image into cells corresponding to the QR code grid.
            cells = []
            for row in range(qr_size):
                for col in range(qr_size):
                    box = (col * cell_size, row * cell_size, (col + 1) * cell_size, (row + 1) * cell_size)
                    cell_rgb = img.crop(box)
                    cell_gray = img_gray.crop(box)
                    mean_val = max(ImageStat.Stat(cell_gray).mean[0], 1e-6)
                    cells.append((cell_rgb, (row, col), mean_val))
            dark_cells = [c for c in cells if c[2] < gray_thresh]
            bright_cells = [c for c in cells if c[2] >= gray_thresh]
            contrast_factor = self.contrast_adjustment_factor.get()

            # Process cells in parallel to adjust brightness and contrast.
            with concurrent.futures.ProcessPoolExecutor(max_workers=os.cpu_count() // 2) as executor:
                dark_cells = list(executor.map(process_dark_cell, dark_cells,
                                               repeat(gray_thresh),
                                               repeat(brightness_dark_max),
                                               repeat(contrast_factor)))
                bright_cells = list(executor.map(process_bright_cell, bright_cells,
                                                 repeat(gray_thresh),
                                                 repeat(brightness_bright_min),
                                                 repeat(contrast_factor)))

            # Create the target image.
            target_img = Image.new('RGB', (qr_size * cell_size, qr_size * cell_size))

            # Process based on the selected position mode.
            if pos_mode == 3:
                # Nearest: standard mapping using original grid positions.
                filled_positions = set()
                for row in range(qr_size):
                    for col in range(qr_size):
                        target_bright = qr_array[row, col]
                        source_cells = bright_cells if target_bright else dark_cells
                        exact_match = next((c for c in source_cells if c[1] == (row, col)), None)
                        if exact_match:
                            target_img.paste(exact_match[0], (col * cell_size, row * cell_size))
                            filled_positions.add((row, col))
                remaining_dark = [c for c in dark_cells if c[1] not in filled_positions]
                remaining_bright = [c for c in bright_cells if c[1] not in filled_positions]
                if not remaining_dark:
                    remaining_dark = dark_cells
                if not remaining_bright:
                    remaining_bright = bright_cells
                dark_index = 0
                bright_index = 0
                for row in range(qr_size):
                    for col in range(qr_size):
                        if (row, col) in filled_positions:
                            continue
                        if qr_array[row, col]:
                            cell = remaining_bright[bright_index][0]
                            bright_index = (bright_index + 1) % len(remaining_bright)
                        else:
                            cell = remaining_dark[dark_index][0]
                            dark_index = (dark_index + 1) % len(remaining_dark)
                        target_img.paste(cell, (col * cell_size, row * cell_size))
            elif pos_mode == 4:
                # vortex mode: fill cells in a vortex order.
                vortex_coords = self.generate_vortex_order(qr_size)
                dark_positions = [pos for pos in vortex_coords if not qr_array[pos[0], pos[1]]]
                bright_positions = [pos for pos in vortex_coords if qr_array[pos[0], pos[1]]]
                for i, pos in enumerate(dark_positions):
                    candidate = dark_cells[i % len(dark_cells)][0]
                    target_img.paste(candidate, (pos[1] * cell_size, pos[0] * cell_size))
                for i, pos in enumerate(bright_positions):
                    candidate = bright_cells[i % len(bright_cells)][0]
                    target_img.paste(candidate, (pos[1] * cell_size, pos[0] * cell_size))
            elif pos_mode == 5:
                # Nearest candidate search: find the closest cell to each QR cell.
                bright_candidates = bright_cells.copy()
                dark_candidates = dark_cells.copy()
                for row in range(qr_size):
                    for col in range(qr_size):
                        target_pos = (row, col)
                        if qr_array[row, col]:
                            candidate = self.find_nearest_candidate(target_pos, bright_candidates, qr_size)
                            if candidate is None:
                                candidate = choice(bright_cells)
                            target_img.paste(candidate[0], (col * cell_size, row * cell_size))
                        else:
                            candidate = self.find_nearest_candidate(target_pos, dark_candidates, qr_size)
                            if candidate is None:
                                candidate = choice(dark_cells)
                            target_img.paste(candidate[0], (col * cell_size, row * cell_size))
            elif pos_mode == 6:
                # "Vertical" bleed mode: for each QR cell, consider only candidates in the same row.
                for row in range(qr_size):
                    for col in range(qr_size):
                        target_bright = qr_array[row, col]
                        candidate_pool = bright_cells if target_bright else dark_cells
                        row_candidates = [c for c in candidate_pool if c[1][0] == row]
                        candidate = None
                        # Look leftward from the current column, wrapping around.
                        for i in range(qr_size):
                            search_col = (col - i) % qr_size
                            found = [c for c in row_candidates if c[1][1] == search_col]
                            if found:
                                candidate = found[0]
                                break
                        if candidate is None and candidate_pool:
                            candidate = choice(candidate_pool)
                        if candidate:
                            target_img.paste(candidate[0], (col * cell_size, row * cell_size))
            elif pos_mode == 7:
                # "Horizontal" blinds mode: for each QR cell, consider only candidates in the same column.
                for row in range(qr_size):
                    for col in range(qr_size):
                        target_bright = qr_array[row, col]
                        candidate_pool = bright_cells if target_bright else dark_cells
                        col_candidates = [c for c in candidate_pool if c[1][1] == col]
                        candidate = None
                        # Look upward from the current row, wrapping around.
                        for i in range(qr_size):
                            search_row = (row - i) % qr_size
                            found = [c for c in col_candidates if c[1][0] == search_row]
                            if found:
                                candidate = found[0]
                                break
                        if candidate is None and candidate_pool:
                            candidate = choice(candidate_pool)
                        if candidate:
                            target_img.paste(candidate[0], (col * cell_size, row * cell_size))
            else:
                # Fallback: simple random or reversed mapping.
                remaining_dark = dark_cells.copy()
                remaining_bright = bright_cells.copy()
                if pos_mode == 1:
                    remaining_dark = remaining_dark[::-1]
                    remaining_bright = remaining_bright[::-1]
                if pos_mode == 0:
                    shuffle(remaining_dark)
                    shuffle(remaining_bright)
                dark_index = 0
                bright_index = 0
                for row in range(qr_size):
                    for col in range(qr_size):
                        if qr_array[row, col]:
                            cell = remaining_bright[bright_index][0]
                            bright_index = (bright_index + 1) % len(remaining_bright)
                        else:
                            cell = remaining_dark[dark_index][0]
                            dark_index = (dark_index + 1) % len(remaining_dark)
                        target_img.paste(cell, (col * cell_size, row * cell_size))
            # Convert to grayscale if the option is selected.
            if self.grayscale.get():
                target_img = target_img.convert('L').convert('RGB')
        except Exception as e:
            # On error, produce a black image of the target width.
            target_img = Image.new('RGB', (out_width, out_width), 'black')
            print()
            print(e, end="\r")
        self.target_img = target_img
        self.display_image(target_img)

    def display_image(self, img):
        """
        Display the target image in the right frame.
        """
        self.tk_img = ImageTk.PhotoImage(img)
        self.image_label.config(image=self.tk_img)

    def save_image(self):
        """
        Open a save dialog and save the target image.
        """
        base_name = os.path.splitext(os.path.basename(self.input_path))[0] if self.input_path else "output"
        initialfile = f"{base_name}_qr_encoded.png"
        file_path = filedialog.asksaveasfilename(
            initialdir="outages",
            initialfile=initialfile,
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("All Files", "*.*")]
        )
        if file_path:
            try:
                self.target_img.save(file_path)
            except:
                pass

if __name__ == "__main__":

    print("QRfypx working. Keep this window opened . . .", end="\r")

    import multiprocessing
    multiprocessing.freeze_support()  # Prevent child processes from re-running the GUI code.
    root = tk.Tk()
    app = App(root)
    root.mainloop()