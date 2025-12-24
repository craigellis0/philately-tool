import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
from pathlib import Path

# Core imports
from philately_tool import (
    extract_stamps,
    perform_search,
    get_db_connection,
    init_project,
    STAMPS_DIR,
    CFG
)

THUMB_SIZE = (128, 128)

# ==========================================================
# Helpers
# ==========================================================

def center_window(window, width, height):
    screen_width = window.winfo_screenwidth()
    screen_height = window.winfo_screenheight()
    x = (screen_width // 2) - (width // 2)
    y = (screen_height // 2) - (height // 2)
    window.geometry(f'{width}x{height}+{x}+{y}')

# ==========================================================
# Tooltip Helper
# ==========================================================

class Tooltip:
    def __init__(self, widget, text_func):
        self.widget = widget
        self.text_func = text_func
        self.tip = None

        widget.bind("<Enter>", self.show)
        widget.bind("<Leave>", self.hide)

    def show(self, event=None):
        if self.tip:
            return

        x = self.widget.winfo_rootx() + 15
        y = self.widget.winfo_rooty() + 15

        self.tip = tk.Toplevel(self.widget)
        self.tip.overrideredirect(True)
        self.tip.geometry(f"+{x}+{y}")

        ttk.Label(
            self.tip,
            text=self.text_func(),
            background="#ffffe0",
            relief="solid",
            borderwidth=1,
            padding=6,
            font=("Segoe UI", 9)
        ).pack()

    def hide(self, event=None):
        if self.tip:
            self.tip.destroy()
            self.tip = None

# ==========================================================
# Custom Search Dialog
# ==========================================================

class SearchDialog(tk.Toplevel):
    def __init__(self, parent, title="Search"):
        super().__init__(parent)
        self.result = None
        self.title(title)

        center_window(self, 260, 200)
        self.resizable(False, False)
        self.transient(parent)
        self.grab_set()

        ttk.Label(self, text="Enter text to search:").pack(pady=10)
        self.text = tk.Text(self, height=5)
        self.text.pack(padx=10, fill="both", expand=True)

        btns = ttk.Frame(self)
        btns.pack(pady=10)
        ttk.Button(btns, text="Search", command=self._ok).pack(side="left", padx=5)
        ttk.Button(btns, text="Cancel", command=self.destroy).pack(side="left", padx=5)

    def _ok(self):
        self.result = self.text.get("1.0", "end").strip()
        self.destroy()

# ==========================================================
# Task Dialog
# ==========================================================

class TaskDialog(tk.Toplevel):
    def __init__(self, parent, title):
        super().__init__(parent)
        self.cancelled = False
        self.title(title)

        center_window(self, 420, 150)
        self.resizable(False, False)
        self.transient(parent)
        self.grab_set()

        ttk.Label(self, text=title, font=("Segoe UI", 11, "bold")).pack(pady=10)
        self.status = ttk.Label(self, text="Starting...")
        self.status.pack()

        self.progress = ttk.Progressbar(self, length=320)
        self.progress.pack(pady=10)

        ttk.Button(self, text="Cancel", command=self._cancel).pack()

    def _cancel(self):
        self.cancelled = True
        self.status.config(text="Cancelling...")

    def set_status(self, text):
        self.after(0, lambda: self.status.config(text=text))

    def set_progress(self, cur, total):
        self.after(0, lambda: (
            self.progress.config(maximum=total),
            self.progress.config(value=cur)
        ))

    def close(self):
        self.after(0, self.destroy)

# ==========================================================
# Main App
# ==========================================================

class PhilatelyApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Philately AI Tool")

        center_window(self, 1150, 750)

        self.album_dirs = []
        self.thumbnails = []

        self._build_ui()
        self.refresh_status_bar()
        self.update_ui_state()

    # ------------------------------------------------------

    def _build_ui(self):
        bar = ttk.Frame(self, padding=5)
        bar.pack(fill="x")

        self.btn_init = ttk.Button(bar, text="Initialize DB", command=self.init_db)
        self.btn_init.pack(side="left")

        self.btn_add = ttk.Button(bar, text="Add Album Folder", command=self.add_folder)
        self.btn_add.pack(side="left")

        self.btn_extract = ttk.Button(bar, text="Start Extraction", command=self.start_extraction)
        self.btn_extract.pack(side="left")

        self.btn_search_text = ttk.Button(bar, text="Search Text", command=self.search_text)
        self.btn_search_text.pack(side="left")

        self.btn_search_img = ttk.Button(bar, text="Search Image", command=self.search_by_image)
        self.btn_search_img.pack(side="left")

        ttk.Label(bar, text="Results").pack(side="left", padx=5)
        self.ent_top = ttk.Entry(bar, width=4)
        self.ent_top.insert(0, CFG["default_top"])
        self.ent_top.pack(side="left")

        ttk.Label(bar, text="Distance").pack(side="left", padx=5)
        self.ent_dist = ttk.Entry(bar, width=4)
        self.ent_dist.insert(0, CFG["default_distance"])
        self.ent_dist.pack(side="left")

        self.status_label = ttk.Label(bar, text="Ready", font=("Segoe UI", 9, "bold"))
        self.status_label.pack(side="right")

        # Left
        left = ttk.LabelFrame(self, text="Albums")
        left.pack(side="left", fill="y", padx=5, pady=5)

        self.album_list = tk.Listbox(left, width=40)
        self.album_list.pack(fill="both", expand=True)
        # Right-click context menu to remove an accidentally added folder
        self.album_list.bind("<Button-3>", self._on_album_right_click)
        self._active_menu = None

        # Main
        main = ttk.Frame(self)
        main.pack(side="right", fill="both", expand=True)

        self.canvas = tk.Canvas(main)
        self.scroll = ttk.Scrollbar(main, command=self.canvas.yview)
        self.grid_frame = ttk.Frame(self.canvas)

        self.grid_frame.bind("<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )

        self.canvas.create_window((0, 0), window=self.grid_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scroll.set)

        self.canvas.pack(side="left", fill="both", expand=True)
        self.scroll.pack(side="right", fill="y")

    # ------------------------------------------------------

    def display_results(self, rows):
        for w in self.grid_frame.winfo_children():
            w.destroy()

        self.thumbnails.clear()

        for idx, row in enumerate(rows):
            img_path = STAMPS_DIR / row[2]
            if not img_path.exists():
                continue

            img = Image.open(img_path).resize(THUMB_SIZE)
            tk_img = ImageTk.PhotoImage(img)
            self.thumbnails.append(tk_img)

            f = ttk.Frame(self.grid_frame)
            f.grid(row=idx // 4, column=idx % 4, padx=5, pady=5)

            lbl = ttk.Label(f, image=tk_img)
            lbl.pack()

            ttk.Label(f, text=f"Dist: {row[3]:.3f}", font=("Segoe UI", 8)).pack()

            album, page, filename = row[0], row[1], img_path.name
            full_path = img_path.resolve()

            # Tooltip
            Tooltip(lbl, lambda a=album, p=page, fn=filename, fp=full_path:
                f"Album: {a}\nPage: {p}\nFile: {fn}\n\n{fp}"
            )

            # Context menu
            menu = tk.Menu(self, tearoff=0)
            menu.add_command(
                label="Copy full path to clipboard",
                command=lambda p=str(full_path): self.copy_to_clipboard(p)
            )

            lbl.bind("<Button-3>", lambda e, m=menu: m.tk_popup(e.x_root, e.y_root))

    # ------------------------------------------------------

    def copy_to_clipboard(self, text):
        self.clipboard_clear()
        self.clipboard_append(text)
        self.update()

    # ------------------------------------------------------

    def init_db(self):
        if Path(CFG["db_path"]).exists():
            if not messagebox.askyesno("Confirm", "Reset database?"):
                return
        init_project()
        self.refresh_status_bar()
        self.update_ui_state()

    def add_folder(self):
        d = filedialog.askdirectory()
        if d and d not in self.album_dirs:
            self.album_dirs.append(d)
            self.album_list.insert("end", d)

    def _on_album_right_click(self, event):
        idx = self.album_list.nearest(event.y)
        if idx is None:
            return
        # select the item under cursor
        self.album_list.selection_clear(0, 'end')
        self.album_list.selection_set(idx)

        # destroy any previously-open menu
        if getattr(self, "_active_menu", None):
            try:
                self._active_menu.destroy()
            except Exception:
                pass
            self._active_menu = None

        menu = tk.Menu(self, tearoff=0)
        menu.add_command(label="Remove Folder", command=lambda i=idx: self._remove_folder(i))

        # Track active menu so we can dismiss it when clicking elsewhere
        self._active_menu = menu

        def _dismiss_active_menu(event=None):
            if getattr(self, "_active_menu", None):
                try:
                    self._active_menu.destroy()
                except Exception:
                    pass
                self._active_menu = None
            try:
                self.unbind("<Button-1>")
            except Exception:
                pass

        # Bind a one-shot left-click on the root window to dismiss the menu
        self.bind("<Button-1>", _dismiss_active_menu)

        try:
            menu.tk_popup(event.x_root, event.y_root)
        finally:
            menu.grab_release()

    def _remove_folder(self, idx):
        try:
            path = self.album_list.get(idx)
        except Exception:
            return
        if not messagebox.askyesno("Confirm", f"Remove folder '{path}'?"):
            return
        if path in self.album_dirs:
            self.album_dirs.remove(path)
        self.album_list.delete(idx)
        # ensure any active context menu is closed after removal
        if getattr(self, "_active_menu", None):
            try:
                self._active_menu.destroy()
            except Exception:
                pass
            self._active_menu = None
            try:
                self.unbind("<Button-1>")
            except Exception:
                pass

    def start_extraction(self):
        dlg = TaskDialog(self, "Extracting")
        threading.Thread(target=self._run_extraction, args=(dlg,), daemon=True).start()

    def _run_extraction(self, dlg):
        for d in self.album_dirs:
            if dlg.cancelled:
                break
            extract_stamps(d, True, False, on_status=dlg.set_status, on_progress=dlg.set_progress)
        dlg.close()
        self.after(0, self.refresh_status_bar)

    def search_text(self):
        dlg = SearchDialog(self)
        self.wait_window(dlg)
        if dlg.result:
            try:
                top = int(self.ent_top.get())
            except Exception:
                top = CFG["default_top"]
            try:
                distance = float(self.ent_dist.get())
            except Exception:
                distance = CFG["default_distance"]

            rows = perform_search("text", dlg.result, top, distance)
            self.display_results(rows)

    def search_by_image(self):
        img = filedialog.askopenfilename()
        if img:
            try:
                top = int(self.ent_top.get())
            except Exception:
                top = CFG["default_top"]
            try:
                distance = float(self.ent_dist.get())
            except Exception:
                distance = CFG["default_distance"]

            rows = perform_search("image", img, top, distance)
            self.display_results(rows)

    def refresh_status_bar(self):
        db = Path(CFG["db_path"])
        if not db.exists():
            self.status_label.config(text="DB Not Found")
            return
        conn = get_db_connection()
        cur = conn.cursor()
        a = cur.execute("SELECT COUNT(DISTINCT album) FROM stamps").fetchone()[0]
        p = cur.execute("SELECT COUNT(DISTINCT page) FROM stamps").fetchone()[0]
        s = cur.execute("SELECT COUNT(*) FROM stamps").fetchone()[0]
        conn.close()
        self.status_label.config(text=f"Albums: {a} | Pages: {p} | Stamps: {s}")

    def update_ui_state(self):
        pass

# ==========================================================
# Run
# ==========================================================

if __name__ == "__main__":
    app = PhilatelyApp()
    app.mainloop()
