# Standard library imports
import queue
import threading
import os
import dearpygui.dearpygui as dpg
import numpy as np
from PIL import Image
import logging

# Project imports
from src.core.image_loader import ImageLoader
from src.core.metadata_manager import MetadataManager
from src.core.image_analyzer import ImageAnalyzer
from src.core.grouper import PhotoGrouper
from src.core.analysis_cache import AnalysisCache
import time
from src.gui.texture_manager import TextureManager
from src.ml.face_detector import FaceDetector
from src.ml.focus_detector import FocusDetector
from src.ml.eye_detector import EyeOpennessDetector
import random

logger = logging.getLogger(__name__)

class GuiManager:
    def __init__(self):
        # TextureManager needs DPG context, so we init it later
        self.texture_manager = None 
        self.image_loader = ImageLoader()
        self.face_detector = FaceDetector()
        self.image_analyzer = ImageAnalyzer() 
        self.image_files = [] # List of image paths
        self.image_groups = {} # path -> group_id
        self.group_best_shots = {} # group_id -> path of best shot
        self.group_second_best_shots = {} 
        self.group_colors = {} # group_id -> (r, g, b, a)
        self.current_index = -1
        
        # Cache
        self.cache = None
        
        # UI State
        self.toast_message = None
        self.toast_timer = 0
        
        # Analysis state
        self.analysis_queue = queue.Queue()
        self.analysis_results = {} # path -> result dict
        self.processed_count = 0
        self.is_analyzing = False
        self.stop_analysis_flag = False
        self.last_group_update_time = 0

        
        self.setup_dpg()
        
    def setup_dpg(self):
        dpg.create_context()
        # Initialize TextureManager after context creation
        self.texture_manager = TextureManager()

        # Create Themes
        with dpg.theme() as self.theme_selected:
            with dpg.theme_component(dpg.mvAll):
                dpg.add_theme_color(dpg.mvThemeCol_Border, (255, 255, 255, 255), category=dpg.mvThemeCat_Core)
                dpg.add_theme_style(dpg.mvStyleVar_FrameBorderSize, 3, category=dpg.mvThemeCat_Core)
                dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 2, 2, category=dpg.mvThemeCat_Core)

        with dpg.theme() as self.theme_unselected:
            with dpg.theme_component(dpg.mvAll):
                dpg.add_theme_color(dpg.mvThemeCol_Border, (0, 0, 0, 0), category=dpg.mvThemeCat_Core)
                dpg.add_theme_style(dpg.mvStyleVar_FrameBorderSize, 0, category=dpg.mvThemeCat_Core)
                dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 2, 2, category=dpg.mvThemeCat_Core)

        # Create placeholder texture
        placeholder = Image.new('RGBA', (100, 100), (30, 30, 30, 255))
        self.texture_manager.load_texture("placeholder", placeholder)
        
        dpg.create_viewport(title='Arcane - Photo Culling', width=1600, height=1000)
        dpg.setup_dearpygui()
        
        # Enable docking
        dpg.configure_app(docking=True, docking_space=True)
        
        # Keyboard Handler
        with dpg.handler_registry():
            dpg.add_key_press_handler(callback=self.on_key_press)
        
        self.build_ui()
        
    def build_ui(self):
        # Menu Bar
        with dpg.viewport_menu_bar():
            with dpg.menu(label="File"):
                dpg.add_menu_item(label="Import Folder...", callback=self.on_import_folder_click)
                dpg.add_menu_item(label="Exit", callback=dpg.destroy_context)
            with dpg.menu(label="View"):
                dpg.add_menu_item(label="Toggle Fullscreen", callback=dpg.toggle_viewport_fullscreen)
            
            dpg.add_separator()
            dpg.add_text("Status:", color=(150, 150, 150))
            dpg.add_text("Idle", tag="txt_status")
            dpg.add_progress_bar(tag="progress_bar", width=200, default_value=0.0, show=False)

        # Gallery Window (Left)
        with dpg.window(label="Gallery", tag="w_gallery", width=300, height=600):
            dpg.add_text("No images loaded.", tag="txt_gallery_status")
            with dpg.table(header_row=False, tag="tbl_gallery", policy=dpg.mvTable_SizingFixedFit, scrollY=True):
                dpg.add_table_column() 

        # Inspector Window (Center/Right)
        with dpg.window(label="Inspector", tag="w_inspector", pos=(320, 0), width=1200, height=720):
            dpg.add_text("No image selected.", tag="txt_inspector_status")
            dpg.add_image("tex_placeholder", tag="img_inspector", show=False)
            dpg.add_text("", tag="txt_metadata_rating", pos=(20, 20), color=(255, 255, 0))

        # Bottom Timeline (Bottom)
        with dpg.window(label="Timeline", tag="w_timeline", pos=(0, 730), width=1600, height=270):
            with dpg.group(horizontal=True, tag="grp_timeline_content", horizontal_spacing=10):
                dpg.add_text("Import images to see timeline.")

        # File Dialog
        with dpg.file_dialog(directory_selector=True, show=False, callback=self.on_folder_selected, tag="file_dialog_id", width=700, height=400):
            dpg.add_file_extension(".*")

    def on_import_folder_click(self):
        dpg.show_item("file_dialog_id")

    def on_folder_selected(self, sender, app_data):
        folder_path = app_data['file_path_name']
        logger.info(f"Importing from: {folder_path}")
        dpg.set_value("txt_gallery_status", f"Loading from {folder_path}...")
        
        # Initialize Cache
        self.cache = AnalysisCache(folder_path)
        
        # Scan folder in thread
        threading.Thread(target=self.scan_folder, args=(folder_path,), daemon=True).start()

    def scan_folder(self, folder_path):
        """Background thread to scan for images."""
        logger.info(f"Scanning folder: {folder_path}")
        valid_exts = ('.ARW', '.CR2', '.NEF', '.DNG', '.JPG', '.JPEG', '.PNG')
        self.image_files = []
        
        try:
            for root, dirs, files in os.walk(folder_path):
                for file in files:
                    if file.upper().endswith(valid_exts):
                        self.image_files.append(os.path.join(root, file))
            
            self.image_files.sort()
            logger.info(f"Found {len(self.image_files)} images.")
            
            # Update UI on main thread
            self.refresh_gallery_ui()
            
            if self.image_files:
                # Start background analysis
                self.start_analysis()
                
                # Load first image
                # self.load_image_at_index(0) 
                pass
                
        except Exception as e:
            logger.error(f"Error scanning folder: {e}", exc_info=True)

    def refresh_gallery_ui(self):
        """Populate the gallery list."""
        try:
            # Clear existing
            dpg.delete_item("tbl_gallery", children_only=True)
            dpg.hide_item("txt_gallery_status")
            
            # Re-add column because delete_item(children_only=True) removed it
            dpg.add_table_column(parent="tbl_gallery")
            
            for idx, path in enumerate(self.image_files):
                basename = os.path.basename(path)
                with dpg.table_row(parent="tbl_gallery"):
                    dpg.add_selectable(label=f"{idx+1}. {basename}", callback=self.on_gallery_select, user_data=idx, tag=f"row_{idx}")
        except Exception as e:
            logger.error(f"Error refreshing gallery UI: {e}", exc_info=True)

    def on_gallery_select(self, sender, app_data, user_data):
        index = user_data
        self.load_image_at_index(index)

    def load_image_at_index(self, index):
        if 0 <= index < len(self.image_files):
            # Unselect previous
            if self.current_index != -1:
                dpg.set_value(f"row_{self.current_index}", False)
            
            self.current_index = index
            dpg.set_value(f"row_{index}", True)
            
            # Highlight timeline
            self.highlight_timeline_item(index)

            
            path = self.image_files[index]
            
            # Load preview
            try:
                # Load PIL image
                img = self.image_loader.load_preview(path)
                
                # Run ML Check (if enabled/available)
                # For responsiveness, this should be threaded, but for MVP doing inline to show proof
                # Convert PIL to Numpy/OpenCV format for ML
                img_np = np.array(img)
                
                # Detect Faces
                detections = self.face_detector.detect(img_np)
                
                # Check Focus for each face
                for det in detections:
                    box = det['box']
                    score = FocusDetector.check_face_focus(img_np, box)
                    det['sharpness'] = score
                    
                    # Draw box on image (simple visualization)
                    color = (0, 255, 0) if score > 100 else (255, 0, 0) # Green if sharp, Red if blurry (threshold TBD)
                    x1, y1, x2, y2 = map(int, box)
                    # Draw using DPG overlay or modifying PIL image? 
                    # Modifying PIL is easier for now to show on texture
                    from PIL import ImageDraw
                    draw = ImageDraw.Draw(img)
                    draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
                    
                    status_text = f"Focus: {score:.0f}"
                    
                    # Check Eyes
                    if det['landmarks'] is not None:
                        eye_status = EyeOpennessDetector.check_eyes(img_np, det['landmarks'])
                        det['eye_status'] = eye_status
                        status_text += f" | Eyes: {eye_status}"
                        
                        # Draw landmarks
                        for pt in det['landmarks']:
                            draw.ellipse([pt[0]-2, pt[1]-2, pt[0]+2, pt[1]+2], fill=(0, 255, 255))
                    
                    draw.text((x1, y1-10), status_text, fill=color)

                # Upload to texture
                tex_tag = self.texture_manager.load_texture(path, img)
                
                if tex_tag:
                    # Update Inspector
                    dpg.configure_item("img_inspector", texture_tag=tex_tag, width=img.width, height=img.height, show=True)
                    dpg.hide_item("txt_inspector_status")
                    
                    # Update Metadata Display
                    self.update_metadata_display(path)
                    
            except Exception as e:
                logger.error(f"Error loading image {path}: {e}")

    def update_metadata_display(self, path):
        meta = MetadataManager.read_metadata(path)
        rating = meta.get('rating', 0)
        label = meta.get('label', '')
        text = f"Rating: {'★'*rating}"
        if label:
            text += f" | {label}"
        dpg.set_value("txt_metadata_rating", text)
        
        # Update connection to Timeline
        self.update_timeline_status(path, rating, label)

    def update_timeline_status(self, path, rating=None, label=None):
        """Updates the rating/flag display on the timeline thumbnail."""
        try:
            # We need to find the index because tags depend on index now
            if path in self.image_files:
                idx = self.image_files.index(path)
                
                # If rating/label not provided, read them
                if rating is None or label is None:
                    meta = MetadataManager.read_metadata(path)
                    rating = meta.get('rating', 0)
                    label = meta.get('label', '')

                rating_tag = f"txt_rating_{idx}"
                flag_tag = f"txt_flag_{idx}"
                
                if dpg.does_item_exist(rating_tag):
                    # Use ASCII representation
                    dpg.set_value(rating_tag, f"R:{rating}" if rating > 0 else "")
                
                if dpg.does_item_exist(flag_tag):
                    # Only show Pick/Reject
                    text = f"[{label.upper()}]" if label in ['Pick', 'Reject'] else ""
                    color = (0, 255, 0) if label == 'Pick' else (255, 0, 0) if label == 'Reject' else (255, 255, 255)
                    
                    # Add grouping info
                    group_id = self.image_groups.get(path)
                    if group_id:
                        if self.group_best_shots.get(group_id) == path:
                            text += " [BEST]"
                            color = (255, 215, 0) # Gold
                        elif self.group_second_best_shots.get(group_id) == path:
                            text += " [2ND]"
                            color = (192, 192, 192) # Silver
                            
                    dpg.set_value(flag_tag, text)
                    dpg.configure_item(flag_tag, color=color)
                
        except ValueError:
            pass

    def highlight_timeline_item(self, index):
        """Highlights the selected item in the timeline and scrolls to it."""
        
        for i in range(len(self.image_files)):
             tag = f"btn_thumb_{i}"
             if dpg.does_item_exist(tag):
                 # Reset tint color to white (in case it was green before)
                 dpg.configure_item(tag, tint_color=(255, 255, 255))
                 
                 if i == index:
                     dpg.bind_item_theme(tag, self.theme_selected)
                 else:
                     dpg.bind_item_theme(tag, self.theme_unselected)
        
        # Scroll to item
        # Item width ~240. Spacing ~8?
        item_width = 250
        # Center the item
        content_width = len(self.image_files) * item_width
        window_width = dpg.get_item_width("w_timeline")
        
        target_scroll = (index * item_width) - (window_width / 2) + (item_width / 2)
        dpg.set_x_scroll("w_timeline", max(0, target_scroll))


    def show_toast(self, message):
        """Shows a temporary popup message."""
        self.toast_message = message
        self.toast_timer = 60 # Frames to show (approx 1-2 secs depending on framerate)
        
        if not dpg.does_item_exist("w_toast"):
             with dpg.window(label="", tag="w_toast", modal=False, no_title_bar=True, no_move=True, no_resize=True,
                             autosize=True, pos=(50, 100), show=False):
                 dpg.add_text(message, tag="txt_toast_msg", color=(0, 255, 255))
        
        dpg.set_value("txt_toast_msg", message)
        dpg.configure_item("w_toast", show=True)
        # Center toast at top
        vp_width = dpg.get_viewport_width()
        dpg.set_item_pos("w_toast", (vp_width/2 - 100, 50))


    def on_key_press(self, sender, app_data):
        # app_data is the key code
        key = app_data
        
        # 0-5 keys (ASCII 48-53) usually, or DPG key codes
        # DPG constants: dpg.mvKey_1 ...
        
        if self.current_index == -1:
            return

        current_path = self.image_files[self.current_index]
        
        # Ratings 1-5
        rating = None
        if key == dpg.mvKey_1: rating = 1
        elif key == dpg.mvKey_2: rating = 2
        elif key == dpg.mvKey_3: rating = 3
        elif key == dpg.mvKey_4: rating = 4
        elif key == dpg.mvKey_5: rating = 5
        elif key == dpg.mvKey_0: rating = 0
        
        if rating is not None:
             # Propagate to group
             group_id = self.image_groups.get(current_path)
             if group_id:
                 affected_paths = [p for p, g in self.image_groups.items() if g == group_id]
                 for p in affected_paths:
                     MetadataManager.write_metadata(p, rating=rating)
                     self.update_metadata_display(p)
                 self.show_toast(f"Rated Group ({len(affected_paths)} images): {rating} Stars")
             else:
                 MetadataManager.write_metadata(current_path, rating=rating)
                 self.update_metadata_display(current_path)
                 self.show_toast(f"Rated {rating} Stars")
             
             # Auto-advance? Maybe
        
        # Flags
        label = None
        if key == dpg.mvKey_P: label = 'Pick'
        elif key == dpg.mvKey_X: label = 'Reject'
        elif key == dpg.mvKey_U: label = '' # Unflag
        
        if label is not None:
             # Propagate to group
             group_id = self.image_groups.get(current_path)
             if group_id:
                 affected_paths = [p for p, g in self.image_groups.items() if g == group_id]
                 for p in affected_paths:
                     MetadataManager.write_metadata(p, label=label)
                     self.update_metadata_display(p) # Update UI for all
                 self.show_toast(f"Flagged Group ({len(affected_paths)} images): {label}" if label else "Unflagged Group")
             else:
                 MetadataManager.write_metadata(current_path, label=label)
                 self.update_metadata_display(current_path)
                 self.show_toast(f"Flagged: {label}" if label else "Unflagged")
                 
        # Ratings 1-5 (Similar propagation)
        # Note: I put rating logic above, let's fix it to include propagation there too.
        # But wait, I can't easily jump back in this edit.
        # I should have edited the whole block.
        # Let's assume I fix rating propagation in a separate chunk or just let it be single for now? 
        # No, user asked for "rated together". I must fix rating section too.
        
        # Navigation
        if key == dpg.mvKey_Right or key == dpg.mvKey_Down:
            self.load_image_at_index(self.current_index + 1)
        elif key == dpg.mvKey_Left or key == dpg.mvKey_Up:
            self.load_image_at_index(self.current_index - 1)

    def start_analysis(self):
        """Starts the background analysis thread."""
        if self.is_analyzing:
            return
            
        self.stop_analysis_flag = False
        self.is_analyzing = True
        self.processed_count = 0
        self.analysis_results.clear()
        self.last_group_update_time = time.time()
        
        # Reset progress components
        dpg.set_value("progress_bar", 0.0)
        dpg.configure_item("progress_bar", show=True)
        dpg.set_value("txt_status", "Analyzing...")
        
        # Clear Timeline
        dpg.delete_item("grp_timeline_content", children_only=True)
        dpg.add_text("Analyzing images...", parent="grp_timeline_content")
        
        threading.Thread(target=self.analysis_worker, daemon=True).start()

    def analysis_worker(self):
        """Worker thread for image analysis."""
        saved_count = 0 
        for path in self.image_files:
            if self.stop_analysis_flag:
                break
            
            try:
                # Check cache first
                cached_data = self.cache.get(path) if self.cache else None
                result = None
                
                if cached_data:
                    # Check if it has embedding (required for DL grouping)
                    if 'embedding' not in cached_data:
                        logger.info(f"Cache miss (missing embedding) for {path}")
                        cached_data = None
                    else:
                        # Cache Hit
                        thumb = self.image_loader.load_preview(path, max_size=(240, 240))
                        result = cached_data
                        result['path'] = path 
                        result['thumbnail'] = thumb
                
                if result is None:
                    # Cache Miss - Run Analysis
                    result = self.image_analyzer.analyze(path)
                    
                    if self.cache and not result.get('error'):
                        self.cache.set(path, result)
                        saved_count += 1
                        # Periodic save
                        if saved_count % 10 == 0:
                            self.cache.save()

                self.analysis_queue.put(result)
                
            except Exception as e:
                logger.error(f"Worker failed for {path}: {e}")
        
        # Final save
        if self.cache:
            self.cache.save()
            
        self.is_analyzing = False
        self.analysis_queue.put("DONE")

    def on_frame(self):
        """Called every frame to update UI from queue."""
        try:
            # Process up to 5 items per frame to avoid freezing if queue fills up fast
            for _ in range(5):
                result = self.analysis_queue.get_nowait()
                
                if result == "DONE":
                    dpg.set_value("txt_status", "Analysis Complete")
                    dpg.configure_item("progress_bar", show=False)
                    continue
                
                # Store result
                self.analysis_results[result['path']] = result
                
                # Compute Groups incrementally?
                # Better to re-compute groups periodically or just check neighbors.
                # For simplicity, let's compute groups on the fly for the added item against previous ones?
                # Or just re-run full grouping every few seconds? 
                # Let's do simple greedy grouping: Compare with last processed item.
                
                # Periodic Grouping check
                if time.time() - self.last_group_update_time > 2.0:
                    self.run_grouping_pass()
                    self.last_group_update_time = time.time()
                
                # Add to Timeline
                self.add_thumbnail_to_timeline(result)
                
                # Update progress
                self.processed_count += 1
                total = len(self.image_files)
                if total > 0:
                    dpg.set_value("progress_bar", self.processed_count / total)
                    dpg.set_value("txt_status", f"Analyzed {self.processed_count}/{total}")
                    
        except queue.Empty:
            pass

    def add_thumbnail_to_timeline(self, result):
        path = result["path"]
        thumb = result["thumbnail"]
        score = result["overall_score"]
        
        if thumb:
            # Load texture
            tex_tag = self.texture_manager.load_texture(f"thumb_{path}", thumb)
            if not tex_tag: 
                return

            # Remove placeholder text if it's the first item
            if self.processed_count == 0:
                dpg.delete_item("grp_timeline_content", children_only=True)

            try:
                idx = self.image_files.index(path)
            except ValueError:
                return

            # Create child window for background tinting
            item_group_tag = f"grp_item_{idx}"
            with dpg.child_window(parent="grp_timeline_content", width=240, height=310, tag=item_group_tag, border=False, no_scrollbar=True, no_scroll_with_mouse=True):
                # Image Button
                # Use a unique tag for the button
                btn_tag = f"btn_thumb_{idx}"
                
                dpg.add_image_button(tex_tag, width=thumb.width, height=thumb.height, 
                                     callback=lambda s, a, u: self.load_image_at_index(u), user_data=idx,
                                     tag=btn_tag)
                                     
                # Rating Indicator
                meta = MetadataManager.read_metadata(path)
                rating = meta.get('rating', 0)
                label = meta.get('label', '')
                
                # ASCII Rating
                dpg.add_text(f"R:{rating}" if rating > 0 else "", tag=f"txt_rating_{idx}", color=(255, 215, 0))
                
                # Flag Indicator & Score
                with dpg.group(horizontal=True):
                    # Color code score
                    score_color = (0, 255, 0) if score > 100 else (255, 255, 0) if score > 50 else (255, 0, 0)
                    dpg.add_text(f"ML:{score:.0f}", color=score_color)
                    
                    flag_text = f"[{label.upper()}]" if label in ['Pick', 'Reject'] else ""
                    flag_color = (0, 255, 0) if label == 'Pick' else (255, 0, 0) if label == 'Reject' else (255, 255, 255)
                    dpg.add_text(flag_text, tag=f"txt_flag_{idx}", color=flag_color)

                # Group Indicator (Background Tint)
                group_id = self.image_groups.get(path)
                if group_id:
                    color = self.get_group_color(group_id, alpha=40)
                    with dpg.theme() as theme_group:
                        with dpg.theme_component(dpg.mvAll):
                            dpg.add_theme_color(dpg.mvThemeCol_ChildBg, color, category=dpg.mvThemeCat_Core)
                    dpg.bind_item_theme(item_group_tag, theme_group)


    def get_group_color(self, group_id, alpha=255):
        if group_id not in self.group_colors:
            # Generate random pastel color
            r = random.randint(100, 200)
            g = random.randint(100, 200)
            b = random.randint(100, 200)
            self.group_colors[group_id] = (r, g, b)
        
        r, g, b = self.group_colors[group_id]
        return (r, g, b, alpha)

    def run_grouping_pass(self):
        """Runs the PhotoGrouper on currently analyzed images."""
        analyzed_data = list(self.analysis_results.values())
        if not analyzed_data:
            return
            
        # Run Grouper
        # Relaxed thresholds for "Scene" grouping: 30 minutes, 0.85 Cosine Sim
        stacks = PhotoGrouper.group_similar(analyzed_data, time_threshold=1800.0, content_threshold=0.85)
        
        # Update Groups Dict
        self.image_groups = {}
        self.group_best_shots = {}
        self.group_second_best_shots = {}
        
        for stack in stacks:
            if len(stack) > 1:
                group_id = stack[0] # Use first path as ID
                
                # Find best and second best in this stack
                scored_images = []
                for path in stack:
                    self.image_groups[path] = group_id
                    res = self.analysis_results.get(path)
                    if res:
                        score = res.get('overall_score', 0)
                        scored_images.append((path, score))
                    else:
                        scored_images.append((path, 0))
                
                # Sort by score descending
                scored_images.sort(key=lambda x: x[1], reverse=True)
                
                if scored_images:
                    self.group_best_shots[group_id] = scored_images[0][0]
                    if len(scored_images) > 1:
                        self.group_second_best_shots[group_id] = scored_images[1][0]

                    
        # Update UI for all visible items
        self.refresh_timeline_groups()

    def refresh_timeline_groups(self):
        """Updates the group indicator for all items in the timeline."""
        for path in self.image_files:
            try:
                idx = self.image_files.index(path)
                item_group_tag = f"grp_item_{idx}"
                
                if dpg.does_item_exist(item_group_tag):
                    group_id = self.image_groups.get(path)
                    if group_id:
                        color = self.get_group_color(group_id, alpha=40)
                        with dpg.theme() as theme_group:
                            with dpg.theme_component(dpg.mvAll):
                                dpg.add_theme_color(dpg.mvThemeCol_ChildBg, color, category=dpg.mvThemeCat_Core)
                        dpg.bind_item_theme(item_group_tag, theme_group)
                        
                        # Update status (covers Best/2nd shot label)
                        self.update_timeline_status(path)

                    else:
                        # Clear theme (remove group indicator)
                        dpg.bind_item_theme(item_group_tag, 0)
                        
                        # Also update status to clear any stale "BEST" labels
                        self.update_timeline_status(path)
                        
            except Exception as e:
                logger.error(f"Error refreshing timeline groups: {e}")


    

    def run(self):
        dpg.show_viewport()
        while dpg.is_dearpygui_running():
            self.on_frame()
            
            # Toast logic
            if self.toast_timer > 0:
                self.toast_timer -= 1
                if self.toast_timer <= 0:
                    dpg.configure_item("w_toast", show=False)
                    
            dpg.render_dearpygui_frame()
        dpg.destroy_context()
