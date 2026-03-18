import inspect
import time
from typing import Any, Callable, Dict, List, Optional, Union

import cv2
import ipywidgets as widgets
import numpy as np
from IPython.display import display

from .utils import normalize_gray


def _array_to_bytes(img: np.ndarray, fmt: str = ".png") -> bytes:
    """Helper to convert numpy array to image bytes for ipywidgets.Image."""
    if img is None:
        return b""
    
    # Ensure image is valid for encoding
    img_disp = img
    if img.dtype != np.uint8:
        img_disp = normalize_gray(img)
    
    # Handle color/grayscale
    if len(img_disp.shape) == 2:
        pass # Grayscale is fine
    elif len(img_disp.shape) == 3 and img_disp.shape[2] == 3:
        # RGB to BGR for OpenCV encoding if needed, but usually we work in gray/RGB
        # Assuming input is RGB for display if 3 channels
        img_disp = cv2.cvtColor(img_disp, cv2.COLOR_RGB2BGR)
        
    _, encoded = cv2.imencode(fmt, img_disp)
    return encoded.tobytes()


class PipelineContext(dict):
    """
    Shared data container for the pipeline.
    Acts like a dictionary but can have helper methods.
    """
    def __getattr__(self, key):
        if key in self:
            return self[key]
        return None

    def reset(self, initial_image: np.ndarray):
        self.clear()
        self['original'] = initial_image
        self['image'] = initial_image.copy()  # The 'current' image flowing through
        self['log'] = [] # Execution log


class Step:
    """
    Represents a single processing block in the pipeline.
    """
    def __init__(
        self,
        name: str,
        func: Callable,
        input_keys: List[str] = None,  # Args to pull from context. e.g. ['image', 'mask']
        output_keys: List[str] = None, # Keys to update in context. e.g. ['image'] or ['mask']
        params: Dict[str, Union[widgets.Widget, Any]] = None, # Param name -> Widget or fixed value
        description: str = ""
    ):
        self.name = name
        self.func = func
        self.input_keys = input_keys if input_keys else ['image']
        self.output_keys = output_keys if output_keys else ['image']
        self.params = params if params else {}
        self.description = description
        
        self.enabled = True
        self.last_execution_time = 0.0
        
        # UI Elements
        self.widget_container = None
        self.image_widget = widgets.Image(format='png', width=300, height=300)
        self.controls_container = None
        self.checkbox = widgets.Checkbox(value=True, description="Active", indent=False)
        self.status_label = widgets.Label(value="")

        # Hook up checkbox
        self.checkbox.observe(self._on_enable_change, names='value')
        
        # Hook up params
        for p in self.params.values():
            if isinstance(p, widgets.Widget):
                p.observe(self._on_param_change, names='value')

        self.parent_pipeline = None

    def _on_enable_change(self, change):
        self.enabled = change['new']
        if self.parent_pipeline:
            self.parent_pipeline.refresh_layout()
            self.parent_pipeline.run()

    def _on_param_change(self, change):
        if self.enabled and self.parent_pipeline:
            self.parent_pipeline.run()

    def get_ui(self, layout_mode: str = 'fixed', image_scale: float = 1.0) -> Optional[widgets.Widget]:
        """
        Constructs the widget for this step.
        """
        # Update image size
        size = int(300 * image_scale)
        self.image_widget.width = str(size) + "px"
        self.image_widget.height = str(size) + "px" # Keep aspect ratio?

        if not self.enabled and layout_mode == 'dynamic':
            return None
        
        # Controls Row
        control_list = [self.checkbox]
        for name, widget_obj in self.params.items():
            if isinstance(widget_obj, widgets.Widget):
                # Add a small label or description if needed, usually widget has description
                control_list.append(widget_obj)
        
        self.controls_container = widgets.VBox(control_list, layout=widgets.Layout(width='100%'))
        
        # Visual State
        if not self.enabled and layout_mode == 'fixed':
            # Placeholder look
            self.controls_container.layout.display = 'none' # Hide controls
            # Create a placeholder image or text
            visual = widgets.HTML(f"<div style='width:{size}px; height:{size}px; background:#eee; display:flex; align-items:center; justify-content:center; color:#aaa;'>Skipped<br>({self.name})</div>")
            return widgets.VBox([widgets.Label(self.name), visual, self.checkbox], layout=widgets.Layout(border='1px solid #ddd', margin='5px', padding='5px'))
        
        # Active State
        visual = self.image_widget
        
        # Main Container
        box = widgets.VBox([
            widgets.Label(self.name, style={'font_weight': 'bold'}),
            visual,
            self.controls_container,
            self.status_label
        ], layout=widgets.Layout(border='1px solid #888', margin='5px', padding='5px', width=f'{size+20}px'))
        
        return box

    def run(self, context: PipelineContext):
        if not self.enabled:
            self.status_label.value = "Skipped"
            return

        # Prepare inputs
        kwargs = {}
        missing_inputs = []
        
        # Map context keys to function arguments
        # If input_key is 'image', we pass it as the first argument or keyword 'img'/'image' if introspected?
        # We assume standard kwargs based on function definition or explicit mapping
        
        # For simplicity in this project, we map input_keys directly to function args
        # But we need to know the arg names of self.func.
        # Let's assume input_keys correspond to the order of arguments, OR use specific mapping.
        # To make it flexible:
        
        # Check function signature
        sig = inspect.signature(self.func)
        func_params = list(sig.parameters.keys())
        
        # 1. Fill from Context
        for i, key in enumerate(self.input_keys):
            val = context.get(key)
            if val is None:
                # Special case: optional inputs like mask
                if 'mask' in key:
                     kwargs[key] = None
                else:
                    missing_inputs.append(key)
            else:
                # Try to match key to function param name. 
                # Simplest heuristic: if key is 'image', pass to first arg or 'img'/'gray'
                # If key matches a param name exactly, use it.
                if key in func_params:
                    kwargs[key] = val
                elif i == 0 and len(func_params) > 0: 
                    # Assume first input key goes to first function argument (e.g. 'image' -> 'img')
                    kwargs[func_params[0]] = val
                else:
                    # Fallback: pass as kwarg matching the key name
                    kwargs[key] = val

        # 2. Fill from Widget Params
        for name, widget_obj in self.params.items():
            val = widget_obj.value if isinstance(widget_obj, widgets.Widget) else widget_obj
            kwargs[name] = val

        if missing_inputs:
            self.status_label.value = f"Missing: {', '.join(missing_inputs)}"
            return

        try:
            t0 = time.time()
            result = self.func(**kwargs)
            dt = time.time() - t0
            self.status_label.value = f"{dt*1000:.1f} ms"
            
            # Update Context
            # Result can be single item or tuple
            if len(self.output_keys) == 1:
                context[self.output_keys[0]] = result
                primary_result = result
            else:
                if not isinstance(result, tuple):
                    # Expecting tuple but got single?
                    context[self.output_keys[0]] = result
                    primary_result = result
                else:
                    for i, out_key in enumerate(self.output_keys):
                        if i < len(result):
                            context[out_key] = result[i]
                    primary_result = result[0]

            # Update Display
            # We display the content of the FIRST output key (usually the image)
            # or the explicitly set display_key
            disp_data = primary_result
            
            # If the result is a dict or complex object, handle it? 
            # Assuming Image (ndarray) for now
            if isinstance(disp_data, np.ndarray):
                self.image_widget.value = _array_to_bytes(disp_data)
                
        except Exception as e:
            self.status_label.value = f"Error: {str(e)}"
            # raise e # Debugging


class InteractivePipeline:
    def __init__(self, steps: List[Step], initial_image: np.ndarray = None):
        self.steps = steps
        self.context = PipelineContext()
        self.initial_image = initial_image
        
        # Link steps to this pipeline
        for s in self.steps:
            s.parent_pipeline = self
            
        # Global Controls
        self.layout_mode = widgets.ToggleButtons(
            options=['fixed', 'dynamic'],
            description='Layout:',
            button_style='', # 'success', 'info', 'warning', 'danger' or ''
        )
        self.layout_mode.observe(self._on_layout_change, names='value')
        
        self.image_scale = widgets.FloatSlider(
            value=1.0,
            min=0.5,
            max=2.0,
            step=0.1,
            description='Zoom:',
            continuous_update=False
        )
        self.image_scale.observe(self._on_scale_change, names='value')

        self.grid_box = widgets.HBox([], layout=widgets.Layout(flex_flow='row wrap'))
        self.container = widgets.VBox([
            widgets.HBox([self.layout_mode, self.image_scale]),
            self.grid_box
        ])

    def _on_layout_change(self, change):
        self.refresh_layout()

    def _on_scale_change(self, change):
        self.refresh_layout()

    def refresh_layout(self):
        """Re-renders the grid based on mode and active steps."""
        mode = self.layout_mode.value
        scale = self.image_scale.value
        
        children = []
        for step in self.steps:
            ui = step.get_ui(layout_mode=mode, image_scale=scale)
            if ui:
                children.append(ui)
        
        self.grid_box.children = children

    def run(self):
        """Runs the entire pipeline from start to finish."""
        if self.initial_image is None:
            return

        self.context.reset(self.initial_image)
        
        for step in self.steps:
            step.run(self.context)

    def display(self):
        self.refresh_layout()
        self.run()
        return self.container

    def set_image(self, img: np.ndarray):
        self.initial_image = img
        self.run()
