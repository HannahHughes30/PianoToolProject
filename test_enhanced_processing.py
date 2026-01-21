import cv2
import numpy as np
from pathlib import Path
from pdf2image import convert_from_path

class SheetMusicProcessor:
    def __init__(self):
        self.debug_mode = True
    
    def load_image(self, image_path):
        image_path_obj = Path(image_path)
        
        if str(image_path_obj).lower().endswith('.pdf'):
            print("   Converting PDF to image...")
            try:
                images = convert_from_path(str(image_path_obj), dpi=300)
                pil_image = images[0]
                img = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
                print(f"   Loaded PDF (first page, {img.shape[1]}x{img.shape[0]} pixels)")
                return img
            except Exception as e:
                raise ValueError(f"Could not load PDF: {image_path_obj}. Error: {e}")
        
        img = cv2.imread(str(image_path_obj))
        if img is None:
            raise ValueError(f"Could not load image: {image_path_obj}")
        return img
    
    def auto_rotate_image(self, image):
        h, w = image.shape[:2]
        
        if w > h:
            image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
            print("   Rotated image to portrait orientation")
        
        return image
    
    def preprocess_for_detection(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        binary = cv2.bitwise_not(binary)
        return binary
    
    def detect_staff_lines(self, binary_image):
        kernel_length = binary_image.shape[1] // 20
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_length, 1))
        horizontal_lines = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
        contours, _ = cv2.findContours(horizontal_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        staff_y_positions = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w > binary_image.shape[1] * 0.5:
                staff_y_positions.append(y + h//2)
        
        staff_y_positions = sorted(set(staff_y_positions))
        print(f"   Detected {len(staff_y_positions)} staff lines")
        return staff_y_positions
    
    def detect_bar_lines(self, binary_image):
        kernel_length = binary_image.shape[0] // 20
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_length))
        vertical_lines = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
        contours, _ = cv2.findContours(vertical_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        bar_x_positions = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if h > binary_image.shape[0] * 0.3:
                bar_x_positions.append(x + w//2)
        
        bar_x_positions = sorted(set(bar_x_positions))
        filtered_bars = []
        for x in bar_x_positions:
            if not filtered_bars or x - filtered_bars[-1] > 20:
                filtered_bars.append(x)
        
        print(f"   Detected {len(filtered_bars)} bar lines")
        return filtered_bars
    
    def preprocess_pipeline(self, image_path, output_dir='data/processed'):
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        base_name = Path(image_path).stem
        
        print("\nStarting image processing pipeline...")
        print(f"File: {base_name}")
        
        print("Loading image...")
        img = self.load_image(image_path)
        
        print("Checking orientation...")
        img = self.auto_rotate_image(img)
        
        print("Converting to binary...")
        binary = self.preprocess_for_detection(img)
        
        print("Detecting staff lines...")
        staff_lines = self.detect_staff_lines(binary)
        
        print("Detecting bar lines...")
        bar_lines = self.detect_bar_lines(binary)
        
        if self.debug_mode:
            rotated_path = f'{output_dir}/{base_name}_rotated.png'
            cv2.imwrite(rotated_path, img)
            print(f"Saved: {rotated_path}")
            
            binary_path = f'{output_dir}/{base_name}_binary.png'
            cv2.imwrite(binary_path, binary)
            print(f"Saved: {binary_path}")
            
            viz = img.copy()
            for y in staff_lines:
                cv2.line(viz, (0, y), (viz.shape[1], y), (0, 255, 0), 3)
            
            detected_path = f'{output_dir}/{base_name}_detected.png'
            cv2.imwrite(detected_path, viz)
            print(f"Saved: {detected_path}")
        
        return {
            'binary_image': binary,
            'staff_lines': staff_lines,
            'bar_lines': bar_lines,
            'original': img
        }


def load_image(image_path):
    return cv2.imread(image_path, cv2.IMREAD_COLOR)

def preprocess_image(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    edges = cv2.Canny(blurred_image, 75, 150)
    return edges

def show_image(title, image):
    cv2.imshow(title, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    processor = SheetMusicProcessor()
    result = processor.preprocess_pipeline('CDLuneScore.png')
    print(f"\nFinal results:")
    print(f"{len(result['staff_lines'])} staff lines detected")
    print(f"{len(result['bar_lines'])} bar lines detected")
