import subprocess
import os

def run_matlab_detection(image_path):
    abs_path = os.path.abspath(image_path)
    script = f"classify_image_simple('{abs_path}')"
    try:
        subprocess.run(
            ["matlab", "-batch", script],
            check=True
        )
        # الصورة المعدلة هترجع بنفس الاسم مع _output
        output_path = abs_path.replace('.', '_output.')
        return output_path if os.path.exists(output_path) else None
    except subprocess.CalledProcessError:
        return None
    except Exception as e:
        print(f"Error running MATLAB script: {e}")
        return None
    "C:\Users\SaWa\OneDrive\Pictures\Saved Pictures\whats\WhatsApp Image 2025-04-11 at 19.03.21_441d354e.jpg"