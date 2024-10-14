import streamlit as st
import subprocess
import os
import pandas as pd
from PIL import Image

def run_script(script_path, env_name, args):
    script_dir = os.path.dirname(script_path)
    cmd = f"conda run -n {env_name} python {script_path} {args}"
    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=script_dir)
    output, error = process.communicate()
    return output.decode(), error.decode()

def main():
    st.title("Text Processing Pipeline")

    # File upload
    uploaded_file = st.file_uploader("Choose an image file", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        # Save the uploaded file
        with open("input_image.jpg", "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Text Detection
        st.header("Step 1: Text Detection")
        if st.button("Run Text Detection"):
            output, error = run_script(
                'C://symbols/TextDetection/TextDetectionMain.py',
                'myenv2',
                '--trained_model="weights/craft_mlt_25k.pth" --low_text=0.4 --text_threshold=0.7 --link_threshold=0.4 --show_time --refiner_model="weights/craft_refiner_CTW1500.pth" --image_path="C://symbols/TextDetection/data/pnid.jpg" --bounds_topx=180 --bounds_topy=63 --bounds_width=2317 --bounds_height=1557 --cuda=False --refine --canvas_size=92288 --mag_ratio=1.5 --poly'
            )
            st.text_area("Output", output)
            if error:
                st.text_area("Error", error)

            # Show result image if exists
            result_image_path = "C://symbols/TextDetection/result/result_image.jpg"  # Update with actual result path
            if os.path.exists(result_image_path):
                st.image(result_image_path, caption='Detected Text Image', use_column_width=True)

        # Text Extraction
        st.header("Step 2: Text Extraction")
        if st.button("Run Text Extraction"):
            output, error = run_script(
                "C://symbols/TextExtraction/Main.py",
                "myenv3",
                '--region_file_path="C://symbols/TextDetection/result/res_pnid.txt" --image_path="C://symbols/TextExtraction/input/pnid.jpg" --bounds_topx=180 --bounds_topy=63 --bounds_width=2317 --bounds_height=1557'
            )
            st.text_area("Output", output)
            if error:
                st.text_area("Error", error)

        # Text Categorization
        st.header("Step 3: Text Categorization")
        if st.button("Run Text Categorization"):
            output, error = run_script(
                "C://symbols/TextCategorisation/TextMapping.py",
                "myenv1",
                ""
            )
            st.text_area("Output", output)
            if error:
                st.text_area("Error", error)

        # Display results
        st.header("Results")
        if os.path.exists("TextCategory.csv"):
            df = pd.read_csv("TextCategory.csv")
            st.dataframe(df)

if __name__ == "__main__":
    main()
