import streamlit as st
# import deeplabcut
from deeplabcut import video_inference_superanimal
import Excel_Generator as exg
import Asymmetry_Detection_V6 as asym
import json
from pathlib import Path, PureWindowsPath
import os
import tempfile
import pandas as pd
import ffmpeg
from PIL import Image
import base64
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Set DeepLabCut checkpoint directory to a writable location
checkpoint_dir = "./checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)
os.environ["DLC_MODELZOO_CHECKPOINTS"] = checkpoint_dir

# Debugging: Log DeepLabCut configuration
st.write(f"DeepLabCut checkpoint directory set to: {checkpoint_dir}")
st.write(f"Directory exists: {os.path.exists(checkpoint_dir)}")
st.write(f"Directory writable: {os.access(checkpoint_dir, os.W_OK)}")
st.write(f"Contents of {checkpoint_dir}: {os.listdir(checkpoint_dir) if os.path.exists(checkpoint_dir) else 'Empty'}")

# Post estimation function
def post_estimation(video_path: str,
                    dest_folder: str,
                    superanimal_name: str = "superanimal_quadruped",
                    model_name: str = "hrnet_w32",
                    detector_name: str = "fasterrcnn_resnet50_fpn_v2"
):
    video_inference_superanimal([video_path],
                                        superanimal_name,
                                        model_name=model_name,
                                        detector_name=detector_name,
                                        scale_list=range(200, 600, 50), 
                                        dest_folder=dest_folder,
                                        plot_trajectories=True,
                                        pcutoff=0.6,
                                        video_adapt=False,
                                        plot_bboxes=True)

def to_posix_rel(path_str: str) -> str:
    p = PureWindowsPath(path_str)
    posix = p.as_posix()
    return posix if (p.drive or posix.startswith(("/", "./", "//"))) else f"./{posix}"

# Function to validate and re-encode video to H.264 MP4
def ensure_playable_mp4(input_path: str, output_path: str) -> bool:
    try:
        probe = ffmpeg.probe(input_path)
        video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
        if not video_stream:
            st.error("No video stream found in the output file.")
            return False
        if video_stream['codec_name'] != 'h264':
            stream = ffmpeg.input(input_path)
            stream = ffmpeg.output(stream, output_path, vcodec='libx264', acodec='aac', format='mp4', strict='experimental')
            ffmpeg.run(stream, overwrite_output=True, capture_stdout=True, capture_stderr=True)
            st.write("Video re-encoded to H.264 MP4 for compatibility.")
        else:
            os.rename(input_path, output_path)
        return True
    except ffmpeg.Error as e:
        st.error(f"FFmpeg error: {e.stderr.decode()}")
        return False
    except Exception as e:
        st.error(f"Error processing video file: {str(e)}")
        return False

# Function to validate image file
def is_valid_image(image_path: str) -> bool:
    try:
        with Image.open(image_path) as img:
            img.verify()
        return True
    except Exception:
        return False

# Function to parse markdown report and extract table
def parse_markdown_report(report: str):
    lines = report.split('\n')
    table_data = []
    table_start = False
    headers = None
    
    for line in lines:
        if line.startswith('| Parameter | Good/Bad | Conformation details | Analysis confidence % |'):
            table_start = True
            headers = ['Parameter', 'Good/Bad', 'Conformation details', 'Analysis confidence %']
            continue
        if table_start and line.startswith('|'):
            if line.strip() == '|---|---|---|---|':
                continue
            row = [cell.strip() for cell in line.split('|')[1:-1]]
            if row:
                table_data.append(row)
        elif table_start and not line.startswith('|'):
            table_start = False
    
    table_df = pd.DataFrame(table_data, columns=headers) if table_data else None
    overall_assessment = lines[0] if lines and lines[0].startswith("Overall conformation:") else ""
    note = next((line for line in lines if line.startswith("Note:")), "") if any(line.startswith("Note:") for line in lines) else ""
    
    return overall_assessment, table_df, note

# Streamlit app
st.title("🐎 Horse Lameness Detection 👨‍⚕️")

# Create temporary directory for file processing
temp_dir = tempfile.mkdtemp()
dest_folder = temp_dir

# Sidebar with tabs
tab1, tab2 = st.sidebar.tabs(["Video Upload", "Image Upload"])

# Video upload tab
with tab1:
    st.header("Upload Video")
    video_file = st.file_uploader("Choose a video file", type=['mp4'])
    if video_file is not None:
        video_path = os.path.join(temp_dir, "input_video.mp4")
        with open(video_path, "wb") as f:
            f.write(video_file.getbuffer())
        st.success("Video uploaded successfully!")
        if st.button("Analyze Video"):
            with st.spinner("Processing video..."):
                try:
                    post_estimation(video_path=video_path, dest_folder=dest_folder)
                    out_csv = exg.convert_dlc_h5s_to_csv(
                        dest_dir=dest_folder,
                        pcutoff=0.6,
                        include_likelihood=True,
                        filter_likelihood_with_cutoff=True,
                        preserve_bodypart_order=True,
                    )
                    csv_path = to_posix_rel(out_csv)
                    res = asym.compute_lameness_from_csv(csv_path=csv_path)
                    json_path = Path(dest_folder) / "Inference.json"
                    with open(json_path, "w", encoding="utf-8") as f:
                        json.dump(res, f, ensure_ascii=False, indent=2)
                    st.success("Video analysis completed!")
                except Exception as e:
                    st.error(f"Error processing video: {str(e)}")

# Image upload tab
with tab2:
    st.header("Upload Image")
    image_file = st.file_uploader("Choose an image file", type=['jpg', 'jpeg', 'png'])
    if image_file is not None:
        image_path = os.path.join(temp_dir, image_file.name)
        with open(image_path, "wb") as f:
            f.write(image_file.getbuffer())
        st.success(f"Image uploaded successfully!")
        st.session_state['uploaded_image_path'] = image_path

# Display Analysis Results
st.header("Analysis Results")
if os.path.exists(dest_folder):
    # Display input video
    input_video_path = os.path.join(dest_folder, "input_video.mp4")
    if os.path.exists(input_video_path):
        try:
            st.subheader("Input Video")
            with open(input_video_path, 'rb') as f:
                video_bytes = f.read()
            st.video(video_bytes, format='video/mp4')
        except Exception as e:
            st.error(f"Error playing input video: {str(e)}")
            st.write("Try downloading the input video to play it locally:")
            with open(input_video_path, 'rb') as f:
                st.download_button(
                    label="Download Input Video",
                    data=f,
                    file_name="input_video.mp4",
                    mime='video/mp4'
                )

    # Display output video
    output_video = [f for f in os.listdir(dest_folder) if f.endswith('.mp4') and f != 'input_video.mp4']
    if output_video:
        st.subheader("Output Video")
        input_video_path = os.path.join(dest_folder, output_video[0])
        output_video_path = os.path.join(dest_folder, "output_video.mp4")
        if ensure_playable_mp4(input_video_path, output_video_path):
            try:
                with open(output_video_path, 'rb') as f:
                    video_bytes = f.read()
                st.video(video_bytes, format='video/mp4')
            except Exception as e:
                st.error(f"Error playing output video: {str(e)}")
                st.write("Try downloading the output video to play it locally:")
                with open(output_video_path, 'rb') as f:
                    st.download_button(
                        label="Download Output Video",
                        data=f,
                        file_name="output_video.mp4",
                        mime='video/mp4'
                    )

    # Handle Inference.json
    json_path = Path(dest_folder) / "Inference.json"
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            json_content = json.load(f)
        if isinstance(json_content, dict):
            def flatten_dict(d, parent_key='', sep='_'):
                items = []
                for k, v in d.items():
                    new_key = f"{parent_key}{sep}{k}" if parent_key else k
                    if isinstance(v, dict):
                        items.extend(flatten_dict(v, new_key, sep).items())
                    else:
                        items.append((new_key, str(v)))
                return dict(items)
            flat_json = flatten_dict(json_content)
            df = pd.DataFrame(list(flat_json.items()), columns=['Key', 'Value'])
            st.markdown(
                """
                <style>
                .stTable table {
                    width: 100%;
                    table-layout: auto;
                }
                .stTable th:nth-child(2), .stTable td:nth-child(2) {
                    white-space: nowrap;
                    width: auto;
                }
                </style>
                """,
                unsafe_allow_html=True
            )
            st.subheader("Inference Results")
            st.table(df)
            st.download_button(
                label="Download Inference.json",
                data=json.dumps(json_content, indent=2),
                file_name="Inference.json",
                mime='application/json'
            )
    else:
        st.write("No Inference.json file found.")

    # Display images from dest_folder in two columns, with "asymmetry" images last
    image_files = [f for f in os.listdir(dest_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    non_asymmetry = [f for f in image_files if "asymmetry" not in f.lower()]
    asymmetry = [f for f in image_files if "asymmetry" in f.lower()]
    sorted_image_files = non_asymmetry + asymmetry
    if sorted_image_files:
        st.subheader("Images")
        col1, col2 = st.columns(2)
        for i, image_file in enumerate(sorted_image_files):
            image_path = os.path.join(dest_folder, image_file)
            if is_valid_image(image_path):
                try:
                    with col1 if i % 2 == 0 else col2:
                        st.image(image_path, caption=image_file, use_container_width=True)
                except Exception as e:
                    st.error(f"Error displaying image {image_file}: {str(e)}")
                    st.write(f"Try downloading the image to view it locally:")
                    with open(image_path, 'rb') as f:
                        st.download_button(
                            label=f"Download {image_file}",
                            data=f,
                            file_name=image_file,
                            mime='image/' + image_file.split('.')[-1].lower()
                        )
            else:
                st.error(f"Invalid or corrupted image: {image_file}")
                st.write(f"Try downloading the image to view it locally:")
                with open(image_path, 'rb') as f:
                    st.download_button(
                        label=f"Download {image_file}",
                        data=f,
                        file_name=image_file,
                        mime='image/' + image_file.split('.')[-1].lower()
                    )

    # Veterinary report generation
    if 'uploaded_image_path' in st.session_state and openai_api_key:
        st.subheader("Veterinary Conformation Analysis")
        if st.button("Generate Veterinary Report"):
            with st.spinner("Generating veterinary report..."):
                try:
                    client = OpenAI(api_key=openai_api_key)
                    image_path = st.session_state['uploaded_image_path']
                    base64_image = base64.b64encode(open(image_path, "rb").read()).decode('utf-8')
                    prompt = """
Veterinary Conformation Analysis Prompt
Instruction to the model:
You are an equine veterinarian performing a conformation analysis. Judge only what is visible in the provided horse image(s). Focus on biomechanics, structural correctness, and potential risk for lameness or injury. If a parameter is not clearly visible, OMIT it completely. Do not ask follow-up questions.
Task:
For this image set, check if the horse has correct conformation across the parameters. Output only the three sections below, in this exact order and format:
1. Overall Assessment
Write:
Overall conformation: Excellent / Good / Bad / Very Bad (choose one, based on soundness risk).
2. Parameter Table
Create a markdown table with columns:
- Parameter (use an appropriate emoji),
- Good/Bad,
- Conformation details (1–2 concise, objective sentences, grounded in veterinary perspective),
- Analysis confidence % (integer 50–100).
Parameters (evaluate only if visible):
Head & Mouth (🦷🙂) – bite alignment, nostril size, ocular placement.
Neck & Throatlatch (🧠➡️) – airway clearance, tie-in to shoulders.
Shoulder & Withers (🪖) – slope/angle, saddle fit implications, stride length.
Back & Topline (〰️) – length, strength, loin coupling.
Hindquarters & Hip (🍑🐾) – pelvis length, croup slope, propulsion.
Forelimb Alignment (🦴) – knees, cannons, toe-in/out.
Hindlimb Alignment (🐾) – hocks, stifles, straightness.
Pasterns & Fetlocks (📐) – angle, strength, suspensory support.
Hooves (🦶) – size, heel depth, medial–lateral balance.
Muscling & Condition (💪) – topline, symmetry, conditioning.
Skin & Soft Tissues (🩹) – swelling, scars, tendon definition.
Movement (🚶) – stride symmetry, tracking up (only if movement is shown).
Other Parameters – if anything missing from the list above.
3. Final Note
After the table, add a single short Note line:
- State the logic behind the overall rating (e.g., 'Overall good balance with minor pastern concerns').
- Suggest any preventive or corrective action if applicable (e.g., hoof trim, conditioning, dental check).
Output Template (must be followed exactly):
```
Overall conformation: <excellent|good|bad|very bad>

| Parameter | Good/Bad | Conformation details | Analysis confidence % |
|---|---|---|---|
| <emoji + parameter name> | <Good/Bad> | <1–2 veterinary observations> | <##> |
| ...additional visible parameters only... |

Note: <one concise sentence with reasoning and action suggestion if relevant>.
```
"""
                    response = client.chat.completions.create(
                        model="gpt-4.1",
                        messages=[
                            {"role": "system", "content": prompt},
                            {"role": "user", "content": [
                                {"type": "text", "text": "Analyze this horse image."},
                                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                            ]}
                        ]
                    )
                    report = response.choices[0].message.content
                    overall_assessment, table_df, note = parse_markdown_report(report)
                    
                    # Display report components
                    if overall_assessment:
                        st.markdown(overall_assessment)
                    if table_df is not None:
                        st.table(table_df)
                    if note:
                        st.markdown(note)
                except Exception as e:
                    st.error(f"Error generating report: {str(e)}")
    elif 'uploaded_image_path' in st.session_state and not openai_api_key:
        st.error("OpenAI API key not found in .env file.")
else:
    st.write("Output directory not found.")