import deeplabcut
import Excel_Generator as exg
import Asymmetry_Detection_V6 as asym
import argparse
import json
from pathlib import Path, PureWindowsPath

# # Initiate variables
# video_path = r"C:\Users\999381\Desktop\Equine\Test_Data\Lame_5.mp4"
# dest_folder=r"C:\Users\999381\Desktop\Equine\apps\Direct_Output"
# superanimal_name = "superanimal_quadruped"
# model_name="hrnet_w32"
# detector_name="fasterrcnn_resnet50_fpn_v2"


# Post estimation
def post_estimation(video_path: str,
                    dest_folder: str,
                    superanimal_name: str = "superanimal_quadruped",
                    model_name: str = "hrnet_w32",
                    detector_name: str = "fasterrcnn_resnet50_fpn_v2"
):
    deeplabcut.video_inference_superanimal([video_path],
                                        superanimal_name,
                                        model_name=model_name,
                                        detector_name=detector_name,
                                        scale_list=range(200, 600, 50), 
                                        dest_folder=dest_folder,
                                        plot_trajectories =True,
                                        pcutoff=0.6,
                                        video_adapt = False,
                                        plot_bboxes = True
                                        )


def parse_args():
    p = argparse.ArgumentParser(
        description="Detect horse lameness (1–5) from DLC trajectories of nose & back base."
    )
    p.add_argument("--video_path", required=True, help="Path to the input vedeo to be analyzed.")
    p.add_argument("--dest_folder", required=True, help="Path to save output files.")
    # p.add_argument("--csv", required=True, help="Path to DLC CSV with nose/back_base keypoints.")
    p.add_argument("--save-prefix", default=None,help="If set, save JSON summary and plots with this prefix (e.g., /path/out/lameness).")
    p.add_argument("--likelihood-threshold", type=float, default=0.6, help="Min keypoint likelihood to keep (if present).")
    p.add_argument("--trend-window", type=int, default=45, help="Rolling window for detrending (frames).")
    p.add_argument("--smooth-window", type=int, default=5, help="Rolling window for short-term smoothing (frames).")
    
    return p.parse_args()

def to_posix_rel(path_str: str) -> str:
    p = PureWindowsPath(path_str)
    posix = p.as_posix()
    # Add "./" only for relative paths without a leading "./"
    return posix if (p.drive or posix.startswith(("/", "./", "//"))) else f"./{posix}"

def main():
    args = parse_args()

    post_estimation(
        video_path=args.video_path,
        dest_folder=args.dest_folder
    )

    # Generate CSV using helper function
    out_csv = exg.convert_dlc_h5s_to_csv(
        dest_dir=args.dest_folder,
        pcutoff=0.6,
        include_likelihood=True,
        filter_likelihood_with_cutoff=True,
        preserve_bodypart_order=True,
    )

    # Calculate Asymmetry
    csv_path = to_posix_rel(out_csv)    
    res = asym.compute_lameness_from_csv(
        csv_path=  csv_path
    )
    print(json.dumps(res, indent=2))
    
    with open(Path(args.dest_folder)/"Inference.json", "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()