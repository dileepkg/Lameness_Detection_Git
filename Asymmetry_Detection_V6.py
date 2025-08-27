import pandas as pd
import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import argparse
import json
from typing import Dict, Optional, Tuple
import os
from pathlib import Path

def compute_lameness_from_csv(
    csv_path: str,
) -> Dict:
    # Load the CSV file
    # csv_filename = "Lame_5_superanimal_quadruped_fasterrcnn_resnet50_fpn_v2_hrnet_w32.csv"
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print("Error: CSV file not found.")
        exit()

    # Threshold for likelihood
    likelihood_threshold = 0.6

    # Filter data based on likelihood for relevant points
    mask = (
        (df['nose_likelihood'] > likelihood_threshold) &
        (df['front_left_knee_likelihood'] > likelihood_threshold) &
        (df['front_right_knee_likelihood'] > likelihood_threshold) &
        (df['back_left_knee_likelihood'] > likelihood_threshold) &
        (df['back_right_knee_likelihood'] > likelihood_threshold)
    )
    df_filtered = df[mask].reset_index(drop=True)

    # Extract relevant series (initialize as empty if filtering removes all data)
    frame = df_filtered['frame'] if not df_filtered.empty else np.array([])
    head_y = df_filtered['nose_y'] if not df_filtered.empty else np.array([])
    left_fore_x = df_filtered['front_left_knee_x'] if not df_filtered.empty else np.array([])
    right_fore_x = df_filtered['front_right_knee_x'] if not df_filtered.empty else np.array([])
    left_hind_x = df_filtered['back_left_knee_x'] if not df_filtered.empty else np.array([])
    right_hind_x = df_filtered['back_right_knee_x'] if not df_filtered.empty else np.array([])

    # Initialize variables to avoid UnboundLocalError
    is_fore_lame = False
    affected_fore = "None"
    fore_score = 0
    diff_fore = 0.0
    mean_head_left = 0.0
    mean_head_right = 0.0
    peaks_left = np.array([])
    peaks_right = np.array([])
    is_hind_lame = False
    affected_hind = "None"
    hind_score = 0
    diff_hind_rel = 0.0
    range_left_hind = 0.0
    range_right_hind = 0.0

    # --- Forelimb Lameness Detection ---
    if len(left_fore_x) > 0 and len(right_fore_x) > 0 and len(head_y) > 0:
        # Find peaks in forelimb x (protraction points, start of stance)
        peaks_left, _ = find_peaks(left_fore_x, prominence=20)
        peaks_right, _ = find_peaks(right_fore_x, prominence=20)

        # Window around peak for average head y during stance
        window = 3

        head_left_stance = []
        for p in peaks_left:
            start = max(0, p - window)
            end = min(len(head_y), p + window + 1)
            head_left_stance.append(np.mean(head_y[start:end]))

        head_right_stance = []
        for p in peaks_right:
            start = max(0, p - window)
            end = min(len(head_y), p + window + 1)
            head_right_stance.append(np.mean(head_y[start:end]))

        if len(head_left_stance) > 0 and len(head_right_stance) > 0:
            mean_head_left = np.mean(head_left_stance)
            mean_head_right = np.mean(head_right_stance)
            diff_fore = np.abs(mean_head_left - mean_head_right) / np.ptp(head_y) if np.ptp(head_y) > 0 else 0
            fore_lame_threshold = 0.05  # ~5% asymmetry threshold
            is_fore_lame = diff_fore > fore_lame_threshold
            if mean_head_left > mean_head_right:
                affected_fore = "Right"
            else:
                affected_fore = "Left"
            fore_score = min(5, int(diff_fore * 100) + 1) if is_fore_lame else 0

    # --- Hindlimb Lameness Detection ---
    if len(left_hind_x) > 0 and len(right_hind_x) > 0:
        range_left_hind = np.ptp(left_hind_x)
        range_right_hind = np.ptp(right_hind_x)
        mean_range_hind = np.mean([range_left_hind, range_right_hind])
        diff_hind_rel = np.abs(range_left_hind - range_right_hind) / mean_range_hind if mean_range_hind > 0 else 0

        hind_lame_threshold = 0.02  # ~2% relative diff threshold
        is_hind_lame = diff_hind_rel > hind_lame_threshold
        if range_left_hind < range_right_hind:
            affected_hind = "Left"
        else:
            affected_hind = "Right"
        if diff_hind_rel < 0.05:
            hind_score = 2
        else:
            hind_score = 4
        if not is_hind_lame:
            hind_score = 0

    # --- Overall Decision ---
    if is_fore_lame or is_hind_lame:
        lameness_flag = "Yes"
        if diff_fore > diff_hind_rel:
            affected_limb = f"Fore - {affected_fore}"
            aaep_score = fore_score
            explanation = f"Forelimb lameness detected based on head movement asymmetry (diff: {diff_fore:.2f}). Head drops more during {'left' if affected_fore == 'Right' else 'right'} stance, indicating {affected_fore.lower()} forelimb lameness."
        else:
            affected_limb = f"Hind - {affected_hind}"
            aaep_score = hind_score
            explanation = f"Hindlimb lameness detected based on stifle horizontal range asymmetry (rel diff: {diff_hind_rel:.2f}). Smaller stride range on {affected_hind.lower()} hindlimb."
    else:
        lameness_flag = "No"
        affected_limb = "None"
        aaep_score = 0
        explanation = "No significant asymmetry detected in head or stifle movements."

    # # Output results
    # print("Lameness flag:", lameness_flag)
    # print("Affected Limb:", affected_limb)
    # print("AAEP Score:", aaep_score)
    # print("Explanation:", explanation)

    # --- Visualizations with Saved Plots in CSV Directory ---
    save_path = Path(csv_path).parent #os.path.join(csv_path, "plots")

    # Create plots directory if it doesn't exist
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Visualization 1: Head Y Position with Forelimb Stance Phases
    if len(frame) > 0 and len(head_y) > 0:
        plt.figure(figsize=(10, 5))
        plt.plot(frame, head_y, label='Head Y Position')
        for p in peaks_left:
            if p < len(frame):
                plt.axvline(frame[p], color='g', linestyle='--', label='Left Fore Stance' if p == peaks_left[0] else "")
        for p in peaks_right:
            if p < len(frame):
                plt.axvline(frame[p], color='r', linestyle='--', label='Right Fore Stance' if p == peaks_right[0] else "")
        plt.xlabel('Frame')
        plt.ylabel('Head Y Coordinate')
        plt.title('Head Movement with Forelimb Stance Phases')
        plt.legend()
        plt.savefig(os.path.join(save_path, 'head_movement_with_stance_phases.png'))
        plt.close()

    # Visualization 2: Mean Head Position During Left vs Right Fore Stance
    if mean_head_left != 0 or mean_head_right != 0:
        plt.figure(figsize=(6, 4))
        plt.bar(['Left Fore Stance', 'Right Fore Stance'], [mean_head_left, mean_head_right], color=['g', 'r'])
        plt.axhline(y=(mean_head_left + mean_head_right) / 2, color='k', linestyle='--', label='Mean')
        plt.ylabel('Mean Head Y')
        plt.title('Asymmetry in Mean Head Position During Forelimb Stance')
        plt.legend()
        plt.savefig(os.path.join(save_path, 'mean_head_position_asymmetry.png'))
        plt.close()

    # Visualization 3: Hindlimb Horizontal Trajectories
    if len(frame) > 0 and len(left_hind_x) > 0 and len(right_hind_x) > 0:
        plt.figure(figsize=(10, 5))
        plt.plot(frame, left_hind_x, label='Left Hind X (Stifle)', color='g')
        plt.plot(frame, right_hind_x, label='Right Hind X (Stifle)', color='r')
        plt.xlabel('Frame')
        plt.ylabel('X Coordinate')
        plt.title('Hindlimb Stifle Horizontal Trajectories')
        plt.legend()
        plt.savefig(os.path.join(save_path, 'hindlimb_trajectories.png'))
        plt.close()

    # Visualization 4: Hindlimb X Range Comparison
    if range_left_hind != 0 or range_right_hind != 0:
        plt.figure(figsize=(6, 4))
        plt.bar(['Left Hind', 'Right Hind'], [range_left_hind, range_right_hind], color=['g', 'r'])
        plt.axhline(y=mean_range_hind, color='k', linestyle='--', label='Mean Range')
        plt.ylabel('X Range (Stride Proxy)')
        plt.title('Asymmetry in Hindlimb Stifle Range')
        plt.legend()
        plt.savefig(os.path.join(save_path, 'hindlimb_range_asymmetry.png'))
        plt.close()
        return {
        "Lameness flag": lameness_flag,
        "Affected Limb": affected_limb,
        "AAEP Score": aaep_score,
        "Explanation": explanation,
        }

def parse_args():
    p = argparse.ArgumentParser(
        description="Detect horse lameness (1–5) from DLC trajectories of nose & back base."
    )
    p.add_argument("--csv", required=True, help="Path to DLC CSV with nose/back_base keypoints.")
    return p.parse_args()


def main():
    args = parse_args()
    res = compute_lameness_from_csv(
        csv_path=args.csv
    )
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()