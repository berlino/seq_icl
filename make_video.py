import argparse
import os
import subprocess
import glob

def prepare_video(glob_str, folder, filename):
    # create a new folder
    files = glob.glob(glob_str)
    os.makedirs(folder, exist_ok=True)
    # remove all files in the folder
    for file in files:
        # copy files to a new folder
        id = int(os.path.basename(file).split("_")[-1].replace(".png", ""))
        # make all ids 3 digit
        id = str(id).zfill(3)
        os.system(f"cp {file} {folder}/{id}.png")


    # Build the FFmpeg command
    ffmpeg_cmd = [
        'ffmpeg',
        '-framerate', '2',
        "-pattern_type", "glob",
        '-i', f'{folder}/*.png',
        '-pix_fmt', 'yuv420p',
        f"{folder}/{filename}.mp4"
    ]

    # Execute the FFmpeg command using subprocess
    try:
        subprocess.run(ffmpeg_cmd, check=True)
        # print(f"Video '{output_video}' created successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")

    # remove png files
    os.system(f"rm {folder}/*.png")
    return f"{folder}/{filename}.mp4"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--glob", type=str, required=True)
    parser.add_argument("--folder", type=str, required=True)

    args = parser.parse_args()

    prepare_video(args.glob, args.folder)


