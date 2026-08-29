import os 
import warnings
import subprocess
from glob import glob
from joblib import Parallel, delayed

class SkullStripVolumes:
    def __init__(self, 
                 volume_directory: str | None = None, 
                 volume_path_list: list[str] | None = None,  # pyright: ignore[reportRedeclaration]
                 output_directory: str = "skull_stripped_volumes",
                 n_jobs: int = -1) -> None:

        if volume_directory == None and volume_path_list == None:
            raise ValueError("Please enter value(s) for either 'volume_directory' or 'volume_path_list'.")
        elif volume_directory == None and volume_path_list != None and len(volume_path_list) == 0:
            raise ValueError("Please enter value(s) for either 'volume_directory' or 'volume_path_list'.")
        elif volume_directory != None and volume_path_list != None and len(volume_path_list) != 0:
            raise ValueError("Please enter value(s) for either 'volume_directory' or 'volume_path_list'.")

        os.makedirs(output_directory, exist_ok=True)

        if volume_directory:
            volume_path_list: list[str] = sorted(glob(os.path.join(volume_directory, "*.nii")) + glob(os.path.join(volume_directory, "*.nii.gz")))
            if len(volume_path_list) == 0:
                raise FileNotFoundError(f"I did not find any volumes in directory: {volume_directory}.")
            print(f"I found {len(volume_path_list)} volumes in directory: {volume_directory}")

        print(f"Skull stripping {len(volume_path_list)} volumes with {n_jobs} Job(s)") # pyright: ignore[reportArgumentType]
        Parallel(n_jobs=n_jobs)(
            delayed(self.skull_strip_single_volume)(
                volume_path=volume_path,
                output_directory=output_directory
            )
            for volume_path in volume_path_list # pyright: ignore[reportOptionalIterable]
        )

        self.output_volume_paths: list[str] = sorted(glob(os.path.join(output_directory, "ss_*.nii.gz")))
        print(f"Created {len(self.output_volume_paths)} skull stripped volumes.")


    def run_command(self, 
                    command: list[str], 
                    verbose: bool = True) -> None:
        if verbose:
            print(f"Running Command: {command}")

        result: subprocess.CompletedProcess = \
            subprocess.run(command, capture_output=True, text=True)
        
        if result.returncode != 0:
            raise CalledProcessError(
                returncode=result.returncode,
                cmd=result.args,
                output=result.stdout,
                stderr=result.stderr
            )
        else:
            print(f"Command was successful.")
            if result.stdout:
                print(f"Stdout:\n{result.stdout}")
            if result.stderr:
                print(f"Stderr:\n{result.stderr}")

                
    def skull_strip_single_volume(self, volume_path: str, output_directory: str):
        
        output_volume_path: str = os.path.join(output_directory, f"ss_{os.path.basename(volume_path)}")
        if not volume_path.endswith(".nii.gz"):
            if not volume_path.endswith(".nii"):
                warnings.warn(
                    message=\
                        f"Input Path: {volume_path} does not end with '.nii' or '.nii.gz'. " \
                        "Skipping skull stripping for this file.",
                    category=UserWarning
                )
            else:
                output_volume_path += ".gz"

        self.run_command(
            command=["bet", volume_path, output_volume_path, "-R"],
            verbose=True
        )
        print(f"Outputted Path: {output_volume_path}")


    def return_output_volume_paths(self) -> list[str]:
        return self.output_volume_paths

class CalledProcessError(subprocess.CalledProcessError):
    def __str__(self) -> str:
        return (
            f"Command resulted in non-zero exit code:\n\n"
            f"Command: {self.cmd}\n\n"
            f"Exit code: {self.returncode}\n\n"
            f"Stdout: {self.output}\n\n"
            f"Stderr: {self.stderr}"
        )


if __name__ == "__main__":
    import argparse
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Skull strip a directory or list of volumes via FSL's 'bet' command"
    )
    parser.add_argument(
        "--volume_directory",
        required=False,
        default=None,
        help=\
            "A directory containing 3D NiFTI files to skull strip. " \
            "Enter value(s) for '--volume_directory' OR '--volume_path_list'."
    )
    parser.add_argument(
        "--volume_path_list",
        required=False,
        nargs="+",
        default=None,
        help=\
            "A list 3D NiFTI file paths to skull strip. " \
            "Enter value(s) for '--volume_directory' OR '--volume_path_list'."
    )
    parser.add_argument(
        "--output_directory",
        required=False,
        default="skull_stripped_volumes",
        help=f"Default: {os.path.abspath('skull_stripped_volumes')}"
    )
    parser.add_argument(
        "--n_jobs",
        required=False,
        type=int,
        default=-1,
        help="Default: -1 (All CPU Cores)"
    )
    args: argparse.Namespace = parser.parse_args()
    SkullStripVolumes(
        volume_directory=\
            os.path.abspath(args.volume_directory)
            if args.volume_directory else None,
        volume_path_list=[
                os.path.abspath(volume_path)
                for volume_path in args.volume_path_list
            ] if args.volume_path_list else None,
        output_directory=os.path.abspath(args.output_directory),
        n_jobs=args.n_jobs
    )
