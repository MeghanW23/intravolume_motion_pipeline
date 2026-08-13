import os 
import subprocess

class RunAlignments:
    def __init__(self,
                 reference_volume_path: str,
                 target_volume_path: str, 
                 target_slice_indices: list[int],
                 initial_transform_path: str,
                 working_directory: str = 'working',
                 output_transform_label: str = 'transform',
                 run_environment: str = 'docker',
                 smsmireg_executable_path: str | None = None,
                 singularity_image_file: str | None = None,
                 optimizer: str = 'retro-motion-measurement') -> None:

        self.reference_volume_path: str = reference_volume_path
        self.target_volume_path: str = target_volume_path
        self.target_slice_indices: list[str] =  [str(index) for index in target_slice_indices]
        self.initial_transform_path: str = initial_transform_path
        self.working_directory: str = working_directory
        self.output_transform_label: str = output_transform_label
        self.run_environment: str = run_environment.strip().lower()
        self.smsmireg_executable_path: str | None = smsmireg_executable_path
        self.singularity_image_file: str | None = singularity_image_file
        self.optimizer: str = optimizer

        os.makedirs(self.working_directory, exist_ok=True)

        print(f"\nYour Inputs to {os.path.basename(__file__)}:")
        print(f"Slice Group Indices: {self.target_slice_indices}")
        print(f"Reference Volume Path: {self.reference_volume_path}")
        print(f"Initial Transform Path: {self.initial_transform_path}")
        print(f"Working Directory Path: {self.working_directory}")
        print(f"Output Label: {self.output_transform_label}")
        print(f"Running via: {self.run_environment}")
        print("\n")

        if self.run_environment == 'docker':
            print("Running Sms-Mi-Reg with Docker")
            self.run_with_docker()

        elif self.run_environment ==  'singularity':
            print(f"Running Sms-Mi-Reg via Singularity image: {self.singularity_image_file}")
            self.run_with_singularity()

        elif self.run_environment == 'local':
            print(f"Running Sms-Mi-Reg locally with file: {self.smsmireg_executable_path}")
            self.run_locally() 

        else:
            raise ValueError(
                "Argument: 'run_environment' must be one of the following values: " + \
                "'docker', 'singularity', or 'local'."
            )

    def run_locally(self):
       if self.smsmireg_executable_path == None:
           raise ValueError(
                "To run locally, please provide the executable " + \
                "Sms-Mi-Reg path on your local machine. This path is outputted " + \
                "by compiling sms-mi-reg.cpp: " + \
                "https://github.com/ComputationalRadiology/sms-mi-reg/blob/main/sms-mi-reg.cpp. "
            )

       self.run_command([
            self.smsmireg_executable_path,
            "--movingVolume", self.reference_volume_path,
            "--fixedVolume", self.target_volume_path,
            "--outputtransformfile", os.path.join(self.working_directory, "alignTransform_" + self.output_transform_label + ".tfm"),
            "--initialtransform", self.initial_transform_path,
            "--activeSlices"] + self.target_slice_indices + [
            "--activeParameters", "111111",
            "--histogramPVAOff",
            "--optimizer", "LN_SBPLX"
            ],
            verbose=True
        )



    def run_with_docker(self):
        """
        Usage: retro-motion-measurement [--help] [--version] --movingVolume VAR --fixedVolume VAR --outputtransformfile VAR [--initialtransform VAR] [--activeSlices VAR...] [--activeParameters VAR] [--histogramPVAOff] [--optimizer VAR]

        Optional arguments:
        -h, --help             shows help message and exits
        -v, --version          prints version information and exits
        -m, --movingVolume     The 3D reference volume to be aligned to the slices. [required]
        -f, --fixedVolume      The 3D volume that will be matched. [required]
        --outputtransformfile  Name of file to write transform. [required]
        --initialtransform     Name of transform file for initialization.
        --activeSlices         List of integer slice indexes to align with. [nargs: 0 or more]
        --activeParameters     Indicate which parameters to optimize with binary string. [default: "111111"]
        --histogramPVAOff      Turn off histogram partial volume updates.
        --optimizer            The name of the optimizer algorithm: (LN_COBYLA, LN_BOBYQA, LN_NELDERMEAD, LN_SBPLX). Default is LN_BOBYQA [default: "LN_BOBYQA"]
        """

        mount_args: list[str] = []
        for path in [self.reference_volume_path, self.initial_transform_path]:
            mount_args += ["-v", f"{path}:/data/{os.path.basename(path)}"]


        self.run_command([
            "docker", "run", "--rm",
            *mount_args,
            "-v", f"{self.working_directory}:/data",
            "crl/sms-mi-reg", self.optimizer,
            "--movingVolume", os.path.basename(self.reference_volume_path),
            "--fixedVolume", os.path.basename(self.target_volume_path),
            "--outputtransformfile", "alignTransform_" + self.output_transform_label + ".tfm",
            "--initialtransform", os.path.basename(self.initial_transform_path),
            "--activeSlices"] + self.target_slice_indices + [
            "--activeParameters", "111111",
            "--histogramPVAOff",
            "--optimizer", "LN_SBPLX"
            ],
            verbose=True
        )

        
    def run_with_singularity(self):
        if self.singularity_image_file is None:
            raise ValueError(
                "If running with Singularity, please provide a .sif " + \
                "file using the 'singularity_image_file' argument."
            )

        mount_args: list[str] = []
        for path in [self.reference_volume_path, self.initial_transform_path]:
            mount_args += ["--bind", f"{path}:/data/{os.path.basename(path)}"]

        # bind working_directory to a writable output dir inside the container
        mount_args += ["--bind", f"{self.working_directory}:/data/out"]

        self.run_command([
            "singularity", "run",
            *mount_args,
            "--pwd", "/data/out",
            self.singularity_image_file, self.optimizer,
            "--movingVolume", os.path.basename(self.reference_volume_path),
            "--fixedVolume", os.path.basename(self.target_volume_path),
            "--outputtransformfile", "alignTransform_" + self.output_transform_label + ".tfm",
            "--initialtransform", os.path.basename(self.initial_transform_path),
            "--activeSlices"] + self.target_slice_indices + [
            "--activeParameters", "111111",
            "--histogramPVAOff",
            "--optimizer", "LN_SBPLX"
            ],
            verbose=True
        )


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
            print(f"Command: {command} was successful.")
            if result.stdout:
                print(f"Stdout:\n{result.stdout}")
            if result.stderr:
                print(f"Stderr:\n{result.stderr}")


    def return_output_transform_path(self) -> str:
        return os.path.join(self.working_directory, f"alignTransform_{self.output_transform_label}.tfm") 


    
# more verbose than subprocess.CalledProcessError
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
        description=\
            "Align the reference volume to a slice group using mutual information. " + \
            "See: https://github.com/ComputationalRadiology/sms-mi-reg/ for more " + \
            "on how the optimizer works."
    )
    parser.add_argument(
        "--reference_volume_path",
        required=True,
        help="The 3D motion-free reference volume."
    )
    parser.add_argument(
        "--target_volume_path",
        required=True,
        help="The 3D target volume."
    )
    parser.add_argument(
        "--target_slice_indices",
        required=True,
        type=int,
        nargs="+",
        help="The indexes to the 2D slices in the slice group."
    )
    parser.add_argument(
        "--initial_transform_path",
        required=True,
        help=\
            "The parameters in this transform will " + \
            "be the starting point for the optimizer."
    )
    parser.add_argument(
        "--working_directory",
        required=False,
        default='working',
        help=f"Default: {os.path.abspath('working')}"
    )
    parser.add_argument(
        "--output_transform_label",
        required=False,
        default="transform",
        help=\
            "Each outputted file will include this label. " + \
            "Default: 'transform'."
    )
    parser.add_argument(
        "--run_environment",
        required=False,
        choices=[
            "docker",
            "singularity",
            "local"
        ],
        default="docker",
        help=\
            "Sms-Mi-Reg can be run in a Docker image, a Singularity image " + \
            "or locally. Please enter one of the following options: " + \
            "'docker', 'singularity', or 'local'. " + \
            "Default: 'docker'"
    )
    parser.add_argument(
        "--smsmireg_executable_path",
        required=False,
        default=None,
        help=\
            "To run locally, please provide the executable " + \
            "Sms-Mi-Reg path on your local machine. This path is outputted " + \
            "by compiling retro-motion-measurement.cxx: " + \
            "https://github.com/ComputationalRadiology/sms-mi-reg/blob/main/retro-motion-measurement.cxx. "
    )
    parser.add_argument(
        "--singularity_image_file",
        required=False,
        default=None,
        help=\
            "To run in a Singularity image, please provide the path " + \
            "to the '.sif' image file."
    )
    args: argparse.Namespace = parser.parse_args()
    RunAlignments(
        reference_volume_path=os.path.abspath(args.reference_volume_path),
        target_volume_path=os.path.abspath(args.target_volume_path),
        target_slice_indices=args.target_slice_indices,
        initial_transform_path=os.path.abspath(args.initial_transform_path),
        working_directory=os.path.abspath(args.working_directory),
        output_transform_label=args.output_transform_label,
        run_environment=args.run_environment,
        smsmireg_executable_path=\
            os.path.abspath(args.smsmireg_executable_path)
            if args.smsmireg_executable_path else None,
        singularity_image_file=\
            os.path.abspath(args.singularity_image_file)
            if args.singularity_image_file else None
    )
