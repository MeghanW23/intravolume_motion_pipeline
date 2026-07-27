import os 
import warnings
import subprocess

class DicomToNifti:
    def __init__(self, 
                 dicom_directory: str, 
                 output_directory: str = 'outputs',
                 dcm2niix_path: str = 'dcm2niix',
                 filename_keyword: str | None = None,
                 return_multiple_ok: bool = False) -> None:
        
        os.makedirs(output_directory, exist_ok=True)

        self.command: list[str] = [
            dcm2niix_path,  # path to dcm2niix
            "-o", output_directory,
            "-z", "y",      # compress images 
            "-b", "y",      # BIDS sidecar
            "-ba", "n",     # don't anonymize BIDS 
            "-w", "1",      # overwrite if name conflict
            dicom_directory # input directory
        ]

        self.run_command()

        self.nifti_outputs: str | list[str] = self.find_output_file(
            output_directory,
            file_extension=".nii.gz",
            filename_keyword=filename_keyword,
            return_multiple_ok=return_multiple_ok
        )
        print(f"Outputted NiFTI File(s): {self.nifti_outputs}")

        self.json_outputs: str | list[str] = self.find_output_file(
            output_directory,
            file_extension=".json",
            filename_keyword=filename_keyword,
            return_multiple_ok=return_multiple_ok
        )
        print(f"Outputted JSON File(s): {self.json_outputs}")


    def run_command(self, verbose: bool = True):
        if verbose:
            print(f"Running Command: {self.command}")

        result: subprocess.CompletedProcess = subprocess.run(
            self.command,
            capture_output=True
        )
        
        if result.returncode != 0:
            raise CalledProcessError(
                returncode=result.returncode,
                cmd=result.args,
                output=result.stdout,
                stderr=result.stderr
            )

        elif verbose: 
            print(f"Command: {self.command} was Sucessful.")
            if result.stdout:
                print(f"Stdout: {result.stdout}")
            if result.stderr:
                print(f"Stderr: {result.stderr}")
    

    def find_output_file(self,
                         output_directory: str, 
                         file_extension: str,
                         filename_keyword: str | None = None,
                         return_multiple_ok: bool = False
                         ) -> str | list[str]:
        
        all_filenames: list[str] = sorted(
            os.listdir(output_directory),
            key=lambda f: os.path.getmtime(os.path.join(output_directory, f))
        )
        all_matching_filenames: list[str] = [
            filename 
            for filename in all_filenames
            if filename.endswith(file_extension) 
            and "epi" not in filename
        ]
        if filename_keyword != None:
            all_matching_filenames: list[str] = [
                    filename 
                    for filename in all_matching_filenames
                    if filename_keyword in filename
                ]

        if len(all_matching_filenames) == 0: 
            raise FileNotFoundError(
                f"No matching files were found in directory: {output_directory}"
                f" with file extension: {file_extension} after dcm2niix"
                f" command: {self.command}."
            )
        
        elif len(all_matching_filenames) > 1:

            if return_multiple_ok:
                print(f"Returning all matching files...")
                return all_matching_filenames
            
            most_recent_file_path: str = os.path.join(output_directory, all_matching_filenames[-1])
            warnings.warn(
                f"More than one matching file was found in directory '{output_directory}' "
                f"with file extension '{file_extension}' "
                f"after running dcm2niix command: {self.command}. "
                f"Matching files (sorted oldest to newest): "
                + "\n".join(all_matching_filenames)
                + f" Returning most recent file: {most_recent_file_path}",
                UserWarning,
            )
            return most_recent_file_path
        
        else:
            return os.path.join(output_directory, all_matching_filenames[0])


    def return_nifti_image(self) -> str | list[str]:
        return self.nifti_outputs


    def return_json_file(self) -> str | list[str]:
        return self.json_outputs
    

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
        description="Run dcm2niix on a DICOM Directory."
    )
    parser.add_argument(
        "--dicom_directory",
        required=True
    )
    parser.add_argument(
        "--output_directory",
        required=False,
        default='outputs',
        help=f"Default: {os.path.abspath('outputs')}"
    )
    parser.add_argument(
        "--dcm2niix",
        required=False,
        default='dcm2niix',
        help="The command name or path for dcm2niix. Default 'dcm2niix'."
    )
    args: argparse.Namespace = parser.parse_args()
    DicomToNifti(
        dicom_directory=os.path.abspath(args.dicom_directory),
        output_directory=os.path.abspath(args.output_directory),
        dcm2niix_path=args.dcm2niix
    )