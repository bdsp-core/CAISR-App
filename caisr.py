import os, sys, docker, subprocess, platform
import pandas as pd
from typing import List, Tuple, Set

# List all Docker images
def list_images(client: docker.DockerClient) -> None:
    '''
    Lists all Docker images including their tags.
    '''
    try:
        images = client.images.list()
        if not images:
            print("No Docker images found.")
            return

        print("Available Docker Images:")
        for image in images:
            tags = ', '.join(image.tags) if image.tags else "<untagged>"
            print(f"Image ID: {image.id[:12]} | Tags: {tags}")

        print(' * If you want to re-install a docker, remove its image first using: "docker rmi <Tag>"')
    
    except docker.errors.DockerException as e:
        print(f"Error communicating with Docker daemon: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

# Match Docker images to corresponding tasks
def match_images_to_tasks(client: docker.DockerClient, tasks: List[str]) -> List[Tuple[str, str]]:
    '''
    Matches Docker images to the specified tasks and identifies missing images.

    Args:
    - client: Docker client object to interact with the Docker daemon.
    - tasks: List of task names (without the 'caisr_' prefix) to be matched with Docker images.

    Returns:
    - matches: List of tuples where each tuple contains a task and its corresponding image tag.
    '''

    matches, missing = [], set()

    # List all Docker images once to avoid repeated calls
    try:
        images = client.images.list()
    except docker.errors.DockerException as e:
        print(f"Error communicating with Docker daemon: {e}")
        return missing

    # Iterate over each task, prepending 'caisr_' to match the image naming convention
    for task in tasks:
        task_image_name = f'caisr_{task}'
        task_found = False

        for image in images:
            for tag in image.tags:
                if tag == f'{task_image_name}:latest':
                    matches.append((task, tag))
                    task_found = True
                    break  # Break inner loop to avoid duplicate matches
                elif tag.split(':')[0] == task_image_name:
                    matches.append((task, tag))
                    task_found = True
                    break  # Break inner loop to avoid duplicate matches
            
            if task_found:
                break  # Break outer loop if task image is found

        # If no image was matched, add to missing
        if not task_found:
            missing.add(task)

    # Extract and load non-installed dockers, if necessary
    if len(missing) > 0:
        images = install_missing_dockers(client, tasks, missing)

    return matches

# Install missing Docker images
def install_missing_dockers(client: docker.DockerClient, tasks: List[str], missing: Set[str]) -> List[Tuple[str, str]]:
    '''
    Install missing Docker images from tar.gz files and match them with tasks.

    Args:
    - client: Docker client object to interact with Docker daemon.
    - tasks: List of all required task names.
    - missing: List of task names that are missing (not yet installed).

    Returns:
    - images: List of images that match the required tasks after installation.
    '''

    print("  Installing missing Dockers:")

    for task in missing:
        # Set path to the zipped Docker file using os.path.join for cross-platform compatibility
        zipped_image = os.path.join('.', 'dockers', f"{task}.tar.gz")
        assert os.path.exists(zipped_image), f'{zipped_image} required but does not exist in your working directory.'

        # Load the Docker image from the tar.gz file
        try:
            print(f"Loading Docker image for task '{task}' from {zipped_image}...")
            subprocess.check_call([
                "docker", "load", "-i", zipped_image,
            ])
            print(f"Docker image for task '{task}' installed successfully.")
        except subprocess.CalledProcessError as e:
            print(f"Error occurred while installing Docker image '{task}': {e}")
            sys.exit(1)  # Exit if any script fails
        except Exception as e:
            print(f"An unexpected error occurred while installing Docker image '{task}': {e}")
            sys.exit(1)

    print()

    # Generate tasks based on available images
    images = match_images_to_tasks(client, tasks)

    return images

# Set run parameters for CAISR tasks
def set_run_parameters(data_folder: str, tasks: List[str]) -> None:
    '''
    Set run parameters for CAISR tasks and save them to a CSV file.

    Args:
    - data_folder: Path to the folder where the run parameters CSV will be saved.
    - tasks: List of task names for which the run parameters are being set.
    '''

    # Set location to save run parameters using os.path.join for cross-platform compatibility
    param_folder = os.path.join(data_folder, 'run_parameters')
    os.makedirs(param_folder, exist_ok=True)

    # Iterate over all tasks and create a run_parameter .csv file for each task
    for task in tasks:
        params_df = pd.DataFrame()
        
        # Set default parameters
        params_df.loc[0, 'overwrite'] = True

        # Task-specific parameters
        if task == 'preprocess':
            params_df.loc[0, 'overwrite'] = False
            params_df.loc[0, 'autoscale_signals'] = True
        elif task == 'resp':
            params_df.loc[0, 'multiprocess'] = True

        # Construct the full path to save the CSV
        csv_path = os.path.join(param_folder, f'{task}.csv')

        # Save parameters to the CSV file
        params_df.to_csv(csv_path, index=False, mode='w+')

# Run a Python script inside a Docker container
def run_python_script_in_docker(image_name: str, task: str, data_folder: str, caisr_output_folder: str) -> None:
    '''
    Runs a Python script inside a Docker container using the specified image and mounts directories for input and output.

    Args:
    - image_name: Name of the Docker image to be used for running the script.
    - task: Name of the task being executed, used for logging purposes.
    - data_folder: Local directory path to be mounted as the data input folder inside the Docker container.
    - caisr_output_folder: Local directory path to be mounted as the output folder inside the Docker container.
    '''
    # Normalize paths for cross-platform compatibility
    data_folder = os.path.abspath(data_folder)
    caisr_output_folder = os.path.abspath(caisr_output_folder)

    # Adjust paths for Windows compatibility
    if platform.system().lower() == 'windows':
        data_folder = data_folder.replace('\\', '/').replace(':', '')
        caisr_output_folder = caisr_output_folder.replace('\\', '/').replace(':', '')
        data_mount = f"/{data_folder[0]}{data_folder[1:]}:/data/data/"
        output_mount = f"/{caisr_output_folder[0]}{caisr_output_folder[1:]}:/data/caisr_output/"
        prompt = ["docker", "run", "--rm", "-v", data_mount, "-v", output_mount, image_name]
    else:
        data_mount = f"{data_folder}:/data/data/"
        output_mount = f"{caisr_output_folder}:/data/caisr_output/"
        prompt = ["docker", "run", "-it", "--rm", "-v", data_mount, "-v", output_mount, image_name]
        
    # Print info about the mounts and running Docker command
    print(f"\n--> Booting Docker '{task}' with image '{image_name}'")

    try:
        # Run CAISR task on all files in the mounted data folder
        subprocess.check_call(prompt)
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while executing '{task}' in Docker: {e}")
        sys.exit(1)  # Exit if any script fails
    except Exception as e:
        print(f"An unexpected error occurred while running '{task}' in Docker: {e}")
        sys.exit(1)

# Cleanup Docker image and associated containers
def cleanup_docker_image(client: docker.DockerClient, image_name: str) -> None:
    """Remove containers and force remove the image."""
    print(f"\nCleaning up containers and image '{image_name}'...")
    try:
        containers = client.containers.list(all=True, filters={"ancestor": image_name})
        for container in containers:
            print(f"Stopping and removing container '{container.name}' using image '{image_name}'...")
            container.stop()
            container.remove()
            print(f"Successfully removed container '{container.name}'")

        # Force remove the image
        image = client.images.get(image_name)
        client.images.remove(image.id, force=True)
        print(f"Successfully removed image '{image_name}'")
    except docker.errors.ImageNotFound:
        print(f"No image named '{image_name}' found.")
    except docker.errors.NotFound:
        print(f"No containers found using image '{image_name}'.")
    except Exception as e:
        print(f"Error occurred while cleaning up image '{image_name}': {e}")



if __name__ == '__main__':
    # TODO: select all tasks, you would like to run. (e.g. remove preprocessing)
    tasks = ['preprocess', 'stage', 'arousal', 'resp', 'limb', 'report']

    # TODO: set paths to input/output folder
    data_folder = os.path.join(os.getcwd(), 'data')  
    caisr_output_folder = os.path.join(os.getcwd(), 'caisr_output') 

    # Initialize Docker client and show available Docker images
    client = docker.from_env()
    list_images(client)

    # Generate tasks to available Docker images
    images = match_images_to_tasks(client, tasks)

    # Set run parameters for all tasks
    set_run_parameters(data_folder, tasks)

    # Run over each CAISR task and Docker image
    for task, (_, image_name) in zip(tasks, images):
        # Create task-specific output folder for intermediate results
        if task in ['stage', 'arousal', 'resp', 'limb']:
            intermediate_folder = os.path.join(caisr_output_folder, 'intermediate', task)
            os.makedirs(intermediate_folder, exist_ok=True)

        # Run the CAISR task inside its respective Docker container
        run_python_script_in_docker(image_name, task, data_folder, caisr_output_folder)

        # Ensure the current user has permission (only on Linux)
        if platform.system().lower() == 'linux':
            try:
                subprocess.run(['sudo', 'chown', '-R', f'{os.getlogin()}:{os.getlogin()}', data_folder], check=True)
                subprocess.run(['sudo', 'chown', '-R', f'{os.getlogin()}:{os.getlogin()}', caisr_output_folder], check=True)
            except subprocess.CalledProcessError as e:
                print(f"Error changing permissions: {e}")

        # Optionally, clean up any existing containers and images related to this task
        # Uncomment the following line if you wish to clean up Docker images/containers after each task to save memory
        # cleanup_docker_image(client, image_name)

    print(f"Completed running all specified tasks: {tasks}")

