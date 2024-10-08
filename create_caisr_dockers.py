import os, sys, docker, shutil, subprocess, platform, gzip
import pandas as pd
from typing import List, Tuple

# List all Docker images
def list_images(client: docker.DockerClient) -> None:
    '''
    List all Docker images including their tags.

    Args:
    - client: Docker client object to interact with the Docker daemon.
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
    - missing: List of task names for which no Docker image was found.
    '''

    matches, missing = [], []

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
                    matches.append(task)
                    task_found = True
                    break  # Break inner loop to avoid duplicate matches
                elif tag.split(':')[0] == task_image_name:
                    matches.append(task)
                    task_found = True
                    break  # Break inner loop to avoid duplicate matches
            
            if task_found:
                break  # Break outer loop if task image is found

        # If no image was matched, add to missing
        if not task_found:
            missing.append(task)

    return matches, missing

# Run a Python script inside a Docker container
def create_docker(Dockerfile: str, task: str) -> None:
    # Create Docker
    print(f"\n--> Creating docker '{task}' (this takes some time..)")
    try:
        # Determine the base command
        docker_command = ['docker', 'build', '-t', f'caisr_{task}', '.']

        # Add 'sudo' if running on Linux
        if platform.system().lower() == 'linux':
            docker_command.insert(0, 'sudo')
        
        # Run the Docker build command
        subprocess.check_call(docker_command)

    except subprocess.CalledProcessError as e:
        print(f"Error occurred while executing '{task}': {e}")
        sys.exit(1)  # Exit if any script fails
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        sys.exit(1)

# Compress and save Docker 
def compress_and_save_docker(save_folder: str, task: str) -> None:
    # Set output path using os.path.join for cross-platform compatibility
    os.makedirs(save_folder, exist_ok=True)
    dst = os.path.join(save_folder, f"caisr_{task}.tar.gz")
    if os.path.exists(dst):
        print(f"\n--> Compressed Docker '{task}' already saved)")
        return

    # Save Docker
    print(f"\n--> Compressing docker '{task}' (this takes a lot of time..)")
    try:
        temp_tar = dst[:-3]  # Temporary tar file path without .gz

        # Save Docker image to a tar file
        docker_save_command = ['docker', 'save', '-o', temp_tar, f'caisr_{task}:latest']
        
        if platform.system().lower() == 'linux':
            # If on Linux, add sudo to the command
            docker_save_command.insert(0, 'sudo')
        
        subprocess.run(docker_save_command, check=True)

        # Ensure the current user has permission (only on Linux)
        if platform.system().lower() == 'linux':
            try:
                subprocess.run(['sudo', 'chown', '-R', f'{os.getlogin()}:{os.getlogin()}', save_folder], check=True)
            except subprocess.CalledProcessError as e:
                print(f"Error changing permissions: {e}")

        # Compress the tar file to tar.gz
        with open(temp_tar, 'rb') as f_in:
            with gzip.open(dst, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)

        # Remove the temporary tar file
        os.remove(temp_tar)

        print(f"--> Docker image '{task}' compressed and saved successfully")

    except subprocess.CalledProcessError as e:
        print(f"Error compressing '{task}' Docker: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")



if __name__ == '__main__':
    # TODO: select all tasks, you would like to create a Docker for. 
    tasks = ['preprocess', 'stage', 'arousal', 'resp', 'limb', 'report']
    
    # Set folder where to save the compressed Docker images
    save_folder = os.path.join(os.getcwd(), 'dockers')
    os.makedirs(save_folder, exist_ok=True)

    # Initialize Docker client and show available Docker images
    client = docker.from_env()
    list_images(client)    

    # Check which Docker images are not installed yet
    matching_tasks, missing_tasks = match_images_to_tasks(client, tasks)

    # Run over all missing tasks
    for task in missing_tasks:
        dockerfile_src = os.path.join(os.getcwd(), task, 'Dockerfile')
        dockerfile_dst = os.path.join(os.getcwd(), 'Dockerfile')

        # Remove existing Dockerfile in the main folder if present
        if os.path.exists(dockerfile_dst):
            os.remove(dockerfile_dst)

        # Copy Dockerfile from the task folder to the main folder
        if os.path.exists(dockerfile_src):
            shutil.copyfile(dockerfile_src, dockerfile_dst)
        else:
            print(f"Error: Dockerfile for task '{task}' not found at {dockerfile_src}. Skipping this task.")
            continue

        # Create Docker image
        create_docker(dockerfile_dst, task)

        # Remove the copied Dockerfile from the main folder
        if os.path.exists(dockerfile_dst):
            os.remove(dockerfile_dst)

        # Compress and save Docker image
        compress_and_save_docker(save_folder, task)

    # Print a completion message
    print(f"Completed processing of missing Docker images: {missing_tasks}")

    # Run over all matched tasks
    for task in matching_tasks:
        # Check if compressed Docker is saved
        compress_and_save_docker(save_folder, task)