import subprocess
import threading
import time
import os
import signal


ENV_DICT = {
    "BrushifyUrban": "ENVS/TRAIN_ENVS/BrushifyUrban/BrushifyUrban.sh",
    "CabinLake": "ENVS/TRAIN_ENVS/CabinLake/CabinLake.sh",
    "CityPark": "ENVS/TRAIN_ENVS/CityPark/CityPark.sh",
    "DownTown": "ENVS/TRAIN_ENVS/DownTown/DownTown1.sh",
    "Neighborhood": "ENVS/TRAIN_ENVS/Neighborhood/NewNeighborhood.sh",
    "Slum": "ENVS/TRAIN_ENVS/Slum/slum1.sh",
    "UrbanJapan": "ENVS/TRAIN_ENVS/UrbanJapan/UrbanJapan.sh",
    "Venice": "ENVS/TRAIN_ENVS/Venice/vinice_new1.sh",
    "WesternTown": "ENVS/TRAIN_ENVS/WesternTown/WesternTown1.sh",
    "WinterTown": "ENVS/TRAIN_ENVS/WinterTown/WinterTown1.sh",
}

class AirSimRunner:
    def __init__(self,env_name):
        self.processes = {}
        exe_path = ENV_DICT[env_name]
        self.base_command = [
            "bash", 
            exe_path,
            "-RenderOffscreen",
            "-NoSound", 
            "-NoVSync"
        ]
        
    def run_single_env(self, gpu_id, settings_file, thread_id):
        """Run a single AirSim environment."""
        command = self.base_command + [
            f"-GraphicsAdapter={gpu_id}",
            f"--settings={settings_file}"
        ]
        print(f"Thread {thread_id}: Starting with GPU {gpu_id}, settings: {settings_file}")
        
        try:
            process = subprocess.Popen(
                command,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                preexec_fn=os.setsid
            )
            
            self.processes[thread_id] = process
            
            # Wait briefly and check startup status.
            time.sleep(5)
            
            if process.poll() is not None:
                stdout, stderr = process.communicate()
                print(f"Thread {thread_id}: Process ended unexpectedly")
                print(f"stdout: {stdout}")
                print(f"stderr: {stderr}")
            else:
                print(f"Thread {thread_id}: AirSim started successfully")
                # Keep the process running.
                process.wait()
                
        except Exception as e:
            print(f"Thread {thread_id}: Error starting AirSim: {e}")
    
    def run_multiple_envs(self, config_list):
        """
        Run multiple AirSim environments in parallel.
        Args:
            config_list: config entries, each as (gpu_id, settings_file_path)
        """
        threads = []
        
        for i, (gpu_id, settings_file) in enumerate(config_list):
            thread = threading.Thread(
                target=self.run_single_env,
                args=(gpu_id, settings_file, i),
                daemon=True
            )
            threads.append(thread)
            thread.start()
            # Stagger startup to avoid resource contention.
            time.sleep(2)
        # Wait for all threads.
        try:
            for thread in threads:
                thread.join()
        except KeyboardInterrupt:
            print("Received interrupt signal, shutting down...")
            self.cleanup()
    
    def cleanup(self):
        """Clean up all processes."""
        for thread_id, process in self.processes.items():
            try:
                if process.poll() is None:
                    print(f"Terminating process for thread {thread_id}")
                    os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                    process.wait(timeout=5)
            except:
                try:
                    os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                except:
                    pass

# Usage example.
if __name__ == "__main__":
    
    import argparse
    parser = argparse.ArgumentParser(description="env name")
    parser.add_argument('--env', type=str, default='env_airsim_16',required=True, help="input env name")
    args = parser.parse_args()
    runner = AirSimRunner(env_name=args.env)
    # Config entries: (GPU_ID, settings file path)
    configs = [
        # (0, "/home/zzz/code/UAV_ON/tools/configs/port_settings/settings_30100.json"),
        # (1, "/home/zzz/code/UAV_ON/tools/configs/port_settings/settings_30101.json"),
        # (2, "/home/zzz/code/UAV_ON/tools/configs/port_settings/settings_30102.json"),
        # (3, "/home/zzz/code/UAV_ON/tools/configs/port_settings/settings_30103.json"),
        (0, "/home/zzz/code/UAV_ON/tools/configs/port_settings/settings_30104.json"),
        (1, "/home/zzz/code/UAV_ON/tools/configs/port_settings/settings_30105.json"),
        (2, "/home/zzz/code/UAV_ON/tools/configs/port_settings/settings_30106.json"),
        (3, "/home/zzz/code/UAV_ON/tools/configs/port_settings/settings_30107.json"),
        
        # (3, "/home/zzz/code/UAV_ON/tools/configs/port_settings/settings_30108.json"),
    ]
    try:
        runner.run_multiple_envs(configs)
    except KeyboardInterrupt:
        print("Shutting down all environments...")
        runner.cleanup()
