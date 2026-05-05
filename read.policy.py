import numpy as np
import os

def read_policies():
    q_table_dir = 'q_table'
    if not os.path.exists(q_table_dir):
        print(f"Directory {q_table_dir} not found.")
        return

    files = [f for f in os.listdir(q_table_dir) if f.endswith('.npy')]
    files.sort()

    for file in files:
        file_path = os.path.join(q_table_dir, file)
        try:
            q_table = np.load(file_path)
            # The best policy is the action with the max Q-value for each state
            policy = np.argmax(q_table, axis=1).tolist()
            print(f"{file}:")
            print(f"policy = {policy}")
            print("-" * 20)
        except Exception as e:
            print(f"Error reading {file}: {e}")

if __name__ == "__main__":
    read_policies()
