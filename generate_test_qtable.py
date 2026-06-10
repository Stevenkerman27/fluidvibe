import numpy as np
import os

def generate_test_q_table():
    """
    Creates a Q-table with the following logic:
    - Neg Vorticity (States 0-3): Action 2 (Left) is best, Action 1 (Up) is second best.
    - Weak Vorticity (States 4-7): Action 1 (Up) is best.
    - Pos Vorticity (States 8-11): Action 0 (Right) is best, Action 1 (Up) is second best.
    
    Actions:
    0: Right, 1: Up, 2: Left, 3: Down
    """
    # 12 states, 4 actions
    q = np.zeros((12, 4))
    
    # Base values for "Other actions all upward"
    # This ensures if the primary action isn't chosen, Up is the fallback.
    # And for Weak Vorticity, Up is the primary.
    
    # 1. 默认所有状态的首选动作均为“向上”(Action 1)
    q[:, 1] = 10.0
    
    # 2. 在“向上移动”的状态下 (y-dominant, vy > 0)，根据涡量调整：
    # 根据 TaylorGreenEnvironment._get_observation:
    # 状态 1: 负涡量 + 向上移动
    # 状态 5: 弱涡量 + 向上移动
    # 状态 9: 正涡量 + 向上移动
    
    # 负涡量且向上移动 -> 左转 (Action 2)
    q[1, 2] = 15.0  # 给予更高分值以覆盖默认的“向上”
    
    # 正涡量且向上移动 -> 右转 (Action 0)
    q[9, 0] = 15.0
    
    # 其他状态（包括弱涡量向上移动的状态 5）均保持“向上”(Action 1) 为最高分值
    
    # Save the Q-table
    save_dir = "q_table"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    save_path = os.path.join(save_dir, "test_policy_vorticity.npy")
    np.save(save_path, q)
    print(f"Manual test Q-table saved to: {save_path}")
    
    # Print the table for verification
    print("\nGenerated Q-Table (12x4):")
    print("State | Right(0) | Up(1) | Left(2) | Down(3)")
    print("-" * 45)
    for i in range(12):
        print(f" {i:2d}   |  {q[i,0]:4.1f}    | {q[i,1]:4.1f}  |  {q[i,2]:4.1f}   |  {q[i,3]:4.1f}")

if __name__ == "__main__":
    generate_test_q_table()
