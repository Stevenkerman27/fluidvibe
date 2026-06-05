# Design Doc: Visualization Update for DQN Policy

**Date**: 2026-06-05  
**Topic**: visualize-dqn-policy-update

## Goal
Enhance the readability and accessibility of the DQN policy visualization in `visualize_dqn_policy.py` by increasing font sizes and adopting a colorblind-friendly color palette.

## Proposed Changes

### 1. Color Palette (Okabe-Ito)
The current palette uses standard Red/Green/Blue/Yellow, which can be problematic for users with color vision deficiencies. We will switch to the Okabe-Ito palette:
- **Right (Action 0)**: `#0072B2` (Blue)
- **Up (Action 1)**: `#E69F00` (Orange)
- **Left (Action 2)**: `#009E73` (Bluish Green)
- **Down (Action 3)**: `#CC79A7` (Reddish Purple)

### 2. Typography and Font Sizes
To improve legibility, especially when plots are embedded in reports or presentations, the following font sizes will be increased:
- **Tick Labels**: 32 (up from 24)
- **Axis Labels**: 44 (up from 36)
- **Plot Title**: 32 (up from 24)
- **Legend Text**: 28 (up from 20)

### 3. Plot Layout
- Maintain the `8x9` figure size.
- Ensure `plt.tight_layout()` is used to prevent overlapping or clipped labels.
- The y-axis orientation labels ('Down', 'Right', 'Up', 'Left', 'Down') will inherit the increased tick label size.

## Success Criteria
- The generated plots (`.png` files) are easily readable without zooming.
- The 4 actions are distinguishable for individuals with common types of color blindness.
- No text elements are cut off at the edges of the image.

## Testing Strategy
- Run `visualize_dqn_policy.py` and manually inspect the output images in `q_table/plots_dqn_jax/` and `q_table/checkpoints/plots/`.
- Verify that the colors correctly correspond to the intended actions in the legend.
