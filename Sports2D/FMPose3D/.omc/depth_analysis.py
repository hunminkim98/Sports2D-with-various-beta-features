"""Analyze depth (z-axis) stability of 3D pose predictions."""
import numpy as np
import os
import glob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# H36M joint names
JOINT_NAMES = [
    "Hip", "RHip", "RKnee", "RFoot", "LHip", "LKnee", "LFoot",
    "Spine", "Thorax", "Neck", "Head", "LShoulder", "LElbow",
    "LWrist", "RShoulder", "RElbow", "RWrist"
]

pose_dir = r"c:\Users\gnsal\OneDrive\Desktop\Projects\FMPose3D\demo\predictions\test1\pose3D"
out_dir = r"c:\Users\gnsal\OneDrive\Desktop\Projects\FMPose3D\demo\predictions\test1"

files = sorted(glob.glob(os.path.join(pose_dir, "*_3D.npz")))
print(f"Found {len(files)} pose files")

poses = []
for f in files:
    data = np.load(f)
    poses.append(data['pose3d'])

poses = np.array(poses)  # (T, 17, 3)
T, J, _ = poses.shape
print(f"Shape: {poses.shape}")

z = poses[:, :, 2]  # (T, 17)

# === 1. Frame-to-frame z jitter ===
z_diff = np.abs(np.diff(z, axis=0))  # (T-1, 17)
mean_jitter = z_diff.mean(axis=0)
max_jitter = z_diff.max(axis=0)
std_jitter = z_diff.std(axis=0)

print("\n" + "="*70)
print("FRAME-TO-FRAME Z JITTER (|z[t] - z[t-1]|)")
print("="*70)
print(f"{'Joint':<12} {'Mean':>8} {'Max':>8} {'Std':>8}")
print("-"*40)
for j in range(J):
    print(f"{JOINT_NAMES[j]:<12} {mean_jitter[j]:8.4f} {max_jitter[j]:8.4f} {std_jitter[j]:8.4f}")
print("-"*40)
print(f"{'OVERALL':<12} {mean_jitter.mean():8.4f} {max_jitter.mean():8.4f} {std_jitter.mean():8.4f}")

# === 2. Depth range per joint ===
z_range = z.max(axis=0) - z.min(axis=0)
print("\n" + "="*70)
print("Z-AXIS RANGE (max - min across all frames)")
print("="*70)
for j in range(J):
    print(f"{JOINT_NAMES[j]:<12} {z_range[j]:8.4f}")
print(f"{'OVERALL AVG':<12} {z_range.mean():8.4f}")

# === 3. Acceleration (2nd derivative) ===
z_accel = np.abs(np.diff(z, n=2, axis=0))  # (T-2, 17)
mean_accel = z_accel.mean(axis=0)
print("\n" + "="*70)
print("Z-AXIS ACCELERATION (|z[t+1] - 2*z[t] + z[t-1]|)")
print("="*70)
for j in range(J):
    print(f"{JOINT_NAMES[j]:<12} {mean_accel[j]:8.4f}")
print(f"{'OVERALL AVG':<12} {mean_accel.mean():8.4f}")

# === 4. Pose scale for reference ===
# Compute average body height (hip to head distance)
hip_to_head = np.linalg.norm(poses[:, 10, :] - poses[:, 0, :], axis=1)  # (T,)
avg_body = hip_to_head.mean()
print(f"\n{'='*70}")
print(f"REFERENCE: Avg body height (hip-to-head): {avg_body:.4f}")
print(f"Jitter as % of body height: {mean_jitter.mean() / avg_body * 100:.1f}%")
print(f"Z-range as % of body height: {z_range.mean() / avg_body * 100:.1f}%")
print(f"{'='*70}")

# === 5. Plot z time series ===
key_joints = [0, 8, 10, 3, 6, 13, 16]
fig, ax = plt.subplots(figsize=(14, 6))
for j in key_joints:
    ax.plot(z[:, j], label=JOINT_NAMES[j], alpha=0.8)
ax.set_xlabel("Frame")
ax.set_ylabel("Z coordinate")
ax.set_title("Z-axis (Depth) over Time — Key Joints")
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "depth_analysis_z_timeseries.png"), dpi=150)
plt.close()
print(f"\nSaved: depth_analysis_z_timeseries.png")

# === 6. Summary bar chart ===
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

x = np.arange(J)
ax1.bar(x, mean_jitter, color='steelblue', alpha=0.8)
ax1.set_xticks(x)
ax1.set_xticklabels(JOINT_NAMES, rotation=45, ha='right', fontsize=8)
ax1.set_ylabel("Mean |dz/dt|")
ax1.set_title("Per-Joint Frame-to-Frame Z Jitter")
ax1.grid(axis='y', alpha=0.3)

ax2.bar(x, z_range, color='coral', alpha=0.8)
ax2.set_xticks(x)
ax2.set_xticklabels(JOINT_NAMES, rotation=45, ha='right', fontsize=8)
ax2.set_ylabel("Z Range (max - min)")
ax2.set_title("Per-Joint Z Range Across All Frames")
ax2.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(out_dir, "depth_analysis_summary.png"), dpi=150)
plt.close()
print(f"Saved: depth_analysis_summary.png")
