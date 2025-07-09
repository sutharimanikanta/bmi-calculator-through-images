import os
import argparse
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
from torchvision import transforms
from network_weight import UNet
from network import UNet as HUNet
from draw_skeleton import create_colors, draw_skeleton

# Ensure reproducibility
np.random.seed(23)


def pad_to_square(image, target_size):
    h, w, _ = image.shape
    scale = target_size / max(h, w)
    resized = cv2.resize(
        image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR
    )
    h_new, w_new = resized.shape[:2]
    delta_h = target_size - h_new
    delta_w = target_size - w_new
    top, bottom = delta_h // 2, delta_h - delta_h // 2
    left, right = delta_w // 2, delta_w - delta_w // 2
    return np.pad(resized, ((top, bottom), (left, right), (0, 0)), mode="constant")


def load_image(image_path, resolution):
    img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB).astype(np.float32)
    img = pad_to_square(img, resolution)
    original = img.copy()
    img /= 255.0
    tensor = transforms.ToTensor()(img).unsqueeze(0)
    return tensor, original


def main():
    parser = argparse.ArgumentParser(
        description="Height and Weight Information from Unconstrained Images"
    )
    parser.add_argument(
        "-i", "--image", type=str, required=True, help="Path to input image"
    )
    parser.add_argument("-g", "--gpu", type=int, default=0, help="GPU id (default=0)")
    parser.add_argument(
        "-r", "--resolution", type=int, required=True, help="Resolution (e.g., 128)"
    )
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    # Load image
    assert args.image.lower().endswith(
        (".jpg", ".jpeg", ".png")
    ), "Use image formats: .jpg/.jpeg/.png"
    img_tensor, original_img = load_image(args.image, args.resolution)
    img_tensor = img_tensor.to(device)

    # Load models
    # model_h = HUNet(128).to(device)
    # model_w = UNet(128, 32, 32).to(device)

    # # model_h.load_state_dict(torch.load('models/model_ep_48.pth.tar', map_location=device)['state_dict'])
    # # model_w.load_state_dict(torch.load('models/model_ep_37.pth.tar', map_location=device)['state_dict'])
    # model_h.load_state_dict(torch.load('/content/drive/MyDrive/model_weights/Copy of model_ep_48.pth.tar', map_location=device)['state_dict'])
    # model_w.load_state_dict(torch.load('/content/drive/MyDrive/model_weights/Copy of model_ep_37.pth.tar', map_location=device)['state_dict'])
    model_h = HUNet(128)  # defined in network.py
    pretrained_model_h = torch.load(
        "/content/drive/MyDrive/model_weights/Copy of model_ep_48.pth.tar",
        map_location=device,
    )
    model_h.load_state_dict(pretrained_model_h["state_dict"], strict=False)
    model_w = UNet(128, 32, 32)  # from network_weight.py
    pretrained_model_w = torch.load(
        "/content/drive/MyDrive/model_weights/Copy of model_ep_37.pth.tar",
        map_location=device,
    )
    model_w.load_state_dict(pretrained_model_w["state_dict"], strict=False)

    model_w.eval()
    with torch.no_grad():
        m_p, j_p, _, w_p = model_w(img_tensor)

    model_h.eval()
    with torch.no_grad():
        _, _, h_p = model_h(img_tensor)

    # Output directories
    os.makedirs("out", exist_ok=True)

    # Outputs
    mask_out = m_p.argmax(1).squeeze().cpu().numpy()
    joint_out = j_p.argmax(1).squeeze().cpu().numpy()
    pred_joints = j_p.squeeze().cpu().numpy()

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask_out.astype("uint8")
    )
    colors = create_colors(30)
    joint_positions = []

    for i in range(1, num_labels):
        part_mask = (labels == i).astype(int)
        joint_heatmap = np.expand_dims(part_mask, axis=0) * pred_joints
        joints = [
            np.unravel_index(joint_heatmap[j].argmax(), joint_heatmap[j].shape)
            for j in range(1, 19)
        ]
        joint_positions.append(joints)

    # Create mask overlay
    mask_rgb = np.stack([255 * mask_out] * 3, axis=-1).astype(np.uint8)
    overlay = cv2.addWeighted(original_img.astype("uint8"), 0.55, mask_rgb, 0.45, 0)

    # Draw skeleton
    skeleton_img = draw_skeleton(overlay / 255.0, joint_positions, colors)

    base_name = os.path.splitext(os.path.basename(args.image))[0]
    out_mask = f"out/{base_name}.mask.png"
    out_joint = f"out/{base_name}.joint.png"
    out_skeleton = f"out/{base_name}.skeleton.png"
    out_info = f"out/{base_name}.info.txt"

    # Save outputs
    cv2.imwrite(out_mask, (255 * mask_out).astype(np.uint8))
    plt.imsave(out_joint, joint_out, cmap="jet")
    plt.imsave(out_skeleton, skeleton_img)

    with open(out_info, "w") as f:
        f.write(f"Image: {args.image}\n")
        f.write(f"Height: {100 * h_p.item():.1f} cm\n")
        f.write(f"Weight: {100 * w_p.item():.1f} kg\n")

    print(f"\nImage: {args.image}")
    print(f"Height: {100 * h_p.item():.1f} cm")
    print(f"Weight: {100 * w_p.item():.1f} kg")
    print("Mask and Joints saved in /out directory")


if __name__ == "__main__":
    main()
