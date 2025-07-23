import cv2
import numpy as np
import os
from tqdm import tqdm


def process_image(img_path, target_size=(28, 28)):
    # 读取图像
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"图像读取失败: {img_path}")

    # 1. 转换为灰度图（保留原始色彩信息）
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2. 噪声抑制（保留边缘的中值滤波）
    denoised = cv2.medianBlur(gray, 3)

    # 3. 对比度增强（CLAHE）
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)

    # 4. 锐化处理（增强字符边缘）
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    sharpened = cv2.filter2D(enhanced, -1, kernel)

    # 5. 严格二值化（确保只有黑白两色）
    _, binary = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 6. 背景检测与强制转换（确保黑底白字）
    if np.mean(binary) > 127:  # 如果白色像素多于黑色
        binary = 255 - binary  # 反转为黑底白字

    # 7. 字符区域提取（去除多余边框）
    # 找到字符轮廓
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.zeros(target_size, dtype=np.uint8)  # 返回全黑图像

    # 获取包含所有轮廓的最小矩形
    all_points = np.vstack([contour.reshape(-1, 2) for contour in contours])
    x, y, w, h = cv2.boundingRect(all_points)

    # 提取纯字符区域（无边框）
    char_region = binary[y:y + h, x:x + w]

    # 8. 智能缩放（保持比例）
    scale = min(target_size[0] / h, target_size[1] / w)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(char_region, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    # 9. 居中放置到纯黑背景
    result = np.zeros(target_size, dtype=np.uint8)
    y_start = (target_size[0] - new_h) // 2
    x_start = (target_size[1] - new_w) // 2
    result[y_start:y_start + new_h, x_start:x_start + new_w] = resized

    # 10. 最终清理（确保纯黑白）
    result[result < 128] = 0  # 所有灰色转为纯黑
    result[result >= 128] = 255  # 所有浅色转为纯白

    return result


def batch_process(input_dir="annCh", output_dir="processed_chars"):
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 遍历所有字符类别文件夹
    char_folders = [d for d in os.listdir(input_dir)
                    if os.path.isdir(os.path.join(input_dir, d))]

    for char_dir in tqdm(char_folders, desc="处理字符类别"):
        char_path = os.path.join(input_dir, char_dir)
        output_char_dir = os.path.join(output_dir, char_dir)
        os.makedirs(output_char_dir, exist_ok=True)

        # 处理当前类别所有图片
        for img_name in os.listdir(char_path):
            if not img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                continue

            input_path = os.path.join(char_path, img_name)
            output_path = os.path.join(output_char_dir, img_name)

            try:
                processed = process_image(input_path)
                cv2.imwrite(output_path, processed)
            except Exception as e:
                print(f"处理失败 {img_name}: {str(e)}")


if __name__ == "__main__":
    print("=" * 60)
    print("专业车牌字符预处理系统 - 严格黑底白字版")
    print("=" * 60)
    batch_process()
    print("\n处理完成！输出目录: processed_chars")
    print("所有图片已转换为: 纯黑背景(0,0,0) + 纯白字符(255,255,255)")